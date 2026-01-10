import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torch import optim
from torch.amp import grad_scaler, autocast_mode
from torchvision import datasets, transforms
import numpy as np
import copy
import pickle
from pathlib import Path
import torch.multiprocessing as mp
import os
import warnings
import csv
import time
import torch.nn.functional as F

# --- 警告过滤 ---
warnings.filterwarnings("ignore", category=np.exceptions.VisibleDeprecationWarning)

from models.student_cnn import StudentCNN
from utils.get_logger import LoggerFactory

CONFIG = {
    "dataset": "CIFAR10",
    "rounds": 200,
    "num_users": 10,
    "frac": 1.0,
    "local_ep": 5,
    "local_bs": 64,
    "lr": 0.01,
    "momentum": 0.9,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "num_workers": 10,
}

DATA_ROOT_DIR = Path(__file__).parent / "data"
PARTITION_DIR = DATA_ROOT_DIR / "partitions"
RESULTS_DIR = Path(__file__).parent / "results" / "fed_baseline"

logger = LoggerFactory.get_logger(__file__)

worker_dataset_train = None


class CachedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, root, train=True, transform=None):
        self.transform = transform
        dataset_name = dataset_name.lower()

        if dataset_name == "cifar10":
            base = datasets.CIFAR10(root, train=train, download=False)
            self.targets = torch.tensor(base.targets)
            # 将 numpy (N, H, W, C) -> Tensor (N, C, H, W) 并归一化到 [0, 1]
            # 这样训练时就不用重复做 ToTensor 了
            self.data = torch.tensor(base.data).permute(0, 3, 1, 2).float() / 255.0

        elif dataset_name == "mnist":
            base = datasets.MNIST(root, train=train, download=False)
            self.targets = base.targets
            # MNIST 原始是 (N, 28, 28)，需要扩充维度和 Resize
            # 预处理阶段一次性完成 Resize 和 Grayscale(3) 的工作
            data = base.data.float().unsqueeze(1) / 255.0  # (N, 1, 28, 28)
            # Resize to 32x32
            data = F.interpolate(
                data, size=(32, 32), mode="bilinear", align_corners=False
            )
            # 复制到 3 通道 (N, 3, 32, 32)
            self.data = data.repeat(1, 3, 1, 1)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)


def get_transforms(dataset_name="cifar10", tensor_input=False):
    """
    tensor_input: 如果为 True，表示输入已经是 Tensor (C, H, W) 格式，
                  不需要再做 ToTensor，且 Transforms 需要支持 Tensor 操作。
    """
    dataset_name = dataset_name.lower()
    train_transform = None
    test_transform = None

    if dataset_name == "cifar10":
        if tensor_input:
            # 针对 CachedDataset 的 Transform (输入已经是 Tensor)
            # 移除了 Cutout
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )
        else:
            # 针对原始 datasets.CIFAR10 的 Transform (输入是 PIL)
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )

        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

    elif dataset_name == "mnist":
        if tensor_input:
            # CachedDataset 已经做好了 Resize 和 Grayscale
            train_transform = transforms.Compose(
                [
                    transforms.Normalize(
                        (0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)
                    ),
                ]
            )
        else:
            train_transform = transforms.Compose(
                [
                    transforms.Resize(32),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)
                    ),
                ]
            )

        test_transform = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)
                ),
            ]
        )

    return train_transform, test_transform


# --- Worker 初始化函数 ---
def init_worker(dataset_name):
    """
    每个 Worker 进程启动时加载一次数据到内存 (CachedDataset)。
    """
    global worker_dataset_train

    # 获取针对 Tensor 输入的 transform (去掉了 ToTensor)
    train_trans, _ = get_transforms(dataset_name, tensor_input=True)

    # 使用 CachedDataset 替代原始 Dataset
    # 这里 download=False，因为主进程已经确保下载了
    worker_dataset_train = CachedDataset(
        dataset_name, DATA_ROOT_DIR, train=True, transform=train_trans
    )


# --- Worker 执行函数 ---
def local_update_handler(args):
    """
    args: tuple (client_id, data_indices, global_state_dict, config)
    """
    client_id, idxs, global_state_dict, config = args
    device = torch.device(config["device"])

    global worker_dataset_train
    if worker_dataset_train is None:
        raise RuntimeError("Worker dataset not initialized!")

    ldr_train = DataLoader(
        Subset(worker_dataset_train, idxs),
        batch_size=config["local_bs"],
        shuffle=True,
        # 优化：既然数据已经是内存 Tensor，关闭 pin_memory 可以减少 CPU 线程开销
        pin_memory=False,
    )

    model = StudentCNN()
    model.load_state_dict(global_state_dict)
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=config["lr"], momentum=config["momentum"]
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["local_ep"]
    )
    scaler = grad_scaler.GradScaler()

    epoch_loss = []
    epoch_acc = []

    for epoch in range(config["local_ep"]):
        batch_loss = []
        correct = 0
        total = 0

        for images, labels in ldr_train:
            images, labels = images.to(device, non_blocking=True), labels.to(
                device, non_blocking=True
            )
            optimizer.zero_grad()

            with autocast_mode.autocast(
                device_type=device.type, enabled=(device.type == "cuda")
            ):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_loss.append(loss.item())

            # 计算准确率
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()

        if batch_loss:
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        else:
            epoch_loss.append(0.0)

        if total > 0:
            epoch_acc.append(100.0 * correct / total)
        else:
            epoch_acc.append(0.0)

    final_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

    avg_loss = sum(epoch_loss) / len(epoch_loss) if epoch_loss else 0.0
    avg_acc = sum(epoch_acc) / len(epoch_acc) if epoch_acc else 0.0

    return final_state_dict, avg_loss, avg_acc, len(idxs)


def fed_avg(clients_weights, client_dataset_sizes):
    total_dataset_size = sum(client_dataset_sizes)
    w_avg = copy.deepcopy(clients_weights[0])

    for key in w_avg.keys():
        w_avg[key] = torch.zeros_like(w_avg[key], dtype=torch.float32)

    for i in range(len(clients_weights)):
        weight = client_dataset_sizes[i] / total_dataset_size
        for key in w_avg.keys():
            w_avg[key] += clients_weights[i][key] * weight

    return w_avg


def evaluate(model, dataset, device, batch_size=128):
    """
    评估全局模型，返回准确率和平均损失
    """
    model.eval()
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()

    correct = 0
    total = 0
    loss_sum = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss_sum += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    avg_loss = loss_sum / total

    return accuracy, avg_loss


def format_time(seconds):
    """辅助函数：将秒格式化为 MM:SS"""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{int(h)}h {int(m)}m {int(s)}s"
    return f"{int(m)}m {int(s)}s"


def main():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    device = torch.device(CONFIG["device"])
    logger.info(f"Using device: {device} | Parallel Workers: {CONFIG['num_workers']}")

    dataset_name = CONFIG["dataset"]
    # 主进程评估使用标准的 Dataset (PIL input)，所以这里 tensor_input=False
    _, trans_test = get_transforms(dataset_name, tensor_input=False)

    # 主进程加载测试集和确保数据下载
    if dataset_name == "CIFAR10":
        datasets.CIFAR10(DATA_ROOT_DIR, train=True, download=True)
        dataset_test = datasets.CIFAR10(
            DATA_ROOT_DIR, train=False, download=True, transform=trans_test
        )
    else:
        datasets.MNIST(DATA_ROOT_DIR, train=True, download=True)
        dataset_test = datasets.MNIST(
            DATA_ROOT_DIR, train=False, download=True, transform=trans_test
        )

    partition_file = (
        PARTITION_DIR / f"noniid_{dataset_name}_k{CONFIG['num_users']}_c2.pkl"
    )
    if not partition_file.exists():
        logger.error(f"Partition file not found: {partition_file}")
        return

    logger.info(f"Loading partitions from {partition_file}")
    with open(partition_file, "rb") as f:
        dict_users = pickle.load(f)

    net_glob = StudentCNN().to(device)
    net_glob.train()
    w_glob = net_glob.state_dict()

    # --- 准备 CSV 文件 ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    csv_filename = f"log_{dataset_name}_users{CONFIG['num_users']}_{timestamp}.csv"
    csv_path = RESULTS_DIR / csv_filename

    logger.info(f"Results will be saved to: {csv_path}")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Round",
                "Client_ID",
                "Train_Loss",
                "Train_Acc",
                "Eval_Loss",
                "Eval_Acc",
                "Time_Sec",
            ]
        )

    logger.info("Start Parallel Federated Learning Training...")

    total_rounds = CONFIG["rounds"]
    training_start_time = time.time()  # 记录总开始时间

    with mp.Pool(
        processes=CONFIG["num_workers"],
        initializer=init_worker,
        initargs=(dataset_name,),
    ) as pool:

        for round_idx in range(CONFIG["rounds"]):
            round_start_time = time.time()  # 记录每轮开始时间

            m = max(int(CONFIG["frac"] * CONFIG["num_users"]), 1)
            idxs_users = np.random.choice(range(CONFIG["num_users"]), m, replace=False)

            # logger.info(
            #     f"--- Round {round_idx+1}/{CONFIG['rounds']} | Selecting {m} Clients ---"
            # )

            w_glob_cpu = {k: v.cpu() for k, v in w_glob.items()}

            tasks = []
            for idx in idxs_users:
                user_idxs = dict_users[idx]
                tasks.append((idx, user_idxs, w_glob_cpu, CONFIG))

            results = pool.map(local_update_handler, tasks)

            w_locals = []
            loss_locals = []
            acc_locals = []
            selected_dataset_sizes = []

            for res_weights, res_loss, res_acc, res_size in results:
                w_locals.append(res_weights)
                loss_locals.append(res_loss)
                acc_locals.append(res_acc)
                selected_dataset_sizes.append(res_size)

            w_glob_new = fed_avg(w_locals, selected_dataset_sizes)
            net_glob.load_state_dict(w_glob_new)
            w_glob = net_glob.state_dict()

            test_acc, test_loss = evaluate(net_glob, dataset_test, device)
            avg_train_loss = sum(loss_locals) / len(loss_locals)

            # --- 计算时间和进度 ---
            round_end_time = time.time()
            round_duration = round_end_time - round_start_time
            total_elapsed = round_end_time - training_start_time

            # 计算 ETA (预计剩余时间)
            avg_time_per_round = total_elapsed / (round_idx + 1)
            remaining_rounds = total_rounds - (round_idx + 1)
            eta_seconds = avg_time_per_round * remaining_rounds

            # 绘制进度条 [====>......]
            bar_len = 20
            progress = (round_idx + 1) / total_rounds
            filled_len = int(bar_len * progress)
            bar = "=" * filled_len + ">" + "." * (bar_len - filled_len - 1)
            if filled_len == bar_len:
                bar = "=" * bar_len  # 完成时去掉箭头

            logger.info(
                f"Round {round_idx+1}/{total_rounds} [{bar}] "
                f"Time: {round_duration:.2f}s (ETA: {format_time(eta_seconds)}) | "
                f"Test Acc: {test_acc:.2f}% | "
                f"Test Loss: {test_loss:.4f}"
            )

            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for i, client_idx in enumerate(idxs_users):
                    c_loss = loss_locals[i]
                    c_acc = acc_locals[i]

                    writer.writerow(
                        [
                            round_idx + 1,
                            client_idx,
                            f"{c_loss:.4f}",
                            f"{c_acc:.2f}",
                            f"{test_loss:.4f}",
                            f"{test_acc:.2f}",
                            f"{round_duration:.2f}",  # 添加本轮耗时
                        ]
                    )

    total_duration = time.time() - training_start_time
    logger.info(f"Training Finished. Total Time: {format_time(total_duration)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(e, exc_info=True)
