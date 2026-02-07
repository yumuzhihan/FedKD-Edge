import sys
from pathlib import Path
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.amp import grad_scaler, autocast_mode
from pandas import DataFrame

sys.path.append("..")
from models.teacher_cnn import TeacherCNN
from utils.get_logger import LoggerFactory

# --- 配置参数 ---
BATCH_SIZE = 128
LEARNING_RATE = 0.1
NUM_CLASSES = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger = LoggerFactory.get_logger("train_teacher_cnn")
data_root_path = Path(__file__).parent / "data"

# 启用 CuDNN 的自动调优以提升性能
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


# Cutout 实现
import torch
import numpy as np


class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        # img 必须已经是 Tensor [C, H, W]
        h, w = img.size(1), img.size(2)

        mask = np.ones((h, w), dtype=np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.0

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


def mixup_data(x, y, alpha=1.0, device="cuda"):
    """返回混合后的输入, 两个对应的标签, 以及混合比例 lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """混合后的 Loss 计算公式"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_transforms(dataset_name="cifar10"):
    """获取数据预处理 transforms"""
    if dataset_name == "cifar10":
        # CIFAR-10 标准增强
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
                Cutout(n_holes=1, length=8),
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
        train_transform = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)
                ),
                Cutout(n_holes=1, length=8),
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
    else:
        raise ValueError("Unknown dataset")

    return train_transform, test_transform


def train(
    model,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    save_path,
    epochs=100,
):
    model.train()
    train_metrics = []

    # 初始化混合精度 Scaler
    scaler = grad_scaler.GradScaler()

    best_acc = 0.0

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(
                device, non_blocking=True
            )

            inputs, targets_a, targets_b, lam = mixup_data(
                inputs, labels, alpha=1.0, device=device
            )
            optimizer.zero_grad()

            with autocast_mode.autocast(device_type=device):
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

            # 使用 scaler 进行反向传播和步进
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 统计
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # 更新学习率
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total

        logger.info(
            f"Epoch [{epoch + 1}/{epochs}] | "
            f"Train Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}% | "
            f"LR: {current_lr:.6f}"
        )

        # 评估
        eval_loss, eval_acc = evaluate(model, test_loader, criterion, device)

        # 记录数据
        train_metrics.append(
            {
                "epoch": epoch + 1,
                "train_loss": epoch_loss,
                "train_accuracy": epoch_acc,
                "eval_loss": eval_loss,
                "eval_accuracy": eval_acc,
                "lr_rate": current_lr,
            }
        )

        # 保存策略：保存最佳模型和最后模型
        if eval_acc > best_acc:
            best_acc = eval_acc
            torch.save(
                model.state_dict(), save_path.with_name(f"{save_path.name}_best.pth")
            )

        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            torch.save(
                model.state_dict(), save_path.with_name(f"{save_path.name}_last.pth")
            )

    return train_metrics


def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(
                device, non_blocking=True
            )

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)  # 修正 Loss 计算
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    accuracy = 100.0 * correct / total

    logger.info(f"   >>> Eval Loss: {epoch_loss:.4f} | Eval Acc: {accuracy:.2f}%")

    model.train()  # 恢复训练模式
    return epoch_loss, accuracy


def run_experiment(dataset_name, model_name_suffix, epochs=100):
    logger.info(f"====== 开始训练任务: {dataset_name.upper()} ======")

    train_transform, test_transform = get_transforms(dataset_name)

    if dataset_name == "cifar10":
        DatasetClass = torchvision.datasets.CIFAR10
    else:
        DatasetClass = torchvision.datasets.MNIST

    train_ds = DatasetClass(
        root=data_root_path, train=True, download=True, transform=train_transform
    )
    test_ds = DatasetClass(root=data_root_path, train=False, transform=test_transform)

    # pin_memory=True 加速数据传输
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

    model = TeacherCNN(num_classes=NUM_CLASSES).to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 标签平滑，加强泛化能力
    optimizer = optim.SGD(
        model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4
    )

    # 使用 CosineAnnealingLR 替代 StepLR，收敛效果更好
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    save_dir = Path(__file__).parent / "weights" / f"{dataset_name}_{model_name_suffix}"
    save_dir.parent.mkdir(parents=True, exist_ok=True)

    results_dir = (
        Path(__file__).parent
        / "results"
        / f"{dataset_name}_{model_name_suffix}_results"
    )
    results_dir.parent.mkdir(parents=True, exist_ok=True)

    metrics = train(
        model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        scheduler,
        DEVICE,
        save_path=save_dir,
        epochs=epochs,
    )

    df = DataFrame(metrics)
    csv_name = (
        f"{results_dir.name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    df.to_csv(results_dir.with_name(csv_name), index=False)
    logger.info(f"训练完成，结果已保存至: {csv_name}\n")


def main():
    logger.info(f"使用计算设备: {DEVICE}")

    # 运行 CIFAR-10
    run_experiment("cifar10", "teacher_cnn", epochs=300)

    # 运行 MNIST
    run_experiment("mnist", "teacher_cnn", epochs=30)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("用户手动中断训练。")
    except Exception as e:
        logger.error(f"训练发生错误: {e}", exc_info=True)
