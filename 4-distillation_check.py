import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from pathlib import Path
import pandas as pd
import numpy as np
import random

from models.student_cnn import StudentCNN
from models.teacher_cnn import TeacherCNN
from utils.get_logger import LoggerFactory

logger = LoggerFactory.get_logger("distillation_check")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
EPOCHS = 30
LR = 0.05

TEMP = 1.0
ALPHA = 0.8

# RUN_BASELINE = True  # 是否运行 Baseline 组
RUN_BASELINE = False
# FIXED_BASELINE_ACC = 0.0  # 如果不运行 Baseline 组，则使用固定的准确率作为对比
FIXED_BASELINE_ACC = 76.44  # 30 epochs
# FIXED_BASELINE_ACC = 0.0  # 50 epochs
SEED = 42

DATA_PATH = Path(__file__).parent / "data"
WEIGHTS_DIR = Path(__file__).parent / "weights"
RESULTS_DIR = Path(__file__).parent / "results"
WEIGHTS_DIR.mkdir(
    parents=True, exist_ok=True
)  # 这里应该是一定存在的，因为需要读取教师模型权重文件
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 由于 MNIST 过于简单，故使用 CIFAR-10 进行蒸馏实验
DATASET_NAME = "CIFAR10"
TEACHER_MODEL_WEIGHTS = WEIGHTS_DIR / "cifar10_teacher_cnn_best.pth"  # 教师模型权重文件


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f">>> 随机种子已固定: {seed}")


class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
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


def distillation_loss(student_logits, teacher_logits, labels, T, alpha):
    """
    params:
    - student_logits: 学生模型的输出 (未经过 softmax)
    - teacher_logits: 教师模型的输出 (未经过 softmax)
    - labels: 真实标签 (Hard Label)
    - T: 温度参数，软化概率分布
    - alpha: 软损失的权重 (1-alpha 为硬损失权重)
    """
    # 1. 与真是标签损失 (Hard Loss)
    hard_loss = F.cross_entropy(student_logits, labels)

    # 2. 与教师 logits 损失 (Soft Loss)
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction="batchmean",
    ) * (T * T)

    return alpha * soft_loss + (1.0 - alpha) * hard_loss


def get_dataloaders():
    # 蒸馏验证不需要太强的数据增强，基础的即可，为了控制变量
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            Cutout(n_holes=1, length=8),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=DATA_PATH, train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root=DATA_PATH, train=False, download=True, transform=transform_test
    )

    trainloader = DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    testloader = DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )
    return trainloader, testloader


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total


def run_training(
    student_model,
    teacher_model=None,
    mode="baseline",
    save_path=None,
    model_save_path=None,
):
    """
    mode: 'baseline' (只用 GT 训练) 或 'distill' (用 KD 训练)
    """
    optimizer = optim.SGD(
        student_model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    train_loader, test_loader = get_dataloaders()

    logger.info(f"\n>>> 开始训练: 模式 [{mode.upper()}]")

    best_acc = 0.0
    history_data = []

    for epoch in range(EPOCHS):
        student_model.train()
        train_loss = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            student_logits = student_model(inputs)

            if mode == "distill" and teacher_model is not None:
                # 获取老师的软标签
                with torch.no_grad():
                    teacher_logits = teacher_model(inputs)

                loss = distillation_loss(
                    student_logits, teacher_logits, labels, TEMP, ALPHA
                )
            else:
                # 普通训练
                loss = F.cross_entropy(student_logits, labels)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # 每 Epoch 评估一次
        acc = evaluate(student_model, test_loader)
        if acc > best_acc:
            best_acc = acc
            if model_save_path:
                torch.save(
                    student_model.state_dict(), model_save_path
                )  # 保存最佳模型权重

        avg_loss = train_loss / len(train_loader)

        # 记录数据
        history_data.append({"epoch": epoch + 1, "loss": avg_loss, "acc": acc})

        logger.info(
            f"Epoch {epoch+1}/{EPOCHS} | Acc: {acc:.2f}% | Best: {best_acc:.2f}% | Loss: {avg_loss:.4f}"
        )

    # 转换为 DataFrame 并保存
    if save_path:
        df = pd.DataFrame(history_data)
        df.to_csv(save_path, index=False)
        logger.info(f"训练记录已保存至: {save_path}")

    return best_acc


def main():
    logger.info(f"使用设备: {DEVICE}")

    # 0. 固定种子
    set_seed(SEED)

    # 1. 准备 Teacher 模型
    if not TEACHER_MODEL_WEIGHTS.exists():
        logger.error(f"错误: 找不到 Teacher 权重文件: {TEACHER_MODEL_WEIGHTS}")
        return

    logger.info("正在加载 Teacher 模型...")
    teacher = TeacherCNN(num_classes=10).to(DEVICE)
    # 加载权重
    state_dict = torch.load(TEACHER_MODEL_WEIGHTS, map_location=DEVICE)
    teacher.load_state_dict(state_dict)

    # 冻结 Teacher，开启 eval 模式
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    # 测试一下 Teacher 的准确率，确保加载对了
    _, test_loader = get_dataloaders()
    teacher_acc = evaluate(teacher, test_loader)
    logger.info(f"Teacher 模型基准准确率: {teacher_acc:.2f}%")

    # 2. 运行 Baseline (Student 自己学)
    if RUN_BASELINE:
        logger.info("\n-------------------------------------------")
        logger.info("对照组 1: Baseline (无蒸馏)")
        logger.info("-------------------------------------------")
        student_baseline = StudentCNN(num_classes=10).to(
            DEVICE
        )  # 初始化一个新的 Student
        torch.save(
            student_baseline.state_dict(), WEIGHTS_DIR / "student_init.pth"
        )  # 存储初始模型权重以备后续对比
        # 指定 Baseline 结果保存路径
        baseline_csv = RESULTS_DIR / f"distillation_baseline_{EPOCHS}.csv"
        acc_baseline = run_training(
            student_baseline,
            mode="baseline",
            save_path=baseline_csv,
            model_save_path=WEIGHTS_DIR / f"student_baseline_{EPOCHS}.pth",
        )
    else:
        logger.info("\n>>> 跳过 Baseline 组训练，使用固定准确率作为对比")
        student_baseline = None
        acc_baseline = FIXED_BASELINE_ACC

    # 3. 运行 Distillation (Student 跟老师学)
    logger.info("\n-------------------------------------------")
    logger.info("实验组 2: Knowledge Distillation (有蒸馏)")
    logger.info("-------------------------------------------")
    student_distill = StudentCNN(num_classes=10).to(DEVICE)  # 初始化一个新的 Student
    student_init_state = torch.load(
        WEIGHTS_DIR / "student_init.pth", map_location=DEVICE
    )
    student_distill.load_state_dict(student_init_state)  # 使用相同的初始权重

    # 指定 Distillation 结果保存路径
    distill_csv = RESULTS_DIR / f"distillation_distill-{ALPHA}-{TEMP}-{EPOCHS}.csv"
    acc_distill = run_training(
        student_distill, teacher_model=teacher, mode="distill", save_path=distill_csv
    )

    # 4. 总结
    logger.info("\n================ 结果汇总 ================")
    logger.info(f"Teacher Accuracy: {teacher_acc:.2f}%")
    logger.info(f"Student Baseline: {acc_baseline:.2f}%")
    logger.info(f"Student Distill : {acc_distill:.2f}%")
    logger.info(f"蒸馏提升 (Gain): {acc_distill - acc_baseline:+.2f}%")
    logger.info(f"ALPHA: {ALPHA}, TEMP: {TEMP}, EPOCHS: {EPOCHS}, LR: {LR}")


if __name__ == "__main__":
    main()
