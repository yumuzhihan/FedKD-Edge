# src/configs/config.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

DEFAULT_CONFIG = {
    # --- 基础设置 ---
    "strategy": "fedavg",  # 策略: fedavg, logit_kd, feature_kd, hybrid_kd
    "dataset": "CIFAR10",  # CIFAR10, MNIST, FashionMNIST, CIFAR100
    "device": "cuda",  # cuda, cpu
    "seed": 42,  # 随机种子
    "num_workers": 10,  # 多进程数量
    "num_classes": 10,  # 类别数量
    "client_classes": 2,  # 每个客户端的类别数
    # --- 联邦学习设置 ---
    "rounds": 100,  # 总通讯轮次
    "num_users": 10,  # 客户端总数
    "frac": 1.0,  # 每轮采样比例
    "local_ep": 5,  # 本地训练 Epoch
    "local_bs": 64,  # 本地 Batch Size
    "lr": 0.01,  # 学习率
    "momentum": 0.9,  # SGD 动量
    # --- 蒸馏相关参数 (策略依赖) ---
    # Logit KD
    "kd_T": 1.0,  # 温度系数
    "kd_alpha": 0.5,  # Logit Loss 权重 (alpha * KD + (1-alpha) * CE)
    # Feature KD
    "feat_alpha": 1.0,  # Feature Loss 权重
    "student_channels": 64,  # Student 倒数第二层通道数
    "teacher_channels": 256,  # Teacher 倒数第二层通道数 (TeacherCNN conv4 输出)
    # Hybrid
    "hybrid_bata": 0.5,
    # --- 路径设置 (自动推导) ---
    "data_root": str(PROJECT_ROOT / "data"),
    "weights_dir": str(PROJECT_ROOT / "weights"),
    "results_dir": str(PROJECT_ROOT / "results" / "unified_logs"),
}
