import torch
import torch.multiprocessing as mp
import pickle
import time
import csv
import numpy as np
import copy
from typing import Optional
from pathlib import Path
from torch.utils.data import DataLoader, Subset, Dataset

from src.utils.get_logger import LoggerFactory
from src.models.student_cnn import StudentCNN
from src.models.teacher_cnn import TeacherCNN
from src.models.feature_adapter import FeatureAdapter
from src.data.dataset import FastTensorDataset, get_fast_transforms
from src.data.partition import ensure_partition_file, get_partition_tag
from src.trainers import (
    CachedKDTrainer,
    FedAvgTrainer,
)
from src.server.checkpoint import CheckpointState, CheckpointManager

logger = LoggerFactory.get_logger("FedServer")

# --- Worker 映射表 ---
TRAINER_MAP = {
    "logit_kd": CachedKDTrainer,
    "feature_kd": CachedKDTrainer,
    "hybrid_kd": CachedKDTrainer,
    "fedavg": FedAvgTrainer,
}

# --- 全局变量 (用于子进程 Copy-On-Write) ---
worker_dataset_train = None


# --- Worker 初始化函数 (必须在 Server 类外部或作为 staticmethod) ---
def init_worker(dataset_name, data_root):
    """
    子进程初始化函数。
    负责在每个 worker 进程中加载数据集到内存。
    由于 Linux 的 fork 机制，这里加载的数据在子进程间是共享内存的(Copy-On-Write)。
    """
    global worker_dataset_train
    # 获取针对 Tensor 的 transform (不含 ToTensor)
    train_trans, _ = get_fast_transforms(dataset_name)

    # 实例化 FastTensorDataset
    # 注意：这里重新构建路径，确保子进程能找到数据
    worker_dataset_train = FastTensorDataset(
        dataset_name,
        root=data_root,
        train=True,
        transform=train_trans,
    )


# --- 通用 Worker 处理函数 ---
def generic_update_handler(args):
    """
    通用的本地训练入口。
    args: (client_id, idxs, payload, config)
    """
    client_id, idxs, payload, config = args
    device = torch.device(config["device"])

    global worker_dataset_train
    if worker_dataset_train is None:
        raise RuntimeError("Worker dataset not initialized!")

    # 创建 DataLoader
    # Subset 非常轻量，只存储索引
    ldr_train = DataLoader(
        Subset(worker_dataset_train, idxs),
        batch_size=config["local_bs"],
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    # 1. 动态获取策略类
    strategy_name = config.get("strategy", "fedavg")
    if strategy_name not in TRAINER_MAP:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    TrainerClass = TRAINER_MAP[strategy_name]
    num_classes = config.get("num_classes", 10)

    # 2. 实例化 Trainer
    trainer = TrainerClass(config, device, client_id, ldr_train, num_classes)

    # 3. 加载权重
    trainer.load_weights(payload["global_state"], payload)

    # 4. 执行训练
    upload_pkg, losses, accs = trainer.train()

    # 5. 清理
    del trainer

    avg_loss = sum(losses) / len(losses) if losses else 0.0
    avg_acc = sum(accs) / len(accs) if accs else 0.0
    dataset_size = len(idxs)

    return upload_pkg, avg_loss, avg_acc, dataset_size


class FederatedServer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config["device"])
        self.strategy = config["strategy"]
        self.partition_tag = config.get("partition_tag") or get_partition_tag(config)
        self.config["partition_tag"] = self.partition_tag

        # 路径设置
        self.data_root = Path(config["data_root"])
        self.weights_dir = Path(config["weights_dir"])
        self.results_dir = Path(config["results_dir"])
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # 1. 初始化模型
        self._init_models()

        self.dataset_test: Optional[Dataset] = None

        # 2. 加载数据分区
        self._load_partitions()

        self._init_test_dataset()
        self.csv_filename = None
        self.csv_path = None

    def _init_models(self):
        """初始化 Global Student, Teacher (可选), Adapter (可选)"""
        logger.info(f"Initializing models for strategy: {self.strategy}")

        # --- A. Global Student (所有策略都需要) ---
        self.net_glob = StudentCNN(num_classes=self.config["num_classes"]).to(
            self.device
        )
        self.w_glob = self.net_glob.state_dict()

        # --- B. Teacher (LogitKD, FeatureKD, Hybrid 需要) ---
        self.w_teacher = None
        if self.strategy in ["logit_kd", "feature_kd", "hybrid_kd"]:
            teacher_path = (
                self.weights_dir
                / f"{self.config['dataset'].lower()}_teacher_cnn_best.pth"
            )
            net_teacher = TeacherCNN(num_classes=self.config["num_classes"]).to(
                self.device
            )
            if teacher_path.exists():
                logger.info(f"Loading Teacher weights from {teacher_path}")
                net_teacher.load_state_dict(
                    torch.load(teacher_path, map_location=self.device)
                )
            else:
                logger.warning(
                    f"Teacher weights not found at {teacher_path}. Using RANDOM weights."
                )
            # 转为 CPU 字典备用，避免多进程传递 CUDA Tensor 报错
            self.w_teacher = {k: v.cpu() for k, v in net_teacher.state_dict().items()}
            del net_teacher

        # --- C. Adapter (FeatureKD, Hybrid 需要) ---
        self.w_adapter = None
        if self.strategy in ["feature_kd", "hybrid_kd"]:
            logger.info("Initializing Feature Adapter...")
            net_adapter = FeatureAdapter(
                self.config["student_channels"], self.config["teacher_channels"]
            ).to(self.device)
            self.w_adapter = {k: v.cpu() for k, v in net_adapter.state_dict().items()}
            del net_adapter

    def _load_partitions(self):
        """自动生成并加载数据分区"""
        partition_file = ensure_partition_file(self.config, self.data_root)

        logger.info(f"Loading partitions from {partition_file}")
        with open(partition_file, "rb") as f:
            self.dict_users = pickle.load(f)

    def _experiment_prefix(self):
        return (
            f"{self.strategy}_{self.config['dataset']}_{self.partition_tag}_"
            f"seed{self.config['seed']}"
        )

    def _init_test_dataset(self):
        """加载测试集用于 Server 端评估"""

        _, trans_test = get_fast_transforms(self.config["dataset"])
        self.dataset_test = FastTensorDataset(
            self.config["dataset"],
            root=self.data_root,
            train=False,
            transform=trans_test,
        )

    def _init_csv_logger(self):
        if self.csv_path is not None:
            return

        timestamp = time.strftime("%Y%m%d-%H%M%S")

        # 优化文件名：包含关键蒸馏参数
        param_suffix = (
            f"T{self.config['kd_T']}_ka{self.config['kd_alpha']}_"
            f"fa{self.config['feat_alpha']}"
        )
        self.csv_filename = (
            f"log_{self._experiment_prefix()}_rounds{self.config['rounds']}_"
            f"hybrid{self.config['hybrid_bata']}_{param_suffix}_{timestamp}.csv"
        )

        self.csv_path = self.results_dir / self.csv_filename

        logger.info(f"Results will be saved to: {self.csv_path}")
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
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

    def restore_from_checkpoint(self, checkpoint: CheckpointState):
        """从检查点恢复服务器状态"""
        logger.info(f"Restoring from checkpoint: round {checkpoint.round_idx + 1}")

        # 恢复全局模型权重
        self.net_glob.load_state_dict(checkpoint.w_glob)
        self.w_glob = self.net_glob.state_dict()

        # 恢复 Adapter 权重（如果存在）
        if checkpoint.w_adapter is not None:
            if self.w_adapter is None:
                raise RuntimeError(
                    f"Checkpoint contains adapter weights but current config "
                    f"({self.config['strategy']}) does not use adapter."
                )
            self.w_adapter = checkpoint.w_adapter

        # 验证 CSV 文件存在且格式正确
        csv_path = Path(checkpoint.csv_path)
        if not csv_path.exists():
            logger.warning(f"Checkpoint CSV not found at {csv_path}. Recreating...")
            self._init_csv_logger()
        else:
            self.csv_path = csv_path
            # 验证CSV格式
            with open(self.csv_path, "r") as f:
                reader = csv.reader(f)
                header = next(reader)
                expected_header = [
                    "Round",
                    "Client_ID",
                    "Train_Loss",
                    "Train_Acc",
                    "Eval_Loss",
                    "Eval_Acc",
                    "Time_Sec",
                ]
                if header != expected_header:
                    raise ValueError(
                        f"CSV header mismatch: {header} vs {expected_header}"
                    )

        logger.info("Checkpoint restoration complete.")

    def get_resume_round(self, checkpoint: CheckpointState) -> int:
        """获取恢复后的起始轮次"""
        return checkpoint.round_idx + 1

    def _checkpoint_path(self, checkpoint_dir: Path) -> Path:
        """为当前实验返回唯一的检查点路径。"""
        return checkpoint_dir / f"checkpoint_{self._experiment_prefix()}.pth"

    def save_checkpoint(self, round_idx: int, checkpoint_dir: Path):
        """保存当前轮次的检查点"""
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 同一实验始终覆盖同一个 checkpoint，避免每轮生成新文件。
        checkpoint_path = self._checkpoint_path(checkpoint_dir)

        # 创建并保存检查点状态
        state = CheckpointState.from_server(self, round_idx)
        state.save(checkpoint_path)

        logger.info(f"Checkpoint saved: {checkpoint_path}")

        return checkpoint_path

    def evaluate(self):
        """Server 端评估函数"""
        if self.dataset_test is None:
            logger.warning("Test dataset not initialized. Skipping evaluation.")
            return 0.0, 0.0

        self.net_glob.eval()
        test_loader = DataLoader(self.dataset_test, batch_size=128, shuffle=False)
        criterion = torch.nn.CrossEntropyLoss()
        correct = 0
        total = 0
        loss_sum = 0.0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.net_glob(images)
                loss = criterion(outputs, labels)

                loss_sum += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100.0 * correct / total
        avg_loss = loss_sum / total
        return accuracy, avg_loss

    def fed_avg(self, w_locals, dataset_sizes):
        """加权平均聚合算法"""
        total_size = sum(dataset_sizes)
        w_avg = copy.deepcopy(w_locals[0])

        for k in w_avg.keys():
            w_avg[k] = w_avg[k] * (dataset_sizes[0] / total_size)

        for i in range(1, len(w_locals)):
            weight = dataset_sizes[i] / total_size
            for k in w_avg.keys():
                w_avg[k] += w_locals[i][k] * weight
        return w_avg

    def run(self, resume_checkpoint: Optional[Path] = None):
        """
        执行联邦学习训练循环。

        Args:
            resume_checkpoint: 如果提供，从该检查点恢复训练
        """
        logger.info(f"Start Federated Learning ({self.strategy})...")
        training_start_time = time.time()

        # === 检查点恢复逻辑 ===
        start_round = 0
        if resume_checkpoint is not None:
            checkpoint = CheckpointState.load(resume_checkpoint)

            # 验证兼容性
            compat_warnings = CheckpointManager.validate_checkpoint(
                checkpoint, self.config
            )
            for warn in compat_warnings:
                logger.warning(warn)

            if checkpoint.round_idx >= self.config["rounds"]:
                logger.info("Checkpoint already at final round. Nothing to do.")
                return

            # 恢复状态
            self.restore_from_checkpoint(checkpoint)
            start_round = self.get_resume_round(checkpoint)

            logger.info(f"Resuming from round {start_round}/{self.config['rounds']}")

        if self.csv_path is None:
            self._init_csv_logger()

        # 启动多进程 Pool
        # 注意: initargs 传入 data_root 确保子进程能找到路径
        with mp.Pool(
            processes=self.config["num_workers"],
            initializer=init_worker,
            initargs=(self.config["dataset"], self.data_root),
        ) as pool:
            for round_idx in range(start_round, self.config["rounds"]):
                round_start_time = time.time()

                # 1. 客户端采样
                m = max(int(self.config["frac"] * self.config["num_users"]), 1)
                idxs_users = np.random.choice(
                    range(self.config["num_users"]), m, replace=False
                )

                # 2. 构造 Payload (包含 Global State 和 Teacher State)
                # 关键：转为 CPU 传输，避免 CUDA IPC 错误
                w_glob_cpu = {k: v.cpu() for k, v in self.w_glob.items()}

                payload_template = {"global_state": w_glob_cpu}
                if self.w_teacher:
                    payload_template["teacher_state"] = self.w_teacher
                if self.w_adapter:
                    # 注意：Adapter 每一轮聚合后更新，所以这里传的是最新的 self.w_adapter
                    payload_template["adapter_state"] = self.w_adapter

                # 3. 构造任务列表
                tasks = []
                for idx in idxs_users:
                    user_idxs = self.dict_users[idx]
                    # 深拷贝 payload 以防止不同任务间修改 (虽然这里是只读，但为了安全)
                    # 对于大数据，copy 开销较大，如果确定只读可以去掉 deepcopy
                    # 但 Python 的 dict 传参是引用，如果 worker 内部修改了 inplace 就糟了
                    # 这里的 payload 主要包含 state_dict，是 Tensor，不会被 worker 修改（worker load_state_dict 是复制值）
                    # 所以直接传 payload_template 是安全的。
                    tasks.append((idx, user_idxs, payload_template, self.config))

                # 4. 并行训练
                results = pool.map(generic_update_handler, tasks)

                # 5. 结果收集与聚合
                w_locals_model = []
                w_locals_adapter = []
                loss_locals = []
                acc_locals = []
                dataset_sizes = []

                for res_pkg, res_loss, res_acc, res_size in results:
                    w_locals_model.append(res_pkg["model"])
                    loss_locals.append(res_loss)
                    acc_locals.append(res_acc)
                    dataset_sizes.append(res_size)

                    if "adapter" in res_pkg:
                        w_locals_adapter.append(res_pkg["adapter"])

                # 聚合 Student 模型
                w_glob_new = self.fed_avg(w_locals_model, dataset_sizes)
                self.net_glob.load_state_dict(w_glob_new)
                self.w_glob = self.net_glob.state_dict()

                # 聚合 Adapter (如果存在)
                if self.w_adapter and w_locals_adapter:
                    w_adapter_new = self.fed_avg(w_locals_adapter, dataset_sizes)
                    self.w_adapter = w_adapter_new  # 更新 Server 端的 Adapter

                # 6. 评估与记录
                test_acc, test_loss = self.evaluate()

                duration = time.time() - round_start_time

                # 计算预计剩余时间 (ETA)
                elapsed_total = time.time() - training_start_time
                avg_time_per_round = elapsed_total / (round_idx + 1)
                remaining_rounds = self.config["rounds"] - (round_idx + 1)
                eta_seconds = avg_time_per_round * remaining_rounds

                logger.info(
                    f"Round {round_idx + 1}/{self.config['rounds']} | "
                    f"Time: {duration:.2f}s | "
                    f"ETA: {eta_seconds / 60:.2f} min | "
                    f"Test Acc: {test_acc:.2f}% | Test Loss: {test_loss:.4f}"
                )

                # 写入 CSV
                with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    for i, client_idx in enumerate(idxs_users):
                        writer.writerow(
                            [
                                round_idx + 1,
                                client_idx,
                                f"{loss_locals[i]:.4f}",
                                f"{acc_locals[i]:.2f}",
                                f"{test_loss:.4f}",
                                f"{test_acc:.2f}",
                                f"{duration:.2f}",
                            ]
                        )

                # 保存检查点
                checkpoint_every = self.config.get("checkpoint_every", 1)
                if (round_idx + 1) % checkpoint_every == 0:
                    checkpoint_dir = Path(
                        self.config.get("checkpoint_dir")
                        or self.results_dir / "checkpoints"
                    )
                    self.save_checkpoint(round_idx, checkpoint_dir)

        total_duration = time.time() - training_start_time
        logger.info(f"Training Finished. Total Time: {total_duration / 60:.2f} min")

        # 保存最终检查点，确保最终轮结果一定落盘。
        final_checkpoint_dir = Path(
            self.config.get("checkpoint_dir") or self.results_dir / "checkpoints"
        )
        final_checkpoint_path = self.save_checkpoint(
            self.config["rounds"] - 1, final_checkpoint_dir
        )
        logger.info(f"Final checkpoint saved: {final_checkpoint_path}")

        # 7. 保存最终模型
        param_suffix = (
            f"T{self.config['kd_T']}_ka{self.config['kd_alpha']}_"
            f"fa{self.config['feat_alpha']}"
        )
        save_name = (
            f"{self._experiment_prefix()}_rounds{self.config['rounds']}_"
            f"hybrid{self.config['hybrid_bata']}_{param_suffix}.pth"
        )
        torch.save(
            self.net_glob.state_dict(), self.results_dir / ("model_" + save_name)
        )

        if self.w_adapter:
            torch.save(self.w_adapter, self.results_dir / ("adapter_" + save_name))
