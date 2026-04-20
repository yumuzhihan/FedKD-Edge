import argparse
import csv
import os
import random
import numpy as np
import torch
import sys
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=np.exceptions.VisibleDeprecationWarning)
warnings.filterwarnings(
    "ignore", message="align should be passed as Python or NumPy boolean"
)

sys.path.append(str(Path(__file__).parent))

from src.configs.config import DEFAULT_CONFIG
from src.data.partition import get_partition_tag
from src.server.worker import FederatedServer
from src.server.checkpoint import CheckpointManager
from src.utils.get_logger import LoggerFactory

logger = LoggerFactory.get_logger("Main")


def set_seed(seed):
    """设置所有随机种子以保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Federated Learning with Knowledge Distillation"
    )

    # 1. 策略选择
    parser.add_argument(
        "--strategy",
        type=str,
        default=DEFAULT_CONFIG["strategy"],
        choices=["fedavg", "logit_kd", "feature_kd", "hybrid_kd"],
        help="FL Strategy to run",
    )

    # 2. 基础参数
    parser.add_argument("--dataset", type=str, default=DEFAULT_CONFIG["dataset"])
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--device", type=str, default=DEFAULT_CONFIG["device"])
    parser.add_argument(
        "--num_workers", type=int, default=DEFAULT_CONFIG["num_workers"]
    )
    parser.add_argument(
        "--num_classes", type=int, default=DEFAULT_CONFIG["num_classes"]
    )
    parser.add_argument(
        "--client_classes", type=int, default=DEFAULT_CONFIG["client_classes"]
    )
    parser.add_argument(
        "--partition_mode",
        type=str,
        default=DEFAULT_CONFIG["partition_mode"],
        choices=["iid", "pathological"],
        help="Dataset partition mode",
    )
    parser.add_argument(
        "--partition_seed",
        type=int,
        default=DEFAULT_CONFIG["partition_seed"],
        help="Random seed used for dataset partitioning",
    )
    parser.add_argument(
        "--partition_path",
        type=str,
        default=DEFAULT_CONFIG["partition_path"],
        help="Optional explicit partition file path",
    )

    # 3. 联邦参数
    parser.add_argument("--rounds", type=int, default=DEFAULT_CONFIG["rounds"])
    parser.add_argument("--num_users", type=int, default=DEFAULT_CONFIG["num_users"])
    parser.add_argument("--frac", type=float, default=DEFAULT_CONFIG["frac"])
    parser.add_argument("--local_ep", type=int, default=DEFAULT_CONFIG["local_ep"])
    parser.add_argument("--local_bs", type=int, default=DEFAULT_CONFIG["local_bs"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--momentum", type=float, default=DEFAULT_CONFIG["momentum"])

    # 4. 蒸馏参数 (Logit)
    parser.add_argument(
        "--kd_T",
        type=float,
        default=DEFAULT_CONFIG["kd_T"],
        help="Temperature for Logit KD",
    )
    parser.add_argument(
        "--kd_alpha",
        type=float,
        default=DEFAULT_CONFIG["kd_alpha"],
        help="Weight for Logit KD Loss",
    )

    # 5. 蒸馏参数 (Feature)
    parser.add_argument(
        "--feat_alpha",
        type=float,
        default=DEFAULT_CONFIG["feat_alpha"],
        help="Weight for Feature MSE Loss",
    )
    parser.add_argument(
        "--student_channels",
        type=int,
        default=DEFAULT_CONFIG["student_channels"],
        help="Student model channel size",
    )
    parser.add_argument(
        "--teacher_channels",
        type=int,
        default=DEFAULT_CONFIG["teacher_channels"],
        help="Teacher model channel size",
    )

    # 6. hybrid 参数
    parser.add_argument(
        "--hybrid_bata",
        type=float,
        default=DEFAULT_CONFIG["hybrid_bata"],
        help="Weight for Hybrid KD Loss",
    )

    # 路径覆盖 (可选)
    parser.add_argument("--data_root", type=str, default=DEFAULT_CONFIG["data_root"])
    parser.add_argument(
        "--weights_dir", type=str, default=DEFAULT_CONFIG["weights_dir"]
    )
    parser.add_argument(
        "--results_dir", type=str, default=DEFAULT_CONFIG["results_dir"]
    )

    # 检查点参数
    parser.add_argument(
        "--resume",
        type=str,
        default="auto",
        help="Path to checkpoint file to resume from. "
        "Use 'auto' to automatically find the latest matching checkpoint. "
        "Use 'none' to always start from scratch.",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=DEFAULT_CONFIG["checkpoint_every"],
        help="Overwrite the experiment checkpoint every N rounds.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=DEFAULT_CONFIG["checkpoint_dir"],
        help="Directory to save/load checkpoints.",
    )

    args = parser.parse_args()
    return args


def _build_result_csv_pattern(config):
    param_suffix = f"T{config['kd_T']}_ka{config['kd_alpha']}_fa{config['feat_alpha']}"
    experiment_prefix = (
        f"{config['strategy']}_{config['dataset']}_{config['partition_tag']}_"
        f"seed{config['seed']}"
    )
    return (
        f"log_{experiment_prefix}_rounds{config['rounds']}_"
        f"hybrid{config['hybrid_bata']}_{param_suffix}_*.csv"
    )


def _get_csv_max_round(csv_path):
    max_round = 0
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "Round" not in reader.fieldnames:
            logger.warning(f"CSV missing 'Round' column, ignoring: {csv_path}")
            return 0

        for row in reader:
            round_value = row.get("Round")
            if not round_value:
                continue
            try:
                max_round = max(max_round, int(round_value))
            except ValueError:
                logger.warning(
                    f"Invalid Round value '{round_value}' in CSV, ignoring: {csv_path}"
                )

    return max_round


def _find_matching_result_csv(config):
    results_dir = Path(config["results_dir"])
    if not results_dir.exists():
        return None

    csv_pattern = _build_result_csv_pattern(config)
    matched_csv_files = sorted(
        results_dir.glob(csv_pattern),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return matched_csv_files[0] if matched_csv_files else None


def should_skip_training(config):
    matched_csv = _find_matching_result_csv(config)
    if matched_csv is None:
        return False

    max_round = _get_csv_max_round(matched_csv)
    if max_round >= config["rounds"]:
        logger.info(
            f"Found completed result CSV: {matched_csv} (max round: {max_round}). "
            f"Requested rounds: {config['rounds']}. Skipping training."
        )
        return True

    return False


def main():
    # 1. 获取参数
    args = parse_args()

    # 2. 合并配置 (命令行参数覆盖默认配置)
    config = DEFAULT_CONFIG.copy()
    config.update(vars(args))
    if config["partition_seed"] is None:
        config["partition_seed"] = config["seed"]
    config["partition_tag"] = get_partition_tag(config)

    # 3. 设置随机种子
    set_seed(config["seed"])
    logger.info(f"Random Seed set to: {config['seed']}")

    # 4. 打印当前配置 (方便查日志)
    logger.info("----------- Configuration -----------")
    for k, v in config.items():
        logger.info(f"{k}: {v}")
    logger.info("-------------------------------------")

    if should_skip_training(config):
        return

    # 5. 实例化并运行 Server
    try:
        server = FederatedServer(config)

        # 处理检查点路径
        resume_path = None
        resume_mode = config.get("resume")
        if isinstance(resume_mode, str):
            resume_mode = resume_mode.strip()

        if resume_mode and str(resume_mode).lower() != "none":
            if str(resume_mode).lower() == "auto":
                # 自动查找最新检查点
                checkpoint_dir = Path(
                    config.get("checkpoint_dir")
                    or Path(config["results_dir"]) / "checkpoints"
                )
                resume_path = CheckpointManager.find_latest_checkpoint(
                    checkpoint_dir,
                    config["strategy"],
                    config["dataset"],
                    config["seed"],
                    config["partition_tag"],
                )
                if resume_path is None:
                    logger.warning(
                        "No matching checkpoint found. Starting from scratch."
                    )
                else:
                    logger.info(f"Auto-found checkpoint: {resume_path}")
            else:
                resume_path = Path(str(resume_mode))
                if not resume_path.exists():
                    raise FileNotFoundError(f"Checkpoint not found: {resume_path}")

        if resume_path is not None:
            resume_csv_path = _find_matching_result_csv(config)
            if resume_csv_path is not None:
                config["resume_csv_path"] = str(resume_csv_path)
                logger.info(
                    f"Resume will append results to existing CSV: {resume_csv_path}"
                )

        server.run(resume_checkpoint=resume_path)
    except Exception as e:
        logger.error("Training crashed!", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    # Windows 下多进程必须保护
    import torch.multiprocessing as mp

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    main()
