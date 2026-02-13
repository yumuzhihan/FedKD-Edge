import argparse
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
from src.server.worker import FederatedServer
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

    # 3. 联邦参数
    parser.add_argument("--rounds", type=int, default=DEFAULT_CONFIG["rounds"])
    parser.add_argument("--num_users", type=int, default=DEFAULT_CONFIG["num_users"])
    parser.add_argument("--frac", type=float, default=DEFAULT_CONFIG["frac"])
    parser.add_argument("--local_ep", type=int, default=DEFAULT_CONFIG["local_ep"])
    parser.add_argument("--local_bs", type=int, default=DEFAULT_CONFIG["local_bs"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])

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

    # 6. hybrid 参数
    parser.add_argument(
        "--hybrid_bata",
        type=float,
        default=DEFAULT_CONFIG["hybrid_bata"],
        help="Weight for Hybrid KD Loss",
    )

    # 路径覆盖 (可选)
    parser.add_argument(
        "--results_dir", type=str, default=DEFAULT_CONFIG["results_dir"]
    )

    args = parser.parse_args()
    return args


def main():
    # 1. 获取参数
    args = parse_args()

    # 2. 合并配置 (命令行参数覆盖默认配置)
    config = DEFAULT_CONFIG.copy()
    config.update(vars(args))

    # 3. 设置随机种子
    set_seed(config["seed"])
    logger.info(f"Random Seed set to: {config['seed']}")

    # 4. 打印当前配置 (方便查日志)
    logger.info("----------- Configuration -----------")
    for k, v in config.items():
        logger.info(f"{k}: {v}")
    logger.info("-------------------------------------")

    # 5. 实例化并运行 Server
    try:
        server = FederatedServer(config)
        server.run()
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
