import argparse
from pathlib import Path

from src.data.partition import (
    DATASET_DEFAULT_CLASSES_PER_USER,
    generate_partition_file,
)
from src.utils.get_logger import LoggerFactory

DATA_ROOT_DIR = Path(__file__).parent / "data"


logger = LoggerFactory.get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate dataset partitions")
    parser.add_argument("--dataset", type=str, default="CIFAR100")
    parser.add_argument("--num_users", type=int, default=10)
    parser.add_argument(
        "--partition_mode",
        type=str,
        default="pathological",
        choices=["iid", "pathological"],
    )
    parser.add_argument("--partition_seed", type=int, default=42)
    parser.add_argument("--client_classes", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def split_dataset(
    dataset="MNIST",
    num_users=10,
    partition_mode="pathological",
    partition_seed=42,
    client_classes=None,
    force=False,
):
    if client_classes is None and partition_mode == "pathological":
        client_classes = DATASET_DEFAULT_CLASSES_PER_USER[dataset]

    partition_file = generate_partition_file(
        dataset_name=dataset,
        num_users=num_users,
        partition_mode=partition_mode,
        data_root=DATA_ROOT_DIR,
        partition_seed=partition_seed,
        client_classes=client_classes,
        force=force,
    )
    logger.info(f"数据集 {dataset} 分割完毕！")
    logger.info(f"保存文件为 {partition_file}")


if __name__ == "__main__":
    args = parse_args()
    split_dataset(
        dataset=args.dataset,
        num_users=args.num_users,
        partition_mode=args.partition_mode,
        partition_seed=args.partition_seed,
        client_classes=args.client_classes,
        force=args.force,
    )
