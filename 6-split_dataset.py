from pathlib import Path
from torchvision import datasets
import numpy as np
import pickle

from src.utils.get_logger import LoggerFactory

K = 10
DATA_ROOT_DIR = Path(__file__).parent / "data"
PARTITION_DIR = DATA_ROOT_DIR / "partitions"

# 不同数据集的默认配置
DATASET_CONFIG = {
    "MNIST": {"classes_per_user": 2},
    "CIFAR10": {"classes_per_user": 2},
    "FashionMNIST": {"classes_per_user": 2},
    "CIFAR100": {
        "classes_per_user": 10
    },  # CIFAR-100 有 100 个类别，每个用户分配 10 个类别
}

PARTITION_DIR.mkdir(parents=True, exist_ok=True)


logger = LoggerFactory.get_logger(__name__)


def non_iid_split(dataset, num_users, classes_per_user):
    # 读取 Labels
    if isinstance(dataset, datasets.MNIST):
        labels = dataset.train_labels
    elif isinstance(dataset, datasets.CIFAR10):
        labels = dataset.targets
    elif isinstance(dataset, datasets.CIFAR100):
        labels = dataset.targets
    else:
        labels = np.array(dataset.targets)

    num_items = int(len(dataset) / num_users)
    dict_users = {i: np.array([], dtype="int64") for i in range(num_users)}

    idxs = np.arange(len(dataset))
    idxs_labels = np.vstack((idxs, labels))

    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    num_shards = num_users * classes_per_user
    num_imgs_per_shard = int(len(dataset) / num_shards)

    idx_shard = [i for i in range(num_shards)]

    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, classes_per_user, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)

        for rand in rand_set:
            shard_idxs = idxs[
                rand * num_imgs_per_shard : (rand + 1) * num_imgs_per_shard
            ]
            dict_users[i] = np.concatenate((dict_users[i], shard_idxs), axis=0)

    return dict_users


def split_dataset(dataset: str = "MNIST"):
    train_dataset = None
    test_dataset = None

    # 获取数据集配置
    if dataset not in DATASET_CONFIG:
        logger.error(f"未知数据集: {dataset}")
        return

    classes_per_user = DATASET_CONFIG[dataset]["classes_per_user"]
    partition_file = PARTITION_DIR / f"noniid_{dataset}_k{K}_c{classes_per_user}.pkl"

    if partition_file.exists():
        logger.info(f"数据集 {dataset} 分割文件已存在，直接读取")
        return

    if dataset == "MNIST":
        train_dataset = datasets.MNIST(
            DATA_ROOT_DIR, train=True, download=True, transform=None
        )
        test_dataset = datasets.MNIST(
            DATA_ROOT_DIR, train=False, download=True, transform=None
        )
    elif dataset == "CIFAR10":
        train_dataset = datasets.CIFAR10(
            DATA_ROOT_DIR, train=True, download=True, transform=None
        )
        test_dataset = datasets.CIFAR10(
            DATA_ROOT_DIR, train=False, download=True, transform=None
        )
    elif dataset == "FashionMNIST":
        train_dataset = datasets.FashionMNIST(
            DATA_ROOT_DIR, train=True, download=True, transform=None
        )
        test_dataset = datasets.FashionMNIST(
            DATA_ROOT_DIR, train=False, download=True, transform=None
        )
    elif dataset == "CIFAR100":
        train_dataset = datasets.CIFAR100(
            DATA_ROOT_DIR, train=True, download=True, transform=None
        )
        test_dataset = datasets.CIFAR100(
            DATA_ROOT_DIR, train=False, download=True, transform=None
        )

    if not train_dataset or not test_dataset:
        logger.error(f"数据集 {dataset} 读取失败！")
        return

    logger.info(f"数据集 {dataset} 读取完毕！")

    user_groups = non_iid_split(train_dataset, K, classes_per_user=classes_per_user)

    with open(partition_file, "wb") as f:
        pickle.dump(user_groups, f)

    logger.info(f"数据集 {dataset} 分割完毕！")
    logger.info(f"保存文件为 {partition_file}")


if __name__ == "__main__":
    # split_dataset("MNIST")
    # split_dataset("CIFAR10")
    # split_dataset("FashionMNIST")
    split_dataset("CIFAR100")
