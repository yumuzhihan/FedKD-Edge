import pickle
from pathlib import Path

import numpy as np
from torchvision import datasets

from src.utils.get_logger import LoggerFactory

logger = LoggerFactory.get_logger("Partition")

DATASET_DEFAULT_CLASSES_PER_USER = {
    "MNIST": 2,
    "CIFAR10": 2,
    "FashionMNIST": 2,
    "CIFAR100": 10,
}


def get_partition_tag(config):
    partition_path = config.get("partition_path")
    if partition_path:
        return Path(partition_path).stem

    mode = config.get("partition_mode", "pathological")
    if mode == "iid":
        return "iid"
    if mode == "pathological":
        return f"pathological_c{config['client_classes']}"
    raise ValueError(f"Unknown partition mode: {mode}")


def build_partition_filename(
    dataset_name,
    num_users,
    partition_mode,
    partition_seed,
    client_classes,
):
    if partition_mode == "iid":
        return f"iid_{dataset_name}_k{num_users}_seed{partition_seed}.pkl"
    if partition_mode == "pathological":
        return (
            f"pathological_{dataset_name}_k{num_users}_"
            f"c{client_classes}_seed{partition_seed}.pkl"
        )
    raise ValueError(f"Unknown partition mode: {partition_mode}")


def get_partition_file_path(config, data_root):
    partition_path = config.get("partition_path")
    if partition_path:
        return Path(partition_path)

    partition_seed = config.get("partition_seed")
    if partition_seed is None:
        partition_seed = config["seed"]

    filename = build_partition_filename(
        dataset_name=config["dataset"],
        num_users=config["num_users"],
        partition_mode=config["partition_mode"],
        partition_seed=partition_seed,
        client_classes=config["client_classes"],
    )
    return Path(data_root) / "partitions" / filename


def _get_dataset_labels(dataset):
    labels = getattr(dataset, "targets", None)
    if labels is None:
        labels = getattr(dataset, "train_labels", None)
    if labels is None:
        raise ValueError("Dataset does not expose labels via targets/train_labels")
    return np.array(labels)


def _load_dataset(dataset_name, data_root):
    dataset_map = {
        "MNIST": datasets.MNIST,
        "CIFAR10": datasets.CIFAR10,
        "FashionMNIST": datasets.FashionMNIST,
        "CIFAR100": datasets.CIFAR100,
    }
    if dataset_name not in dataset_map:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataset_cls = dataset_map[dataset_name]
    return dataset_cls(data_root, train=True, download=True, transform=None)


def iid_split(dataset, num_users, rng):
    idxs = rng.permutation(len(dataset))
    split_idxs = np.array_split(idxs, num_users)
    return {user_id: split.astype("int64") for user_id, split in enumerate(split_idxs)}


def pathological_split(dataset, num_users, classes_per_user, rng):
    labels = _get_dataset_labels(dataset)
    dict_users = {i: np.array([], dtype="int64") for i in range(num_users)}

    idxs = np.arange(len(dataset))
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    num_shards = num_users * classes_per_user
    num_imgs_per_shard = int(len(dataset) / num_shards)
    idx_shard = list(range(num_shards))

    for user_id in range(num_users):
        rand_set = set(rng.choice(idx_shard, classes_per_user, replace=False).tolist())
        idx_shard = list(set(idx_shard) - rand_set)

        for rand in rand_set:
            shard_idxs = idxs[
                rand * num_imgs_per_shard : (rand + 1) * num_imgs_per_shard
            ]
            dict_users[user_id] = np.concatenate(
                (dict_users[user_id], shard_idxs), axis=0
            )

    return dict_users


def generate_partition_file(
    dataset_name,
    num_users,
    partition_mode,
    data_root,
    partition_seed,
    client_classes=None,
    partition_path=None,
    force=False,
):
    if partition_path is None:
        filename = build_partition_filename(
            dataset_name=dataset_name,
            num_users=num_users,
            partition_mode=partition_mode,
            partition_seed=partition_seed,
            client_classes=client_classes,
        )
        partition_path = Path(data_root) / "partitions" / filename
    else:
        partition_path = Path(partition_path)

    partition_path.parent.mkdir(parents=True, exist_ok=True)
    if partition_path.exists() and not force:
        logger.info(f"Partition file already exists: {partition_path}")
        return partition_path

    dataset = _load_dataset(dataset_name, data_root)
    rng = np.random.default_rng(partition_seed)

    if partition_mode == "iid":
        user_groups = iid_split(dataset, num_users, rng)
    elif partition_mode == "pathological":
        if client_classes is None:
            client_classes = DATASET_DEFAULT_CLASSES_PER_USER[dataset_name]
        user_groups = pathological_split(dataset, num_users, client_classes, rng)
    else:
        raise ValueError(f"Unknown partition mode: {partition_mode}")

    with open(partition_path, "wb") as f:
        pickle.dump(user_groups, f)

    logger.info(f"Saved partition file to {partition_path}")
    return partition_path


def ensure_partition_file(config, data_root):
    partition_path = get_partition_file_path(config, data_root)
    if partition_path.exists():
        return partition_path

    partition_seed = config.get("partition_seed")
    if partition_seed is None:
        partition_seed = config["seed"]

    return generate_partition_file(
        dataset_name=config["dataset"],
        num_users=config["num_users"],
        partition_mode=config["partition_mode"],
        data_root=data_root,
        partition_seed=partition_seed,
        client_classes=config.get("client_classes"),
        partition_path=partition_path,
    )
