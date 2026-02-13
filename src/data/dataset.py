import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class FastTensorDataset(Dataset):
    def __init__(self, dataset_name, root, train=True, transform=None):
        self.transform = transform
        dataset_name = dataset_name.lower()

        data = None
        targets = None

        # 1. 加载原始数据
        if dataset_name == "cifar10":
            base_dataset = datasets.CIFAR10(root, train=train, download=False)
            data = base_dataset.data
            targets = base_dataset.targets
        elif dataset_name == "mnist":
            base_dataset = datasets.MNIST(root, train=train, download=False)
            data = base_dataset.data
            targets = base_dataset.targets
            if isinstance(data, torch.Tensor):
                data = data.numpy()
            data = np.expand_dims(data, axis=-1)
        elif dataset_name == "fashionmnist":
            base_dataset = datasets.FashionMNIST(root, train=train, download=False)
            data = base_dataset.data
            targets = base_dataset.targets
            if isinstance(data, torch.Tensor):
                data = data.numpy()
            data = np.expand_dims(data, axis=-1)
        elif dataset_name == "cifar100":
            base_dataset = datasets.CIFAR100(root, train=train, download=False)
            data = base_dataset.data
            targets = base_dataset.targets

        if data is None or targets is None:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

        self.data = (
            torch.from_numpy(data).permute(0, 3, 1, 2).contiguous().float().div(255.0)
        )

        if dataset_name == "mnist":
            self.data = self.data.repeat(1, 3, 1, 1)
        elif dataset_name == "fashionmnist":
            self.data = self.data.repeat(1, 3, 1, 1)

        self.targets = torch.tensor(targets, dtype=torch.long)

    def __getitem__(self, index):
        # 【极速读取】现在这里没有任何计算，只有内存寻址
        img = self.data[index]
        target = self.targets[index]

        if self.transform is not None:
            # transform 接收的已经是 float tensor 了
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)


def get_fast_transforms(dataset_name="cifar10"):
    dataset_name = dataset_name.lower()

    train_transform = None
    test_transform = None

    if dataset_name == "cifar10":
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

    elif dataset_name == "mnist":
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(
                    (0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)
                ),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.Normalize(
                    (0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)
                ),
            ]
        )
    elif dataset_name == "fashionmnist":
        train_transform = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize((0.2860,), (0.3530,)),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.Grayscale(num_output_channels=3),
                transforms.Normalize((0.2860,), (0.3530,)),
            ]
        )
    elif dataset_name == "cifar100":
        stats = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(*stats),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Normalize(*stats),
            ]
        )

    if train_transform is None or test_transform is None:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    return train_transform, test_transform
