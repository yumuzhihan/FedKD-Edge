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

        if data is None or targets is None:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

        # 2. 【核心优化】立刻转为 Float Tensor 并归一化到 [0, 1]
        # 这一步会消耗一些内存 (CIFAR10 约 600MB)，但彻底消除了训练时的 CPU 转换开销
        self.data = (
            torch.from_numpy(data).permute(0, 3, 1, 2).contiguous().float().div(255.0)
        )

        # 针对 MNIST 的特殊处理
        if dataset_name == "mnist":
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
        # 注意：这里不需要 ToTensor，也不需要 div(255)，因为 Dataset 里已经做好了
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

    if train_transform is None or test_transform is None:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    return train_transform, test_transform
