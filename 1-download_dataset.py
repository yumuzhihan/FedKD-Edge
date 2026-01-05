import torchvision
import torchvision.transforms as transforms
import os
from pathlib import Path

data_root_path = Path(__file__).parent / "data"


def download_datasets(root_path=data_root_path):
    print(f"准备将数据集下载到: {os.path.abspath(root_path)}")

    if not os.path.exists(root_path):
        os.makedirs(root_path)

    # 1. 下载 MNIST
    print("-" * 30)
    print("开始下载 MNIST 数据集...")
    try:
        # train=True 下载训练集，download=True 如果没有则下载
        mnist_train = torchvision.datasets.MNIST(
            root=root_path, train=True, download=True, transform=transforms.ToTensor()
        )
        mnist_test = torchvision.datasets.MNIST(
            root=root_path, train=False, download=True, transform=transforms.ToTensor()
        )
        print("✅ MNIST 下载并加载成功！")
        print(f"   训练集大小: {len(mnist_train)}")
        print(f"   测试集大小: {len(mnist_test)}")
    except Exception as e:
        print(f"❌ MNIST 下载失败: {e}")

    # 2. 下载 CIFAR-10
    print("-" * 30)
    print("开始下载 CIFAR-10 数据集...")
    try:
        cifar_train = torchvision.datasets.CIFAR10(
            root=root_path, train=True, download=True, transform=transforms.ToTensor()
        )
        cifar_test = torchvision.datasets.CIFAR10(
            root=root_path, train=False, download=True, transform=transforms.ToTensor()
        )
        print("✅ CIFAR-10 下载并加载成功！")
        print(f"   训练集大小: {len(cifar_train)}")
        print(f"   测试集大小: {len(cifar_test)}")
    except Exception as e:
        print(f"❌ CIFAR-10 下载失败: {e}")

    print("-" * 30)
    print("所有任务完成。")


if __name__ == "__main__":
    download_datasets()
