from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from src.models.teacher_cnn import TeacherCNN


BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
WEIGHTS_DIR = ROOT_DIR / "weights"

DATASET_CONFIG = {
    "cifar10": {
        "dataset_class": torchvision.datasets.CIFAR10,
        "num_classes": 10,
        "transform": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        ),
    },
    "mnist": {
        "dataset_class": torchvision.datasets.MNIST,
        "num_classes": 10,
        "transform": transforms.Compose(
            [
                transforms.Resize(32),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)
                ),
            ]
        ),
    },
    "fashionmnist": {
        "dataset_class": torchvision.datasets.FashionMNIST,
        "num_classes": 10,
        "transform": transforms.Compose(
            [
                transforms.Resize(32),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,)),
            ]
        ),
    },
    "cifar100": {
        "dataset_class": torchvision.datasets.CIFAR100,
        "num_classes": 100,
        "transform": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        ),
    },
}


def get_test_loader(dataset_name: str) -> DataLoader:
    config = DATASET_CONFIG[dataset_name]
    dataset = config["dataset_class"](
        root=DATA_DIR,
        train=False,
        download=True,
        transform=config["transform"],
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )


def evaluate(model: nn.Module, data_loader: DataLoader) -> tuple[float, float, int, int]:
    criterion = nn.CrossEntropyLoss()
    model.eval()

    total_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            correct += outputs.argmax(dim=1).eq(labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total, total_loss / total, correct, total


def evaluate_weight(weight_path: Path) -> tuple[str, float, float, int, int]:
    dataset_name = weight_path.name.replace("_teacher_cnn_best.pth", "")
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unsupported dataset from weight file: {weight_path.name}")

    num_classes = DATASET_CONFIG[dataset_name]["num_classes"]
    model = TeacherCNN(num_classes=num_classes).to(DEVICE)
    state_dict = torch.load(weight_path, map_location=DEVICE)
    model.load_state_dict(state_dict)

    test_loader = get_test_loader(dataset_name)
    accuracy, avg_loss, correct, total = evaluate(model, test_loader)
    return dataset_name, accuracy, avg_loss, correct, total


def main() -> None:
    weight_paths = sorted(WEIGHTS_DIR.glob("*_teacher_cnn_best.pth"))
    if not weight_paths:
        raise FileNotFoundError(f"No teacher best weights found in: {WEIGHTS_DIR}")

    print(f"Device: {DEVICE}")
    print("Teacher CNN best model accuracy on test sets")
    print("-" * 72)

    for weight_path in weight_paths:
        dataset_name, accuracy, avg_loss, correct, total = evaluate_weight(weight_path)
        print(
            f"{dataset_name:<12} | acc: {accuracy:>6.2f}% | "
            f"loss: {avg_loss:>7.4f} | correct: {correct:>5}/{total:<5} | file: {weight_path.name}"
        )


if __name__ == "__main__":
    main()
