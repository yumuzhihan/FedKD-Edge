import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# 定义文件路径
cifar_train_res = Path(__file__).parent / "results" / "cifar10_teacher_cnn_results.csv"
mnist_train_res = Path(__file__).parent / "results" / "mnist_teacher_cnn_results.csv"


def annotate_max_point(df, x_col, y_col, ax_color, label_prefix=""):
    """
    在图表中找到最高点并添加标注
    """
    max_idx = df[y_col].idxmax()
    max_x = df.loc[max_idx, x_col]
    max_y = df.loc[max_idx, y_col]

    plt.plot(max_x, max_y, "o", color=ax_color)

    plt.annotate(
        f"{label_prefix}Max: {max_y:.4f}\nEpoch: {max_x}",
        xy=(max_x, max_y),
        xytext=(0, 15),
        textcoords="offset points",
        ha="center",
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color=ax_color),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=ax_color, alpha=0.8),
    )


def draw_teacher_train_graph(file_path: Path, dataset_name: str):
    if not file_path.exists():
        print(f"Warning: File not found {file_path}")
        return

    df = pd.read_csv(file_path)

    output_dir = Path(__file__).parent / "results" / "graphs"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss", color="blue")
    plt.plot(df["epoch"], df["eval_loss"], label="Validation Loss", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{dataset_name} Teacher Model Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    output_path_loss = output_dir / f"{dataset_name.lower()}_teacher_train_loss.png"
    plt.savefig(output_path_loss)
    plt.close()

    plt.figure(figsize=(10, 6))

    # 绘图
    plt.plot(df["epoch"], df["train_accuracy"], label="Train Accuracy", color="green")
    plt.plot(df["epoch"], df["eval_accuracy"], label="Validation Accuracy", color="red")

    annotate_max_point(df, "epoch", "train_accuracy", "green", "Train ")

    annotate_max_point(df, "epoch", "eval_accuracy", "red", "Val ")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{dataset_name} Teacher Model Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)

    output_path_acc = output_dir / f"{dataset_name.lower()}_teacher_train_accuracy.png"
    plt.savefig(output_path_acc)
    plt.close()

    print(f"Saved graphs to {output_dir}")


def main():
    draw_teacher_train_graph(cifar_train_res, "CIFAR-10")
    draw_teacher_train_graph(mnist_train_res, "MNIST")


if __name__ == "__main__":
    main()
