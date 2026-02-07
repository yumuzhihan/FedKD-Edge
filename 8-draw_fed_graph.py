import matplotlib

matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_fl_metrics(file_path: Path):

    # 1. 读取数据
    try:
        df = pd.read_csv(file_path, header=0)

        # 清理列名空格
        df.columns = df.columns.str.strip()

        # 强制转换为数值类型，无法转换的变NaN
        numeric_cols = ["Round", "Train_Loss", "Train_Acc", "Eval_Loss", "Eval_Acc"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # 删除坏数据行
        df.dropna(subset=numeric_cols, inplace=True)

    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    plt.style.use("ggplot")

    # ==========================================
    # 图表 1: Client Training Metrics
    # ==========================================
    fig1, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_xlabel("Round")
    ax1.set_ylabel("Train Loss", color="tab:red", fontweight="bold")
    ax2 = ax1.twinx()
    ax2.set_ylabel("Train Accuracy (%)", color="tab:blue", fontweight="bold")

    clients = df["Client_ID"].unique()
    for client in clients:
        client_data = df[df["Client_ID"] == client].sort_values("Round")

        ax1.plot(
            client_data["Round"],
            client_data["Train_Loss"],
            color="tab:red",
            alpha=0.3,
            linewidth=1,
            label="Loss" if client == clients[0] else "",
        )
        ax2.plot(
            client_data["Round"],
            client_data["Train_Acc"],
            color="tab:blue",
            alpha=0.3,
            linewidth=1,
            linestyle="--",
            label="Acc" if client == clients[0] else "",
        )

    if not df.empty:
        best_train_row = df.sort_values("Train_Acc", ascending=False).iloc[0]

        # 此时取出来的一定是标量（单个数字）
        max_train_acc = float(best_train_row["Train_Acc"])
        max_train_round = float(best_train_row["Round"])

        ax2.annotate(
            f"Max Train Acc: {max_train_acc:.2f}%\n(Round {int(max_train_round)})",
            xy=(max_train_round, max_train_acc),
            xytext=(max_train_round, max_train_acc + 5),
            arrowprops=dict(facecolor="black", shrink=0.05),
            horizontalalignment="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8),
        )

    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    # 构造图例
    from matplotlib.lines import Line2D

    custom_lines = [
        Line2D([0], [0], color="tab:red", lw=2),
        Line2D([0], [0], color="tab:blue", lw=2, linestyle="--"),
    ]
    ax1.legend(custom_lines, ["Train Loss", "Train Accuracy"], loc="upper left")

    plt.title("Client Training Metrics")
    fig1.tight_layout()
    save_path1 = file_path.parent / "plot_client_training.png"
    plt.savefig(save_path1, dpi=300)
    print(f"图表1已保存: {save_path1}")
    plt.close(fig1)

    # ==========================================
    # 图表 2: Global Eval Metrics
    # ==========================================
    global_df = df.drop_duplicates(subset=["Round"]).sort_values("Round")
    global_df = global_df.reset_index(drop=True)

    fig2, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_xlabel("Round")
    ax1.set_ylabel("Global Eval Loss", color="tab:orange", fontweight="bold")
    l1 = ax1.plot(
        global_df["Round"],
        global_df["Eval_Loss"],
        color="tab:orange",
        linewidth=2,
        marker="o",
        markersize=4,
        label="Eval Loss",
    )

    ax2 = ax1.twinx()
    ax2.set_ylabel("Global Eval Accuracy (%)", color="tab:green", fontweight="bold")
    l2 = ax2.plot(
        global_df["Round"],
        global_df["Eval_Acc"],
        color="tab:green",
        linewidth=2,
        marker="s",
        markersize=4,
        label="Eval Acc",
    )

    if not global_df.empty:
        best_eval_row = global_df.sort_values("Eval_Acc", ascending=False).iloc[0]

        max_eval_acc = float(best_eval_row["Eval_Acc"])
        max_eval_round = float(best_eval_row["Round"])

        ax2.annotate(
            f"Max Test Acc: {max_eval_acc:.2f}%\n(Round {int(max_eval_round)})",
            xy=(max_eval_round, max_eval_acc),
            xytext=(max_eval_round, max_eval_acc + 2),
            arrowprops=dict(facecolor="darkgreen", shrink=0.05),
            horizontalalignment="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="darkgreen", alpha=0.8),
        )

    ax1.tick_params(axis="y", labelcolor="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:green")

    lines = l1 + l2
    labels = [str(l.get_label()) for l in lines]
    ax1.legend(lines, labels, loc="center right")

    plt.title("Global Evaluation Metrics")
    fig2.tight_layout()

    save_path2 = file_path.parent / "plot_global_eval.png"
    plt.savefig(save_path2, dpi=300)
    print(f"图表2已保存: {save_path2}")
    plt.close(fig2)


if __name__ == "__main__":
    # csv_file = (
    #     Path(__file__).parent
    #     / "results"
    #     / "fed_baseline"
    #     / "log_CIFAR10_users10_20260110-094028.csv"
    # )
    csv_file = (
        Path(__file__).parent
        / "results"
        / "unified_logs"
        / "log_feature_kd_CIFAR10_seed42_T1.0_ka0.5_fa0.3_20260128-205628.csv"
    )

    if csv_file.exists():
        plot_fl_metrics(csv_file)
    else:
        print(f"未找到文件: {csv_file}")
