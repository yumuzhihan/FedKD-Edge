from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

table_path = Path(__file__).parent / "results" / "DistillationTest.xlsx"
df = pd.read_excel(table_path)
output_path = Path(__file__).parent / "results" / "graphs" / "distillation_heatmap.png"


def draw_heatmap():
    data = df.to_dict()
    temps = data["TEMP"].values()
    alphas = data["ALPHA"].values()
    gains = data["Gain"].values()

    temps = list(temps)
    alphas = list(alphas)
    gains = list(gains)

    data_count = len(temps)

    # 准备热力图，横轴是 TEMP，纵轴是 ALPHA，值是 Gain
    temp_values = sorted(set(temps))
    alpha_values = sorted(set(alphas))
    heatmap_data = [
        [0 for _ in range(len(temp_values))] for _ in range(len(alpha_values))
    ]

    for i in range(data_count):
        temp_idx = temp_values.index(temps[i])
        alpha_idx = alpha_values.index(alphas[i])
        heatmap_data[alpha_idx][temp_idx] = gains[i]

    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap_data, cmap="viridis", aspect="auto")
    plt.colorbar(label="Gain")
    plt.xticks(range(len(temp_values)), temp_values)
    plt.yticks(range(len(alpha_values)), alpha_values)
    plt.xlabel("TEMP")
    plt.ylabel("ALPHA")
    plt.title("Distillation Test Heatmap")

    for i in range(len(alpha_values)):
        for j in range(len(temp_values)):
            plt.text(
                j,
                i,
                f"{heatmap_data[i][j]:.2f}",
                ha="center",
                va="center",
                color=(
                    "white"
                    if heatmap_data[i][j] < max(map(max, heatmap_data)) / 2
                    else "black"
                ),
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    draw_heatmap()
