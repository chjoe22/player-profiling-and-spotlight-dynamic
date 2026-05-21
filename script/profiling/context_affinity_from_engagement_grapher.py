import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


AFFINITY_COLUMNS = [
    "combat_affinity",
    "exploration_affinity",
    "social_affinity"
]


def create_affinity_barplots(parent_folder: str):
    sns.set_theme(style="whitegrid")

    for root, _, files in os.walk(parent_folder):
        for file in files:

            if not file.endswith("_context_affinity.csv"):
                continue

            csv_path = os.path.join(root, file)

            try:
                df = pd.read_csv(csv_path)

                # Remove empty / baseline rows safely
                df = df[df["episode"].notna()]
                df = df[df["episode"] != "AVERAGE"]

                if df.empty:
                    continue

                # Compute averages directly (more reliable than last row)
                values = [
                    df["combat_affinity"].mean(),
                    df["exploration_affinity"].mean(),
                    df["social_affinity"].mean(),
                ]

                plot_df = pd.DataFrame({
                    "Category": ["Combat", "Exploration", "Social"],
                    "Value": values
                })

                plt.figure(figsize=(4, 5))

                ax = sns.barplot(
                    data=plot_df,
                    x="Category",
                    y="Value"
                )

                # Symmetric axis around 0
                max_abs = max(abs(plot_df["Value"].min()), abs(plot_df["Value"].max()))
                ax.set_ylim(-max_abs * 1.2, max_abs * 1.2)

                ax.axhline(0, color="black", linewidth=1)

                # Value labels
                for container in ax.containers:
                    ax.bar_label(container, fmt="%.2f", padding=3)

                folder_name = os.path.basename(root)
                base_name = os.path.splitext(file)[0]

                plt.title(f"{folder_name} - Engagement/Context Affinity")

                plt.tight_layout()

                png_filename = f"{base_name}.png"
                png_path = os.path.join(root, png_filename)

                plt.savefig(png_path)
                plt.close()

                print(f"Saved: {png_path}")

            except Exception as e:
                print(f"Failed processing {csv_path}: {e}")


if __name__ == "__main__":
    parent_folder = "../../resources/speaker_stats"
    create_affinity_barplots(parent_folder)