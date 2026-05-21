import os
import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_sentiment_context_affinity(folder_path: str):
    sns.set_theme(style="whitegrid")

    for file_name in os.listdir(folder_path):

        if not file_name.endswith("_sentiment_context_affinity.csv"):
            continue

        csv_path = os.path.join(folder_path, file_name)

        try:
            with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)

                header = next(reader)

                combat_idx = header.index("combat_affinity")
                exploration_idx = header.index("exploration_affinity")
                social_idx = header.index("social_affinity")

                combat_val = None
                exploration_val = None
                social_val = None

                for row in reader:
                    if not row:
                        continue

                    if row[0] == "AVERAGE":
                        combat_val = float(row[combat_idx])
                        exploration_val = float(row[exploration_idx])
                        social_val = float(row[social_idx])
                        break

            if combat_val is None:
                continue

            plot_df = pd.DataFrame({
                "Category": ["Combat", "Exploration", "Social"],
                "Value": [combat_val, exploration_val, social_val]
            })

            plt.figure(figsize=(4, 5))

            ax = sns.barplot(
                data=plot_df,
                x="Category",
                y="Value"
            )

            # Center around zero
            max_abs = max(abs(plot_df["Value"].min()), abs(plot_df["Value"].max()))
            ax.set_ylim(-max_abs * 1.2, max_abs * 1.2)

            ax.axhline(0, color="black", linewidth=1)

            # Value labels
            for container in ax.containers:
                ax.bar_label(container, fmt="%.2f", padding=3)

            player_name = file_name.replace("_sentiment_context_affinity.csv", "")

            plt.title(f"{player_name} - Sentiment/Context Affinity")

            plt.tight_layout()

            output_path = os.path.join(
                folder_path,
                f"{player_name}_sentiment_affinity.png"
            )

            plt.savefig(output_path)
            plt.close()

            print(f"Saved: {output_path}")

        except Exception as e:
            print(f"Failed processing {file_name}: {e}")


if __name__ == "__main__":
    folder_path = "../../resources/sentiment_analysis/"
    plot_sentiment_context_affinity(folder_path)