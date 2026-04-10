import os
import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def make_average_barplot(folder_path: str):
    sns.set_theme(style="whitegrid")

    for root, _, files in os.walk(folder_path):
        for file_name in files:

            if "average_comparison" not in file_name or not file_name.endswith(".csv"):
                continue

            csv_path = os.path.join(root, file_name)

            turns_change = []
            total_sec_change = []
            duration_change = []

            avg_turns = avg_total_sec = avg_duration = None

            with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)

                idx_turns_change = header.index("turns_change_from_avg")
                idx_total_sec_change = header.index("total_sec_change_from_avg")
                idx_duration_change = header.index("avg_turn_duration_change_from_avg")

                idx_turns = header.index("turns")
                idx_total_sec = header.index("total_sec")
                idx_duration = header.index("avg_turn_duration")

                rows = [row for row in reader if row]
                if not rows:
                    continue

                avg_row = rows[-1]
                try:
                    avg_turns = float(avg_row[idx_turns])
                    avg_total_sec = float(avg_row[idx_total_sec])
                    avg_duration = float(avg_row[idx_duration])
                except (ValueError, TypeError):
                    continue

                episode_rows = rows[:-1]

                for row in episode_rows:
                    try:
                        turns_change.append(float(row[idx_turns_change]))
                        total_sec_change.append(float(row[idx_total_sec_change]))
                        duration_change.append(float(row[idx_duration_change]))
                    except (ValueError, TypeError):
                        continue

            if not turns_change:
                continue

            # Build dataframe in long format (required for seaborn grouped bars)
            df = pd.DataFrame({
                "Episode": list(range(1, len(turns_change) + 1)),
                "Turns": turns_change,
                "Total Seconds": total_sec_change,
                "Avg Turn Duration": duration_change
            })

            df_long = df.melt(
                id_vars="Episode",
                var_name="Metric",
                value_name="Percent Change"
            )

            player_name = file_name.replace("_average_comparison.csv", "")

            plt.figure(figsize=(12, 6))
            ax = sns.barplot(
                data=df_long,
                x="Episode",
                y="Percent Change",
                hue="Metric"
            )

            # Horizontal baseline at 0%
            ax.axhline(0, color="black", linestyle="--", linewidth=1.5)

            ax.set_title(
                f"{player_name} - % Change from Average\n"
                f"(Avg Turns: {avg_turns:.2f}, "
                f"Avg Total Sec: {avg_total_sec:.2f}, "
                f"Avg Duration: {avg_duration:.2f})"
            )

            ax.set_xlabel("Episode")
            ax.set_ylabel("% Change from Average")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            episodes = sorted(df["Episode"].unique())
            ax.set_xticks(range(len(episodes)))
            ax.set_xticklabels(episodes)

            for i in range(len(episodes) + 1):
                ax.axvline(i - 0.5, color="gray", linestyle="--", linewidth=1, alpha=1)

            plt.tight_layout()

            output_file = os.path.join(
                root,
                file_name.replace(".csv", "_barplot.png")
            )
            plt.savefig(output_file)
            plt.close()

    print("Seaborn average barplots created successfully!")



def make_baseline_barplot(folder_path: str):
    sns.set_theme(style="whitegrid")

    for root, _, files in os.walk(folder_path):
        for file_name in files:

            if "baseline_comparison" not in file_name or not file_name.endswith(".csv"):
                continue

            csv_path = os.path.join(root, file_name)

            turns_change = []
            total_sec_change = []
            duration_change = []

            with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)

                idx_turns = header.index("turns_change_from_baseline")
                idx_total_sec = header.index("total_sec_change_from_baseline")
                idx_duration = header.index("avg_turn_duration_change_from_baseline")

                for row in reader:
                    if not row:
                        continue
                    try:
                        turns_change.append(float(row[idx_turns]))
                        total_sec_change.append(float(row[idx_total_sec]))
                        duration_change.append(float(row[idx_duration]))
                    except (ValueError, IndexError):
                        continue

            if not turns_change:
                continue

            # Build dataframe
            df = pd.DataFrame({
                "Episode": list(range(1, len(turns_change) + 1)),
                "Turns": turns_change,
                "Total Seconds": total_sec_change,
                "Avg Turn Duration": duration_change
            })

            # Convert to long format (required for seaborn grouped bars)
            df_long = df.melt(
                id_vars="Episode",
                var_name="Metric",
                value_name="Percent Change"
            )

            player_name = file_name.replace("_baseline_comparison.csv", "")

            plt.figure(figsize=(12, 6))

            ax = sns.barplot(
                data=df_long,
                x="Episode",
                y="Percent Change",
                hue="Metric"
            )

            # Baseline reference line
            ax.axhline(0, color="black", linestyle="--", linewidth=1.5)

            ax.set_title(f"{player_name} - % Change Over Episodes (vs Baseline)")
            ax.set_xlabel("Episode")
            ax.set_ylabel("% Change")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            episodes = sorted(df["Episode"].unique())
            ax.set_xticks(range(len(episodes)))
            ax.set_xticklabels(episodes)

            for i in range(len(episodes) + 1):
                ax.axvline(i - 0.5, color="gray", linestyle="--", linewidth=1, alpha=1)

            plt.tight_layout()

            output_file = os.path.join(
                root,
                file_name.replace(".csv", "_barplot.png")
            )
            plt.savefig(output_file)
            plt.close()

    print("Seaborn baseline barplots created successfully!")

if __name__ == "__main__":
    folder_path = "../../resources/speaker_stats/"
    make_average_barplot(folder_path)
    make_baseline_barplot(folder_path)
