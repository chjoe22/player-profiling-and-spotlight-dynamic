import os
import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

LEGEND_LABELS = {
    "turns": "Turns",
    "total_sec_spoken": "Total Time (s)",
    "avg_turn_duration": "Avg Duration (s)",

    "turns_per_hour": "Turns / Hour",
    "total_sec_spoken_per_hour": "Time / Hour (s)",
}

def make_average_barplot(folder_path: str):
    sns.set_theme(style="whitegrid")

    for root, _, files in os.walk(folder_path):
        for file_name in files:

            if "average_comparison" not in file_name or not file_name.endswith(".csv"):
                continue

            csv_path = os.path.join(root, file_name)

            with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)

                metric_names = [
                    "turns",
                    "total_sec_spoken",
                    "avg_turn_duration"
                ]

                change_columns = [f"{m}_change_from_avg" for m in metric_names]

                metric_indices = [header.index(m) for m in metric_names]
                change_indices = [header.index(c) for c in change_columns]

                rows = [row for row in reader if row and any(cell.strip() for cell in row)]
                if len(rows) < 2:
                    continue

                avg_row = rows[-1]
                episode_rows = rows[:-1]

                avg_values = []
                for i in metric_indices:
                    try:
                        avg_values.append(float(avg_row[i]))
                    except (ValueError, TypeError):
                        avg_values.append(np.nan)

                data = {name: [] for name in metric_names}

                for row in episode_rows:
                    for name, idx in zip(metric_names, change_indices):
                        try:
                            data[name].append(float(row[idx]))
                        except (ValueError, TypeError):
                            data[name].append(np.nan)

            df = pd.DataFrame({
                "Episode": list(range(1, len(episode_rows) + 1)),
                **data
            })

            df_long = df.melt(
                id_vars="Episode",
                var_name="Metric",
                value_name="Percent Change"
            )

            player_name = file_name.replace("_average_comparison.csv", "")

            plt.figure(figsize=(12, 6))
            ax = sns.barplot(data=df_long, x="Episode", y="Percent Change", hue="Metric")

            ax.axhline(0, color="black", linestyle="--", linewidth=1.5)

            avg_text = (
                f"Across Episodes Averages:\n"
                f"Avg Turns: {avg_values[0]:.2f}, "
                f"Avg Total Time (sec): {avg_values[1]:.2f}, "
                f"Avg Turn Duration (sec): {avg_values[2]:.2f}"
            )

            handles, labels = ax.get_legend_handles_labels()
            new_labels = [LEGEND_LABELS.get(label, label) for label in labels]
            ax.legend(handles, new_labels, title="Metric")

            ax.set_title(f"{player_name} - % Change from Average\n({avg_text})")
            ax.set_xlabel("Episode")
            ax.set_ylabel("% Change from Average")

            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            episodes = sorted(df["Episode"].unique())
            ax.set_xticks(range(len(episodes)))
            ax.set_xticklabels(episodes)

            for i in range(len(episodes) + 1):
                ax.axvline(i - 0.5, color="gray", linestyle="--", linewidth=1, alpha=1)

            plt.tight_layout()

            plt.savefig(os.path.join(root, file_name.replace(".csv", "_barplot.png")))
            plt.close()

    print("Non-hourly average barplots created successfully!")




def make_average_barplot_per_hour(folder_path: str):
    sns.set_theme(style="whitegrid")

    for root, _, files in os.walk(folder_path):
        for file_name in files:

            if "average_comparison" not in file_name or not file_name.endswith(".csv"):
                continue

            csv_path = os.path.join(root, file_name)

            with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)

                metric_names = [
                    "turns_per_hour",
                    "total_sec_spoken_per_hour",
                    "avg_turn_duration"
                ]

                change_columns = [f"{m}_change_from_avg" for m in metric_names]

                metric_indices = [header.index(m) for m in metric_names]
                change_indices = [header.index(c) for c in change_columns]

                rows = [row for row in reader if row and any(cell.strip() for cell in row)]
                if len(rows) < 2:
                    continue

                avg_row = rows[-1]
                episode_rows = rows[:-1]

                avg_values = []
                for i in metric_indices:
                    try:
                        avg_values.append(float(avg_row[i]))
                    except (ValueError, TypeError):
                        avg_values.append(np.nan)

                data = {name: [] for name in metric_names}

                for row in episode_rows:
                    for name, idx in zip(metric_names, change_indices):
                        try:
                            data[name].append(float(row[idx]))
                        except (ValueError, TypeError):
                            data[name].append(np.nan)

            df = pd.DataFrame({
                "Episode": list(range(1, len(episode_rows) + 1)),
                **data
            })

            df_long = df.melt(
                id_vars="Episode",
                var_name="Metric",
                value_name="Percent Change"
            )

            player_name = file_name.replace("_average_comparison.csv", "")

            plt.figure(figsize=(12, 6))
            ax = sns.barplot(data=df_long, x="Episode", y="Percent Change", hue="Metric")

            ax.axhline(0, color="black", linestyle="--", linewidth=1.5)

            avg_text = (
                f"Across Episodes Averages:\n"
                f"Avg Turns per Hour: {avg_values[0]:.2f}, "
                f"Avg Time per Hour (sec): {avg_values[1]:.2f}, "
                f"Avg Turn Duration (sec): {avg_values[2]:.2f}"
            )

            handles, labels = ax.get_legend_handles_labels()
            new_labels = [LEGEND_LABELS.get(label, label) for label in labels]
            ax.legend(handles, new_labels, title="Metric")

            ax.set_title(f"{player_name} - Per Hour % Change\n({avg_text})")
            ax.set_xlabel("Episode")
            ax.set_ylabel("% Change from Average")

            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            episodes = sorted(df["Episode"].unique())
            ax.set_xticks(range(len(episodes)))
            ax.set_xticklabels(episodes)

            for i in range(len(episodes) + 1):
                ax.axvline(i - 0.5, color="gray", linestyle="--", linewidth=1, alpha=1)

            plt.tight_layout()

            plt.savefig(os.path.join(root, file_name.replace(".csv", "_per_hour_barplot.png")))
            plt.close()

    print("Per-hour average barplots created successfully!")



#DEPRECATED
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
    make_average_barplot_per_hour(folder_path)
    #make_baseline_barplot(folder_path) #DEPRECATED
