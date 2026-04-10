import csv
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import MaxNLocator

def make_baseline_plot(folder_path: str):
    """
    Generates percentage-change plots for each player's baseline comparison CSV.
    The function searches recursively through subfolders and saves each plot
    in the same directory as its corresponding CSV file.
    """

    # Recursively walk through all subdirectories
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            # Process only relevant CSV files
            if "baseline_comparison" not in file_name or not file_name.endswith(".csv"):
                continue

            csv_path = os.path.join(root, file_name)

            # Lists to store %change values for each data column
            turns_change = []
            total_sec_change = []
            avg_duration_change = []

            # Read the CSV
            with open(csv_path, mode='r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)  # Skip header

                # Find the indices of %change columns
                idx_turns_change = header.index("turns_change_from_baseline")
                idx_total_sec_change = header.index("total_sec_change_from_baseline")
                idx_avg_duration_change = header.index(
                    "avg_turn_duration_change_from_baseline"
                )

                for row in reader:
                    if not row:
                        continue
                    try:
                        turns_change.append(float(row[idx_turns_change]))
                        total_sec_change.append(float(row[idx_total_sec_change]))
                        avg_duration_change.append(
                            float(row[idx_avg_duration_change])
                        )
                    except (ValueError, IndexError):
                        # Skip malformed rows
                        continue

            if not turns_change:
                continue

            # Plotting
            plt.figure(figsize=(10, 6))
            x = list(range(1, len(turns_change) + 1))  # X-axis = episode numbers

            plt.plot(x, turns_change, marker='o', color='red', label='Turns taken')
            plt.plot(x, total_sec_change, marker='s', color='green', label='Total seconds')
            plt.plot(x, avg_duration_change, marker='^', color='blue', label='Avg turn duration')

            plt.title("Percentage Change Over Episodes (Compared to Baseline)")
            plt.xlabel("Episode")
            plt.ylabel("% Change")
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend()
            plt.tight_layout()
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

            # Save the plot in the same subfolder as the CSV
            output_file = os.path.join(root, file_name.replace(".csv", "_lineplot.png"))
            plt.savefig(output_file)
            plt.close()

    print("Created baseline plots successfully!")


def make_average_plot(folder_path: str):
    """
    Generates percentage-change plots for each player's average baseline comparison CSV.
    Each plot includes:
        - Percentage change per episode for each metric.
        - A horizontal dotted line at 0% representing the average baseline.
        - Average metric values displayed in the plot title.
    """

    # Walk through all subdirectories
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            # Process only the relevant CSV files
            if "average_comparison" not in file_name or not file_name.endswith(".csv"):
                continue

            csv_path = os.path.join(root, file_name)

            # Lists to store percentage change values
            turns_change = []
            total_sec_change = []
            avg_duration_change = []

            # Variables to store actual average values (from the last row)
            avg_turns = avg_total_sec = avg_duration = None

            with open(csv_path, mode='r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)

                # Identify indices of percentage change columns
                idx_turns_change = header.index("turns_change_from_avg")
                idx_total_sec_change = header.index("total_sec_change_from_avg")
                idx_avg_duration_change = header.index(
                    "avg_turn_duration_change_from_avg"
                )

                idx_turns = header.index("turns")
                idx_total_sec = header.index("total_sec")
                idx_avg_duration = header.index("avg_turn_duration")

                # Read and clean rows (remove blank lines)
                rows = [row for row in reader if row]

                if not rows:
                    continue

                # The last row contains the actual averages
                avg_row = rows[-1]
                try:
                    avg_turns = float(avg_row[idx_turns])
                    avg_total_sec = float(avg_row[idx_total_sec])
                    avg_duration = float(avg_row[idx_avg_duration])
                except (ValueError, TypeError):
                    # If parsing fails, skip this file
                    continue

                # All preceding rows correspond to episodes
                episode_rows = rows[:-1]

                for row in episode_rows:
                    try:
                        turns_change.append(float(row[idx_turns_change]))
                        total_sec_change.append(float(row[idx_total_sec_change]))
                        avg_duration_change.append(
                            float(row[idx_avg_duration_change])
                        )
                    except (ValueError, TypeError):
                        # Skip malformed rows
                        continue

            if not turns_change:
                continue

            # X-axis: Episode numbers
            episodes = list(range(1, len(turns_change) + 1))

            # Plotting
            plt.figure(figsize=(10, 6))
            plt.plot(episodes, turns_change, marker='o', color='red', label='Turns % Change')
            plt.plot(episodes, total_sec_change, marker='s', color='green', label='Total Seconds % Change')
            plt.plot(episodes, avg_duration_change, marker='^', color='blue', label='Avg Duration % Change')

            # Horizontal baseline at 0% (average)
            plt.axhline(
                y=0,
                color='black',
                linestyle=':',
                linewidth=2,
                label='Average Baseline (0%)'
            )

            # Extract player name from the file name
            player_name = file_name.replace("_average_comparison.csv", "")

            # Titles and labels
            plt.title(
                f"{player_name} - % Change from Average\n"
                f"(Avg Turns: {avg_turns:.2f}, "
                f"Avg Total Sec: {avg_total_sec:.2f}, "
                f"Avg Duration: {avg_duration:.2f})"
            )
            plt.xlabel("Episode")
            plt.ylabel("% Change from Average")
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend()
            plt.tight_layout()
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

            # Save the plot in the same subfolder as the CSV
            output_file = os.path.join(
                root,
                file_name.replace(".csv", "_lineplot.png")
            )
            plt.savefig(output_file)
            plt.close()

    print("Created average plots successfully!")


if __name__ == "__main__":
    folder_path = "../../resources/speaker_stats/"
    make_baseline_plot(folder_path)
    make_average_plot(folder_path)