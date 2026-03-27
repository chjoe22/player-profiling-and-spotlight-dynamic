import csv
import matplotlib.pyplot as plt
import os

# Path to the baseline CSV file
#csv_file = "../../resources/speaker_stats/ASHLEY_basline_comparison.csv"
folder_path = "../../resources/speaker_stats/"

csv_files = [f for f in os.listdir(folder_path) if "baseline_comparison" in f and f.endswith(".csv")]
for file_name in csv_files:
    csv_path = os.path.join(folder_path, file_name)

    # Lists to store %change values for each data column
    turns_change = []
    total_sec_change = []
    avg_duration_change = []

    # Read the CSV
    with open(csv_path, mode='r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header

        # Find the indices of %change columns
        # Assuming header like: data1, data1%change, data2, data2%change, data3, data3%change
        idx_turns_change = header.index("turns_change_from_baseline")
        idx_total_sec_change = header.index("total_sec_change_from_baseline")
        idx_avg_duration_change = header.index("avg_turn_duration_change_from_baseline")

        for row in reader:
            if not row:
                continue
            turns_change.append(float(row[idx_turns_change]))
            total_sec_change.append(float(row[idx_total_sec_change]))
            avg_duration_change.append(float(row[idx_avg_duration_change]))

    # Plotting
    plt.figure(figsize=(10, 6))
    x = list(range(1, len(turns_change) + 1))  # X-axis = row numbers / files

    plt.plot(x, turns_change, marker='o', color='red', label='Turns taken')
    plt.plot(x, total_sec_change, marker='s', color='green', label='Total seconds')
    plt.plot(x, avg_duration_change, marker='^', color='blue', label='Avg turn duration')

    plt.title("Percentage Change Over episodes (Compared to baseline)")
    plt.xlabel("Episode")
    plt.ylabel("% Change")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    #plt.tight_layout()
    #plt.show()

    output_file = os.path.join(folder_path, file_name.replace(".csv", ".png"))
    plt.savefig(output_file)
    plt.close()

print("Created plots")
