import csv
import os
import re
from pprint import pprint
from statistics import mean
from typing import Dict, List, Tuple, Optional


def get_player_names(path: str):
    players = ["LAURA", "LIAM", "MARISHA", "SAM", "TALIESIN", "TRAVIS", "ASHLEY"]

    return list(players)



def extract_data(path: str, player_names):
    player_data = {name: [] for name in player_names}

    # Handle ALL csv files in folder
    csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
    csv_files.sort()  # simple alphabetical ordering

    for file_name in csv_files:
        file_path = os.path.join(path, file_name)

        with open(file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader, None)  # skip header safely

            for row in reader:
                if not row:
                    continue

                name_in_row = row[0]

                if name_in_row in player_data:
                    datapoints = []

                    # Extract ALL numeric columns (skip speaker column)
                    for value in row[1:]:
                        try:
                            datapoints.append(float(value))
                        except ValueError:
                            datapoints.append(None)

                    # Store episode/file name with datapoints
                    player_data[name_in_row].append({
                        "episode": os.path.splitext(file_name)[0],
                        "data": datapoints
                    })

    return player_data


#DEPRECATED
def analyse_baseline_comparison(output_path: str, player_data):
	os.makedirs(output_path, exist_ok=True)  # create folder if it doesn't exist
	
	for player, data_rows in player_data.items():
		if not data_rows:
			continue  # skip players with no data
	    
	    # Take first row as baseline
		baseline = data_rows[0]

		player_folder = os.path.join(output_path, player)
		os.makedirs(player_folder, exist_ok=True)

		# Save the CSV inside the player's folder
		output_file = os.path.join(
    		player_folder, f"{player}_baseline_comparison.csv"
		)

		with open(output_file, mode='w', newline='', encoding='utf-8') as f:
			writer = csv.writer(f)
	        
	        # Write header
			writer.writerow([
	            "turns", "turns_change_from_baseline",
	            "total_sec", "total_sec_change_from_baseline",
	            "avg_turn_duration", "avg_turn_duration_change_from_baseline"
	        ])
	        
	        # Write data rows
			for row in data_rows:
				new_row = []
				for i in range(3):
					value = row[i]
					change = ((value - baseline[i]) / baseline[i] * 100) if baseline[i] != 0 else None
					new_row.extend([value, change])
				writer.writerow(new_row)

	print("Player speaker frequency baseline analysis CSV files generated successfully!")


#DEPRECATED!
def analyse_rolling_change_comparison(output_path: str, player_data):
	os.makedirs(output_path, exist_ok=True)  # create folder if it doesn't exist
	
	for player, data_rows in player_data.items():
		if not data_rows:
			continue  # skip players with no data

		player_folder = os.path.join(output_path, player)
		os.makedirs(player_folder, exist_ok=True)

		# Save the CSV inside the player's folder
		output_file = os.path.join(
    		player_folder, f"{player}_rolling_comparison.csv"
		)

		with open(output_file, mode='w', newline='', encoding='utf-8') as f:
			writer = csv.writer(f)

			# Write header
			writer.writerow([
				"turns", "turns_change",
				"total_sec", "total_sec_change",
				"avg_turn_duration", "avg_turn_duration_change"
			])

			previous_row = data_rows[0]  # first row has no previous, so %change = 0
			# Write the first row (with 0% change)
			writer.writerow([val for x in previous_row for val in (x, 0.0)])

			# Process remaining rows
			for row in data_rows[1:]:
				new_row = []
				for i in range(3):
					value = row[i]
					prev_value = previous_row[i]
					change = ((value - prev_value) / prev_value * 100) if prev_value != 0 else None
					new_row.extend([value, change])
				writer.writerow(new_row)
				previous_row = row  # update previous_row for next iteration

	print("Player speaker frequency rolling change analysis CSV files generated successfully!")



def analyse_average_change_comparison(output_path: str, player_data):
    os.makedirs(output_path, exist_ok=True)

    metric_names = [
        "turns",
        "total_sec_spoken",
        "avg_turn_duration",
        "total_sec_spoken_per_hour",
        "turns_per_hour"
    ]

    for player, entries in player_data.items():
        if not entries:
            continue

        num_metrics = len(entries[0]["data"])

        # --- Calculate baseline averages (ignore None) ---
        baseline = []

        for i in range(num_metrics):
            values = [
                entry["data"][i]
                for entry in entries
                if entry["data"][i] is not None
            ]

            baseline.append(mean(values) if values else 0)

        player_folder = os.path.join(output_path, player)
        os.makedirs(player_folder, exist_ok=True)

        output_file = os.path.join(
            player_folder,
            f"{player}_average_comparison.csv"
        )

        with open(output_file, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # --- Header ---
            header = ["episode"]

            for name in metric_names:
                header.extend([name, f"{name}_change_from_avg"])

            writer.writerow(header)

            # --- Compare each episode to average baseline ---
            sorted_entries = sorted(
                entries,
                key=lambda e: int(re.match(r"(\d+)", e["episode"]).group(1))
                if re.match(r"(\d+)", e["episode"]) else float("inf")
            )
            for entry in sorted_entries:
                episode = entry["episode"]
                row = entry["data"]

                new_row = [episode]

                for i in range(num_metrics):
                    value = row[i]
                    baseline_value = baseline[i]

                    if value is None or baseline_value == 0:
                        change = None
                    else:
                        change = ((value - baseline_value) / baseline_value) * 100

                    new_row.extend([value, change])

                writer.writerow(new_row)

            # --- Baseline row ---
            writer.writerow([])

            baseline_row = ["AVERAGE_BASELINE"]

            for value in baseline:
                baseline_row.extend([value, 0.0])

            writer.writerow(baseline_row)

    print("Player speaker frequency average baseline comparison CSV files generated successfully!")


if __name__ == "__main__":
	speaker_stats_path = "../../resources/transcripts_stats/"
	output_folder = "../../resources/speaker_stats/"
	
	player_names = get_player_names(speaker_stats_path)
	player_data = extract_data(speaker_stats_path, player_names)
	#analyse_baseline_comparison(output_folder, player_data) #DEPRECATED
	#analyse_rolling_change_comparison(output_folder, player_data) #DEPRECATED
	analyse_average_change_comparison(output_folder, player_data)

