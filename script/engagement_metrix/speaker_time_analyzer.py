import csv
import os
import re
from pprint import pprint
from statistics import mean
from typing import Dict, List, Tuple, Optional


def get_player_names(path: str):
	csv_file_path = os.path.join(path, "100_transcript_stats.csv") #Yes, the starting file is currently hardcoded
	
	# Read the first column into a list
	first_column_values = []

	with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
		reader = csv.reader(file)

    	# Skip header if the CSV has one
		header = next(reader)
    
		for row in reader:
			if row:
				first_column_values.append(row[0])

	return first_column_values



def extract_data(path: str, player_names):
	player_data = {name: [] for name in player_names}

	# List all CSV files
	csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]

	# Sort files numerically by the starting number
	csv_files.sort(key=lambda x: int(re.match(r"^(\d+)", x).group(1)))

	# Loop through each file in order
	for file_name in csv_files:
		file_path = os.path.join(path, file_name)
		with open(file_path, mode='r', newline='', encoding='utf-8') as file:
			reader = csv.reader(file)
			header = next(reader)  # skip header if present

			for row in reader:
				if not row:
					continue  # skip empty rows
				name_in_row = row[0]
				if name_in_row in player_data:
                
                	# Extract the 3 datapoints from the row (convert to float)
					try:
						datapoints = [float(x) for x in row[1:4]]
					except ValueError:
						datapoints = [None, None, None]  # handle missing or invalid numbers
					player_data[name_in_row].append(datapoints)
	return player_data



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
	os.makedirs(output_path, exist_ok=True)  # Create folder if it doesn't exist

	for player, data_rows in player_data.items():
		if not data_rows:
			continue  # Skip players with no data

        # --- Calculate baseline averages ---
		avg_turns = mean(row[0] for row in data_rows)
		avg_total_sec = mean(row[1] for row in data_rows)
		avg_avg_turn_duration = mean(row[2] for row in data_rows)

		baseline = [avg_turns, avg_total_sec, avg_avg_turn_duration]

		player_folder = os.path.join(output_path, player)
		os.makedirs(player_folder, exist_ok=True)

		output_file = os.path.join(
    		player_folder, f"{player}_average_comparison.csv"
		)

		with open(output_file, mode='w', newline='', encoding='utf-8') as f:
			writer = csv.writer(f)

            # Write header
			writer.writerow([
                "turns", "turns_change_from_avg",
                "total_sec", "total_sec_change_from_avg",
                "avg_turn_duration", "avg_turn_duration_change_from_avg"
			])

            # --- Compare each episode to the baseline ---
			for row in data_rows:
				new_row = []
				for i in range(3):
					value = row[i]
					baseline_value = baseline[i]

                    # Calculate percentage change from the average baseline
					change: Optional[float]
					if baseline_value != 0:
						change = ((value - baseline_value) / baseline_value) * 100
					else:
						change = None  # Avoid division by zero

					new_row.extend([value, change])

				writer.writerow(new_row)

            # Optionally, append a row showing the baseline averages
			writer.writerow([])
			writer.writerow([
                avg_turns, 0.0,
                avg_total_sec, 0.0,
                avg_avg_turn_duration, 0.0
			])

	print("Player speaker frequency average baseline comparison CSV files generated successfully!")


if __name__ == "__main__":
	speaker_stats_path = "../../resources/transcripts_stats/"
	output_folder = "../../resources/speaker_stats/"
	
	player_names = get_player_names(speaker_stats_path)
	player_data = extract_data(speaker_stats_path, player_names)
	analyse_baseline_comparison(output_folder, player_data)
	analyse_rolling_change_comparison(output_folder, player_data)
	analyse_average_change_comparison(output_folder, player_data)

