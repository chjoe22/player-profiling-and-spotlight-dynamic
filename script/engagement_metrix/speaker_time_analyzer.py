import csv
import os

def get_player_names(path):
	csv_file_path = os.path.join(path, "100_transcritp_stats.csv")
	
	# Read the first column into a list
	first_column_values = []

	with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
		reader = csv.reader(file)

    	# Skip header if the CSV has one
    	header = next(reader)
    
    	for row in reader:
        	if row:
            	first_column_values.append(row[0])

	print(first_column_values)


if __name__ == "__main__":
	speaker_stats_path = "../../resources/transcipts_stats"
	get_player_names(speaker_stats_path)

