import glob
import os

import pandas as pd


def find_combat(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    start_mask = (
            df["text"].str.contains("roll", case=False, na=False) &
            df["text"].str.contains("initiative", case=False, na=False)
    )
    end_mask = (
            df["text"].str.contains("How do you want to do this", case=False, na=False) &
            df["speaker"].str.contains("matt", case=False, na=False)
    )
    starts = df[start_mask][["start_time", "speaker", "text"]].reset_index(drop=True)
    ends = df[end_mask][["start_time", "speaker", "text"]].reset_index(drop=True)

    starts = starts.rename(columns={"start_time": "combat_start", "speaker": "start_speaker", "text": "start_text"})
    ends = ends.rename(columns={"start_time": "combat_end", "speaker": "end_speaker", "text": "end_text"})

    combats = pd.concat([starts, ends], axis=1)
    return combats

if __name__ == "__main__":
    folder = "../../resources/transcripts"
    csv_files = glob.glob(f"{folder}/*.csv")
    output_folder_combat = "../../resources/transcripts_context/combat"
    os.makedirs(output_folder_combat, exist_ok=True)


