import os
from datetime import datetime

import pandas as pd

def parse_time(value) -> int:
    if pd.isna(value):
        return 0
    value = str(value).strip()
    t = datetime.strptime(value, "%H:%M:%S")
    return t.hour * 3600 + t.minute * 60 + t.second


def time(csv_path: str):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    df = df.dropna(subset=["speaker", "start_time", "end_time"])
    df["start_sec"] = df["start_time"].apply(parse_time)
    df["end_sec"] = df["end_time"].apply(parse_time)
    df["duration_sec"] = df["end_sec"] - df["start_sec"]

    rows = []
    for _, row in df.iterrows():
        speakers = [s.strip() for s in row["speaker"].split(",")]
        for speaker in speakers:
            rows.append({
                "speaker": speaker,
                "duration_sec": row["duration_sec"],
            })
    expanded_df = pd.DataFrame(rows)

    stats = (
        expanded_df.groupby("speaker")
        .agg(
            turns=("duration_sec", "count"),
            total_sec=("duration_sec", "sum"),
            avg_turn_duration=("duration_sec", "mean"),
        )
        .reset_index()
    )

    stats = stats.sort_values("total_sec", ascending=False).reset_index(drop=True)

    for _, row in stats.iterrows():
        print(f"Speaker: {row['speaker']}")
        print(f"Turns: {int(row['turns'])}")
        print(f"Total time: {int(row['total_sec'])}")
        print(f"Average time: {row['avg_turn_duration']:.2f}")
        print()

    output_path = csv_path.replace(".csv", "_stats.csv")
    output_path = output_path.replace("transcripts", "transcripts_stats")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    stats.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

    return stats


def compare(csv_path):
    df1 = pd.read_csv(csv_path)
    path2 = csv_path.replace(".csv", "_stats.csv")
    df2 = pd.read_csv(csv_path)



if __name__ == "__main__":
    import glob

    folder = "../../resources/transcripts"
    folder_stats = "../../resources/transcripts_stats"

    csv_files = glob.glob(f"{folder}/*.csv")

    if not csv_files:
        print(f"No CSV files found in {folder}")
    else:
        for path in sorted(csv_files):
            print(f"Processing {path}...")
            time(path)
    csv_stats = glob.glob(f"{folder_stats}/*.csv")


    #for path in csv_stats:
        #compare(path)


