import glob
import os
from datetime import datetime

import pandas as pd

def parse_time(value) -> int:
    value = str(value).strip()
    t = datetime.strptime(value, "%H:%M:%S")
    return t.hour * 3600 + t.minute * 60 + t.second

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
    starts = df[start_mask][["start_time"]].reset_index(drop=True)
    ends = df[end_mask][["start_time"]].reset_index(drop=True)

    starts = starts.rename(columns={"start_time": "combat_start"})
    ends = ends.rename(columns={"start_time": "combat_end"})

    combats = pd.concat([starts, ends], axis=1).dropna()
    return combats

def get_speaking_under_combat(transcript_path, combat):
    df = pd.read_csv(transcript_path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.dropna(subset=["speaker", "start_time", "end_time"])
    df["start_sec"] = df["start_time"].apply(parse_time)
    df["end_sec"] = df["end_time"].apply(parse_time)
    df["duration"] = df["end_sec"] - df["start_sec"]

    rows = []
    for _, combat in combat.iterrows():
        combat_start = parse_time(combat["combat_start"])
        combat_end = parse_time(combat["combat_end"])


        mask = (df["start_sec"] >= combat_start) & (df["end_sec"] <= combat_end)
        in_combat = df[mask]

        for _, row in in_combat.iterrows():
            for speaker in [s.strip() for s in row["speaker"].split(",")]:
                rows.append({
                        "speaker": speaker,
                        "combat_start": combat["combat_start"],
                        "combat_end": combat["combat_end"],
                        "duration": row["duration"],
                })

    return pd.DataFrame(rows)

if __name__ == "__main__":
    transcript_folder = "../../resources/transcripts"
    output_folder = "../../resources/transcripts_context/combat_speaking"
    os.makedirs(output_folder, exist_ok=True)

    all_rows = []
    combat_hours = {}  # store total combat duration per episode
    all_durations = []

    for transcript_path in sorted(glob.glob(f"{transcript_folder}/*.csv")):
        episode_name = os.path.basename(transcript_path).replace(".csv", "")
        combats = find_combat(transcript_path)

        if combats.empty:
            print(f"{episode_name}: no complete combat pairs found, skipping")
            continue

        for _, combat in combats.iterrows():
            start_sec = parse_time(combat["combat_start"])
            end_sec = parse_time(combat["combat_end"])
            all_durations.append({
                "episode": episode_name,
                "combat_start": combat["combat_start"],
                "combat_end": combat["combat_end"],
                "duration_sec": end_sec - start_sec,
            })
        duration_folder = "../../resources/transcripts_context/combat_duration"
        os.makedirs(duration_folder, exist_ok=True)
        pd.DataFrame(all_durations).to_csv(os.path.join(duration_folder, "combat_durations.csv"), index=False)
        print(f"Saved combat durations")

        # Calculate total combat time in hours for this episode
        total_combat_sec = sum(
            parse_time(row["combat_end"]) - parse_time(row["combat_start"])
            for _, row in combats.iterrows()
        )
        combat_hours[episode_name] = total_combat_sec / 3600

        print(f"{episode_name}: {len(combats)} combat(s) found, {total_combat_sec}s total combat")

        speaking_df = get_speaking_under_combat(transcript_path, combats)
        speaking_df["episode"] = episode_name
        all_rows.append(speaking_df)



    if not all_rows:
        print("No combat data found.")
    else:
        combined = pd.concat(all_rows, ignore_index=True)

        stats = (
            combined.groupby(["speaker", "episode"])
            .agg(
                turns=("duration", "count"),
                total_sec_spoken=("duration", "sum"),
            )
            .reset_index()
        )

        stats["combat_hours"] = stats["episode"].map(combat_hours)
        stats["turns_per_hour"] = stats["turns"] / stats["combat_hours"]
        stats["total_sec_spoken_per_hour"] = stats["total_sec_spoken"] / stats["combat_hours"]


        for speaker, speaker_df in stats.groupby("speaker"):
            output_path = os.path.join(output_folder, f"{speaker}.csv")
            speaker_df.to_csv(output_path, index=False)
            print(f"Saved {output_path}")