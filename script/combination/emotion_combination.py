import glob
import os
import pandas as pd
from pathlib import Path

AUDIO_WEIGHT = 0.38
VIDEO_WEIGHT = 0.55

# Parser
def parse_scores(s):
    s = str(s).strip().strip("{}")
    scores = {}

    if not s:
        return scores

    for part in s.split(","):
        part = part.strip()
        if ":" not in part:
            continue

        key, value = part.split(":", 1)
        key = key.strip().strip("'").strip('"')

        try:
            scores[key] = float(value.strip())
        except ValueError:
            continue

    return scores

# Gets all the csv files
def get_csv_files(folder_path):
    folder = Path(folder_path)
    csv_files = sorted(folder.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in folder: {folder}")

    return csv_files

# Gets first number in files
def get_first_number(filename):
    return filename.split("_")[0]

# looks in audio/video folder, combines weighted graph in new csv file
audio_folder = "../../resources/results/audio"
video_folder = "../../resources/results/video"

audio_files = glob.glob(f"{audio_folder}/**/*.csv", recursive=True)
video_files = glob.glob(f"{video_folder}/**/*.csv", recursive=True)

os.makedirs("../../resources/results/combined/", exist_ok=True)

print(len(audio_files))

folder_count = False

# Goes through all audio files
for audio_file in audio_files:


    audio_number = get_first_number(os.path.basename(audio_file).replace(".csv", ""))
    video_file = None
    if folder_count:
        break
    elif audio_number == "120":
        folder_count = True

    # Looks for matching video file
    for v_file in video_files:
        video_number = get_first_number(os.path.basename(v_file).replace(".csv", ""))
        if audio_number == video_number:
            video_file = v_file
            break

    # skips if missing
    if video_file is None:
        print(f"Ingen matchende video fil til {audio_file}")
        continue

    # naming
    episode_name = audio_number
    output_path = f"../../resources/results/combined/{episode_name}_weighted.csv"

    audio = pd.read_csv(audio_file)
    video = pd.read_csv(video_file)

    audio["scores"] = audio["scores"].apply(parse_scores)
    video["scores"] = video["scores"].apply(parse_scores)

    # Normalize timestamps to HH:MM:SS
    audio["start_time"] = pd.to_datetime(audio["start_time"], format="mixed").dt.strftime("%H:%M:%S")
    video["timestamp"] = pd.to_datetime(video["timestamp"], format="mixed").dt.strftime("%H:%M:%S")
    video["speaker"] = video["speaker"].str.upper().str.strip()

    audio_keys = set(zip(audio["speaker"], audio["start_time"]))

    results = []

    for _, a in audio.iterrows():
        v_match = video[
            (video["speaker"] == a["speaker"]) &
            (video["timestamp"] >= a["start_time"] & video["timestamp"] <= a["end_time"])
            ]

        if not v_match.empty:
            video_avg = {}
            for _, v_row in v_match.iterrows():
                for emotion, score in v_row["scores"].items():
                    video_avg[emotion] = video_avg.get(emotion, 0) + score
            for emotion in video_avg:
                video_avg[emotion] /= len(v_match)
            video_scores = video_avg
            
            combined = {}
            all_emotions = set(a["scores"]) | set(video_scores)
            for emotion in all_emotions:
                combined[emotion] = a["scores"].get(emotion, 0) * AUDIO_WEIGHT + video_scores.get(emotion,
                                                                                                  0) * VIDEO_WEIGHT
        else:
            combined = a["scores"]

        combined = dict(sorted(combined.items(), key=lambda x: x[1], reverse=True))
        final_emotion = max(combined, key=combined.get) if combined else None
        results.append({
            "speaker": a["speaker"],
            "final_emotion": final_emotion,
            "start_time": a["start_time"],
            "end_time": a["end_time"],
            "combined_scores": str(combined)
        })


    for _, v in video.iterrows():
        if (v["speaker"], v["timestamp"]) not in audio_keys:
            scores = v["scores"]
            scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
            final_emotion = max(scores, key=scores.get) if scores else None
            results.append({
                "speaker": v["speaker"],
                "final_emotion": final_emotion,
                "start_time": v["timestamp"],
                "end_time": v["timestamp"],
                "combined_scores": str(scores)
            })

    print(f"Audio rows: {len(audio)}")
    print(f"Video rows: {len(video)}")
    print(f"Combined rows: {len(results)}")
    result_df = pd.DataFrame(results)
    result_df.sort_values(["speaker", "start_time"]).reset_index(drop=True)
    result_df.to_csv(output_path, index=False)

# Legally blind så for brug for dem her nogle gange
# print(result_df.head())
# print(f"Loaded audio from: {audio_file}")
# print(f"Loaded video from: {video_file}")
# print(f"Saved to {output_path}")