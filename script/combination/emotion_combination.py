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

# Kigger på csv filer i folder
def get_csv_files(folder_path):
    folder = Path(folder_path)
    csv_files = sorted(folder.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in folder: {folder}")

    return csv_files

# Henter nummer først i filnavn
def get_first_number(filename):
    return filename.split("_")[0]

# Kigger i audio/video folder, combiner weighted graf i ny csv fil
audio_folder = "../../resources/results/audio"
video_folder = "../../resources/results/video"

audio_files = glob.glob(f"{audio_folder}/**/*.csv", recursive=True)
video_files = glob.glob(f"{video_folder}/**/*.csv", recursive=True)

os.makedirs("../../resources/results/combined/", exist_ok=True)

print(len(audio_files))

# Går igennem alle audio filer
for audio_file in audio_files:
    audio_number = get_first_number(os.path.basename(audio_file).replace(".csv", ""))
    video_file = None

    # Kigger efter video fil med samme første nummer
    for v_file in video_files:
        video_number = get_first_number(os.path.basename(v_file).replace(".csv", ""))
        if audio_number == video_number:
            video_file = v_file
            break

    # Hvis der ikke er match i video så skipper den
    if video_file is None:
        print(f"Ingen matchende video fil til {audio_file}")
        continue

    # Navngivet ud fra audio, evt skift
    episode_name = audio_number
    output_path = f"../../resources/results/combined/{episode_name}_weighted.csv"

    audio = pd.read_csv(audio_file)
    video = pd.read_csv(video_file)

    audio["scores"] = audio["scores"].apply(parse_scores)
    video["scores"] = video["scores"].apply(parse_scores)

    audio["speaker"] = audio["speaker"].astype(str).str.strip()
    video["speaker"] = video["speaker"].astype(str).str.strip()

    results = []

    # Finder match i video for speaker i audio, laver en weighted udregning for emotion samlet
    for _, a in audio.iterrows():
        matches = video[video["speaker"] == a["speaker"]]

        video_avg = {}
        for row in matches["scores"]:
            for emotion, score in row.items():
                video_avg[emotion] = video_avg.get(emotion, 0) + score

        if not matches.empty:
            for emotion in video_avg:
                video_avg[emotion] /= len(matches)

        combined = {}
        all_emotions = set(a["scores"]) | set(video_avg)

        for emotion in all_emotions:
            audio_score = a["scores"].get(emotion, 0)
            video_score = video_avg.get(emotion, 0)
            combined[emotion] = audio_score * AUDIO_WEIGHT + video_score * VIDEO_WEIGHT

        combined = dict(sorted(combined.items(), key=lambda x: x[1], reverse=True))
        final_emotion = max(combined, key=combined.get) if combined else None

        results.append({
            "speaker": a["speaker"],
            "final_emotion": final_emotion,
            "start_time": a["start_time"],
            "end_time": a["end_time"],
            "combined_scores": str(combined)
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_path, index=False)

# Legally blind så for brug for dem her nogle gange
# print(result_df.head())
# print(f"Loaded audio from: {audio_file}")
# print(f"Loaded video from: {video_file}")
# print(f"Saved to {output_path}")