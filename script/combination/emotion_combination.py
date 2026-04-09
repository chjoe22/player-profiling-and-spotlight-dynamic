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

# Kigger på første csv fil først osv
def get_first_csv(folder_path):
    folder = Path(folder_path)
    csv_files = sorted(folder.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in folder: {folder}")

    return csv_files[0]

# Kigger i audio/video folder, combiner weighted graf i compared_results.csv
audio_folder = "../../resources/results/audio"
video_folder = "../../resources/results/video"
output_path = "../../resources/results/combined/compared_results.csv"

audio_file = get_first_csv(audio_folder)
video_file = get_first_csv(video_folder)

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