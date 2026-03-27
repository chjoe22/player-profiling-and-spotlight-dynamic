import pandas as pd

AUDIO_WEIGHT = 0.38
VIDEO_WEIGHT = 0.55


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


audio = pd.read_csv("../../resources/results/audio/episode0_results.csv")
video = pd.read_csv("../../resources/results/video/episode100_results.csv")

audio["scores"] = audio["scores"].apply(parse_scores)
video["scores"] = video["scores"].apply(parse_scores)

audio["speaker"] = audio["speaker"].astype(str).str.strip()
video["speaker"] = video["speaker"].astype(str).str.strip()

results = []

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
result_df.to_csv("../../resources/results/combined/compared_results.csv", index=False)

print(result_df.head())
print("Saved to ../../resources/results/combined/compared_results.csv")