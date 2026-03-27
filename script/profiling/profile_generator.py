import os
import glob
import pandas as pd
from profile import profile

POSITIVE_EMOTIONS = {"happy", "surprised"}
NEGATIVE_EMOTIONS = {"angry", "disgust", "fearful", "sad"}

def load_emotions(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    return df[["speaker", "start_time", "emotion"]]

def load_context(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = [c.strip().lower() for c in df.columns]
    return df[["skill", "speaker", "start_time", "roll_start_time"]]

def generate_profile(emotion_path: str, context_path: str, episode: str) -> list[profile]:
    emotions = load_emotions(emotion_path)
    context = load_context(context_path)

    merged = pd.merge(context, emotions, on=["speaker", "start_time"])

    profiles = []
    for _, row in merged.iterrows():
        speaker = row["speaker"]
        if speaker not in profiles:
            profiles[speaker] = profile(name=speaker, episode=episode)

        emotion = emotions[row["emotion"]]

def save_profiles(profiles: list[profile], output_path: str):
    df = pd.DataFrame([p.to_dict() for p in profiles])
    df = df.sort_values("player").reset_index(drop=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {output_path}")

