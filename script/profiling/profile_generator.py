import os
import glob
import pandas as pd
from profile import profile
from collections import defaultdict

POSITIVE_EMOTIONS = {"happy", "surprised"}
NEGATIVE_EMOTIONS = {"angry", "disgust", "fearful", "sad"}
from collections import defaultdict

EMOTION_MAP = {
    "fearful": "fear",
    "happy": "happy",
    "angry": "angry",
    "disgust": "disgust",
    "sad": "sad",
    "surprised": "surprise",
    "neutral": "neutral",
}


def get_emotion_at(emotions: pd.DataFrame, speaker: str, time: str) -> str | None:
    mask = (
        (emotions["speaker"] == speaker) &
        (emotions["start_time"] <= time) &
        (emotions["end_time"] >= time)
    )
    matches = emotions[mask]
    if matches.empty:
        return None
    return matches.iloc[0]["emotion"]

def load_emotions(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    print("Combined columns:", df.columns.tolist())
    df = df.rename(columns={"timestamp": "start_time", "final_emotion": "emotion"})
    df["start_time"] = pd.to_datetime(df["start_time"], format="mixed").dt.strftime("%H:%M:%S")
    return df[["speaker", "start_time", "end_time", "emotion"]]

def load_context(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    return df[["skill", "speaker", "start_time"]]

def generate_profile(emotion_path: str, context_path: str, episode: int):
    emotions = load_emotions(emotion_path)
    context = load_context(context_path)

    print("Emotion columns:", emotions.columns.tolist())
    print("context columns:", context.columns.tolist())
    print("Emotion start_time sample:", emotions["start_time"].head())
    print("context start_time sample:", context["start_time"].head())

    profiles = {}
    for _, row in context.iterrows():
        speakers = [s.strip() for s in row["speaker"].split(",")]
        for speaker in speakers:
            emotion = get_emotion_at(emotions, speaker, row["start_time"])
            if emotion is None:
                continue

            if speaker not in profiles:
                profiles[speaker] = profile(name=speaker, episode=episode)

            if emotion in POSITIVE_EMOTIONS:
                profiles[speaker].update_scenario(row["skill"], positive=True)
            elif emotion in NEGATIVE_EMOTIONS:
                profiles[speaker].update_scenario(row["skill"], positive=False)

    for _, row in emotions.iterrows():
        speakers = [s.strip() for s in row["speaker"].split(",")]
        for speaker in speakers:
            emotion = EMOTION_MAP.get(row["emotion"], None)
            if emotion is None:
                continue
            if speaker not in profiles:
                profiles[speaker] = profile(name=speaker, episode=episode)
            profiles[speaker].update_emotion(emotion)

    """emotion = row["emotion"].lower()
    if emotion in POSITIVE_EMOTIONS:
        profiles[speaker].update_scenario(row["skill"], positive=True)"""

    return list(profiles.values())


def save_profiles(profiles: list[profile], output_path: str):
    df = pd.DataFrame([p.to_dict() for p in profiles])
    df = df.sort_values("player").reset_index(drop=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {output_path}")

if __name__ == "__main__":
    emotion_folder = "../../resources/results/combined"
    context_folder = "../../resources/transcripts_context/skills"
    combat_folder = "../../resources/transcripts_context/combat"
    output_folder = "../../resources/profiles"
    os.makedirs(output_folder, exist_ok=True)

    emotion_files = {
        os.path.basename(p).replace("_weighted.csv", ""): p
        for p in glob.glob(f"{emotion_folder}/*_weighted.csv")
    }
    context_files = {
        os.path.basename(p).replace("_transcript_skill.csv", ""): p
        for p in glob.glob(f"{context_folder}/*_transcript_skill.csv")
    }

    print("Emotion files:", emotion_files)
    print("context files:", context_files)
    episodes = sorted(set(emotion_files) & set(context_files))

    print("Emotion files:", emotion_files)
    print("context files:", list(context_files.keys())[:5])
    print("Matching episodes:", episodes)

    all_profiles = defaultdict(list)

    for episode in episodes:
        print(f"Processing {episode}...")
        profiles = generate_profile(
            emotion_path=emotion_files[episode],
            context_path=context_files[episode],
            episode=episode,
        )
        for p in profiles:
            all_profiles[p.name].append(p.to_dict())

    os.makedirs(output_folder, exist_ok=True)
    for player, dicts in all_profiles.items():
        df = pd.DataFrame(dicts).sort_values("episode").reset_index(drop=True)
        output_path = os.path.join(output_folder, f"{player}.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved {output_path}")