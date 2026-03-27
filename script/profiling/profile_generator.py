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

def generate_profile(emotion_path: str, context_path: str, episode: int) -> list[profile]:
    emotions = load_emotions(emotion_path)
    context = load_context(context_path)

    merged = pd.merge(context, emotions, on=["speaker", "start_time"])

    profiles = []
    for _, row in merged.iterrows():
        speaker = row["speaker"]
        if speaker not in profiles:
            profiles[speaker] = profile(name=speaker, episode=episode)

        emotion = emotions[row["emotion"]]
        if emotion in POSITIVE_EMOTIONS:
            profiles[speaker].update(row["skill"], positive=True)
        elif emotion in NEGATIVE_EMOTIONS:
            profiles[speaker].update(row["skill"], positive=False)

    for emotion, row in emotions.iterrows():
        speaker = row["speaker"]
        profiles[speaker].update_emotion(emotion)

    return list(profiles.values())


def save_profiles(profiles: list[profile], output_path: str):
    df = pd.DataFrame([p.to_dict() for p in profiles])
    df = df.sort_values("player").reset_index(drop=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {output_path}")

if __name__ == "__main__":
    emotion_folder = "../../resources/results/video"
    context_folder = "../../resources/transcripts_context"
    output_folder = "../../resources/profiles"
    os.makedirs(output_folder, exist_ok=True)

    emotion_files = {
        os.path.basename(p).replace("_results.csv", ""): p
        for p in glob.glob(f"{emotion_folder}/*_results.csv")
    }
    context_files = {
        os.path.basename(p).replace("_context.csv", ""): p
        for p in glob.glob(f"{context_folder}/*_context.csv")
    }

    episodes = sorted(set(emotion_files) & set(context_files))

    for episode in episodes:
        print(f"Processing {episode}...")
        profiles = generate_profile(
            emotion_path=emotion_files[episode],
            context_path=context_files[episode],
            episode=episode,
        )
        output_path = os.path.join(output_folder, f"{episode}_profiles.csv")
        save_profiles(profiles=profiles, output_path=output_path)

    generate_profile(emotion_folder, context_folder, "100")