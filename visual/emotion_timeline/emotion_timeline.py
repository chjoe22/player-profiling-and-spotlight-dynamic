import os
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch



# Paths

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

INPUT_CSV = os.path.join(PROJECT_ROOT, "results", "video", "episode100_results.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "generated")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# How many minutes each "act" should cover
CHUNK_MINUTES = 10



# Helpers
def is_single_speaker(speaker):
    if pd.isna(speaker):
        return False

    speaker = str(speaker).strip()

    if not speaker:
        return False

    if speaker.upper() == "ALL":
        return False

    return not any(x in speaker for x in [",", "&", "/"])


def time_to_seconds(value):
    if pd.isna(value):
        return None

    value = str(value).strip()
    if not value:
        return None

    parts = value.split(":")
    if len(parts) != 3:
        return None

    try:
        h = float(parts[0])
        m = float(parts[1])
        s = float(parts[2])
        return h * 3600 + m * 60 + s
    except ValueError:
        return None


def seconds_to_hms(seconds):
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"



# Load & clean data

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)

    print("Columns found:", list(df.columns))

    required_cols = {"timestamp", "speaker", "emotion"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df = df.dropna(subset=["timestamp", "speaker", "emotion"]).copy()

    df["speaker"] = df["speaker"].astype(str).str.strip()
    df["emotion"] = df["emotion"].astype(str).str.strip().str.lower()

    df = df[df["speaker"].apply(is_single_speaker)].copy()

    df["time_sec"] = df["timestamp"].apply(time_to_seconds)
    df = df.dropna(subset=["time_sec"]).copy()

    if "frame_id" in df.columns:
        df["frame_id_num"] = pd.to_numeric(df["frame_id"], errors="coerce")
        df = df.sort_values(["speaker", "time_sec", "frame_id_num"]).reset_index(drop=True)
    else:
        df = df.sort_values(["speaker", "time_sec"]).reset_index(drop=True)

    print("Rows after cleaning:", len(df))
    print("Speakers found:", sorted(df["speaker"].unique()))
    print("Time range:", seconds_to_hms(df["time_sec"].min()), "to", seconds_to_hms(df["time_sec"].max()))

    return df



# Merge nearby segments with same emotions

def merge_segments(df, max_gap=1.5, min_duration=0.8):
    if df.empty:
        return pd.DataFrame(columns=["speaker", "emotion", "start_sec", "end_sec", "duration_sec"])

    merged_rows = []

    for speaker, group in df.groupby("speaker"):
        group = group.sort_values("time_sec").reset_index(drop=True)

        current_emotion = group.loc[0, "emotion"]
        current_start = group.loc[0, "time_sec"]
        current_end = group.loc[0, "time_sec"]

        for i in range(1, len(group)):
            row = group.loc[i]
            emotion = row["emotion"]
            t = row["time_sec"]

            prev_t = group.loc[i - 1, "time_sec"]
            gap = t - prev_t

            if emotion == current_emotion and gap <= max_gap:
                current_end = t
            else:
                duration = max(current_end - current_start, min_duration)
                merged_rows.append({
                    "speaker": speaker,
                    "emotion": current_emotion,
                    "start_sec": current_start,
                    "end_sec": current_start + duration,
                    "duration_sec": duration
                })

                current_emotion = emotion
                current_start = t
                current_end = t

        duration = max(current_end - current_start, min_duration)
        merged_rows.append({
            "speaker": speaker,
            "emotion": current_emotion,
            "start_sec": current_start,
            "end_sec": current_start + duration,
            "duration_sec": duration
        })

    merged_df = pd.DataFrame(merged_rows)
    merged_df = merged_df.sort_values(["speaker", "start_sec"]).reset_index(drop=True)

    return merged_df



# Plotting

def plot_emotion_timeline_chunk(df, output_path, title, chunk_start, chunk_end, all_speakers):
    if df.empty:
        print(f"No data to plot for {title}")
        return

    emotion_colors = {
        "anger": "#d62728",
        "angry": "#d62728",
        "joy": "#ffbf00",
        "happy": "#ffbf00",
        "sadness": "#1f77b4",
        "sad": "#1f77b4",
        "fear": "#9467bd",
        "surprise": "#ff7f0e",
        "disgust": "#2ca02c",
        "neutral": "#7f7f7f",
        "calm": "#8c564b",
    }

    speaker_to_y = {speaker: i for i, speaker in enumerate(all_speakers)}

    fig_height = max(6, len(all_speakers) * 0.9)
    fig, ax = plt.subplots(figsize=(20, fig_height))

    used_emotions = set()

    for _, row in df.iterrows():
        speaker = row["speaker"]
        emotion = row["emotion"]
        start = max(row["start_sec"], chunk_start)
        end = min(row["end_sec"], chunk_end)
        duration = end - start

        if duration <= 0:
            continue

        y = speaker_to_y[speaker]
        color = emotion_colors.get(emotion, "#cccccc")
        used_emotions.add(emotion)

        ax.broken_barh(
            [(start, duration)],
            (y - 0.35, 0.7),
            facecolors=color,
            edgecolors="black",
            linewidth=0.3
        )

    tick_step = 60
    chunk_duration = chunk_end - chunk_start
    if chunk_duration > 1800:
        tick_step = 300
    if chunk_duration > 3600:
        tick_step = 600

    ticks = list(range(int(chunk_start), int(chunk_end) + tick_step, tick_step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([seconds_to_hms(t) for t in ticks], rotation=45, ha="right")

    ax.set_yticks(range(len(all_speakers)))
    ax.set_yticklabels(all_speakers)

    ax.set_xlim(chunk_start, chunk_end)
    ax.set_ylim(-1, len(all_speakers))
    ax.set_xlabel("Time")
    ax.set_ylabel("Speaker")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    legend_handles = [
        Patch(facecolor=emotion_colors.get(emotion, "#cccccc"), edgecolor="black", label=emotion)
        for emotion in sorted(used_emotions)
    ]

    if legend_handles:
        ax.legend(
            handles=legend_handles,
            title="Emotion",
            bbox_to_anchor=(1.02, 1),
            loc="upper left"
        )

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")



# Split into acts/chunks
def save_timeline_chunks(df, output_dir, chunk_minutes=10):
    if df.empty:
        print("No data available for chunking.")
        return

    all_speakers = sorted(df["speaker"].unique())

    min_time = int(df["start_sec"].min())
    max_time = int(df["end_sec"].max())

    chunk_size = chunk_minutes * 60
    num_chunks = math.ceil((max_time - min_time) / chunk_size)

    print(f"Creating {num_chunks} act(s) of {chunk_minutes} minute(s) each")

    for i in range(num_chunks):
        chunk_start = min_time + i * chunk_size
        chunk_end = min(chunk_start + chunk_size, max_time)

        chunk_df = df[
            (df["end_sec"] >= chunk_start) &
            (df["start_sec"] <= chunk_end)
        ].copy()

        act_num = i + 1
        title = f"Emotion Timeline by Speaker — Act {act_num} ({seconds_to_hms(chunk_start)} to {seconds_to_hms(chunk_end)})"
        output_path = os.path.join(output_dir, f"emotion_timeline_act_{act_num}.png")

        plot_emotion_timeline_chunk(
            chunk_df,
            output_path,
            title,
            chunk_start,
            chunk_end,
            all_speakers
        )




def main():
    print("Looking for CSV at:", INPUT_CSV)
    print("CSV exists:", os.path.exists(INPUT_CSV))
    print("Saving output to:", OUTPUT_DIR)

    df = load_data(INPUT_CSV)
    merged_df = merge_segments(df, max_gap=1.5, min_duration=0.8)

    print("Segments after merging:", len(merged_df))
    print(merged_df.head(10).to_string(index=False))

    save_timeline_chunks(merged_df, OUTPUT_DIR, chunk_minutes=CHUNK_MINUTES)


if __name__ == "__main__":
    main()