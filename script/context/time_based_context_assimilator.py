from pathlib import Path
import pandas as pd
from datetime import timedelta


# =========================
# CONFIG
# =========================

TRANSCRIPT_FOLDER = r"../../resources/transcripts"
COMBAT_CSV = r"../../resources/transcripts_context/combat_duration/combat_durations.csv"
METRICS_CSV = r"../../resources/transcripts_context/scenario_counts/all_scenario_counts.csv"
OUTPUT_CSV = r"../../resources/transcripts_context/assimilated_time_based_context.csv"


# =========================
# HELPERS
# =========================

def time_to_seconds(time_str):
    """
    Convert HH:MM:SS to total seconds.
    """
    if pd.isna(time_str) or time_str == "":
        return 0

    h, m, s = map(int, time_str.split(":"))
    return h * 3600 + m * 60 + s


def seconds_to_hms(seconds):
    """
    Convert seconds to HH:MM:SS format.
    """
    seconds = int(round(seconds))

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    return f"{hours:02}:{minutes:02}:{secs:02}"


def get_episode_duration(transcript_path):
    """
    Read transcript CSV and return the final valid end_time as seconds.
    Handles missing end_time in the last row by falling back to:
    - second-to-last row end_time
    - or last row start_time
    """
    df = pd.read_csv(transcript_path)

    if df.empty:
        return 0

    # Try last row end_time
    last_end = df.iloc[-1].get("end_time", None)

    if pd.notna(last_end) and str(last_end).strip() != "":
        return time_to_seconds(last_end)

    # Fallback: second last row end_time
    if len(df) >= 2:
        prev_end = df.iloc[-2].get("end_time", None)
        if pd.notna(prev_end) and str(prev_end).strip() != "":
            return time_to_seconds(prev_end)

    # Final fallback: last row start_time
    last_start = df.iloc[-1].get("start_time", None)
    if pd.notna(last_start) and str(last_start).strip() != "":
        return time_to_seconds(last_start)

    return 0


# =========================
# LOAD DATA
# =========================

combat_df = pd.read_csv(COMBAT_CSV)
metrics_df = pd.read_csv(METRICS_CSV)

# Group combat durations by episode
combat_totals = (
    combat_df.groupby("episode")["duration_sec"]
    .sum()
    .to_dict()
)

# Metrics lookup
metrics_lookup = metrics_df.set_index("episode").to_dict(orient="index")


# =========================
# PROCESS EPISODES
# =========================

results = []

transcript_folder = Path(TRANSCRIPT_FOLDER)

for transcript_file in transcript_folder.glob("*.csv"):

    episode_name = transcript_file.stem

    print(f"Processing: {episode_name}")

    # -------------------------
    # Total episode duration
    # -------------------------
    total_duration_sec = get_episode_duration(transcript_file)

    # -------------------------
    # Combat duration
    # -------------------------
    combat_duration_sec = combat_totals.get(episode_name, 0)

    # Prevent negative durations
    non_combat_sec = max(total_duration_sec - combat_duration_sec, 0)

    # -------------------------
    # Metrics
    # -------------------------
    metric_data = metrics_lookup.get(episode_name)

    if metric_data is None:
        print(f"  No metrics found for {episode_name}, skipping.")
        continue

    exploration_metric = metric_data["exploration"]
    social_metric = metric_data["social"]

    total_metric = exploration_metric + social_metric

    # -------------------------
    # Allocate time
    # -------------------------
    if total_metric == 0:
        exploration_sec = 0
        social_sec = 0
    else:
        exploration_sec = (
            non_combat_sec * exploration_metric / total_metric
        )

        social_sec = (
            non_combat_sec * social_metric / total_metric
        )

    # -------------------------
    # Store result
    # -------------------------
    results.append({
        "episode": episode_name,
        "total_time": seconds_to_hms(total_duration_sec),
        "combat_time": seconds_to_hms(combat_duration_sec),
        "exploration_time": seconds_to_hms(exploration_sec),
        "social_time": seconds_to_hms(social_sec),
    })


# =========================
# SAVE OUTPUT
# =========================

output_df = pd.DataFrame(results)

# Optional sorting
output_df = output_df.sort_values("episode")

output_df.to_csv(OUTPUT_CSV, index=False)

print("\nDone.")
print(f"Saved output to: {OUTPUT_CSV}")