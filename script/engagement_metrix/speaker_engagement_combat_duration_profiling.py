import os
import pandas as pd
import re


def find_engagement_file(folder):
    candidates = [
        f for f in os.listdir(folder)
        if f.endswith(".csv") and "average_comparison" in f
    ]

    if not candidates:
        raise FileNotFoundError(f"No engagement CSV with 'average_comparison' found in {folder}")

    if len(candidates) > 1:
        print(f"Warning: multiple candidates in {folder}, using {candidates[0]}")

    return os.path.join(folder, candidates[0])


def extract_episode_index(episode_name, base_episode=100):
    """
    Converts '102_transcript' → index 2 (if base_episode=100)
    """
    match = re.match(r"(\d+)", episode_name)
    if not match:
        raise ValueError(f"Invalid episode format: {episode_name}")
    episode_num = int(match.group(1))
    return episode_num - base_episode


def process_person(person_path, combat_df, base_episode=100):
    engagement_file = find_engagement_file(person_path)
    df_eng = pd.read_csv(engagement_file)

    metric = df_eng["total_sec_spoken_per_hour_change_from_avg"].values

    weighted_values = []

    for _, row in combat_df.iterrows():
        try:
            idx = extract_episode_index(row["episode"], base_episode)

            if idx < 0 or idx >= len(metric):
                continue

            weight = metric[idx]
            duration = row["duration_sec"]

            weighted_values.append(duration * weight)

        except Exception:
            continue

    if not weighted_values:
        return None

    return sum(weighted_values) / len(weighted_values)


def save_result(person_path, value):
    output_file = os.path.join(person_path, "combat_engagement_profile.csv")
    pd.DataFrame([{"combat_interest": value}]).to_csv(output_file, index=False)


def process_all(root_folder, combat_file):
    combat_df = pd.read_csv(combat_file)

    all_results = []

    for person in os.listdir(root_folder):
        person_path = os.path.join(root_folder, person)

        if not os.path.isdir(person_path):
            continue

        try:
            result = process_person(person_path, combat_df)

            if result is not None:
                save_result(person_path, result)  # per-player file still written

                all_results.append({
                    "player": person,
                    "combat_interest": result
                })

                print(f"Saved combat result for {person}")
            else:
                print(f"No valid combat data for {person}")

        except Exception as e:
            print(f"Skipping {person}: {type(e).__name__} - {e}")

    # NEW: save global summary
    summary_df = pd.DataFrame(all_results)
    summary_path = os.path.join(root_folder, "all_players_combat_engagement.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"\nSaved global summary to {summary_path}")


if __name__ == "__main__":
    root_folder = "../../resources/speaker_stats"
    combat_file = "../../resources/transcripts_context/combat_duration/combat_durations.csv"

    process_all(root_folder, combat_file)