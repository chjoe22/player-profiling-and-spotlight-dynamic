import os
import pandas as pd


def find_csv_file(folder):
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            return os.path.join(folder, file)
    raise FileNotFoundError(f"No CSV file found in {folder}")


def process_person(person_path, context_df):
    # Load engagement data
    engagement_file = find_csv_file(person_path)
    df_eng = pd.read_csv(engagement_file)

    metric = df_eng["total_sec_spoken_per_hour_change_from_avg"].values

    results = {"combat": 0, "exploration": 0, "social": 0}
    episode_count = 0

    max_len = min(len(metric), len(context_df))

    for i in range(max_len):
        weight = metric[i]

        for col in ["combat", "exploration", "social"]:
            results[col] += context_df[col].iloc[i] * weight

        episode_count += 1

    if episode_count > 0:
        for key in results:
            results[key] /= episode_count

    return results


def save_results(person_path, results):
    output_file = os.path.join(person_path, "engagement_profile.csv")

    df_out = pd.DataFrame([results])
    df_out.to_csv(output_file, index=False)


def process_all(root_folder, context_file):
    context_df = pd.read_csv(context_file)

    for person in os.listdir(root_folder):
        person_path = os.path.join(root_folder, person)

        if not os.path.isdir(person_path):
            continue

        try:
            results = process_person(person_path, context_df)
            save_results(person_path, results)
            print(f"Saved results for {person}")
        except Exception as e:
            print(f"Skipping {person}: {e}")


if __name__ == "__main__":
    player_engagement_stats_root = "../../resources/speaker_stats"
    episode_context_folder = "../../resources/transcripts_context/scenario_counts/all_scenario_counts.csv"

    process_all(player_engagement_stats_root, episode_context_folder)

    #for person, metrics in results.items():
    #    print(f"\n{person}")
    #    for k, v in metrics.items():
    #        print(f"  {k}: {v:.4f}")