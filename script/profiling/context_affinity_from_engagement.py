import os
import csv
from statistics import mean


def time_to_seconds(time_str: str) -> int:
    h, m, s = map(int, time_str.split(":"))
    return h * 3600 + m * 60 + s


def load_context_data(context_csv_path: str):
    context_data = {}

    with open(context_csv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            episode = row["episode"]

            total_time = time_to_seconds(row["total_time"])

            combat_time = time_to_seconds(row["combat_time"])
            exploration_time = time_to_seconds(row["exploration_time"])
            social_time = time_to_seconds(row["social_time"])

            if total_time == 0:
                continue

            context_data[episode] = {
                "combat_ratio": combat_time / total_time,
                "exploration_ratio": exploration_time / total_time,
                "social_ratio": social_time / total_time,
            }

    return context_data


def analyze_player_context_affinity(
    speaker_stats_folder: str,
    context_csv_path: str,
):
    context_data = load_context_data(context_csv_path)

    for player_name in os.listdir(speaker_stats_folder):
        player_folder = os.path.join(speaker_stats_folder, player_name)

        if not os.path.isdir(player_folder):
            continue

        engagement_file = None

        for file_name in os.listdir(player_folder):
            if file_name.endswith("_average_comparison.csv"):
                engagement_file = os.path.join(player_folder, file_name)
                break

        if engagement_file is None:
            continue

        episode_results = []

        combat_scores = []
        exploration_scores = []
        social_scores = []

        with open(engagement_file, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                episode = row["episode"]

                if episode == "AVERAGE_BASELINE":
                    continue

                # Match context file episode names
                normalized_episode = episode.replace("_stats", "")

                if normalized_episode not in context_data:
                    continue

                try:
                    engagement = float(
                        row["total_sec_spoken_per_hour_change_from_avg"]
                    )
                except (ValueError, TypeError):
                    continue

                context = context_data[normalized_episode]

                combat_affinity = (
                    engagement * context["combat_ratio"]
                )

                exploration_affinity = (
                    engagement * context["exploration_ratio"]
                )

                social_affinity = (
                    engagement * context["social_ratio"]
                )

                combat_scores.append(combat_affinity)
                exploration_scores.append(exploration_affinity)
                social_scores.append(social_affinity)

                episode_results.append({
                    "episode": normalized_episode,
                    "engagement_change": engagement,
                    "combat_ratio": context["combat_ratio"],
                    "exploration_ratio": context["exploration_ratio"],
                    "social_ratio": context["social_ratio"],
                    "combat_affinity": combat_affinity,
                    "exploration_affinity": exploration_affinity,
                    "social_affinity": social_affinity,
                })

        if not episode_results:
            continue

        output_file = os.path.join(
            player_folder,
            f"{player_name}_context_affinity.csv"
        )

        with open(output_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            writer.writerow([
                "episode",
                "engagement_change",
                "combat_ratio",
                "exploration_ratio",
                "social_ratio",
                "combat_affinity",
                "exploration_affinity",
                "social_affinity"
            ])

            for result in episode_results:
                writer.writerow([
                    result["episode"],
                    result["engagement_change"],
                    result["combat_ratio"],
                    result["exploration_ratio"],
                    result["social_ratio"],
                    result["combat_affinity"],
                    result["exploration_affinity"],
                    result["social_affinity"],
                ])

            # Final averages row
            writer.writerow([])

            writer.writerow([
                "AVERAGE",
                "",
                "",
                "",
                "",
                mean(combat_scores),
                mean(exploration_scores),
                mean(social_scores),
            ])

        print(f"Generated context affinity file for {player_name}")

    print("Player context affinity analysis completed successfully!")


if __name__ == "__main__":
    speaker_stats_folder = "../../resources/speaker_stats/"
    context_csv_path = "../../resources/transcripts_context/assimilated_time_based_context.csv"

    analyze_player_context_affinity(
        speaker_stats_folder,
        context_csv_path,
    )