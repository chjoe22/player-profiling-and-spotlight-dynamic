import os
import csv
from statistics import mean, stdev


def time_to_seconds(time_str: str) -> int:
    h, m, s = map(int, time_str.split(":"))
    return h * 3600 + m * 60 + s


def normalize_episode(ep: str) -> str:
    ep = ep.lower()
    ep = ep.replace("_transcript", "")
    ep = ep.replace("_stats", "")
    ep = ep.replace("episode_", "")
    return ep


def load_context_data(context_csv_path: str):
    context_data = {}
    episode_order = []

    with open(context_csv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            ep = normalize_episode(row["episode"])
            episode_order.append(ep)

            total_time = time_to_seconds(row["total_time"])
            if total_time == 0:
                continue

            context_data[ep] = {
                "combat_ratio": time_to_seconds(row["combat_time"]) / total_time,
                "exploration_ratio": time_to_seconds(row["exploration_time"]) / total_time,
                "social_ratio": time_to_seconds(row["social_time"]) / total_time,
            }

    return episode_order, context_data


def load_sentiment_file(path: str):
    with open(path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)

        header = next(reader)
        values = next(reader)

    sentiment = {}

    for col, val in zip(header, values):
        ep = normalize_episode(col)
        try:
            sentiment[ep] = float(val)
        except ValueError:
            continue

    return sentiment


def zscore(values):
    if len(values) < 2:
        return [0.0] * len(values)

    m = mean(values)
    s = stdev(values)

    if s == 0:
        return [0.0] * len(values)

    return [(v - m) / s for v in values]


def analyze_sentiment_context_affinity(
    sentiment_folder: str,
    context_csv_path: str
):
    episode_order, context_data = load_context_data(context_csv_path)

    for file_name in os.listdir(sentiment_folder):
        if not file_name.endswith(".csv"):
            continue

        sentiment_path = os.path.join(sentiment_folder, file_name)
        player_name = os.path.splitext(file_name)[0]

        sentiment_data = load_sentiment_file(sentiment_path)

        aligned_episodes = []
        raw_sentiment = []
        context_list = []

        # Context defines universe
        for ep in episode_order:
            if ep not in context_data:
                continue
            if ep not in sentiment_data:
                continue

            aligned_episodes.append(ep)
            raw_sentiment.append(sentiment_data[ep])
            context_list.append(context_data[ep])

        if len(raw_sentiment) < 2:
            print(f"Skipping {player_name} (not enough overlapping episodes)")
            continue

        sentiment_z = zscore(raw_sentiment)

        combat_scores = []
        exploration_scores = []
        social_scores = []

        rows = []

        for ep, sent, ctx in zip(aligned_episodes, sentiment_z, context_list):
            combat_aff = sent * ctx["combat_ratio"]
            exploration_aff = sent * ctx["exploration_ratio"]
            social_aff = sent * ctx["social_ratio"]

            combat_scores.append(combat_aff)
            exploration_scores.append(exploration_aff)
            social_scores.append(social_aff)

            rows.append([
                ep,
                sent,
                ctx["combat_ratio"],
                ctx["exploration_ratio"],
                ctx["social_ratio"],
                combat_aff,
                exploration_aff,
                social_aff
            ])

        output_file = os.path.join(
            sentiment_folder,
            f"{player_name}_sentiment_context_affinity.csv"
        )

        with open(output_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            writer.writerow([
                "episode",
                "sentiment_z",
                "combat_ratio",
                "exploration_ratio",
                "social_ratio",
                "combat_affinity",
                "exploration_affinity",
                "social_affinity"
            ])

            writer.writerows(rows)

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

        print(f"Saved: {output_file}")


if __name__ == "__main__":
    sentiment_folder = "../../resources/sentiment_analysis/"
    context_csv_path = "../../resources/transcripts_context/assimilated_time_based_context.csv"

    analyze_sentiment_context_affinity(
        sentiment_folder,
        context_csv_path
    )