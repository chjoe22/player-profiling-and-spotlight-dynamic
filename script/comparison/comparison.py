import os
from scipy import stats
import pandas as pd


EMOTION_MAPPING = {
    "Happy": "Positive",
    "Neutral": "Neutral",
    "Sad": "Negative",
    "Surprised": "Neutral",
    "Disgust": "Negative",
    "Disgusted": "Negative",
    "Fearful": "Negative",
    "Angry": "Negative",
    "Other": "Neutral",
    "Unknown": "Neutral",
}
players = ["LAURA", "LIAM", "MARISHA", "SAM", "TALIESIN", "TRAVIS"]

def emotion_score(name: str):
    results = {}
    for folder in os.scandir("../../resources/emotion_output"):
        if folder.is_dir():
            input_path = os.path.join(folder.path, name + ".csv")
            if not os.path.exists(input_path):
                continue
            df = pd.read_csv(input_path)
            print(input_path, df.columns.tolist())
            count = 0
            for index, row in df.iterrows():
                sentiment = EMOTION_MAPPING[row["final_emotion"]]
                print(f"  {row['final_emotion']} -> {sentiment}, count={row['count']}")
                if sentiment == "Positive":
                    count += row["count"]
                elif sentiment == "Negative":
                    count -= row["count"]
            print(f"{folder.name}: total = {count}")
            results[folder.name] = count
    print(results)
    return pd.DataFrame([results])


if __name__ == "__main__":
    output_folder = "../../resources/sentiment_analysis"
    os.makedirs(output_folder, exist_ok=True)
    results = []
    for player in players:
        df = emotion_score(player)
        output_path = os.path.join(output_folder, f"{player}.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved {output_path}")

        stats_path = os.path.join("../../resources", "speaker_stats", player.upper(), f"{player.upper()}_average_comparison.csv")
        if not os.path.exists(stats_path):
            print(f"  No transcript stats found for {player}, skipping correlation.")
            continue

        df_stats = pd.read_csv(stats_path)
        df_stats = df_stats[df_stats['episode'].str.contains('transcript_stats', na=False)]
        df_stats['ep_num'] = df_stats['episode'].str.extract(r'(\d+)').astype(int)

        score_cols = [c for c in df.columns if c.startswith("episode_")]
        scores = {}
        for col in score_cols:
            ep_num = int(col.replace("episode_", ""))
            scores[ep_num] = df[col].iloc[0]
        df_scores = pd.DataFrame(list(scores.items()), columns=['ep_num', 'score'])

        merged_df = df_stats.merge(df_scores, on='ep_num')

        r, p = stats.pearsonr(merged_df['score'], merged_df['total_sec_spoken_per_hour_change_from_avg'])
        rho, p_spear = stats.spearmanr(merged_df['score'], merged_df['total_sec_spoken_per_hour_change_from_avg'])

        results.append({
            'player': player,
            'n_episodes': len(merged_df),
            'pearson_r': r,
            'pearson_p': p,
            'spearman_rho': rho,
            'spearman_p': p_spear
        })

        print(f"  {player} correlation (n={len(merged_df)}):")
        print(f"    Pearson r={r:.4f}, p={p:.4f}")
        print(f"    Spearman rho={rho:.4f}, p={p_spear:.4f}")

    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(output_folder, "correlation_results.csv"), index=False)
    print(f"Saved correlation results to {output_folder}/correlation_results.csv")