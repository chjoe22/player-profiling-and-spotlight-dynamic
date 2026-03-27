import os
import glob
import re
import pandas as pd

WORD_TO_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
}


def extract_number(text: str):
    if pd.isna(text):
        return None
    text = str(text)

    digit_match = pd.Series([text]).str.extractall(r'\b(\d+)\b')
    if not digit_match.empty:
        return int(digit_match[0].tolist()[-1])

    text_lower = text.lower()
    for word, num in WORD_TO_NUM.items():
        if re.search(rf'\b{word}\b', text_lower):
            return num

    return None


def find_combat(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    mask = (
            df["text"].str.contains("roll", case=False, na=False) &
            df["text"].str.contains("initiative", case=False, na=False)
    )
    start_matches = df[mask]

    return start_matches

    """print(f"\n{csv_path} — {len(start_matches)} match(es)")
    for _, row in start_matches.iterrows():
        print(f"  [{row['start_time']}] {row['speaker']}: {row['text'][:120]}")"""


def find_skill(csv_path, window):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    skills = ["athletics", "acrobatics", "sleight of hand", "stealth", "arcana", "history", "religion", "nature",
              "investigation", "animal handling", "insight", "medicine", "perception", "survival", "deception",
              "intimidation", "performance", "persuasion"]

    rows = []
    for skill in skills:
        mask = df["text"].str.contains(skill, case=False, na=False)
        matches = df[mask]
        for idx, row in matches.iterrows():
            roll_result = None
            roll_speaker = None
            roll_time = None

            window_rows = df.iloc[idx + 1: idx + 1 + window]
            for _, w_row in window_rows.iterrows():
                result = extract_number(w_row["text"])
                if result is not None:
                    roll_result = result
                    roll_speaker = w_row["speaker"]
                    roll_time = w_row["start_time"]
                    break

            rows.append({
                "skill": skill,
                "speaker": row["speaker"],
                "start_time": row["start_time"],
                "roll_result": roll_result,
                "roll_speaker": roll_speaker,
                "roll_start_time": roll_time,
            })
    return pd.DataFrame(rows)



if __name__ == "__main__":
    folder = "../../resources/transcripts"
    csv_files = glob.glob(f"{folder}/*.csv")
    output_folder = "../../resources/transcripts_context"
    os.makedirs(output_folder, exist_ok=True)

    all_skills = []
    for csv_path in csv_files:

        episode_name = os.path.basename(csv_path).replace(".csv", "")
        skill_df = find_skill(csv_path, 20)
        combat_df = find_combat(csv_path)


        output_path = os.path.join(output_folder, f"{episode_name}_skill.csv")
        skill_df.to_csv(output_path, index=False)
        output_path = os.path.join(output_folder, f"{episode_name}_combat.csv")
        combat_df.to_csv(output_path, index=False)
        print(f"Saved {output_path}")


