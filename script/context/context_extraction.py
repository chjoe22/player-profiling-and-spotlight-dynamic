import os
import glob
import re
import pandas as pd

WORD_TO_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
}
SCENARIOS = [
    "combat",
    "exploration",
    "social",
]

SKILL_SCENARIOS = {
    "athletics":       ["combat"],
    "acrobatics":      ["combat"],
    "medicine":        ["combat"],
    "perception":      ["exploration"],
    "survival":        ["exploration"],
    "nature":          ["exploration"],
    "animal handling": ["exploration"],
    "stealth":         ["exploration"],
    "sleight of hand": ["exploration"],
    "investigation":   ["exploration"],
    "arcana":          ["exploration"],
    "history":         ["exploration"],
    "religion":        ["exploration"],
    "persuasion":      ["social"],
    "deception":       ["social"],
    "intimidation":    ["social"],
    "performance":     ["social"],
    "insight":         ["social"],
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

    scenario_counts = {scenario: 0 for scenario in SCENARIOS}
    for row in rows:
        scenarios = SKILL_SCENARIOS.get(row["skill"].lower(), [])
        for scenario in scenarios:
            scenario_counts[scenario] += 1

    print(f"\nScenario counts for {csv_path}:")
    for scenario, count in sorted(scenario_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {scenario:<25} {count}")

    episode_name = os.path.basename(csv_path).replace(".csv", "")
    counts_df = pd.DataFrame([{"episode": episode_name, **{s: scenario_counts[s] for s in SCENARIOS}}])
    counts_output = os.path.join("../../resources/transcripts_context/scenario_counts",
                                 f"{episode_name}_scenario_counts.csv")
    os.makedirs(os.path.dirname(counts_output), exist_ok=True)
    counts_df.to_csv(counts_output, index=False)

    return pd.DataFrame(rows), scenario_counts



if __name__ == "__main__":
    folder = "../../resources/transcripts"
    csv_files = glob.glob(f"{folder}/*.csv")
    output_folder_skills = "../../resources/transcripts_context/skills"
    os.makedirs(output_folder_skills, exist_ok=True)

    all_counts = []  # ← add this

    for csv_path in sorted(csv_files):  # ← add sorted
        episode_name = os.path.basename(csv_path).replace(".csv", "")
        skill_df, scenario_counts = find_skill(csv_path, 20)  # ← unpack both

        output_path = os.path.join(output_folder_skills, f"{episode_name}_skill.csv")
        skill_df.to_csv(output_path, index=False)
        print(f"Saved {output_path}")

        all_counts.append({"episode": episode_name, **{s: scenario_counts[s] for s in SCENARIOS}})  # ← add this

    # After loop ← add this block
    pd.DataFrame(all_counts).to_csv(
        "../../resources/transcripts_context/scenario_counts/all_scenario_counts.csv", index=False
    )
    print("Saved all_scenario_counts.csv")


