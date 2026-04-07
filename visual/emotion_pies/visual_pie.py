import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

INPUT_CSV = os.path.join(PROJECT_ROOT, "results", "video", "100_episode_results.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "generated")

# functions
def clean_filename(name):
    name = str(name).strip()
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    return name.replace(" ", "_")

def is_single_speaker(speaker):
    if pd.isna(speaker):
        return False
    speaker = str(speaker).strip()
    return not any(x in speaker for x in [",", "&", "/"])

def make_folder(path):
    os.makedirs(path, exist_ok=True)

def plot_pie(speaker, emotion_counts, output_dir):
    if emotion_counts.empty:
        return

    labels = emotion_counts.index.tolist()
    values = emotion_counts.values.tolist()
    explode = [0.06 if i == 0 else 0 for i in range(len(values))]

    fig, ax = plt.subplots(figsize=(11, 9))

    wedges, _ = ax.pie(
        values,
        labels=None,
        startangle=90,
        explode=explode,
        wedgeprops={"edgecolor": "white", "linewidth": 2}
    )

    centre_circle = plt.Circle((0, 0), 0.45, fc="white")
    ax.add_artist(centre_circle)

    ax.set_title(f"{speaker}", fontsize=18, pad=20, weight="bold")
    ax.axis("equal")

    legend_labels = [f"{label}: {count}" for label, count in zip(labels, values)]
    ax.legend(
        wedges,
        legend_labels,
        title="Emotions",
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
        frameon=False,
        fontsize=10,
        title_fontsize=11
    )

    plt.subplots_adjust(right=0.78)

    filename = clean_filename(speaker)
    save_path = os.path.join(output_dir, f"{filename}_emotion_pie.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_path}")

if __name__ == "__main__":
    plt.style.use("ggplot")

    print("Script dir:", SCRIPT_DIR)
    print("Project root:", PROJECT_ROOT)
    print("Looking for CSV at:", INPUT_CSV)
    print("CSV exists:", os.path.exists(INPUT_CSV))
    print("Saving output to:", OUTPUT_DIR)

    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"CSV not found: {INPUT_CSV}")

    data = pd.read_csv(INPUT_CSV)

    print("Columns found:", list(data.columns))
    print("Number of rows before cleaning:", len(data))

    required_cols = {"speaker", "emotion"}
    missing = required_cols - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = data.dropna(subset=["speaker", "emotion"]).copy()

    data["speaker"] = data["speaker"].astype(str).str.strip()
    data["emotion"] = data["emotion"].astype(str).str.strip()

    data = data[data["speaker"].apply(is_single_speaker)]

    print("Number of rows after cleaning/filtering:", len(data))
    print("Unique speakers after filtering:", data["speaker"].nunique())

    if data.empty:
        raise ValueError("No rows left after filtering. Check speaker/emotion values.")

    make_folder(OUTPUT_DIR)

    for speaker, group in data.groupby("speaker"):
        emotion_counts = group["emotion"].value_counts()
        plot_pie(speaker, emotion_counts, OUTPUT_DIR)

    print(f"Done. Saved in: {OUTPUT_DIR}")