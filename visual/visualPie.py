import os
import re
import pandas as pd
import matplotlib.pyplot as plt

INPUT_CSV = "../emotion_results.csv"
OUTPUT_DIR = "../visual/emotion_pies"

# Utility functions
def clean_filename(name):
    name = str(name).strip()
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    return name.replace(" ", "_")

# Check if more than one speaker, gets removed if more for now 
# (ALL is also funky, so remove when generating)
def is_single_speaker(speaker):
    if pd.isna(speaker):
        return False
    speaker = str(speaker).strip()
    return not any(x in speaker for x in [",", "&", "/"])

# Check if dir is there or create one
def make_folder(path):
    os.makedirs(path, exist_ok=True)

# Plotter
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
        bbox_to_anchor=(1.25, 0.5),
        frameon=False,
        fontsize=10,
        title_fontsize=11
    )

    plt.subplots_adjust(right=0.72)

    filename = clean_filename(speaker)
    save_path = os.path.join(output_dir, f"{filename}_emotion_pie.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    plt.style.use("ggplot")

    data = pd.read_csv(INPUT_CSV)
    data = data.dropna(subset=["speaker", "emotion"]).copy()

    data["speaker"] = data["speaker"].astype(str).str.strip()
    data["emotion"] = data["emotion"].astype(str).str.strip()

    data = data[data["speaker"].apply(is_single_speaker)]

    make_folder(OUTPUT_DIR)

    for speaker, group in data.groupby("speaker"):
        emotion_counts = group["emotion"].value_counts()
        plot_pie(speaker, emotion_counts, OUTPUT_DIR)

    print(f"Done. Saved in: {OUTPUT_DIR}")