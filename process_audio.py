import pandas as pd
from transformers import pipeline

# Load the emotion model once
emotion_classifier = pipeline(
    "text-classification",
    model="SamLowe/roberta-base-go_emotions",
    top_k=None
)

def hhmmss_to_seconds(time_str):
    """
    Convert HH:MM:SS to total seconds.
    """
    h, m, s = map(int, time_str.strip().split(":"))
    return h * 3600 + m * 60 + s

def classify_emotion_with_label(text, threshold=0.5):
    """
    Classify text with GoEmotions.
    Returns:
        top_label: highest scoring label
        passing_labels: all labels above threshold
        raw_scores: all labels with scores
    """
    if not text or not text.strip():
        raise ValueError("No text to classify.")

    outputs = emotion_classifier(text)[0]  # list of dicts
    outputs = sorted(outputs, key=lambda x: x["score"], reverse=True)

    top_label = outputs[0]["label"]
    passing_labels = [x["label"] for x in outputs if x["score"] >= threshold]

    return top_label, passing_labels, outputs

def process_transcript_range(csv_file, timestamp_range, threshold=0.5, show_rows=True):
    """
    Read transcript CSV, collect rows overlapping the given range,
    combine text, and classify emotion.

    Args:
        csv_file (str): e.g. "0_transcript.csv"
        timestamp_range (str): e.g. "00:26:33,00:26:35"
        threshold (float): threshold for multi-label output
        show_rows (bool): whether to print matched rows
    """
    # Parse input range like "00:26:33,00:26:35"
    try:
        from_time, to_time = [x.strip() for x in timestamp_range.split(",", 1)]
    except ValueError:
        raise ValueError("timestamp_range must be in format 'HH:MM:SS,HH:MM:SS'")

    df = pd.read_csv(csv_file)

    # Check required columns
    required = {"speaker", "start_time", "end_time", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # Convert CSV times to seconds
    df["_start_sec"] = df["start_time"].apply(hhmmss_to_seconds)
    df["_end_sec"] = df["end_time"].apply(
        lambda x: hhmmss_to_seconds(x) if pd.notna(x) and str(x).strip() else None
    )

    # If end_time is missing, treat it as equal to start_time
    df["_end_sec"] = df["_end_sec"].fillna(df["_start_sec"])

    range_start = hhmmss_to_seconds(from_time)
    range_end = hhmmss_to_seconds(to_time)

    if range_start >= range_end:
        raise ValueError("Start time must be earlier than end time")

    # Find rows that overlap the requested time window
    selected = df[
        (df["_end_sec"] > range_start) &
        (df["_start_sec"] < range_end)
    ].copy()

    if selected.empty:
        print("No transcript rows found in that range.")
        return None

    selected = selected.sort_values("_start_sec")

    if show_rows:
        print("\nMatched rows:")
        for _, row in selected.iterrows():
            print(f"[{row['start_time']} - {row['end_time']}] {row['speaker']}: {row['text']}")

    combined_text = " ".join(
        str(x).strip() for x in selected["text"].tolist() if pd.notna(x)
    ).strip()

    if not combined_text:
        print("Rows matched, but no usable text was found.")
        return None

    print("\nCombined text:")
    print(combined_text)

    top_label, passing_labels, raw_scores = classify_emotion_with_label(
        combined_text,
        threshold=threshold
    )

    print("\nEmotion results:")
    print(f"Top emotion: {top_label}")
    print(f"Labels above threshold ({threshold}): {passing_labels if passing_labels else ['none']}")

    print("\nTop 5 scores:")
    for item in raw_scores[:5]:
        print(f"  {item['label']}: {item['score']:.4f}")

    return {
        "matched_rows": selected[["speaker", "start_time", "end_time", "text"]].to_dict("records"),
        "combined_text": combined_text,
        "top_emotion": top_label,
        "labels_above_threshold": passing_labels,
        "raw_scores": raw_scores
    }

if __name__ == "__main__":
    csv_file = "transcripts/0_transcript.csv"

    while True:
        user_input = input("\nEnter time range (HH:MM:SS,HH:MM:SS) or 'q' to quit: ").strip()

        if user_input.lower() == "q":
            print("Exiting.")
            break

        try:
            process_transcript_range(csv_file, user_input)
        except Exception as e:
            print(f"Error: {e}")