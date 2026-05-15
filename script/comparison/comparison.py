import os

import pandas as pd
from Cython.Build.Dependencies import nonempty

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
players = ["laura", "liam", "marisha", "sam", "taliesin", "travis"]

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
    for player in players:
        df = emotion_score(player)
        output_path = os.path.join(output_folder, f"{player}.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved {output_path}")



