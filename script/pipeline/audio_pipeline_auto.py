import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import csv
import os
import numpy as np
import re
from datetime import datetime
from pydub import AudioSegment
from tqdm import tqdm

# Model switching - change which ones are commented out to test other models
# from models.audio.dpngtm_model import predict_emotion; model_name = 'dpngtm'
from models.audio.firdhokk_model import predict_emotion; model_name = 'firdhokk'
# from models.audio.prithivMLmods_model import predict_emotion; model_name = 'prithivMLmods'
# from models.audio.emotionWav2vec_model import predict_emotion; model_name = 'emotionWav2vec'

def hhmmss_to_ms(timestamp: str) -> int:
    t = datetime.strptime(timestamp, "%H:%M:%S")
    return (t.hour * 3600 + t.minute * 60 + t.second) * 1000

audio_root = "../../segmented-audio/"
audio_files = [f for f in os.listdir(audio_root) if f.endswith(".wav") and "episode" in f.lower()]
audio_files.sort(key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0)

print(f"Found {len(audio_files)} episodes in {audio_root}")

for audio_filename in audio_files:
    number = re.search(r'\d+', audio_filename).group()
    episode_number = f"episode{number}"
    
    transcripts_path = f"../../resources/transcripts/{number}_transcript.csv"
    audio_path = os.path.join(audio_root, audio_filename)
    results_dir = f"../../resources/results/audio/{model_name}"
    overlap_dir = f"../../resources/results/overlap/{model_name}"
    
    if not os.path.exists(transcripts_path):
        print(f"Skipping {episode_number}: Transcript not found at {transcripts_path}")
        continue

    print(f"\n--- Starting {episode_number} ---")
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(16000).set_channels(1)

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(overlap_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"{number}_episode_results.csv")
    overlap_path = os.path.join(overlap_dir, f"{number}_episode_overlap_results.csv")

    previous_time_ms = 0
    with open(transcripts_path, encoding="utf-8") as file, \
         open(results_path, "w", newline="", encoding="utf-8") as output, \
         open(overlap_path, "w", newline="", encoding="utf-8") as overlap:

        reader = list(csv.reader(file))[1:] # will ignore the header
        writer = csv.writer(output)
        overlap_writer = csv.writer(overlap)
        writer.writerow(["speaker", "start_time", "end_time", "emotion", "scores", "text"])
        overlap_writer.writerow(["speaker", "start_time", "end_time", "reason"])

        for speaker, start_time, end_time, text in tqdm(reader, desc=f"Processing {episode_number}", unit="segment"):

            # Checks if end_time is null or none and continue if it doesnt exists
            if not end_time.strip():
                overlap_writer.writerow([speaker, start_time, end_time, "missing_end_time"])
                continue

            start_ms = hhmmss_to_ms(start_time)
            end_ms = hhmmss_to_ms(end_time)

            if end_ms <= start_ms:
                overlap_writer.writerow([speaker, start_time, end_time, "invalid_range"])
                continue

            if start_ms < previous_time_ms:
                overlap_writer.writerow([speaker, start_time, end_time, "overlap_with_previous"])
                continue

            audio_clip = audio[start_ms:end_ms]
            if len(audio_clip) == 0:
                continue

            samples = np.array(audio_clip.get_array_of_samples()).astype("float32")
            if samples.size == 0:
                continue
                
            max_peak = np.max(np.abs(samples))

            if max_peak > 0:
                samples /= max_peak
            else:
                overlap_writer.writerow([speaker, start_time, end_time, "silence"])
                continue

            emotion, scores = predict_emotion(samples)
            writer.writerow([speaker, start_time, end_time, emotion, scores, text])
            previous_time_ms = end_ms

    print(f"Finished {episode_number}")

print("\nAll detected audio files have been processed.")