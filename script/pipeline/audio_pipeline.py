import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import csv
import os
import numpy as np
from datetime import datetime
from pydub import AudioSegment

# Model switching - change which ones are commented out to test other models
# from models.dpngtmModel import predict_emotion
# from models.firdhokkModel import predict_emotion
# from models.prithivMLmodsModel import predict_emotion
from models.emotionWav2vec import predict_emotion


episode_number = "episode100"
transcripts_path = "../../transcripts/0_transcript.csv"
audio_path = "../../audio/episode0.wav"
results_dir = "../../results/audio"
overlap_dir = "../../results/overlap"
audio = AudioSegment.from_file(audio_path)
audio = audio.set_frame_rate(16000).set_channels(1)

os.makedirs(results_dir, exist_ok=True)
os.makedirs(overlap_dir, exist_ok=True)
results_path = os.path.join(results_dir, f"{episode_number}_results.csv")
overlap_path = os.path.join(overlap_dir, f"{episode_number}_overlap_results.csv")

def hhmmss_to_ms(timestamp: str) -> int:
    t = datetime.strptime(timestamp, "%H:%M:%S")
    return (t.hour * 3600 + t.minute * 60 + t.second) * 1000

previous_time_ms = 0
with open(transcripts_path, encoding="utf-8") as file, \
    open(results_path, "w", newline="", encoding="utf-8") as output, \
    open(overlap_path, "w", newline="", encoding="utf-8") as overlap:

    reader = list(csv.reader(file))[1:] # will ignore the header
    writer = csv.writer(output)
    overlap_writer = csv.writer(overlap)
    writer.writerow(["speaker", "start_time", "end_time", "emotion", "scores"])
    overlap_writer.writerow(["speaker", "start_time", "end_time", "reason"])

    for speaker, start_time, end_time, text in reader:

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

        #print(speaker, f"{start_time} : {start_ms} | {end_time} : {end_ms}")

        audio_clip = audio[start_ms:end_ms]
        if len(audio_clip) == 0:
            continue

        samples = np.array(audio_clip.get_array_of_samples()).astype("float32")
        max_peak = np.max(np.abs(samples))
        if samples.size == 0:
            continue

        if max_peak > 0:
            samples /= max_peak
        else:
            overlap_writer.writerow([speaker, start_time, end_time, "silence"])
            continue

        emotion, scores = predict_emotion(samples)
        writer.writerow([speaker, start_time, end_time, emotion, scores])
        previous_time_ms = end_ms


