import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import csv
import numpy as np
from datetime import datetime
from pydub import AudioSegment
from models.dpngtmModel import predict_emotion


transcripts_path = "../transcripts/0_transcript.csv"
audio_path = "../audio/episode0.wav"
results_path = "../emotion_results.csv"
audio = AudioSegment.from_file(audio_path)
audio = audio.set_frame_rate(16000)

def hhmmss_to_ms(timestamp: str) -> int:
    t = datetime.strptime(timestamp, "%H:%M:%S")
    return (t.hour * 3600 + t.minute * 60 + t.second) * 1000


with open(transcripts_path, encoding="utf-8") as file, \
    open(results_path, "w", newline="", encoding="utf-8") as output:

    reader = list(csv.reader(file))[1:] # will ignore the header
    writer = csv.writer(output)
    writer.writerow(["speaker", "start_time", "end_time", "emotion", "scores"])

    for speaker, start_time, end_time, text in reader:
        start_ms = hhmmss_to_ms(start_time)

        # Checks if end_time is null or none and continue if it doesnt exists
        if not end_time.strip():
            continue

        end_ms = hhmmss_to_ms(end_time)
        
        #print(speaker, f"{start_time} : {start_ms} | {end_time} : {end_ms}")
        
        audio_clip = audio[start_ms:end_ms]
        if len(audio_clip) == 0:
            continue

        samples = np.array(audio_clip.get_array_of_samples()).astype("float32")
        if samples.size == 0:
            continue
        samples /= 32768.0

        emotion, scores = predict_emotion(samples)
        writer.writerow([speaker, start_time, end_time, emotion, scores])
        


