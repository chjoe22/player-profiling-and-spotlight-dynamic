import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))


import os
import cv2
import numpy as np
import csv
import torch
import gc
from models.insightfaceModel import identify_face, identify_all_faces
from insightface.app import FaceAnalysis
# from models.<MODEL_NAME> import predict_video_emotion

app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

frames_root = "../frames/episode1/"
video_segment_path = "../segmented-video"
data = np.load("cast_embeddings.npz")
gallery = {name: data[name] for name in data.files}
results_path = "../final_video_emotions.csv"

video_files = sorted([f for f in os.listdir(video_segment_path) if f.endswith(".mp4")])

with open(results_path, "w", newline="", encoding="utf-8") as output:
    writer = csv.writer(output)
    writer.writerow(["segment", "frame_id", "speaker", "emotion", "scores"])

    for video_file in sorted(video_files):
        video_path = os.path.join(video_segment_path, video_file)
        cap = cv2.VideoCapture(video_path)
        print(f"\nStreaming Segment: {video_file}")
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % 10 == 0:
                results = identify_all_faces(frame, app, gallery)
                for name, face_crop in results:
                    # emotion, scores = predict_face_emotion(face_crop)
                    debug_dir = f"../debug_crops/{name}/"
                    os.makedirs(debug_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(debug_dir, f"{video_file}_f{frame_idx}.jpg"), face_crop)
                    #writer.writerow([video_file, frame_idx, name, emotion, scores])

                if frame_idx % 500 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
            frame_idx += 1
        cap.release()