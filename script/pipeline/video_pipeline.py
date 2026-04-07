import sys
from pathlib import Path
# To find the path in the root folder - too many .parent but they are necessary for it to work
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import os
import cv2
import numpy as np
import csv
import torch
import gc
from tqdm import tqdm # Library for process-bar
import warnings
import datetime

from insightface.app import FaceAnalysis
from models.video.insightface_model import identify_face, identify_all_faces
from models.video.dima806_model import predict_video_emotion; model_name = "dima806"
#from models.video.BEiT_model import predict_video_emotion; model_name = "BEiT"

# Warnings ignores to make sure that the process bar and area is free and is easily readable - unnecessary
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["HF_HUB_DISABLE_SYMLINK_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Loading face model
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

# frames_root = "../frames/episode1/"
episode_number = "107" # Change number to reflect the episode running
video_segment_path = f"../../segmented-video/episode{episode_number}"
results_dir = f"../../resources/results/video/{model_name}"
data = np.load("../helper/cast_embeddings.npz")
gallery = {name: data[name] for name in data.files}

os.makedirs(results_dir, exist_ok=True)

results_path = os.path.join(results_dir, f"{episode_number}_episode_results.csv")

with open(results_path, "w", newline="", encoding="utf-8") as output:
    writer = csv.writer(output)
    writer.writerow(["segment", "frame_id", "timestamp", "speaker", "emotion", "scores"])
    video_files = sorted([f for f in os.listdir(video_segment_path) if f.endswith(".mp4")])

    overall_process_bar = tqdm(video_files, desc=f"Total Process ({episode_number})", unit="clip")

    print(f"Starting model on {episode_number}")

    # Instead of "overall_process_bar" use sorted(video_files)
    for video_file in overall_process_bar:
        video_path = os.path.join(video_segment_path, video_file)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps == 0: 
            fps = 30.0

        # Remove these two if sorted(video_files) is used
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        inner_process_bar = tqdm(total=total_frames, desc=f" > {video_file}", leave=False, unit="fr")

        print(f"\nStreaming Segment: {video_file}")

        segment_num = int(video_file.split('_')[-1].replace('.mp4', ''))
        segment_offset = segment_num * 600

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Determines how many frames are to be before they are read - current is every 10 frames are read
            if frame_idx % 10 == 0:

                sec_in_segment = frame_idx / fps
                total_sec = segment_offset + sec_in_segment
            
                timestamp = str(datetime.timedelta(seconds=int(total_sec)))

                results = identify_all_faces(frame, app, gallery)
                for name, face_crop in results:
                    emotion, scores = predict_video_emotion(face_crop)

                    # Ignore these they are/were here for debugging purposes
                    #debug_dir = f"../debug_crops/{name}/"
                    #os.makedirs(debug_dir, exist_ok=True)
                    #cv2.imwrite(os.path.join(debug_dir, f"{video_file}_f{frame_idx}.jpg"), face_crop)

                    writer.writerow([video_file, frame_idx, timestamp, name, emotion, scores])

                # Added to include VRAM usage onto the process bar 
                vram_mb = torch.cuda.memory_allocated() / 1024**2
                inner_process_bar.set_postfix({"VRAM": f"{vram_mb:.0f}MB"})
                if frame_idx % 500 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

            inner_process_bar.update(1)
            frame_idx += 1

        cap.release()
        inner_process_bar.close()
        print(f"Segment done: {video_file}")
print(f"Finished all segments for {episode_number}")