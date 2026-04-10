import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import os
import cv2
import numpy as np
import csv
import torch
import gc
from tqdm import tqdm
import warnings
import datetime
import re

from insightface.app import FaceAnalysis
from models.video.insightface_model import identify_face, identify_all_faces
from models.video.dima806_model import predict_video_emotion; model_name = "dima806"
#from models.video.BEiT_model import predict_video_emotion; model_name = "BEiT"

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["HF_HUB_DISABLE_SYMLINK_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

data = np.load("../helper/cast_embeddings.npz")
gallery = {name: data[name] for name in data.files}

episodes_root = "../../segmented-video/"
results_base_dir = f"../../resources/results/video/{model_name}"
os.makedirs(results_base_dir, exist_ok=True)

episode_folders = sorted(
    [f for f in os.listdir(episodes_root) if os.path.isdir(os.path.join(episodes_root, f)) and "episode" in f],
    key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0
)

for folder_name in episode_folders:
    episode_number = re.search(r'\d+', folder_name).group() if re.search(r'\d+', folder_name) else folder_name
    
    video_segment_path = os.path.join(episodes_root, folder_name)
    results_path = os.path.join(results_base_dir, f"{episode_number}_episode_results.csv")

    with open(results_path, "w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["segment", "frame_id", "timestamp", "speaker", "emotion", "scores"])
        
        video_files = sorted([f for f in os.listdir(video_segment_path) if f.endswith(".mp4")])

        overall_process_bar = tqdm(video_files, desc=f"Total Process ({episode_number})", unit="clip")

        print(f"Starting model on {episode_number}")

        for video_file in overall_process_bar:
            video_path = os.path.join(video_segment_path, video_file)
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)

            if fps == 0:
                fps = 30.0

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
                
                if frame_idx % 30 == 0:
                    sec_in_segment = frame_idx / fps
                    total_sec = segment_offset + sec_in_segment
                    timestamp = str(datetime.timedelta(seconds=int(total_sec)))

                    results = identify_all_faces(frame, app, gallery)
                    for name, face_crop in results:
                        emotion, scores = predict_video_emotion(face_crop)
                        writer.writerow([video_file, frame_idx, timestamp, name, emotion, scores])

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