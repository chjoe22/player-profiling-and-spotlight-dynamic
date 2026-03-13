import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

photo_folder = "../cast_folder/"
gallery_data = {}

for filename in os.listdir(photo_folder):
    if filename.endswith((".jpg", ".png")):
        name = os.path.splitext(filename)[0]
        img = cv2.imread(os.path.join(photo_folder, filename))

        faces = app.get(img)
        if faces:
            gallery_data[name] = faces[0].normed_embedding
            print(f"Stored embedding for {name}")

np.savez("cast_embeddings.npz", **gallery_data)