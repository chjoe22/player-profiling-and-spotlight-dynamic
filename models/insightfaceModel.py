import numpy as np
import cv2

def identify_face(face_embedding, gallery, threshold=0.6):
    """
    Compares a new face from a video frame to your pre-saved gallery.
    """
    best_match = "Unknown"
    highest_score = -1

    for name, gallery_emb in gallery.items():
        score = np.dot(face_embedding, gallery_emb)

        if score > highest_score:
            highest_score = score
            best_name = name

    if highest_score > threshold:
        return best_name
    return "Unknown"

def identify_all_faces(frame, app, gallery, threshold=0.5):

    faces = app.get(frame)
    results = []
    h, w, _ = frame.shape

    for face in faces:
        emb = face.normed_embedding
        best_name = "Unknown"
        max_sim = -1

        for name, g_emb in gallery.items():
            sim = np.dot(emb, g_emb)
            if sim > max_sim:
                max_sim = sim
                best_name = name

        if max_sim > threshold:
            bbox = face.bbox.astype(int)

            x1, y1, x2, y2 = np.maximum(bbox, 0)
            x2, y2 = min(w, x2), min(h, y2)

            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size > 0:
                results.append((best_name, face_crop))

    return results