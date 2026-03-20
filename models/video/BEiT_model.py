import torch
import cv2
from transformers import BeitImageProcessor, BeitForImageClassification
from PIL import Image
import numpy as np

model_name = "Tanneru/Facial-Emotion-Detection-FER-RAFDB-AffectNet-BEIT-Large"
processor = BeitImageProcessor.from_pretrained(model_name)
model = BeitForImageClassification.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

EMOTION_MAP = {
    "LABEL_0": "Surprise",
    "LABEL_1": "Fear",
    "LABEL_2": "Disgust",
    "LABEL_3": "Happy",
    "LABEL_4": "Sad",
    "LABEL_5": "Angry",
    "LABEL_6": "Neutral"
}

def predict_video_emotion(face_crop):
    if face_crop is None or face_crop.size == 0:
        return "Unknown", []

    image = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    probs = torch.nn.functional.softmax(logits, dim=-1)
    top_class_idx = probs.argmax(-1).item()
    
    raw_label = model.config.id2label[top_class_idx]

    emotion_label = EMOTION_MAP.get(raw_label, raw_label)
    
    return emotion_label, probs.cpu().numpy().tolist()[0]