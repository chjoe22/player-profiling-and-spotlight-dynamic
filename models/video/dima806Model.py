import torch
import cv2
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import numpy as np

model_name = "dima806/facial_emotions_image_detection"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

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
    emotion_label = model.config.id2label[top_class_idx]
    
    return emotion_label, probs.cpu().numpy().tolist()[0]