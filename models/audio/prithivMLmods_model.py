import torch
import numpy as np
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

MODEL_ID = "prithivMLmods/Speech-Emotion-Classification"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_ID).to(device).eval()

emotion_label = {
    0: "Anger",
    1: "Calm",
    2: "Disgust",
    3: "Fear",
    4: "Happy",
    5: "Neutral",
    6: "Sad",
    7: "Surprised"
}

model.config.id2label = emotion_label
model.config.label2id = {v: k for k, v in emotion_label.items()}

id2label = model.config.id2label


def predict_emotion(audio_array, sr=16000):

    inputs = extractor(audio_array, sampling_rate=sr, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0)

    scores = {id2label[i]: float(probs[i]) for i in range(len(probs))}
    
    emotion = id2label[torch.argmax(probs).item()]

    return emotion, scores