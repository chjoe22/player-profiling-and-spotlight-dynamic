import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

MODEL_ID = "Dpngtm/wav2vec2-emotion-recognition"
extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)

def predict_emotion(audio_array, sr=16000):
    inputs = extractor(audio_array, sampling_rate=sr, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

    scores = {
        model.config.id2label[i]: float(probs[i])
        for i in range(probs.numel())
    }
    emotion = model.config.id2label[int(torch.argmax(probs))]

    return emotion, scores
