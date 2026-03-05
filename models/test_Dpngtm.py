import torch
import librosa
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

MODEL_ID = "Dpngtm/wav2vec2-emotion-recognition"
extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)

def predict_emotion(audio_path: str):
    audio, sr = librosa.load(audio_path, sr=16000)
    inputs = extractor(audio, sampling_rate=sr, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred_id = int(torch.argmax(probs).item())

    return model.config.id2label[pred_id], probs

emotion, probs = predict_emotion("../segmented-audio/output.wav")
print("Predicted emotion:", emotion)
print({model.config.id2label[i]: float(probs[i]) for i in range(probs.numel())})