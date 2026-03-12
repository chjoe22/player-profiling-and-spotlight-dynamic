import torch
import numpy as np
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

MODEL_ID = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID, do_normalize=True)
model = AutoModelForAudioClassification.from_pretrained(MODEL_ID).to(device).eval()
id2label = model.config.id2label

def predict_emotion(audio_array, sr=16000):
    max_length = int(extractor.sampling_rate * 30.0)
    
    if len(audio_array) > max_length:
        audio_array = audio_array[:max_length]
    else:
        audio_array = np.pad(audio_array, (0, max_length - len(audio_array)))

    inputs = extractor(audio_array, sampling_rate=sr, return_tensors="pt").to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0)

    scores = {id2label[i]: float(probs[i]) for i in range(len(probs))}
    
    emotion = id2label[torch.argmax(probs).item()]

    return emotion, scores