import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

MODEL_ID = "Dpngtm/wav2vec2-emotion-recognition"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
model = AutoModelForAudioClassification.from_pretrained(MODEL_ID).to(device).eval()

def predict_emotion(audio_array, sr=16000):
    inputs = extractor(audio_array, sampling_rate=sr, return_tensors="pt", padding=True)
    
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

    probs = probs.cpu()

    scores = {
        model.config.id2label[i]: float(probs[i])
        for i in range(probs.numel())
    }
    emotion = model.config.id2label[int(torch.argmax(probs))]

    return emotion, scores