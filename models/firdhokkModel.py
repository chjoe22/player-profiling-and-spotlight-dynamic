# Requires: librosa
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import librosa
import torch
import numpy as np

model_id = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
model = AutoModelForAudioClassification.from_pretrained(model_id)

feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, do_normalize=True)
id2label = model.config.id2label


def preprocess_audio(audio_path, feature_extractor, max_duration=30.0):
    audio_array, sampling_rate = librosa.load(audio_path, sr=16000, mono=True)

    max_length = int(feature_extractor.sampling_rate * max_duration)
    if len(audio_array) > max_length:
        audio_array = audio_array[:max_length]
    else:
        audio_array = np.pad(audio_array, (0, max_length - len(audio_array)))

    inputs = feature_extractor(
        audio_array,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    return inputs

def predict_emotion(audio_path, model, feature_extractor, id2label, max_duration=30.0):
    inputs = preprocess_audio(audio_path, feature_extractor, max_duration)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_id = torch.argmax(logits, dim=-1).item()
    predicted_label = id2label[predicted_id]

    return predicted_label

def predict_emotions_all(audio_path, model, feature_extractor, id2label, max_duration=30.0):
    inputs = preprocess_audio(audio_path, feature_extractor, max_duration)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits.squeeze(0)  # [num_labels]

    # Most SER models are single-label => softmax
    probs = torch.softmax(logits, dim=-1)

    results = {id2label[i]: float(probs[i].cpu()) for i in range(probs.shape[0])}

    # Sort high -> low
    results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
    return results

audio_path = "../segmented-audio/output.wav"

predicted_emotion = predict_emotion(audio_path, model, feature_extractor, id2label)
print(f"Predicted Emotion: {predicted_emotion}")

predicted_emotion_values = predict_emotions_all(audio_path, model, feature_extractor, id2label)
for label, p in predicted_emotion_values.items():
    print(f"{label:>12}: {p:.4f}")
