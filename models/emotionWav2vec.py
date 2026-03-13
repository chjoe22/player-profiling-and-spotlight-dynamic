import torch
import numpy as np
from funasr import AutoModel

#MODEL_ID = "iic/emotion2vec_plus_large"
MODEL_ID = "iic/emotion2vec_plus_base"
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    device_str = "cuda"
else:
    device_str = "cpu"

model = AutoModel(model=MODEL_ID, hub="hf", device=device_str, disable_update=True)

id2label = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Other",
    6: "Sad",
    7: "Surprised",
    8: "Unknown"
}

def predict_emotion(audio_array, sr=16000):

    try:
        res = model.generate(input=audio_array, granularity="utterance")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        probs = res[0]['scores']
        scores = {id2label[i]: float(probs[i]) for i in range(len(probs))}
        emotion = id2label[np.argmax(probs)]
        return emotion, scores
    except torch.cuda.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return "MemoryError", {label: 0.0 for label in id2label.values()}
