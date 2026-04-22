import torch
import numpy as np
import logging
from funasr import AutoModel

logging.getLogger('funasr').setLevel(logging.ERROR)

# MODEL_ID = "iic/emotion2vec_plus_large"
MODEL_ID = "iic/emotion2vec_plus_base"

if torch.cuda.is_available():
    device_str = "cuda"
else:
    device_str = "cpu"


model = AutoModel(
    model=MODEL_ID, 
    hub="hf", 
    device=device_str, 
    disable_update=True, 
    disable_pbar=True
)

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
        if isinstance(audio_array, np.ndarray):
            audio_array = audio_array.astype(np.float32)

        res = model.generate(
            input=audio_array, 
            granularity="utterance", 
            disable_pbar=True
        )
        
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
    except Exception as e:
        return f"Error: {str(e)}", {label: 0.0 for label in id2label.values()}