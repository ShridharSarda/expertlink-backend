# ml/matcher.py
import os
import joblib
from typing import List

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
model = None

def load_model():
    global model
    if model is None:
        model = joblib.load(MODEL_PATH)
    return model

def load_model_and_match(text: str, subject: str) -> List[str]:
    m = load_model()
    vec = m["vectorizer"].transform([text])
    dists, inds = m["nn"].kneighbors(vec, n_neighbors=min(3, m["nn"].n_neighbors))
    ids = []
    for i in inds[0]:
        ids.append(m["mentor_ids"][i])
    return ids
