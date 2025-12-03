# ml/train.py
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

def train_and_save():
    # simple synthetic dataset mapping sample texts -> mentor ids (strings)
    mentor_texts = [
        ("m_math_1", "algebra calculus geometry equations integrals derivatives"),
        ("m_math_2", "calculus integrals limits series"),
        ("m_phys_1", "mechanics kinematics dynamics forces"),
        ("m_chem_1", "organic chemistry reaction mechanism"),
        ("m_cs_1", "graphs dp algorithms data structures")
    ]
    texts = [t for _, t in mentor_texts]
    ids = [mid for mid, _ in mentor_texts]
    vec = TfidfVectorizer(ngram_range=(1,2), max_features=2000)
    X = vec.fit_transform(texts)
    nn = NearestNeighbors(n_neighbors=3, metric="cosine").fit(X)
    joblib.dump({"vectorizer": vec, "nn": nn, "mentor_ids": ids}, MODEL_PATH)
    print("ML model trained and saved to", MODEL_PATH)

def ensure_model_trained():
    if not os.path.exists(MODEL_PATH):
        train_and_save()
