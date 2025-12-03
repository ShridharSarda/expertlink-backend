# ml/train_advanced.py
import os
import joblib
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_advanced.pkl")

def generate_synthetic_questions(num=500):
    subjects = ["math", "physics", "chemistry", "cs"]
    topics = {
        "math": ["algebra", "calculus integrals", "geometry theorems", "differential equations"],
        "physics": ["mechanics dynamics", "optics light", "thermodynamics", "electricity magnetism"],
        "chemistry": ["organic reactions", "periodic table", "chemical bonding", "titration"],
        "cs": ["graphs dp", "recursion", "sorting algorithms", "data structures"]
    }

    data = []
    for _ in range(num):
        subject = random.choice(subjects)
        text = random.choice(topics[subject])
        complexity = random.uniform(0.5, 3.0)  # ML feature
        length = random.randint(5, 30)         # ML feature
        demand = random.uniform(0.5, 2.0)      # ML feature

        # Price rule for synthetic training
        price = 20 + (complexity * 50) + (length * 2) + (demand * 20) + random.uniform(-10, 10)

        data.append((text, subject, length, complexity, demand, price))

    return data


def train_and_save_advanced():
    print("Training advanced ML model...")

    data = generate_synthetic_questions(500)

    texts = [d[0] for d in data]
    subjects = [d[1] for d in data]
    lengths = [d[2] for d in data]
    complexity = [d[3] for d in data]
    demand = [d[4] for d in data]
    prices = [d[5] for d in data]

    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_text = vectorizer.fit_transform(texts)

    # Mentor matching model (Nearest Neighbors)
    nn = NearestNeighbors(n_neighbors=3, metric="cosine")
    nn.fit(X_text)

    # Subject encoding
    subject_map = {s: i for i, s in enumerate(set(subjects))}
    X_subject = np.array([subject_map[s] for s in subjects]).reshape(-1, 1)

    # Combine structured features
    X_extra = np.column_stack((lengths, complexity, demand, X_subject.flatten()))

    # Price prediction model
    reg = RandomForestRegressor(n_estimators=200)
    reg.fit(X_extra, prices)

    # Save all models
    joblib.dump({
        "vectorizer": vectorizer,
        "nn": nn,
        "price_model": reg,
        "subject_map": subject_map
    }, MODEL_PATH)

    print("Advanced model trained and saved.")


def ensure_advanced_trained():
    if not os.path.exists(MODEL_PATH):
        train_and_save_advanced()

if __name__ == "__main__":
    train_and_save_advanced()
