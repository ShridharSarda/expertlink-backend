# ml/advanced_matcher.py
import os
import joblib
import numpy as np
import yake
from models import User


MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_advanced.pkl")
model = None

def load_model():
    global model
    if model is None:
        model = joblib.load(MODEL_PATH)
    return model

# Extract keywords from TF-IDF

def extract_keywords(text, top_k=5):
    kw_extractor = yake.KeywordExtractor(lan="en", n=1, top=top_k)
    keywords = kw_extractor.extract_keywords(text)
    return [k for k, score in keywords]


def predict_price(text, subject):
    m = load_model()
    vectorizer = m["vectorizer"]
    reg = m["price_model"]
    subject_map = m["subject_map"]

    # features for ML regression
    length = len(text.split())
    complexity = length / 5
    demand = 1.5
    subj_enc = subject_map.get(subject.lower(), 0)

    X = np.array([[length, complexity, demand, subj_enc]])
    price = reg.predict(X)[0]
    return max(10, round(price, 2))

def match_mentors(text, subject, db):
    keywords = extract_keywords(text)

    mentors = db.query(User).filter(User.role == "mentor").all()

    scored = []

    for m in mentors:
        mentor_keywords = (m.subjects or "").lower().split(",")
        mentor_keywords = [k.strip() for k in mentor_keywords]

        # match score
        match_score = len(set(keywords) & set(mentor_keywords))

        # load balancing = fewer solved questions => higher score
        load_score = 1 / (1 + (m.experience_years or 0))  # you can use solved_count instead

        final_score = (match_score * 0.8) + (load_score * 0.2)

        scored.append((final_score, m.id))

    # sort best first
    scored.sort(reverse=True)

    # return top 5 mentors
    return [mid for score, mid in scored[:5]]

 