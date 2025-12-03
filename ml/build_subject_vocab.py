# backend/ml/build_subject_vocab.py
import os, collections, re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_questions.csv")
OUT_JSON = os.path.join(os.path.dirname(__file__), "subject_vocab.json")

def normalize_text(s):
    s = (s or "").lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def top_tokens_for_subject(df, subject, top_k=300):
    # collect texts for subject
    texts = df[df["subject"].astype(str).str.lower() == subject]["text"].astype(str).tolist()
    if not texts:
        return []
    # use simple TF-IDF to get top tokens across docs
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=2000, stop_words='english')
    X = vectorizer.fit_transform([normalize_text(t) for t in texts])
    # sum tfidf per feature
    scores = X.sum(axis=0).A1
    features = vectorizer.get_feature_names_out()
    pairs = sorted(zip(features, scores), key=lambda x: x[1], reverse=True)
    top = [f for f, _ in pairs[:top_k]]
    # also include single-word tokens from texts frequency
    freq = collections.Counter()
    for t in texts:
        for w in normalize_text(t).split():
            if len(w) > 2:
                freq[w] += 1
    freq_top = [w for w, _ in freq.most_common(200)]
    # merge and dedupe
    out = []
    for w in top + freq_top:
        if w not in out:
            out.append(w)
        if len(out) >= top_k:
            break
    return out

def build_vocab():
    df = pd.read_csv(DATA_CSV)
    subjects = sorted(df['subject'].dropna().astype(str).str.lower().unique())
    vocab = {}
    for s in subjects:
        vocab[s] = top_tokens_for_subject(df, s, top_k=400)
        print(f"built vocab for {s}: {len(vocab[s])} tokens")
    # write to json
    import json
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print("wrote subject vocab to", OUT_JSON)

if __name__ == "__main__":
    build_vocab()
