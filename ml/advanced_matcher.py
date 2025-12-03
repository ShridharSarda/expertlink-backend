# backend/ml/advanced_matcher.py
import os
import json
import joblib
import re
import numpy as np
import difflib
import yake

from models import User
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# Config / paths / state
# -------------------------
HERE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(HERE, "model_advanced.pkl")
VOCAB_PATH = os.path.join(HERE, "subject_vocab.json")  # built by build_subject_vocab.py
_kw_extractor = yake.KeywordExtractor(lan="en", n=1, top=12)

model = None
_subject_vocab = None  # lazy-loaded dict: { subject: [token1, token2, ...] }

# -------------------------
# Utilities
# -------------------------
def load_model():
    global model
    if model is None and os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    return model

def _normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def _load_subject_vocab():
    """Load the subject_vocab.json produced by build_subject_vocab.py"""
    global _subject_vocab
    if _subject_vocab is not None:
        return _subject_vocab
    if os.path.exists(VOCAB_PATH):
        try:
            with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
                _subject_vocab = json.load(f)
        except Exception:
            _subject_vocab = {}
    else:
        _subject_vocab = {}
    return _subject_vocab

def map_token_to_subject_vocab(token: str, cutoff: float = 0.62):
    """
    Map arbitrary token -> (subject, canonical_token, similarity_score)
    Returns (None, None, 0.0) if no good mapping found.
    """
    token = _normalize_text(token)
    if not token:
        return None, None, 0.0

    vocab = _load_subject_vocab()
    best = (None, None, 0.0)
    for subj, toks in vocab.items():
        if not toks: 
            continue
        # use difflib to get close matches in this subject's token list
        matches = difflib.get_close_matches(token, toks, n=1, cutoff=cutoff)
        if matches:
            candidate = matches[0]
            ratio = difflib.SequenceMatcher(None, token, candidate).ratio()
            if ratio > best[2]:
                best = (subj, candidate, ratio)
    return best  # (subject, canonical_token, ratio)

# -------------------------
# Keyword extraction (data-driven)
# -------------------------
STOPWORDS = set([
    "doubt","question","query","please","help","solve","problem","show","example",
    "sr", "sir", "urgent", "plz"
])

def extract_keywords(text: str, top_k: int = 6):
    """
    Data-driven keyword extraction:
      - If comma-separated short list, prefer that splitting.
      - Otherwise use YAKE to extract candidate tokens.
      - Map tokens to canonical subject tokens via the subject_vocab.json (fuzzy).
      - Return deduped canonical tokens (or normalized tokens if no mapping).
    """
    if not text or not text.strip():
        return []

    txt = text.strip()
    tokens = []

    # If looks like a short comma list, split and use parts directly
    if ',' in txt and len(txt.split(',')) <= 12:
        parts = [p.strip() for p in txt.split(',') if p.strip()]
        tokens = parts
    else:
        # try YAKE
        try:
            kws = _kw_extractor.extract_keywords(txt)
            tokens = [k for k, score in kws][: top_k * 2]
            if not tokens:
                # fallback to whitespace splitting
                tokens = txt.split()
        except Exception:
            tokens = txt.split()

    # Normalize tokens, filter stopwords and very short tokens
    normalized = []
    for t in tokens:
        tnorm = _normalize_text(t)
        if not tnorm or tnorm in STOPWORDS or len(tnorm) <= 2:
            continue
        normalized.append(tnorm)

    out = []
    seen = set()
    for tok in normalized:
        subj, canon, score = map_token_to_subject_vocab(tok)
        if canon:
            val = canon
        else:
            val = tok
        if val not in seen:
            out.append(val)
            seen.add(val)
        if len(out) >= top_k:
            break

    return out

# -------------------------
# Price prediction (unchanged semantics)
# -------------------------
def predict_price(text, subject):
    """
    Keep your existing price estimator intact (uses model_advanced.pkl if available).
    If no model present, use a simple heuristic fallback.
    """
    m = load_model()
    if not m:
        # fallback heuristic (very simple) if model missing
        length = max(1, len((text or "").split()))
        return max(10, round(20 + length * 2, 2))

    vectorizer = m.get("vectorizer")
    reg = m.get("price_model")
    subject_map = m.get("subject_map", {})

    length = len((text or "").split())
    complexity = length / 5
    demand = 1.5
    subj_enc = subject_map.get((subject or "").lower(), 0)

    X = np.array([[length, complexity, demand, subj_enc]])
    price = reg.predict(X)[0]
    return max(10, round(price, 2))

# -------------------------
# Matching function (robust)
# -------------------------
def match_mentors(question_text: str, subject: str, db, top_k: int = 5):
    """
    Robust matching using:
     - data-driven keyword extraction (extract_keywords)
     - augment question with subject + canonical keywords
     - compute word-level TF-IDF and char_wb TF-IDF and combine
     - boost mentors that explicitly list the subject or share canonical keywords

    Returns list of dicts: [{"mentor_id": <int>, "score": <0..100 float>}, ...]
    """
    question_text = (question_text or "").strip()
    subject = (subject or "").strip().lower()

    mentors = db.query(User).filter(User.role == "mentor").all()
    if not mentors:
        return []

    # Extract keywords and canonicalize
    keywords = extract_keywords(question_text, top_k=8)

    # Augment question text so TF-IDF vocabulary includes subject + detected keywords
    augmented_question = " ".join(filter(None, [question_text, subject, " ".join(keywords)]))
    augmented_question = _normalize_text(augmented_question)

    # Build mentor profile texts (subjects + solved_keywords)
    mentor_texts = []
    mentor_ids = []
    for m in mentors:
        subj_text = (m.subjects or "")
        solved_text = (m.solved_keywords or "")
        profile = f"{subj_text} {solved_text}"
        mentor_texts.append(_normalize_text(profile))
        mentor_ids.append(m.id)

    # Corpus for TF-IDF
    corpus_word = [augmented_question] + mentor_texts
    corpus_char = [augmented_question] + mentor_texts

    # Word-level TF-IDF similarity
    try:
        vec_word = TfidfVectorizer(ngram_range=(1,2), max_features=5000, stop_words='english')
        Xw = vec_word.fit_transform(corpus_word)
        qw = Xw[0]
        mw = Xw[1:]
        sims_word = cosine_similarity(qw, mw).flatten()
    except Exception:
        sims_word = np.zeros(len(mentor_texts))

    # Char n-gram (char_wb) TF-IDF for robustness to misspellings / short tokens
    try:
        vec_char = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), max_features=5000)
        Xc = vec_char.fit_transform(corpus_char)
        qc = Xc[0]
        mc = Xc[1:]
        sims_char = cosine_similarity(qc, mc).flatten()
    except Exception:
        sims_char = np.zeros(len(mentor_texts))

    # Combine scores with weighting (tweakable)
    sims_combined = (0.75 * sims_word) + (0.25 * sims_char)

    # Convert to percent base
    base_percent = (sims_combined * 100.0).round(4)

    # Apply boosting for subject matches / keyword matches
    final_scores = []
    for idx, mid in enumerate(mentor_ids):
        m = mentors[idx]
        score = float(base_percent[idx])

        # boost if mentor explicitly lists the subject token (exact token match)
        if subject:
            mentor_subj_tokens = [s.strip().lower() for s in (m.subjects or "").split(",") if s.strip()]
            if subject in mentor_subj_tokens:
                score += 12.0  # absolute boost

        # small boost for each canonical keyword match in mentor profile
        profile_text = ( (m.subjects or "") + " " + (m.solved_keywords or "") ).lower()
        for kw in keywords:
            if kw and kw in profile_text:
                score += 6.0

        # cap
        score = min(100.0, round(score, 4))
        final_scores.append(score)

    # Build return list
    scored = [{"mentor_id": int(mid), "score": float(sc)} for mid, sc in zip(mentor_ids, final_scores)]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


