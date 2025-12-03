"""
ml/utils.py

Utilities for subject-vocabulary mapping and data-driven keyword extraction.

Usage:
    from ml.utils import load_subject_vocab, map_token_to_subject_vocab, extract_keywords_data_driven

Notes:
 - Run build_subject_vocab.py first to create ml/subject_vocab.json.
 - This module uses only stdlib + pandas optional (for other scripts), no heavy deps.
"""

import os
import json
import re
import difflib
from typing import Tuple, Optional, List

BASE = os.path.dirname(__file__)
VOCAB_FILE = os.path.join(BASE, "subject_vocab.json")

# Normalization helpers
def _normalize_token(t: str) -> str:
    if not t:
        return ""
    t = t.lower().strip()
    # keep alphanumerics and simple punctuation removed
    t = re.sub(r'[^a-z0-9\s]', '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

# Load subject vocab (cached)
_subject_vocab = None
def load_subject_vocab() -> dict:
    """
    Load subject_vocab.json into memory. Returns dict of {subject: [tokens...]}.
    If file missing, returns empty dict.
    """
    global _subject_vocab
    if _subject_vocab is None:
        if os.path.exists(VOCAB_FILE):
            try:
                with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
                    _subject_vocab = json.load(f)
            except Exception:
                _subject_vocab = {}
        else:
            _subject_vocab = {}
    return _subject_vocab

def clear_subject_vocab_cache():
    """Clear in-memory cache (if you re-generate subject_vocab.json during runtime)."""
    global _subject_vocab
    _subject_vocab = None

def _all_vocab_tokens():
    """Return flattened list of (subject, token) tuples for faster scanning if needed."""
    vocab = load_subject_vocab()
    out = []
    for subj, toks in vocab.items():
        for t in toks:
            out.append((subj, t))
    return out

def map_token_to_subject_vocab(token: str, cutoff: float = 0.62) -> Tuple[Optional[str], Optional[str], float]:
    """
    Map a token -> (subject, canonical_token, similarity_score).
    Uses difflib fuzzy matching against the canonical subject vocab tokens.
    Returns (None, None, 0.0) if no good match found.

    cutoff: minimum difflib matching ratio to consider as a match (0..1).
    """
    token = _normalize_token(token)
    if not token:
        return None, None, 0.0

    vocab = load_subject_vocab()
    if not vocab:
        return None, None, 0.0

    best_subject = None
    best_token = None
    best_ratio = 0.0

    # Iterate subjects, use get_close_matches per subject for speed and locality
    for subj, tokens in vocab.items():
        # difflib requires sequence of strings; tokens ideally are pre-normalized when built
        match = difflib.get_close_matches(token, tokens, n=1, cutoff=cutoff)
        if match:
            cand = match[0]
            ratio = difflib.SequenceMatcher(None, token, cand).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_token = cand
                best_subject = subj

    if best_ratio >= cutoff:
        return best_subject, best_token, float(best_ratio)
    return None, None, 0.0

# Lightweight keyword extractor fallback (no external deps)
def _simple_token_extractor(text: str, max_k: int = 10) -> List[str]:
    """
    Extract candidate tokens from text:
     - split on punctuation/space
     - collect n-grams of length 1..2 (phrases) filtered by length
     - sort by simple frequency and return top-k
    """
    if not text:
        return []
    s = text.lower()
    s = re.sub(r'[^a-z0-9\s,]', ' ', s)
    tokens = [t.strip() for t in re.split(r'[,\\s]+', s) if t.strip()]
    # filter very short tokens
    tokens = [t for t in tokens if len(t) > 2]
    # build unigrams and bigrams frequency
    freq = {}
    for i, t in enumerate(tokens):
        freq[t] = freq.get(t, 0) + 1
        if i + 1 < len(tokens):
            big = t + " " + tokens[i + 1]
            freq[big] = freq.get(big, 0) + 1
    # sort by frequency then length/lexicographically
    items = sorted(freq.items(), key=lambda kv: (-kv[1], -len(kv[0]), kv[0]))
    return [k for k, _ in items][:max_k]

# stopwords / noisy tokens you might want to ignore
_DEFAULT_STOPWORDS = set([
    "doubt", "question", "query", "please", "help", "solve", "problem", "show",
    "explain", "with", "example", "detailed", "step", "steps", "stepby", "stepbystep"
])

def extract_keywords_data_driven(text: str, top_k: int = 6, stopwords: set = None) -> List[str]:
    """
    Data-driven keyword extraction:
     - Accepts free-text input (question)
     - Attempts: (1) comma-split quick parse, (2) simple token extractor fallback
     - Normalizes tokens, removes stopwords, maps tokens to canonical subject tokens using subject vocab
     - Returns up to top_k canonical tokens (or normalized tokens if no mapping)
    """
    if stopwords is None:
        stopwords = _DEFAULT_STOPWORDS

    if not text or not text.strip():
        return []

    t = text.strip()

    # 1) If user provided comma-separated keywords, prefer them (and keep short phrase tokens)
    candidate_tokens = []
    if ',' in t and len(t.split(',')) <= 10:
        candidate_tokens = [p.strip() for p in t.split(',') if p.strip()]
    else:
        # fallback to simple extractor
        candidate_tokens = _simple_token_extractor(t, max_k=top_k * 5)

    out_tokens = []
    for tok in candidate_tokens:
        tok_norm = _normalize_token(tok)
        if not tok_norm or tok_norm in stopwords:
            continue
        subj, canon, ratio = map_token_to_subject_vocab(tok_norm)
        if canon:
            if canon not in out_tokens:
                out_tokens.append(canon)
        else:
            # no mapping found; keep normalized token to preserve signal
            if tok_norm not in out_tokens:
                out_tokens.append(tok_norm)
        if len(out_tokens) >= top_k:
            break

    return out_tokens

# convenience wrapper name used by your routes
def extract_keywords(text: str, top_k: int = 6) -> List[str]:
    """
    Backwards-compatible wrapper. Use this name from routers/questions.py
    """
    return extract_keywords_data_driven(text, top_k=top_k)
