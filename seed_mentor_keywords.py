# backend/seed_mentor_keywords.py
import os, collections
import pandas as pd
from db import SessionLocal
from models import User

CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "synthetic_questions.csv")

def seed_keywords_per_subject(top_k_per_subject=30):
    df = pd.read_csv(CSV_PATH)
    db = SessionLocal()
    # collect common keywords per subject
    by_sub = {}
    for subj, g in df.groupby("subject"):
        all_kws = []
        for ks in g["keywords"].dropna().astype(str):
            all_kws += [k.strip().lower() for k in ks.split(",") if k.strip()]
        counter = collections.Counter(all_kws)
        common = [k for k,_ in counter.most_common(top_k_per_subject)]
        by_sub[subj] = common

    mentors = db.query(User).filter(User.role == "mentor").all()
    for m in mentors:
        subj_list = [s.strip().lower() for s in (m.subjects or "").split(",") if s.strip()]
        combined = []
        for s in subj_list:
            combined += by_sub.get(s, [])[:10]
        # dedupe and assign
        final = ",".join(list(dict.fromkeys([k for k in combined if k])))
        if final:
            m.solved_keywords = final
    try:
        db.commit()
        print("Mentor keywords seeded.")
    except Exception as e:
        db.rollback()
        print("Failed to commit mentor updates:", e)
    finally:
        db.close()

if __name__ == "__main__":
    seed_keywords_per_subject()
