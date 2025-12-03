# backend/load_questions_from_csv.py
import os
import pandas as pd
from db import SessionLocal, engine
from models import Base, Question, User
# make sure Base.metadata.create_all(bind=engine) already executed elsewhere (app startup or seed)
from sqlalchemy.exc import IntegrityError

CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "synthetic_questions.csv")

def load_questions(limit=None, assign_student_id=1):
    df = pd.read_csv(CSV_PATH)
    if limit:
        df = df.head(limit)
    db = SessionLocal()
    created = 0
    for _, row in df.iterrows():
        try:
            q = Question(
                student_id=assign_student_id,
                text=str(row["text"]),
                subject=str(row["subject"]),
                keywords=str(row["keywords"]),
                price=float(row["price"]),
                matched_mentors="",  # leave blank - matching logic will fill later
                status="matched"
            )
            db.add(q)
            created += 1
        except Exception as e:
            print("Skipping row due to error:", e)
    try:
        db.commit()
    except Exception as e:
        db.rollback()
        print("Commit failed:", e)
    finally:
        db.close()
    print("Imported", created, "questions into DB.")

if __name__ == "__main__":
    load_questions(limit=None, assign_student_id=1)
