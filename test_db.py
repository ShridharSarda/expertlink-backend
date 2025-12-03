# test_db.py
from db import SessionLocal
from models import Question, User

db = SessionLocal()
try:
    qcount = db.query(Question).count()
    mentors = db.query(User).filter(User.role == 'mentor').all()
    print("Questions in DB:", qcount)
    print("Mentors in DB:", len(mentors))
    for m in mentors:
        print(m.id, m.name, "subjects=", m.subjects, "solved_keywords_len=", len((m.solved_keywords or "").split(",")))
finally:
    db.close()
