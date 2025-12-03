# list_mentors.py
from db import SessionLocal, engine
from models import User
from sqlalchemy import select

def main():
    print("DB engine:", getattr(engine, "url", str(engine)))
    db = SessionLocal()
    try:
        # total mentors
        total = db.query(User).filter(User.role == "mentor").count()
        print("Total mentors with role='mentor':", total)
        print("-" * 50)

        # show 10 newest mentors by id (assumes auto-increment id)
        q = db.query(User).filter(User.role == "mentor").order_by(User.id.desc()).limit(10)
        rows = q.all()
        print("10 newest mentors (id, name, email, subjects):")
        for r in rows:
            print(r.id, r.name, r.email, getattr(r, "subjects", None))
        print("-" * 50)

        # show 10 oldest mentors
        q2 = db.query(User).filter(User.role == "mentor").order_by(User.id.asc()).limit(10)
        print("10 oldest mentors:")
        for r in q2.all():
            print(r.id, r.name, r.email, getattr(r, "subjects", None))
    finally:
        db.close()

if __name__ == "__main__":
    main()
