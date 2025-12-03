# seed_demo.py
from db import SessionLocal, engine
from models import Base, User
from utils import hash_password

Base.metadata.create_all(bind=engine)

def seed():
    db = SessionLocal()
    # check if any users
    if db.query(User).count() > 0:
        print("DB already seeded")
        return
    demo_mentors = [
        {"name":"Alice Mentor","email":"alice@demo.com","password":"pass","role":"mentor","subjects":"math,physics","experience_years":5},
        {"name":"Bob Mentor","email":"bob@demo.com","password":"pass","role":"mentor","subjects":"chemistry","experience_years":3},
        {"name":"Carol Mentor","email":"carol@demo.com","password":"pass","role":"mentor","subjects":"math","experience_years":4},
        {"name":"Dave Mentor","email":"dave@demo.com","password":"pass","role":"mentor","subjects":"cs","experience_years":6},
        {"name":"Eve Mentor","email":"eve@demo.com","password":"pass","role":"mentor","subjects":"math,cs","experience_years":2}
    ]
    demo_students = [
        {"name":"Student One","email":"s1@demo.com","password":"pass","role":"student"},
        {"name":"Student Two","email":"s2@demo.com","password":"pass","role":"student"}
    ]
    for u in demo_mentors + demo_students:
        user = User(
            name=u["name"], email=u["email"],
            password=hash_password(u["password"]),
            role=u["role"],
            subjects=u.get("subjects",""),
            experience_years=u.get("experience_years",0)
        )
        db.add(user)
    db.commit()
    db.close()
    print("Seed complete")

if __name__ == "__main__":
    seed()
