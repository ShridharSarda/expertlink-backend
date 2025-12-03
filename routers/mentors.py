# routers/mentors.py
from fastapi import APIRouter, Depends
from typing import List
from db import SessionLocal
from sqlalchemy.orm import Session
from models import User
from utils import generate_meeting_link

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/")
def get_mentors(db: Session = Depends(get_db)):
    mentors = db.query(User).filter(User.role == "mentor").all()
    out = []
    for m in mentors:
        out.append({
            "id": m.id,
            "name": m.name,
            "email": m.email,
            "subjects": m.subjects,
            "rating": m.rating,
            "experience_years": m.experience_years
        })
    return {"mentors": out}
