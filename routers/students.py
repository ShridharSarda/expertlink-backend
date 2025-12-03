# routers/students.py
from fastapi import APIRouter, Depends, HTTPException
from db import SessionLocal
from sqlalchemy.orm import Session
from models import User

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/{student_id}")
def get_student(student_id: int, db: Session = Depends(get_db)):
    s = db.query(User).filter(User.id == student_id, User.role == "student").first()
    if not s:
        raise HTTPException(status_code=404, detail="Student not found")
    return {"id": s.id, "name": s.name, "email": s.email}
