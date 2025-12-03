# routers/auth.py
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from db import SessionLocal, engine, Base
from models import User
from schemas import RegisterIn, LoginIn
from utils import hash_password, verify_password, create_jwt
from typing import Generator

# create DB tables if not exist
Base.metadata.create_all(bind=engine)

router = APIRouter()

def get_db() -> Generator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/register")
def register(payload: RegisterIn, db: Session = Depends(get_db)):
    # check exists
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(
        name=payload.name,
        email=payload.email,
        password=hash_password(payload.password),
        role=payload.role,
        subjects=payload.subjects or ""
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"status": "registered", "user_id": user.id}

@router.post("/login")
def login(payload: LoginIn, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user:
        raise HTTPException(status_code=400, detail="User not found")
    if not verify_password(payload.password, user.password):
        raise HTTPException(status_code=400, detail="Wrong credentials")
    token = create_jwt(user.id, user.role)
    return {"token": token, "user": {"id": user.id, "name": user.name, "role": user.role}}
