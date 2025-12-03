# utils.py
from passlib.context import CryptContext
import jwt
from datetime import datetime, timedelta
from uuid import uuid4

# Using pbkdf2_sha256 (PURE PYTHON — works on Python 3.12 and Windows)
pwd = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

JWT_SECRET = "replace_with_a_strong_secret"
JWT_ALGO = "HS256"

def hash_password(password: str) -> str:
    return pwd.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd.verify(plain, hashed)

def create_jwt(user_id: int, role: str, expires_minutes: int = 60*24*7):
    payload = {
        "sub": str(user_id),
        "role": role,
        "exp": datetime.utcnow() + timedelta(minutes=expires_minutes)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)

def generate_meeting_link():
    return f"https://meet.jit.si/{uuid4()}"
