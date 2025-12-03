# schemas.py
from pydantic import BaseModel
from typing import List, Optional

class RegisterIn(BaseModel):
    name: str
    email: str
    password: str
    role: str  # 'student' or 'mentor'
    subjects: Optional[str] = None

class LoginIn(BaseModel):
    email: str
    password: str

class QuestionIn(BaseModel):
    student_id: int
    text: Optional[str] = None
    subject: Optional[str] = None
    price: Optional[float] = 0.0
