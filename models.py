# models.py

from sqlalchemy import Column, Integer, String, Text, Float, Enum, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from db import Base
from sqlalchemy.sql import func
from sqlalchemy import DateTime

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=False)
    role = Column(String, nullable=False)  # student or mentor

    # mentor-specific fields
    subjects = Column(String, nullable=True)
    experience_years = Column(Integer, default=0)

    solved_count = Column(Integer, default=0)                  # total number solved
    solved_keywords = Column(Text, default="")                 # comma-separated keywords
    balance = Column(Float, default=0.0)                       # earnings
    expertise_score = Column(Float, default=1.0)               # ML score

    rating = Column(Float, default=4.5)
    is_active = Column(Boolean, default=True)


class Question(Base):
    __tablename__ = "questions"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    text = Column(Text, nullable=True)
    subject = Column(String, nullable=True)
    keywords = Column(String, nullable=True)
    price = Column(Float, default=0.0)

    difficulty = Column(Float, default=1.0)  # NEW FIELD

    accepted_mentor = Column(Integer, nullable=True)
    matched_mentors = Column(String, nullable=True)
    meeting_link = Column(String, nullable=True)
    status = Column(String, default="matched")
