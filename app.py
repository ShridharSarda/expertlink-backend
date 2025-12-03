# app.py
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import auth, mentors, questions, students
from ml.train import ensure_model_trained

# ensure model exists before app starts (non-blocking simple check)
ensure_model_trained()

app = FastAPI(title="Expert Link (SQLite)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # change to your React origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(mentors.router, prefix="/mentors", tags=["mentors"])
app.include_router(students.router, prefix="/students", tags=["students"])
app.include_router(questions.router, prefix="/questions", tags=["questions"])


@app.get("/")
def index():
    return {"message": "Expert Link backend (SQLite) running"}
