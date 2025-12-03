from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from db import SessionLocal, engine
from models import Question, User, Base
from schemas import QuestionIn
from ml.advanced_matcher import extract_keywords, match_mentors, predict_price
from utils import generate_meeting_link
from sqlalchemy.exc import IntegrityError

Base.metadata.create_all(bind=engine)

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/post")
def post_question(payload: QuestionIn, db: Session = Depends(get_db)):
    student = db.query(User).filter(User.id == payload.student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    # ML keyword extraction
    keywords = extract_keywords(payload.text)

    # ML price prediction (unchanged technique)
    price = predict_price(payload.text, payload.subject)

    # ML mentor matching - returns list of {"mentor_id":.., "score":..}
    ml_matches = match_mentors(payload.text, payload.subject, db) or []

    # Build maps/lists from ML output (handle older format if ml returned ints)
    ml_ids = []
    ml_scores_map = {}
    for entry in ml_matches:
        # entry may be dict {"mentor_id":..., "score":...} or an int string/id
        if isinstance(entry, dict) and "mentor_id" in entry:
            mid = int(entry["mentor_id"])
            ml_ids.append(mid)
            ml_scores_map[mid] = float(entry.get("score", 0.0))
        else:
            # try convert raw value to int
            try:
                mid = int(entry)
                ml_ids.append(mid)
            except Exception:
                pass

    # DB mentors who teach the subject (fallback / boost)
    db_matched = db.query(User).filter(
        User.role == "mentor",
        User.subjects.contains(payload.subject)
    ).all()
    db_ids = [m.id for m in db_matched] if db_matched else []

    # Combine: keep ML ordering, but ensure DB subject mentors are included
    combined_order = []
    for mid in ml_ids:
        if mid not in combined_order:
            combined_order.append(mid)
    for mid in db_ids:
        if mid not in combined_order:
            combined_order.append(mid)

    # absolute fallback: all mentors
    if not combined_order:
        all_mentors = db.query(User).filter(User.role == "mentor").all()
        combined_order = [m.id for m in all_mentors]

    matched_ids = combined_order

    # Save question (store matched mentors as CSV for compatibility)
    q = Question(
        student_id=payload.student_id,
        text=payload.text,
        subject=payload.subject,
        keywords=",".join(keywords),
        price=price,
        matched_mentors=",".join([str(x) for x in matched_ids]),
        status="matched"
    )
    db.add(q)
    db.commit()
    db.refresh(q)

    # Build mentor objects for frontend (name, subjects, computed score 0..1)
    mentor_rows = db.query(User).filter(User.id.in_(matched_ids)).all()
    mentor_map = {m.id: m for m in mentor_rows}

    def compute_overlap_score(mentor: User, question_keywords):
        # fallback overlap-based score (0..1)
        mk = []
        if mentor.solved_keywords:
            mk += [k.strip().lower() for k in mentor.solved_keywords.split(",") if k.strip()]
        if mentor.subjects:
            mk += [s.strip().lower() for s in mentor.subjects.split(",") if s.strip()]
        mk = list(dict.fromkeys(mk))
        if not mk or not question_keywords:
            return 0.0
        overlap = sum(1 for k in question_keywords if k.lower() in mk)
        return round(overlap / max(1, len(question_keywords)), 4)

    mentors_list = []
    for mid in matched_ids:
        m = mentor_map.get(int(mid))
        if not m:
            continue
        # prefer ML-provided score if present (ml_scores_map stores percent 0..100)
        raw_score = ml_scores_map.get(m.id)
        if raw_score is None:
            # fallback to overlap 0..1
            normalized_score = compute_overlap_score(m, keywords)
        else:
            # ml returned percent (0..100) — convert to 0..1 to keep frontend expectation
            try:
                normalized_score = float(raw_score) / 100.0
            except Exception:
                normalized_score = compute_overlap_score(m, keywords)

        mentors_list.append({
            "id": m.id,
            "name": m.name,
            "subjects": m.subjects.split(",") if m.subjects else [],
            "score": normalized_score  # 0..1 float (frontend expects this)
        })

    return {
        "question_id": q.id,
        "keywords": keywords,
        "price": price,
        "matched": mentors_list
    }


@router.post("/accept")
def accept_question(body: dict, db: Session = Depends(get_db)):
    """
    Mentor accepts a question. To avoid races (especially on SQLite which lacks
    strong row-level locking), we:
      1. load the question,
      2. ensure it's unaccepted,
      3. apply changes in-session,
      4. flush and re-check the DB row,
      5. commit or rollback on collision/error.
    """
    qid = int(body["question_id"])
    mid = int(body["mentor_id"])

    # initial load
    q = db.query(Question).filter(Question.id == qid).first()
    if not q:
        raise HTTPException(status_code=404, detail="Question not found")

    if q.accepted_mentor:
        raise HTTPException(status_code=400, detail="Already accepted")

    mentor = db.query(User).filter(User.id == mid, User.role == "mentor").first()
    if not mentor:
        raise HTTPException(status_code=404, detail="Mentor not found")

    # Apply updates (in-memory)
    q.accepted_mentor = mid
    q.status = "accepted"
    q.meeting_link = generate_meeting_link()

    # Update mentor stats safely (handle None values)
    mentor.solved_count = (mentor.solved_count or 0) + 1

    new_keywords = [k.strip().lower() for k in (q.keywords or "").split(",") if k.strip()]
    existing_keywords = [k.strip().lower() for k in (mentor.solved_keywords or "").split(",") if k.strip()]
    updated_keywords = list(dict.fromkeys(existing_keywords + new_keywords))
    mentor.solved_keywords = ",".join(updated_keywords)

    mentor.balance = (mentor.balance or 0.0) + (q.price or 0.0)

    db.add(mentor)
    db.add(q)

    try:
        # flush to push to DB layer (not commit). This will raise on constraint errors.
        db.flush()

        # Re-check the question from DB — if another process accepted it meanwhile, abort.
        recheck = db.query(Question).filter(Question.id == qid).first()
        if recheck.accepted_mentor and recheck.accepted_mentor != mid:
            db.rollback()
            raise HTTPException(status_code=400, detail="Collision detected: already accepted by another mentor")

        db.commit()
    except HTTPException:
        # propagate HTTP exceptions raised above (like collision)
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail="Failed to accept question")

    return {
        "status": "accepted",
        "meeting_link": q.meeting_link,
        "mentor_balance": mentor.balance,
        "mentor_keywords": updated_keywords,
        "solved_count": mentor.solved_count
    }

@router.get("/{question_id}")
def get_question(question_id: int, db: Session = Depends(get_db)):
    """
    Return a single question by id.
    Endpoint path will be /questions/{question_id} because router is included
    with prefix '/questions' in app.py.
    """
    q = db.query(Question).filter(Question.id == question_id).first()
    if not q:
        raise HTTPException(status_code=404, detail="Question not found")

    mentor_info = None
    if q.accepted_mentor:
        m = db.query(User).filter(User.id == q.accepted_mentor).first()
        if m:
            mentor_info = {
                "id": m.id,
                "name": m.name,
                "subjects": m.subjects.split(",") if m.subjects else [],
                # add more fields if you want (email, experience, rating...)
            }

    return {
        "id": q.id,
        "student_id": q.student_id,
        "text": q.text,
        "subject": q.subject,
        "keywords": q.keywords,
        "price": q.price,
        "status": q.status,
        "matched_mentors": q.matched_mentors,
        "accepted_mentor": q.accepted_mentor,
        "meeting_link": q.meeting_link,
        "mentor": mentor_info
    }


@router.get("/student/{student_id}")
def get_questions_by_student(student_id: int, db: Session = Depends(get_db)):
    """
    Return all questions posted by a student (most recent first)
    """
    qs = db.query(Question).filter(Question.student_id == student_id).order_by(Question.id.desc()).all()
    out = []
    for q in qs:
        out.append({
            "id": q.id,
            "text": q.text,
            "subject": q.subject,
            "keywords": q.keywords,
            "price": q.price,
            "status": q.status,
            "accepted_mentor": q.accepted_mentor,
            "meeting_link": q.meeting_link
        })
    return {"questions": out}

@router.get("/for_mentor/{mentor_id}")
def get_questions_for_mentor(mentor_id: int, db: Session = Depends(get_db)):
    mentor = db.query(User).filter(User.id == mentor_id).first()
    if not mentor:
        raise HTTPException(status_code=404, detail="Mentor not found")

    questions = db.query(Question).filter(
        Question.accepted_mentor.is_(None),
        Question.status != "closed"
    ).all()

    result = []
    for q in questions:
        if not q.matched_mentors:
            continue

        # normalize stored ids
        ids = [s.strip() for s in q.matched_mentors.split(",") if s.strip()]

        # convert safely to int
        try:
            ids = list(map(int, ids))
        except:
            continue  # skip invalid rows

        if mentor_id in ids:
            result.append({
                "id": q.id,
                "text": q.text,
                "subject": q.subject,
                "keywords": q.keywords,
                "price": q.price,
                "status": q.status,
            })
    
    return {"pending": result}
