from dotenv import load_dotenv
load_dotenv()
from __future__ import annotations
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.models import Student, Team
from app.database import init_db, SessionLocal, StudentDB

# ---------------------------------------------------------------------------
# Lifespan: seed data on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()

    db = SessionLocal()
    try:
        existing = db.query(StudentDB).first()
        if not existing:
            seed_path = Path(__file__).parent.parent / "sample_students.json"
            with open(seed_path) as f:
                data = json.load(f)
            for raw in data:
                student = Student(**raw)
                db_student = StudentDB(
                    id=student.id,
                    name=student.name,
                    competence_signature=student.competence_signature.model_dump(),
                    work_rhythm_signature=student.work_rhythm_signature.model_dump(),
                    collaboration_signature=student.collaboration_signature.model_dump(),
                    motivation_layer=student.motivation_layer.model_dump(),
                    confidence_layer=student.confidence_layer.model_dump(),
                )
                db.add(db_student)
            db.commit()
    finally:
        db.close()

    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Behavioral Team Formation API",
    description="Research prototype for AI-assisted behavioral team formation",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import routers after app is created to avoid circular imports
from app import behavioral_analysis, matching, explainability  # noqa: E402

app.include_router(behavioral_analysis.router)
app.include_router(matching.router)
app.include_router(explainability.router)


# ---------------------------------------------------------------------------
# Core student routes
# ---------------------------------------------------------------------------
@app.get("/")
def health():
    return {"status": "ok", "service": "behavioral-team-formation"}


@app.get("/students", response_model=List[Student])
def list_students():
    db = SessionLocal()
    try:
        students = db.query(StudentDB).all()
        return [
            Student(
                id=s.id,
                name=s.name,
                competence_signature=s.competence_signature,
                work_rhythm_signature=s.work_rhythm_signature,
                collaboration_signature=s.collaboration_signature,
                motivation_layer=s.motivation_layer,
                confidence_layer=s.confidence_layer,
            )
            for s in students
        ]
    finally:
        db.close()


@app.post("/students", response_model=Student)
def add_student(student: Student):
    db = SessionLocal()
    try:
        db_student = StudentDB(
            id=student.id,
            name=student.name,
            competence_signature=student.competence_signature.model_dump(),
            work_rhythm_signature=student.work_rhythm_signature.model_dump(),
            collaboration_signature=student.collaboration_signature.model_dump(),
            motivation_layer=student.motivation_layer.model_dump(),
            confidence_layer=student.confidence_layer.model_dump(),
        )
        db.add(db_student)
        db.commit()
        return student
    finally:
        db.close()