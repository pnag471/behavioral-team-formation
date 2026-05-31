from __future__ import annotations
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.models import Student, Team

# ---------------------------------------------------------------------------
# Shared in-memory state — imported by reference in other modules
# ---------------------------------------------------------------------------
students_store: dict[str, Student] = {}
teams_store: dict[str, Team] = {}


# ---------------------------------------------------------------------------
# Lifespan: seed data on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    seed_path = Path(__file__).parent.parent / "sample_students.json"
    with open(seed_path) as f:
        data = json.load(f)
    for raw in data:
        student = Student(**raw)
        students_store[student.id] = student
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
    return list(students_store.values())


@app.post("/students", response_model=Student)
def add_student(student: Student):
    students_store[student.id] = student
    return student
