from __future__ import annotations
from sqlalchemy import create_engine, Column, String, Float, JSON, DateTime, UniqueConstraint
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./behavioral_team_formation.db")

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# Tables

class StudentDB(Base):
    __tablename__ = "students"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    competence_signature = Column(JSON, default={})
    work_rhythm_signature = Column(JSON, default={})
    collaboration_signature = Column(JSON, default={})
    motivation_layer = Column(JSON, default={})
    confidence_layer = Column(JSON, default={})
    # Dimensions for which the last scorer run had zero MCQ signals.
    under_determined_dims = Column(JSON, default=[])


class AssessmentDB(Base):
    __tablename__ = "assessments"

    id = Column(String, primary_key=True)
    student_id = Column(String, nullable=False)
    status = Column(String, default="complete")
    # Retained for explainability LLM calls; null for deterministic scoring sessions.
    model_version = Column(String, nullable=True, default=None)
    created_at = Column(DateTime, default=datetime.utcnow)


class ResponseDB(Base):
    __tablename__ = "responses"

    id = Column(String, primary_key=True)         # UUID — surrogate key
    assessment_id = Column(String, nullable=False)
    student_id = Column(String, nullable=False)
    scenario_id = Column(String, nullable=False)
    option_id = Column(String, nullable=False)     # anonymous: opt_a … opt_d
    response_text = Column(String, default="")
    timestamp = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("assessment_id", "scenario_id", name="uq_response_session_scenario"),
    )


class BehavioralSignatureDB(Base):
    __tablename__ = "behavioral_signatures"

    id = Column(String, primary_key=True)          # UUID
    student_id = Column(String, nullable=False, index=True)
    signature = Column(JSON, nullable=False)
    # Scoring provenance — no model/prompt fields; scoring is deterministic code.
    rubric_version = Column(String, nullable=False)
    bank_version = Column(String, nullable=False)
    scorer_version = Column(String, nullable=False)
    scored_at = Column(DateTime, default=datetime.utcnow)


class TeamDB(Base):
    __tablename__ = "teams"

    id = Column(String, primary_key=True)
    members = Column(JSON, nullable=False)
    score_breakdown = Column(JSON, nullable=False)
    team_norms = Column(JSON, default=[])
    formation_mode = Column(String, default="behavioral")
    created_at = Column(DateTime, default=datetime.utcnow)


class TeamScoreDB(Base):
    __tablename__ = "team_scores"

    team_id = Column(String, primary_key=True)
    scores = Column(JSON, nullable=False)
    confidence_score = Column(Float, default=0.0)


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(bind=engine)
