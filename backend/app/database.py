from __future__ import annotations
from sqlalchemy import create_engine, Column, String, Float, JSON, DateTime
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


class AssessmentDB(Base):
    __tablename__ = "assessments"

    id = Column(String, primary_key=True)
    student_id = Column(String, nullable=False)
    status = Column(String, default="complete")
    model_version = Column(String, default="mock")
    created_at = Column(DateTime, default=datetime.utcnow)


class ResponseDB(Base):
    __tablename__ = "responses"

    id = Column(String, primary_key=True)
    assessment_id = Column(String, nullable=False)
    student_id = Column(String, nullable=False)
    scenario_id = Column(String, nullable=False)
    option_id = Column(String, nullable=False)
    response_text = Column(String, default="")
    timestamp = Column(DateTime, default=datetime.utcnow)


class BehavioralSignatureDB(Base):
    __tablename__ = "behavioral_signatures"

    student_id = Column(String, primary_key=True)
    signature = Column(JSON, nullable=False)
    model_version = Column(String, default="mock")
    created_at = Column(DateTime, default=datetime.utcnow)


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
