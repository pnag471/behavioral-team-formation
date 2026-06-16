from __future__ import annotations
from sqlalchemy import create_engine, Column, String, Float, JSON, DateTime
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timezone
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
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class ResponseDB(Base):
    __tablename__ = "responses"

    id = Column(String, primary_key=True)
    assessment_id = Column(String, nullable=False)
    student_id = Column(String, nullable=False)
    scenario_id = Column(String, nullable=False)
    option_id = Column(String, nullable=False)
    response_text = Column(String, default="")
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class BehavioralSignatureDB(Base):
    __tablename__ = "behavioral_signatures"

    student_id = Column(String, primary_key=True)
    signature = Column(JSON, nullable=False)
    model_version = Column(String, default="mock")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class TeamDB(Base):
    __tablename__ = "teams"

    id = Column(String, primary_key=True)
    members = Column(JSON, nullable=False)
    score_breakdown = Column(JSON, nullable=False)
    team_norms = Column(JSON, default=[])
    formation_mode = Column(String, default="behavioral")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


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


# ---------------------------------------------------------------------------
# Research infrastructure tables (per evaluation architecture doc)
# ---------------------------------------------------------------------------

class ParticipantDB(Base):
    __tablename__ = "participants"
    id = Column(String, primary_key=True)  # surrogate key, used everywhere
    student_id = Column(String, nullable=True)  # links to students table
    cohort = Column(String, nullable=True)  # e.g. '2026-fall-cse310'
    enrolled_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    consent_version = Column(String, nullable=True)
    consent_scope = Column(JSON, default=[])  # list of consented scopes


class ParticipantAttributesDB(Base):
    __tablename__ = "participant_attributes"
    participant_id = Column(String, primary_key=True)
    language_background = Column(String, nullable=True)
    is_native_english = Column(String, nullable=True)
    demographic_json = Column(JSON, default={})


class ScenarioDB(Base):
    __tablename__ = "scenarios"
    id = Column(String, primary_key=True)
    version = Column(String, default="1.0")
    target_dimensions = Column(JSON, default=[])
    prompt_text = Column(String, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class ConversationTurnDB(Base):
    __tablename__ = "conversation_turns"
    id = Column(String, primary_key=True)
    session_id = Column(String, nullable=False)
    turn_index = Column(String, nullable=False)
    role = Column(String, nullable=False)  # 'interviewer' | 'student'
    text = Column(String, nullable=False)
    shown_at = Column(DateTime, nullable=True)
    response_at = Column(DateTime, nullable=True)
    char_count = Column(String, nullable=True)


class ExtractionRunDB(Base):
    __tablename__ = "extraction_runs"
    id = Column(String, primary_key=True)
    session_id = Column(String, nullable=False)
    model_name = Column(String, nullable=False)
    prompt_version = Column(String, default="1.0")
    params_json = Column(JSON, default={})
    run_purpose = Column(String, default="production")  # production | rerun | ablation
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class ProfileVersionDB(Base):
    __tablename__ = "profile_versions"
    id = Column(String, primary_key=True)
    session_id = Column(String, nullable=False)
    student_id = Column(String, nullable=False)
    run_id = Column(String, nullable=False)  # links to extraction_runs
    signature_json = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class DimensionScoreDB(Base):
    __tablename__ = "dimension_scores"
    id = Column(String, primary_key=True)
    profile_version_id = Column(String, nullable=False)
    dimension = Column(String, nullable=False)
    value = Column(String, nullable=False)
    confidence = Column(Float, nullable=True)
    evidence_span = Column(String, nullable=True)  # supporting quote from transcript


class AnnotatorDB(Base):
    __tablename__ = "annotators"
    id = Column(String, primary_key=True)
    expertise = Column(String, nullable=True)
    training_version = Column(String, nullable=True)


class HumanAnnotationDB(Base):
    __tablename__ = "human_annotations"
    id = Column(String, primary_key=True)
    session_id = Column(String, nullable=False)
    annotator_id = Column(String, nullable=False)
    dimension = Column(String, nullable=False)
    value = Column(String, nullable=False)
    evidence_spans_json = Column(JSON, default=[])
    annotated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    is_adjudicated = Column(String, default="false")


class SelfReportResponseDB(Base):
    __tablename__ = "self_report_responses"
    id = Column(String, primary_key=True)
    session_id = Column(String, nullable=False)
    instrument = Column(String, nullable=False)  # 'mcq_v1' | 'social_desirability'
    item_id = Column(String, nullable=False)
    value = Column(String, nullable=False)
    collected_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


# Stub tables — create now, populate post-collaboration

class PeerEvaluationDB(Base):
    __tablename__ = "peer_evaluations"
    id = Column(String, primary_key=True)
    team_id = Column(String, nullable=False)
    rater_id = Column(String, nullable=False)
    ratee_id = Column(String, nullable=False)
    dimension = Column(String, nullable=False)
    value = Column(Float, nullable=False)
    collected_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class OutcomeMeasureDB(Base):
    __tablename__ = "outcome_measures"
    id = Column(String, primary_key=True)
    team_id = Column(String, nullable=True)
    participant_id = Column(String, nullable=True)
    measure_type = Column(String, nullable=False)
    value = Column(Float, nullable=False)
    collected_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class ContributionLogDB(Base):
    __tablename__ = "contribution_logs"
    id = Column(String, primary_key=True)
    participant_id = Column(String, nullable=False)
    source = Column(String, nullable=False)  # 'github' | 'slack'
    metric = Column(String, nullable=False)
    value = Column(Float, nullable=False)
    window_start = Column(DateTime, nullable=True)
    window_end = Column(DateTime, nullable=True)