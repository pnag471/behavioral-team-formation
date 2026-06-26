"""
Unit tests for the assessment pipeline.

Run with:
    cd backend && venv/bin/pytest tests/test_assessment.py -v
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.database as db_module
from app.database import Base
from app.main import app

# ---------------------------------------------------------------------------
# Test responses using ANONYMOUS option IDs (opt_a … opt_d).
# Mapping for reference (see bank_v1.yaml):
#   missed_deadline     opt_c → conflict_style=collaborative, cooperativeness=high
#   workload_imbalance  opt_c → conflict_style=collaborative, cooperativeness=high, accountability=high
#   comm_breakdown      opt_c → leadership_style=facilitative, planning_style=planner, communication_style=mixed
#   leadership_vacuum   opt_b → leadership_style=facilitative, communication_style=sync
#   technical_disagreement opt_c → conflict_style=collaborative, execution_style=methodical
#   ambiguous_requirements opt_b → leadership_style=facilitative, communication_style=sync
# ---------------------------------------------------------------------------
ALL_RESPONSES = [
    {"scenario_id": "missed_deadline",         "option_id": "opt_c", "response_text": ""},
    {"scenario_id": "workload_imbalance",       "option_id": "opt_c", "response_text": ""},
    {"scenario_id": "communication_breakdown",  "option_id": "opt_c", "response_text": ""},
    {"scenario_id": "leadership_vacuum",        "option_id": "opt_b", "response_text": ""},
    {"scenario_id": "technical_disagreement",   "option_id": "opt_c", "response_text": ""},
    {"scenario_id": "ambiguous_requirements",   "option_id": "opt_b", "response_text": ""},
]

# Expected scorer output for the above responses (deterministic):
#   conflict_style:       collaborative (3 signals)
#   accountability:       high (1 signal)
#   cooperativeness:      high (2 signals)
#   leadership_style:     facilitative (3 signals)
#   planning_style:       planner (2 signals)
#   communication_style:  mixed (1) vs sync (2) → sync
#   execution_style:      methodical (1 signal)
#   help_seeking:         no signals → default "proactive"
EXPECTED_PLANNING_STYLE = "planner"


# ---------------------------------------------------------------------------
# Fixture: fresh in-memory DB for each test
# No LLM mock needed — scoring is now deterministic code.
# ---------------------------------------------------------------------------
@pytest.fixture
def client():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    original_session = db_module.SessionLocal
    original_init_db = db_module.init_db
    db_module.SessionLocal = TestSession
    db_module.init_db = lambda: None

    with TestClient(app) as c:
        yield c

    db_module.SessionLocal = original_session
    db_module.init_db = original_init_db
    Base.metadata.drop_all(engine)


# ---------------------------------------------------------------------------
# 1. Scenarios endpoint
# ---------------------------------------------------------------------------
def test_get_scenarios(client):
    resp = client.get("/assessment/scenarios")
    assert resp.status_code == 200
    scenarios = resp.json()
    assert len(scenarios) == 6
    for s in scenarios:
        assert "id" in s
        assert "title" in s
        assert "description" in s
        assert len(s["options"]) == 4
        for opt in s["options"]:
            # option IDs must be anonymous
            assert opt["id"].startswith("opt_"), f"Expected anonymous ID, got {opt['id']!r}"


# ---------------------------------------------------------------------------
# 2. Analyze creates all DB rows for a new student
# ---------------------------------------------------------------------------
def test_analyze_creates_student(client):
    from app.database import StudentDB, AssessmentDB, ResponseDB, BehavioralSignatureDB

    resp = client.post("/assessment/analyze", json={
        "student_name": "Test Student Alpha",
        "responses": ALL_RESPONSES,
    })
    assert resp.status_code == 200
    data = resp.json()
    student_id = data["student_id"]
    session_id = data["session_id"]

    db = db_module.SessionLocal()
    try:
        assert db.query(StudentDB).filter(StudentDB.id == student_id).first() is not None
        assessment = db.query(AssessmentDB).filter(AssessmentDB.id == session_id).first()
        assert assessment is not None
        assert assessment.status == "complete"
        responses = db.query(ResponseDB).filter(ResponseDB.assessment_id == session_id).all()
        assert len(responses) == 6
        sig_row = (
            db.query(BehavioralSignatureDB)
            .filter(BehavioralSignatureDB.student_id == student_id)
            .first()
        )
        assert sig_row is not None
        assert sig_row.rubric_version == "v1"
        assert sig_row.bank_version == "v1"
        assert sig_row.scorer_version == "v1"
        assert sig_row.signature["planning_style"] == EXPECTED_PLANNING_STYLE
    finally:
        db.close()


# ---------------------------------------------------------------------------
# 3. Analyze updates an existing student rather than creating a duplicate
# ---------------------------------------------------------------------------
def test_analyze_updates_existing_student(client):
    from app.database import StudentDB

    resp1 = client.post("/assessment/analyze", json={
        "student_name": "Test Student Beta",
        "responses": ALL_RESPONSES,
    })
    assert resp1.status_code == 200
    student_id = resp1.json()["student_id"]

    resp2 = client.post("/assessment/analyze", json={
        "student_name": "Test Student Beta",
        "student_id": student_id,
        "responses": ALL_RESPONSES,
    })
    assert resp2.status_code == 200
    assert resp2.json()["student_id"] == student_id

    db = db_module.SessionLocal()
    try:
        count = db.query(StudentDB).filter(StudentDB.id == student_id).count()
        assert count == 1, f"Expected 1 StudentDB row, got {count}"
        row = db.query(StudentDB).filter(StudentDB.id == student_id).first()
        assert row.work_rhythm_signature["planning_style"] == EXPECTED_PLANNING_STYLE
    finally:
        db.close()


# ---------------------------------------------------------------------------
# 4. Start session creates an incomplete AssessmentDB row
# ---------------------------------------------------------------------------
def test_start_session(client):
    from app.database import AssessmentDB

    resp = client.post("/assessment/start", json={"student_name": "Test Student Gamma"})
    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    assert "student_id" in data

    db = db_module.SessionLocal()
    try:
        session = db.query(AssessmentDB).filter(AssessmentDB.id == data["session_id"]).first()
        assert session is not None
        assert session.status == "incomplete"
        assert session.student_id == data["student_id"]
    finally:
        db.close()


# ---------------------------------------------------------------------------
# 5. Save-progress stores responses without changing session status
# ---------------------------------------------------------------------------
def test_save_progress(client):
    from app.database import AssessmentDB, ResponseDB

    start = client.post("/assessment/start", json={"student_name": "Test Student Delta"})
    assert start.status_code == 200
    session_id = start.json()["session_id"]
    student_id = start.json()["student_id"]

    partial = ALL_RESPONSES[:3]
    resp = client.post("/assessment/save-progress", json={
        "session_id": session_id,
        "student_id": student_id,
        "responses": partial,
    })
    assert resp.status_code == 200
    assert resp.json()["saved"] == 3

    db = db_module.SessionLocal()
    try:
        session = db.query(AssessmentDB).filter(AssessmentDB.id == session_id).first()
        assert session.status == "incomplete", "Status must remain incomplete after save-progress"
        saved = db.query(ResponseDB).filter(ResponseDB.assessment_id == session_id).all()
        assert len(saved) == 3
    finally:
        db.close()


# ---------------------------------------------------------------------------
# 6. Analyze with session_id completes the existing session (no duplicate)
# ---------------------------------------------------------------------------
def test_analyze_completes_existing_session(client):
    from app.database import AssessmentDB

    start = client.post("/assessment/start", json={"student_name": "Test Student Epsilon"})
    assert start.status_code == 200
    session_id = start.json()["session_id"]
    student_id = start.json()["student_id"]

    resp = client.post("/assessment/analyze", json={
        "student_name": "Test Student Epsilon",
        "student_id": student_id,
        "session_id": session_id,
        "responses": ALL_RESPONSES,
    })
    assert resp.status_code == 200
    assert resp.json()["session_id"] == session_id

    db = db_module.SessionLocal()
    try:
        rows = db.query(AssessmentDB).filter(AssessmentDB.student_id == student_id).all()
        assert len(rows) == 1, f"Expected 1 AssessmentDB row, got {len(rows)}"
        assert rows[0].status == "complete"
    finally:
        db.close()


# ---------------------------------------------------------------------------
# 7. Get session returns status and saved responses
# ---------------------------------------------------------------------------
def test_get_session(client):
    start = client.post("/assessment/start", json={"student_name": "Test Student Zeta"})
    assert start.status_code == 200
    session_id = start.json()["session_id"]
    student_id = start.json()["student_id"]

    client.post("/assessment/save-progress", json={
        "session_id": session_id,
        "student_id": student_id,
        "responses": ALL_RESPONSES[:2],
    })

    resp = client.get(f"/assessment/session/{session_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == session_id
    assert data["student_id"] == student_id
    assert data["status"] == "incomplete"
    assert len(data["responses"]) == 2
    assert data["responses"][0]["scenario_id"] == "missed_deadline"


# ---------------------------------------------------------------------------
# 8. Scorer determinism — same inputs must yield the same profile
# ---------------------------------------------------------------------------
def test_analyze_is_deterministic(client):
    """The scoring path contains no random or LLM components."""
    from app.database import StudentDB

    resp1 = client.post("/assessment/analyze", json={
        "student_name": "Determinism Test A",
        "responses": ALL_RESPONSES,
    })
    resp2 = client.post("/assessment/analyze", json={
        "student_name": "Determinism Test B",
        "responses": ALL_RESPONSES,
    })
    assert resp1.status_code == 200
    assert resp2.status_code == 200

    sig1 = resp1.json()["signature"]
    sig2 = resp2.json()["signature"]

    for key in ("planning_style", "conflict_style", "accountability",
                "leadership_style", "cooperativeness", "execution_style"):
        assert sig1[key] == sig2[key], f"Non-deterministic result for {key!r}"
