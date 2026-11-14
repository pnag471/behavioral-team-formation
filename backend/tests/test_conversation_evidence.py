"""
GET /conversation/{session_id}/evidence: read-only reviewer view of the
evidence stored for a session's extraction run.

No real Gemini: one test runs the real /extract with the client mocked, the
others seed evidence_units rows directly.

Run with:
    cd backend && python3 -m pytest tests/test_conversation_evidence.py -q
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.conversation as conversation
import app.database as db_module
from app.database import Base, EvidenceUnitDB, ExtractionRunDB
from app.main import app

T0 = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def db_factory():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    original_session, original_init = db_module.SessionLocal, db_module.init_db
    db_module.SessionLocal, db_module.init_db = TestSession, (lambda: None)
    yield TestSession
    db_module.SessionLocal, db_module.init_db = original_session, original_init
    Base.metadata.drop_all(engine)


@pytest.fixture
def client(db_factory):
    with TestClient(app) as c:
        yield c


def add_run(db, run_id, session_id, created_at, error=False):
    db.add(ExtractionRunDB(
        id=run_id, session_id=session_id, model_name="gemini-test", prompt_version="p",
        params_json={"temperature": 0.1, **({"error": "boom"} if error else {})},
        rubric_version="v1", created_at=created_at,
    ))


def add_unit(db, run_id, session_id, trait, indicator, decision, needs_review, quote_start=0,
             polarity="supporting", strength="medium", quote="q", confidence=0.8):
    db.add(EvidenceUnitDB(
        id=f"{run_id}-{trait}-{indicator}-{decision}-{quote_start}", run_id=run_id, session_id=session_id,
        trait=trait, rubric_indicator=indicator, polarity=polarity, strength=strength, span=quote,
        rejected="true" if decision.startswith("rejected") else "false",
        confidence=confidence, needs_review=needs_review, decision=decision, quote_start=quote_start,
    ))


# ---------------------------------------------------------------------------
# Seeded rows
# ---------------------------------------------------------------------------

def seed_session(db_factory):
    db = db_factory()
    # Older run with different evidence: must NOT be shown.
    add_run(db, "run-old", "sess-1", T0)
    add_unit(db, "run-old", "sess-1", "planning_style", "spontaneous", "accepted", "false")
    # Latest successful run.
    add_run(db, "run-new", "sess-1", T0 + timedelta(hours=1))
    add_unit(db, "run-new", "sess-1", "planning_style", "planner", "accepted", "false", quote_start=10,
             strength="strong", quote="I made a shared calendar", confidence=0.9)
    add_unit(db, "run-new", "sess-1", "accountability", "high", "excluded:contradicting", "true",
             quote_start=40, polarity="contradicting")
    add_unit(db, "run-new", "sess-1", "conflict_style", "confrontational", "rejected:indicator_not_allowed", "true",
             quote_start=None)
    add_unit(db, "run-new", "sess-1", "cooperativeness", "high", "accepted", "false", quote_start=5)
    add_unit(db, "run-new", "sess-1", "cooperativeness", "high", "excluded:over_cap", "false", quote_start=80)
    # A later FAILED run (502): skipped, so it doesn't hide the evidence.
    add_run(db, "run-failed", "sess-1", T0 + timedelta(hours=2), error=True)
    db.commit()
    db.close()


def test_returns_latest_successful_run_items_with_fields(client, db_factory):
    seed_session(db_factory)
    body = client.get("/conversation/sess-1/evidence").json()
    assert body["session_id"] == "sess-1"
    assert body["run_id"] == "run-new"
    assert body["runs_for_session"] == 3
    assert body["rubric_version"] == "v1" and body["model_name"] == "gemini-test"
    assert len(body["items"]) == 5
    assert all(i["indicator"] != "spontaneous" for i in body["items"])   # old run's evidence not shown

    planner = next(i for i in body["items"] if i["indicator"] == "planner")
    assert planner == {
        "dimension": "planning_style", "indicator": "planner", "polarity": "supporting",
        "strength": "strong", "quote": "I made a shared calendar", "confidence": 0.9,
        "decision": "accepted", "needs_review": False,
    }


def test_flagged_items_come_first_then_by_dimension(client, db_factory):
    seed_session(db_factory)
    items = client.get("/conversation/sess-1/evidence").json()["items"]
    flags = [i["needs_review"] for i in items]
    assert flags == [True, True, False, False, False]
    assert [i["dimension"] for i in items[:2]] == ["accountability", "conflict_style"]
    assert [i["dimension"] for i in items[2:]] == ["cooperativeness", "cooperativeness", "planning_style"]
    assert [i["decision"] for i in items[2:4]] == ["accepted", "excluded:over_cap"]   # transcript order


def test_summary_counts_match(client, db_factory):
    seed_session(db_factory)
    body = client.get("/conversation/sess-1/evidence").json()
    assert body["summary"] == {
        "total": 5,
        "needs_review": 2,
        "by_decision": {
            "accepted": 2,
            "excluded:contradicting": 1,
            "excluded:over_cap": 1,
            "rejected:indicator_not_allowed": 1,
        },
    }
    assert body["summary"]["needs_review"] == sum(1 for i in body["items"] if i["needs_review"])


def test_legacy_rows_without_flags(client, db_factory):
    db = db_factory()
    add_run(db, "run-legacy", "sess-legacy", T0)
    db.add(EvidenceUnitDB(id="u1", run_id="run-legacy", session_id="sess-legacy", trait="accountability",
                          rubric_indicator="high", polarity="supporting", strength="weak", span="x"))
    db.commit()
    db.close()
    body = client.get("/conversation/sess-legacy/evidence").json()
    assert body["items"][0]["needs_review"] is None and body["items"][0]["decision"] is None
    assert body["summary"] == {"total": 1, "needs_review": 0, "by_decision": {"unknown": 1}}


def test_no_run_is_404(client):
    resp = client.get("/conversation/no-such-session/evidence")
    assert resp.status_code == 404
    assert "No extraction run" in resp.json()["detail"]


def test_only_failed_runs_is_404(client, db_factory):
    db = db_factory()
    add_run(db, "run-x", "sess-failed", T0, error=True)
    db.commit()
    db.close()
    resp = client.get("/conversation/sess-failed/evidence")
    assert resp.status_code == 404
    assert "1 failed run" in resp.json()["detail"]


def test_endpoint_is_read_only(client, db_factory):
    seed_session(db_factory)
    db = db_factory()
    before = (db.query(ExtractionRunDB).count(), db.query(EvidenceUnitDB).count())
    db.close()
    client.get("/conversation/sess-1/evidence")
    client.get("/conversation/no-such-session/evidence")
    db = db_factory()
    assert (db.query(ExtractionRunDB).count(), db.query(EvidenceUnitDB).count()) == before
    db.close()


# ---------------------------------------------------------------------------
# End to end: real /extract (Gemini mocked) -> reviewer view
# ---------------------------------------------------------------------------

def test_reviewer_view_after_mocked_extract(client, monkeypatch):
    history = [
        {"role": "interviewer", "content": "Tell me about a project."},
        {"role": "student", "content": "Last spring I made a shared calendar for our group the first day. "
                                       "Honestly I missed the deadline once and didn't tell anyone."},
    ]
    evidence = [
        {"quote": "I made a shared calendar for our group the first day", "dimension": "planning_style",
         "indicator": "planner", "polarity": "supporting", "strength": "strong", "confidence": 0.9},
        {"quote": "I missed the deadline once", "dimension": "accountability", "indicator": "high",
         "polarity": "contradicting", "strength": "strong", "confidence": 0.7},
        {"quote": "Tell me about a project", "dimension": "leadership_style", "indicator": "directive",
         "polarity": "supporting", "strength": "medium", "confidence": 0.2},
    ]
    raw = json.dumps({"evidence": evidence})
    fake = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **_: SimpleNamespace(text=raw)))
    monkeypatch.setattr(conversation, "_get_gemini_client", lambda: fake)

    extract = client.post("/conversation/extract", params={"session_id": "sess-e2e", "student_name": "Jo"},
                          json=history)
    assert extract.status_code == 200, extract.text

    body = client.get("/conversation/sess-e2e/evidence").json()
    assert body["run_id"] == extract.json()["run_id"]
    assert body["summary"] == {
        "total": 3, "needs_review": 2,
        "by_decision": {"accepted": 1, "excluded:contradicting": 1, "rejected:not_verbatim": 1},
    }
    assert [i["needs_review"] for i in body["items"]] == [True, True, False]
    assert body["items"][-1]["indicator"] == "planner"
