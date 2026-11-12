"""
Tests for POST /conversation/extract (evidence-only extraction -> deterministic scorer).

The Gemini client is MOCKED: nothing touches the network or needs an API key.
Each test gets a fresh in-memory DB (same fixture pattern as test_assessment.py).

Run with:
    cd backend && python3 -m pytest tests/test_conversation_extract.py -q
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.conversation as conversation
import app.database as db_module
from app.database import (
    Base, BehavioralSignatureDB, DimensionScoreDB, EvidenceUnitDB, ExtractionRunDB,
    ProfileVersionDB, StudentDB,
)
from app.main import app
from scoring.dimension_scorer import get_rubric, score
from scoring.interview_adapter import EvidenceItem, evidence_to_signals

RUBRIC = get_rubric()
SESSION_ID = "sess-test-1"
NAME = "Jordan Lee"

HISTORY = [
    {"role": "interviewer", "content": "Tell me about a recent group project."},
    {"role": "student", "content": "Last spring I made a shared calendar for our group the first day."},
    {"role": "interviewer", "content": "How did disagreements go?"},
    {"role": "student", "content": "When we disagreed about the database, I told them straight out I didn’t agree."},
    {"role": "interviewer", "content": "What about deadlines?"},
    {"role": "student", "content": "Honestly I missed the deadline once and didn't tell anyone. "
                                   "I always reply to the group chat within an hour."},
    {"role": "interviewer", "content": "Anything else?"},
    {"role": "student", "content": "I check in with everyone before our weekly meeting. "
                                   "I also review my teammates' pull requests the same day. "
                                   "I helped my teammate debug her code at midnight."},
]


def ev(quote, dimension, indicator, polarity="supporting", strength="medium", confidence=0.8):
    return {"quote": quote, "dimension": dimension, "indicator": indicator,
            "polarity": polarity, "strength": strength, "confidence": confidence}


# Exercises every adapter outcome.
EVIDENCE = [
    ev("I made a shared calendar for our group the first day", "planning_style", "planner", strength="strong"),
    ev("I told them straight out I didn't agree", "conflict_style", "assertive", strength="strong"),
    ev("I missed the deadline once and didn't tell anyone", "accountability", "low", strength="strong"),
    ev("I always reply to the group chat within an hour", "accountability", "high", polarity="contradicting"),
    ev("I check in with everyone before our weekly meeting", "cooperativeness", "high"),
    ev("I check in with everyone", "cooperativeness", "high"),                                  # duplicate
    ev("I also review my teammates' pull requests the same day", "cooperativeness", "high"),
    ev("I helped my teammate debug her code at midnight", "cooperativeness", "high"),           # over cap
    ev("I made a detailed Gantt chart", "planning_style", "planner"),                           # not verbatim
    ev("Tell me about a recent group project", "leadership_style", "directive"),                # interviewer's words
    ev("I told them straight out", "conflict_style", "confrontational"),                        # old vocabulary
]

EXPECTED_DECISIONS = sorted([
    "accepted", "accepted", "accepted", "accepted", "accepted",
    "excluded:contradicting", "excluded:duplicate", "excluded:over_cap",
    "rejected:not_verbatim", "rejected:not_verbatim", "rejected:indicator_not_allowed",
])


def model_json(evidence=EVIDENCE, **extra):
    return json.dumps({
        "evidence": evidence,
        "primary_skills": ["python", "sql"],
        "skill_depth": "advanced",
        "skill_no_gos": ["frontend"],
        "unavailable_times": ["Friday evenings"],
        "timezone_notes": "PST",
        **extra,
    })


class FakeModels:
    def __init__(self, outputs):
        self.outputs = list(outputs)   # str -> response text; Exception -> raised
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return SimpleNamespace(text=out)


@pytest.fixture
def db_session_factory():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    original_session, original_init = db_module.SessionLocal, db_module.init_db
    db_module.SessionLocal, db_module.init_db = TestSession, (lambda: None)
    yield TestSession
    db_module.SessionLocal, db_module.init_db = original_session, original_init
    Base.metadata.drop_all(engine)


@pytest.fixture
def run_extract(db_session_factory, monkeypatch):
    """Call the route with the Gemini client mocked. Returns (http_response, fake_models)."""
    monkeypatch.setattr(conversation, "RETRY_DELAY_SECONDS", 0)

    def _run(*outputs, history=HISTORY, session_id=SESSION_ID):
        fake = FakeModels(outputs)
        monkeypatch.setattr(conversation, "_get_gemini_client", lambda: SimpleNamespace(models=fake))
        with TestClient(app) as c:
            resp = c.post("/conversation/extract",
                          params={"session_id": session_id, "student_name": NAME}, json=history)
        return resp, fake
    return _run


def student_text():
    return conversation.STUDENT_TURN_SEPARATOR.join(t["content"] for t in HISTORY if t["role"] == "student")


# ===========================================================================
# Labels come from score(), not the model
# ===========================================================================

def test_labels_come_from_scorer_not_model(run_extract, db_session_factory):
    # The model ALSO emits old-style labels that disagree with its own evidence.
    # They must be ignored: only the evidence is scored.
    raw = model_json(conflict_style="avoidant", accountability="high", planning_style="spontaneous",
                     overall_confidence=0.99)
    resp, _ = run_extract(raw)
    assert resp.status_code == 200, resp.text
    body = resp.json()

    expected = score(evidence_to_signals([EvidenceItem(**e) for e in EVIDENCE], student_text(), RUBRIC)[0], RUBRIC)
    assert body["signature"] == expected

    db = db_session_factory()
    s = db.get(StudentDB, body["student_id"])
    assert s.collaboration_signature["conflict_style"] == "assertive"    # model said "avoidant"
    assert s.collaboration_signature["accountability"] == "low"          # model said "high"
    assert s.work_rhythm_signature["planning_style"] == "planner"        # model said "spontaneous"
    assert s.collaboration_signature["cooperativeness"] == "high"
    # Coverage under both names, same value; model's self-confidence not written.
    assert s.confidence_layer["measurement_coverage"] == expected["confidence_score"]
    assert s.confidence_layer["confidence_score"] == expected["confidence_score"]
    assert expected["confidence_score"] != 0.99
    # Facts are stored; skill_depth is NOT copied into experience_level (TODO-9).
    assert s.competence_signature["skills"] == ["python", "sql"]
    assert s.competence_signature["skill_depth"] == "advanced"
    assert s.competence_signature["experience_level"] == "intermediate"
    assert s.work_rhythm_signature["unavailable_times"] == ["Friday evenings"]
    # safety_contribution / stress_response are written too (B3), here as unknown defaults.
    assert "safety_contribution" in s.collaboration_signature
    assert "stress_response" in s.collaboration_signature

    sig = db.query(BehavioralSignatureDB).filter_by(student_id=body["student_id"]).one()
    assert sig.signature == expected
    assert (sig.rubric_version, sig.scorer_version, sig.model_version) == (
        RUBRIC["schema_version"], "v1", conversation.EXTRACTION_MODEL)
    db.close()


def test_confidence_does_not_change_the_profile(run_extract):
    lo = [{**e, "confidence": 0.0} for e in EVIDENCE]
    hi = [{**e, "confidence": 1.0} for e in EVIDENCE]
    # Separate sessions: a second extract on the SAME session hits the existing
    # responses (assessment_id, scenario_id) unique constraint, as it did before.
    r_lo, _ = run_extract(model_json(lo), session_id="sess-lo")
    r_hi, _ = run_extract(model_json(hi), session_id="sess-hi")
    assert r_lo.status_code == r_hi.status_code == 200
    assert r_lo.json()["signature"] == r_hi.json()["signature"]


def test_prompt_and_schema_are_built_from_rubric(run_extract):
    _, fake = run_extract(model_json())
    call = fake.calls[0]
    prompt = call["contents"]
    for value in ("assertive", "developing", "cooperativeness", "safety_contribution"):
        assert value in prompt
    # The old value never appears as an indicator (the word itself does occur in
    # the rubric's prose description of stress_response).
    assert '"confrontational"' not in prompt
    assert "A specific past example with concrete detail" in prompt   # rubric strength definition
    assert "Last spring I made a shared calendar" in prompt           # transcript interpolated
    props = call["config"].response_schema.properties["evidence"].items.properties
    scorable = [n for n, d in RUBRIC["dimensions"].items() if d.get("type") != "derived"]
    assert props["dimension"].enum == scorable
    assert props["polarity"].enum == ["supporting", "contradicting"]
    assert props["strength"].enum == list(RUBRIC["evidence_strength"])
    assert props["indicator"].enum is None    # checked by the adapter instead


# ===========================================================================
# Unknown dimensions
# ===========================================================================

def test_under_determined_dims_set_when_no_evidence(run_extract, db_session_factory):
    resp, _ = run_extract(model_json())
    body = resp.json()
    with_evidence = {"planning_style", "conflict_style", "accountability", "cooperativeness"}
    scorable = {n for n, d in RUBRIC["dimensions"].items() if d.get("type") != "derived"}
    assert set(body["under_determined_dims"]) == scorable - with_evidence
    assert "leadership_style" in body["under_determined_dims"]   # its only evidence was rejected

    db = db_session_factory()
    s = db.get(StudentDB, body["student_id"])
    assert set(s.under_determined_dims) == scorable - with_evidence
    # dimension_scores rows only for measured dimensions, with the scorer's value and no LLM confidence.
    rows = db.query(DimensionScoreDB).all()
    assert {r.dimension for r in rows} == with_evidence
    by_dim = {r.dimension: r for r in rows}
    assert by_dim["conflict_style"].value == "assertive"
    assert all(r.confidence is None for r in rows)
    assert by_dim["planning_style"].evidence_span == "I made a shared calendar for our group the first day"
    db.close()


# ===========================================================================
# Every evidence item is persisted
# ===========================================================================

def test_every_evidence_item_persisted(run_extract, db_session_factory):
    resp, _ = run_extract(model_json())
    body = resp.json()
    db = db_session_factory()
    rows = db.query(EvidenceUnitDB).all()
    assert len(rows) == len(EVIDENCE)
    assert sorted(r.decision for r in rows) == EXPECTED_DECISIONS
    assert all(r.run_id == body["run_id"] and r.session_id == SESSION_ID for r in rows)
    assert sum(body["evidence_counts"].values()) == len(EVIDENCE)

    by_decision = {}
    for r in rows:
        by_decision.setdefault(r.decision, []).append(r)
    for r in rows:
        assert r.rejected == ("true" if r.decision.startswith("rejected") else "false")
        assert r.confidence == 0.8
        assert r.needs_review in ("true", "false")
    assert all(r.needs_review == "true" for r in rows if r.decision.startswith("rejected"))
    contra = by_decision["excluded:contradicting"][0]
    assert (contra.trait, contra.needs_review) == ("accountability", "true")   # floor dimension
    assert by_decision["rejected:indicator_not_allowed"][0].rubric_indicator == "confrontational"
    assert all(r.quote_start is not None for r in by_decision["accepted"])

    # Contradicting items become consistency flags on the student (decision doc §5).
    s = db.get(StudentDB, body["student_id"])
    flags = s.confidence_layer["consistency_flags"]
    assert len(flags) == 1
    assert flags[0]["dimension"] == "accountability" and flags[0]["floor_dimension"] is True
    assert "note" in flags[0]   # shape the profile page renders

    run = db.get(ExtractionRunDB, body["run_id"])
    assert run.raw_response == model_json()
    assert run.rubric_version == RUBRIC["schema_version"]
    assert run.prompt_hash and len(run.prompt_hash) == 64
    assert run.prompt_version.startswith("evidence_extraction:")
    db.close()


# ===========================================================================
# rescore_run reproduces the stored profile byte for byte
# ===========================================================================

def test_rescore_run_reproduces_signature_byte_for_byte(run_extract, db_session_factory):
    resp, _ = run_extract(model_json())
    run_id = resp.json()["run_id"]
    db = db_session_factory()
    stored_bytes = db.execute(
        text("SELECT signature_json FROM profile_versions WHERE run_id = :r"), {"r": run_id}
    ).scalar_one()
    rescored = conversation.rescore_run(run_id, db=db)
    # Same serializer SQLAlchemy's JSON type uses for SQLite (json.dumps).
    assert json.dumps(rescored) == stored_bytes

    # Independent of the order the evidence rows come back in.
    for r in db.query(EvidenceUnitDB).all():
        r.id = "z" + r.id[::-1]
    db.commit()
    assert json.dumps(conversation.rescore_run(run_id, db=db)) == stored_bytes
    db.close()


def test_rescore_run_without_db_argument_and_errors(run_extract, db_session_factory):
    resp, _ = run_extract(model_json())
    run_id = resp.json()["run_id"]
    pv = db_session_factory().query(ProfileVersionDB).filter_by(run_id=run_id).one()
    assert conversation.rescore_run(run_id) == pv.signature_json
    with pytest.raises(LookupError):
        conversation.rescore_run("no-such-run")


# ===========================================================================
# Malformed output: run row persisted with raw_response, no profile written
# ===========================================================================

@pytest.mark.parametrize("raw", [
    "this is not json {",
    json.dumps({"evidence": [{"dimension": "planning_style"}]}),          # schema fail: fields missing
    json.dumps({"evidence": [ev("I made a shared calendar", "planning_style", "planner", confidence=7)]}),
    "",
])
def test_malformed_output_persists_run_and_writes_no_profile(run_extract, db_session_factory, raw):
    resp, _ = run_extract(raw)
    assert resp.status_code == 502
    db = db_session_factory()
    runs = db.query(ExtractionRunDB).all()
    assert len(runs) == 1
    assert runs[0].raw_response == raw
    assert "error" in runs[0].params_json
    for model in (ProfileVersionDB, StudentDB, EvidenceUnitDB, DimensionScoreDB, BehavioralSignatureDB):
        assert db.query(model).count() == 0, model.__name__
    db.close()


def test_retries_on_503_then_succeeds(run_extract):
    resp, fake = run_extract(Exception("503 UNAVAILABLE"), model_json())
    assert resp.status_code == 200
    assert len(fake.calls) == 2


def test_non_retryable_api_error_is_502_and_run_persisted(run_extract, db_session_factory):
    resp, fake = run_extract(Exception("400 INVALID_ARGUMENT"))
    assert resp.status_code == 502
    assert len(fake.calls) == 1
    db = db_session_factory()
    run = db.query(ExtractionRunDB).one()
    assert run.raw_response is None and "400" in run.params_json["error"]
    assert db.query(StudentDB).count() == 0
    db.close()
