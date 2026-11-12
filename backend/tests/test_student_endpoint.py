"""
GET /students/{id} returns the fields the profile page needs:
under_determined_dims and confidence_layer.consistency_flags (both were
previously dropped), with confidence_score / measurement_coverage unchanged.

Run with:
    cd backend && python3 -m pytest tests/test_student_endpoint.py -q
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.database as db_module
import app.main as main_module
from app.database import Base, StudentDB

FLAGS = [
    {"dimension": "accountability", "note": 'Contradicting evidence against "high": "..."',
     "indicator": "high", "quote": "...", "floor_dimension": True, "needs_review": True},
]


@pytest.fixture
def client():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    # main.py imports SessionLocal by name, so patch it there as well as in app.database.
    originals = (db_module.SessionLocal, main_module.SessionLocal, db_module.init_db)
    db_module.SessionLocal = main_module.SessionLocal = TestSession
    db_module.init_db = lambda: None

    db = TestSession()
    db.add(StudentDB(
        id="s_interview", name="Interview Student",
        competence_signature={}, work_rhythm_signature={}, collaboration_signature={},
        motivation_layer={},
        confidence_layer={"measurement_coverage": 0.4, "confidence_score": 0.4, "consistency_flags": FLAGS},
        under_determined_dims=["help_seeking", "execution_style"],
    ))
    db.add(StudentDB(   # older shape: no measurement_coverage, no flags, no unknowns column value
        id="s_legacy", name="Legacy Student",
        competence_signature={}, work_rhythm_signature={}, collaboration_signature={},
        motivation_layer={}, confidence_layer={"confidence_score": 0.7},
        under_determined_dims=None,
    ))
    db.commit()
    db.close()

    with TestClient(main_module.app) as c:
        yield c

    db_module.SessionLocal, main_module.SessionLocal, db_module.init_db = originals
    Base.metadata.drop_all(engine)


def test_under_determined_dims_returned(client):
    body = client.get("/students/s_interview").json()
    assert body["under_determined_dims"] == ["help_seeking", "execution_style"]


def test_consistency_flags_returned(client):
    body = client.get("/students/s_interview").json()
    assert body["confidence_layer"]["consistency_flags"] == FLAGS


def test_confidence_values_serialize_as_before(client):
    cl = client.get("/students/s_interview").json()["confidence_layer"]
    assert cl["confidence_score"] == 0.4
    assert cl["measurement_coverage"] == 0.4


def test_legacy_profile_still_loads(client):
    body = client.get("/students/s_legacy").json()
    assert body["under_determined_dims"] == []          # NULL column -> []
    assert body["confidence_layer"] == {
        "confidence_score": 0.7, "measurement_coverage": None, "consistency_flags": [],
    }


def test_missing_student_still_404(client):
    assert client.get("/students/nope").status_code == 404
