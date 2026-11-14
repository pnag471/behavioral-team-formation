"""
POST /conversation/turn: the hard turn cap (MAX_INTERVIEW_HISTORY) is a safety
backstop that closes the interview BEFORE any LLM call.

Run with:
    cd backend && python3 -m pytest tests/test_conversation_turn.py -q
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.conversation as conversation
import app.database as db_module
from app.database import Base
from app.main import app


@pytest.fixture
def client(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    original_session, original_init = db_module.SessionLocal, db_module.init_db
    db_module.SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    db_module.init_db = lambda: None

    # Any Gemini use would fail the test: the cap must return before the LLM is touched.
    def _no_llm():
        raise AssertionError("the turn cap must return before any LLM call")
    monkeypatch.setattr(conversation, "_get_gemini_client", _no_llm)

    with TestClient(app) as c:
        yield c

    db_module.SessionLocal, db_module.init_db = original_session, original_init
    Base.metadata.drop_all(engine)


def test_turn_cap_closes_interview_without_llm(client):
    history = [
        {"role": "interviewer" if i % 2 == 0 else "student", "content": f"turn {i}"}
        for i in range(conversation.MAX_INTERVIEW_HISTORY)
    ]
    resp = client.post("/conversation/turn", json={
        "session_id": "sess-cap", "student_name": "Jordan Lee",
        "message": "one more answer", "history": history,
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["is_complete"] is True
    assert body["session_id"] == "sess-cap"
