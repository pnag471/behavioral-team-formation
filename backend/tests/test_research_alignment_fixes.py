"""
Tests for the Part 1 research-alignment fixes:

  1. The accountability/cooperativeness floor-policy contradiction
     (rubric_v1.yaml's zero-tolerance wording vs. matching.py's one-per-team
     implementation) — now a configurable, consistently-applied policy.
  2. The uneven-team-size bug (11 students / team_size=4 producing [7, 4]).
  3. The broken norms-generation wiring in the live (Gemini) explanation path.
  4. The missing google-genai dependency.

See research_alignment_audit.md, recommended_changes.md #1/#6/#8/#14, and
research_traceability_matrix.md rows 3/4/9 for the gaps these tests cover.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.database as db_module
from app.database import Base
from app.main import app
from app import matching as prod
from evaluation import fixtures as fx
from evaluation.metrics import EvalConfig, constraint_violations


# ===========================================================================
# 0. Dependency fix: google-genai importable in this environment
# ===========================================================================
def test_google_genai_import_available():
    """requirements.txt must list google-genai — ai/gemini_explainer.py
    imports it unconditionally at module load (recommended_changes.md #14)."""
    from google import genai
    from google.genai import types

    assert genai is not None
    assert types is not None
    # The live code path constructs a client and a config object from these.
    assert hasattr(genai, "Client")
    assert hasattr(types, "GenerateContentConfig")


# ===========================================================================
# 1. Floor policy — both options, applied consistently
# ===========================================================================
def _team_with_floor_members(n_floor: int, n_high: int = 2, dim="accountability"):
    """One team: n_floor members at the floor level on `dim`, n_high members
    comfortably above it."""
    floor = [
        fx._s(f"fl{i}", f"Floor {i}", ["frontend"], **{dim: "low"})
        for i in range(n_floor)
    ]
    high = [
        fx._s(f"hi{i}", f"High {i}", ["backend"], **{dim: "high"})
        for i in range(n_high)
    ]
    return [floor + high]  # a single team


def test_one_low_per_team_is_the_production_default():
    assert prod.DEFAULT_FLOOR_POLICY == prod.FLOOR_POLICY_ONE_PER_TEAM


def test_one_low_per_team_tolerates_a_single_floor_member():
    teams = _team_with_floor_members(n_floor=1)
    ok, violations, _ = prod._check_floor_violations(teams, prod.FLOOR_POLICY_ONE_PER_TEAM)
    assert ok is True
    assert violations == []


def test_one_low_per_team_flags_two_floor_members_on_the_same_team():
    teams = _team_with_floor_members(n_floor=2)
    ok, violations, _ = prod._check_floor_violations(teams, prod.FLOOR_POLICY_ONE_PER_TEAM)
    assert ok is False
    assert len(violations) == 2
    assert all("accountability=low" in v for v in violations)


def test_zero_tolerance_flags_a_single_floor_member():
    teams = _team_with_floor_members(n_floor=1)
    ok, violations, _ = prod._check_floor_violations(teams, prod.FLOOR_POLICY_ZERO_TOLERANCE)
    assert ok is False
    assert len(violations) == 1


def test_zero_tolerance_and_one_low_per_team_disagree_on_the_same_team():
    """This is the exact contradiction research_alignment_audit.md Critical #1
    documents: the same team is reported differently depending on which
    policy is in force. Both must now be reachable, not silently merged."""
    teams = _team_with_floor_members(n_floor=1)
    zero_ok, _, _ = prod._check_floor_violations(teams, prod.FLOOR_POLICY_ZERO_TOLERANCE)
    one_ok, _, _ = prod._check_floor_violations(teams, prod.FLOOR_POLICY_ONE_PER_TEAM)
    assert zero_ok is False
    assert one_ok is True


def test_feasibility_one_low_per_team_allows_up_to_n_teams_floor_students():
    students = [fx._s(f"s{i}", f"S{i}", ["frontend"], accountability="low") for i in range(2)]
    students += [fx._s(f"h{i}", f"H{i}", ["backend"]) for i in range(6)]
    feasible, infeasible = prod._check_feasibility(students, n_teams=2, policy=prod.FLOOR_POLICY_ONE_PER_TEAM)
    assert feasible is True
    assert infeasible == {}


def test_feasibility_zero_tolerance_rejects_any_floor_student():
    students = [fx._s("s0", "S0", ["frontend"], accountability="low")]
    students += [fx._s(f"h{i}", f"H{i}", ["backend"]) for i in range(7)]
    feasible, infeasible = prod._check_feasibility(students, n_teams=2, policy=prod.FLOOR_POLICY_ZERO_TOLERANCE)
    assert feasible is False
    assert infeasible.get("accountability") == 1


def test_evaluation_harness_constraint_violations_respects_floor_policy():
    """The evaluation harness ('scoring'/validation layer) must use the same
    policy semantics as production matching, not a hardcoded one."""
    teams = _team_with_floor_members(n_floor=1)
    cfg = EvalConfig()
    zero = constraint_violations(teams, cfg, floor_policy=prod.FLOOR_POLICY_ZERO_TOLERANCE)
    one = constraint_violations(teams, cfg, floor_policy=prod.FLOOR_POLICY_ONE_PER_TEAM)
    assert zero["floor_ok"] is False
    assert one["floor_ok"] is True
    assert zero["floor_policy"] == prod.FLOOR_POLICY_ZERO_TOLERANCE
    assert one["floor_policy"] == prod.FLOOR_POLICY_ONE_PER_TEAM


# ---------------------------------------------------------------------------
# Floor policy end-to-end via the API
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


def _seed_student(sid, name, cooperativeness="developing"):
    """
    Insert a StudentDB row using the currently-patched (in-memory)
    db_module.SessionLocal.

    NOT done via `POST /students`: app/main.py binds `SessionLocal` at
    module-import time and its route handlers use that stale reference
    directly (unlike matching.py/explainability.py, which re-import it
    lazily inside each request) — a pre-existing isolation gap
    (recommended_changes.md #17, out of scope for this fix set) that would
    otherwise make `POST /students` write to the real on-disk database
    instead of this test's isolated engine.
    """
    from app.database import StudentDB

    db = db_module.SessionLocal()
    try:
        db.add(StudentDB(
            id=sid,
            name=name,
            competence_signature={"skills": ["frontend"], "roles": [], "experience_level": "intermediate"},
            work_rhythm_signature={"availability": ["Mon-morning"]},
            collaboration_signature={"cooperativeness": cooperativeness},
            motivation_layer={},
            confidence_layer={},
            under_determined_dims=[],
        ))
        db.commit()
    finally:
        db.close()


def _seed_roster_with_one_floor_student(n_students=8):
    _seed_student("s0", "Floor Student", cooperativeness="low")
    for i in range(1, n_students):
        _seed_student(f"s{i}", f"Student {i}", cooperativeness="developing")


def test_generate_teams_default_policy_accepts_one_floor_student(client):
    _seed_roster_with_one_floor_student(n_students=8)
    resp = client.post("/teams/generate", json={"team_size": 4, "formation_mode": "behavioral"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["floor_policy"] == "one_low_per_team"
    assert body["feasible"] is True
    assert body["floor_violations"] == []


def test_generate_teams_zero_tolerance_rejects_the_same_roster(client):
    _seed_roster_with_one_floor_student(n_students=8)
    resp = client.post("/teams/generate", json={
        "team_size": 4, "formation_mode": "behavioral", "floor_policy": "zero_tolerance",
    })
    assert resp.status_code == 422
    assert "infeasible_dims" in resp.json()["detail"]


# ===========================================================================
# 2. Team-size balancing (11-student roster and neighbors)
# ===========================================================================
def _sizes(teams):
    return sorted(len(t) for t in teams)


def _no_student_duplicated_or_dropped(students, teams):
    placed = [s.id for t in teams for s in t]
    assert sorted(placed) == sorted(s.id for s in students)
    assert len(placed) == len(set(placed))


@pytest.mark.parametrize("n_students,team_size", [
    (3, 4),    # below team_size -> single team
    (4, 4),    # exactly team_size -> single team
    (8, 4),    # exact multiple -> even split
    (11, 4),   # the documented bug case
    (9, 4),
    (13, 4),
    (10, 3),
    (17, 5),
])
def test_greedy_build_produces_balanced_team_sizes(n_students, team_size):
    students = [fx._s(f"rs{i}", f"Roster {i}", ["frontend"]) for i in range(n_students)]
    teams = prod._greedy_build(students, team_size, prod._normalize_weights({
        "skill_coverage": 0.4, "behavioral_compat": 0.3,
        "availability_overlap": 0.2, "shared_interests": 0.1,
    }))
    sizes = _sizes(teams)
    assert max(sizes) - min(sizes) <= 1, f"n={n_students}, team_size={team_size}: sizes={sizes}"
    _no_student_duplicated_or_dropped(students, teams)


def test_eleven_students_team_size_four_no_longer_produces_seven_and_four():
    case = fx.fixture_uneven_roster_size()
    teams = prod._greedy_build(case.students, case.team_size, prod._normalize_weights({
        "skill_coverage": 0.4, "behavioral_compat": 0.3,
        "availability_overlap": 0.2, "shared_interests": 0.1,
    }))
    sizes = _sizes(teams)
    assert sizes == [3, 4, 4], f"expected a balanced 3-team split, got {sizes}"
    assert len(teams) == 3


def test_compute_n_teams_rounds_to_nearest_not_floor():
    assert prod._compute_n_teams(11, 4) == 3   # 11/4 = 2.75 -> 3, not floor(2.75)=2
    assert prod._compute_n_teams(8, 4) == 2
    assert prod._compute_n_teams(3, 4) == 1
    assert prod._compute_n_teams(0, 4) == 0


def test_target_team_sizes_never_differ_by_more_than_one():
    for n, k in [(11, 3), (13, 3), (7, 1), (20, 5), (9, 4)]:
        sizes = prod._target_team_sizes(n, k)
        assert sum(sizes) == n
        assert len(sizes) == k
        assert max(sizes) - min(sizes) <= 1


def test_generate_teams_endpoint_balances_eleven_students(client):
    for i in range(11):
        _seed_student(f"rs{i}", f"Roster {i}")
    resp = client.post("/teams/generate", json={"team_size": 4, "formation_mode": "behavioral"})
    assert resp.status_code == 200
    sizes = sorted(len(t["members"]) for t in resp.json()["teams"])
    assert sizes == [3, 4, 4]


# ===========================================================================
# 3. Norms-generation wiring in the live (Gemini) explanation path
# ===========================================================================
_FAKE_NORMS = [
    {"category": "Communication", "norm": "Async-first status updates."},
    {"category": "Conflict Resolution", "norm": "Name tensions in retros."},
    {"category": "Decision Making", "norm": "Rotating decision-owner."},
    {"category": "Planning", "norm": "Rolling two-week plan."},
    {"category": "Accountability", "norm": "Twice-weekly check-ins."},
]
_FAKE_EXPLANATION = {
    "strengths": ["Broad skill coverage across the team."],
    "risks": ["Low scheduling overlap."],
    "explanation": "This team pairs complementary skills with compatible working styles.",
}


def _fake_generate_content(*args, **kwargs):
    prompt = kwargs.get("contents", "")
    response = MagicMock()
    if "working norms" in prompt:
        response.text = json.dumps(_FAKE_NORMS)
    else:
        response.text = json.dumps(_FAKE_EXPLANATION)
    return response


def test_live_gemini_path_populates_team_norms(client, monkeypatch):
    """Integration test for the fixed wiring: GET /teams/{id}/explanation on
    the GEMINI_API_KEY-configured (live) path must return the 5 suggested
    norms, not an empty list (recommended_changes.md #8). No real API
    credentials or network calls are used — google.genai.Client is mocked."""
    import ai.gemini_explainer as ge_module
    import app.explainability as expl_module

    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")

    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = _fake_generate_content
    monkeypatch.setattr(ge_module.genai, "Client", lambda **kw: fake_client)

    live_explainer = ge_module.GeminiExplainer()
    assert live_explainer._fallback is None, "should be on the live path, not the MockExplainer fallback"
    monkeypatch.setattr(expl_module, "explainer", live_explainer)

    for i in range(4):
        _seed_student(f"ln{i}", f"Live Norms {i}")
    gen_resp = client.post("/teams/generate", json={"team_size": 4, "formation_mode": "behavioral"})
    assert gen_resp.status_code == 200
    team_id = gen_resp.json()["teams"][0]["id"]

    # Persisted team_norms is [] at formation time (norms are generated
    # on-demand) — this is the state that used to leak through unfixed.
    assert gen_resp.json()["teams"][0]["team_norms"] == []

    expl_resp = client.get(f"/teams/{team_id}/explanation")
    assert expl_resp.status_code == 200
    body = expl_resp.json()
    assert len(body["team_norms"]) == 5
    assert {n["category"] for n in body["team_norms"]} == {
        "Communication", "Conflict Resolution", "Decision Making", "Planning", "Accountability",
    }
    assert body["explanation"] == _FAKE_EXPLANATION["explanation"]
    # Both the norms call and the explanation call went through the (mocked) live client.
    assert fake_client.models.generate_content.call_count == 2
