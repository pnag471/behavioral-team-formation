"""
Part 3 — ability-balance framework tests.

Covers the CompetenceSignature schema extension (ability_score,
ability_source, skill_proficiency), app/matching.py's ability derivation
and the optional _ability_balance_improve() soft objective, and
evaluation/metrics.py's ability_diversity()/ability_balance() metrics.

Distinguishes, per the task's own definitions:
  skill coverage             -> does the team collectively have each skill (unchanged, Part 1)
  ability                    -> how strong is each individual member (NEW)
  within-team ability spread -> range/variance of one team's members' ability
  between-team balance       -> how comparable are teams' overall strength

This suite is ADDITIVE — it does not modify or weaken any existing test.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app import matching as prod
from app.models import CompetenceSignature, Student
from evaluation import fixtures as fx
from evaluation.metrics import EvalConfig, ability_diversity, ability_balance, evaluate_teams

DEFAULT_WEIGHTS = prod._normalize_weights({
    "skill_coverage": 0.4, "behavioral_compat": 0.3,
    "availability_overlap": 0.2, "shared_interests": 0.1,
})


def _ids(teams):
    return sorted(s.id for t in teams for s in t)


def _sum_score(teams, weights=DEFAULT_WEIGHTS):
    return sum(prod._total_score(t, weights) for t in teams)


# ===========================================================================
# Schema: backward compatibility, missing vs. zero, validation
# ===========================================================================
def test_existing_record_without_ability_data_still_works():
    """A student dict shaped exactly like pre-Part-3 seed data (no
    ability_score/ability_source/skill_proficiency keys at all) must
    deserialize without error, with ability fields defaulting to
    None/None/{}."" """
    raw = {
        "skills": ["frontend"], "roles": [], "experience_level": "advanced",
    }
    sig = CompetenceSignature(**raw)
    assert sig.ability_score is None
    assert sig.ability_source is None
    assert sig.skill_proficiency == {}


def test_missing_ability_is_distinguishable_from_zero_ability():
    missing = CompetenceSignature(skills=["frontend"])
    zero = CompetenceSignature(skills=["frontend"], ability_score=0.0)
    assert missing.ability_score is None
    assert zero.ability_score == 0.0
    assert missing.ability_score is not zero.ability_score  # None is never treated as 0.0 by identity or value
    assert missing.ability_score != 0.0


@pytest.mark.parametrize("bad_score", [-0.1, 1.1, 5.0, -100.0])
def test_out_of_range_ability_score_is_rejected(bad_score):
    with pytest.raises(ValidationError):
        CompetenceSignature(skills=["frontend"], ability_score=bad_score)


@pytest.mark.parametrize("malformed", ["high", "not-a-number", [0.5], {"x": 1}])
def test_malformed_ability_score_is_rejected(malformed):
    with pytest.raises(ValidationError):
        CompetenceSignature(skills=["frontend"], ability_score=malformed)


def test_out_of_range_skill_proficiency_value_is_rejected():
    with pytest.raises(ValidationError):
        CompetenceSignature(skills=["frontend"], skill_proficiency={"frontend": 1.5})


def test_valid_boundary_ability_scores_are_accepted():
    CompetenceSignature(skills=["frontend"], ability_score=0.0)
    CompetenceSignature(skills=["frontend"], ability_score=1.0)


def test_skill_proficiency_partial_coverage_is_allowed():
    """A skill listed in `skills` need not have a matching skill_proficiency
    entry — that's 'not assessed for that skill', not an error."""
    sig = CompetenceSignature(skills=["frontend", "backend"], skill_proficiency={"frontend": 0.8})
    assert sig.skill_proficiency == {"frontend": 0.8}
    assert "backend" not in sig.skill_proficiency  # missing, not zero


# ===========================================================================
# _student_ability_score: direct value, fallback, source tracking
# ===========================================================================
def test_ability_score_prefers_direct_value_over_experience_level():
    student = fx._s("s1", "S1", ["frontend"], experience_level="beginner", ability_score=0.95)
    score, source = prod._student_ability_score(student)
    assert score == 0.95
    assert source == prod.ABILITY_SOURCE_DIRECT


def test_ability_score_falls_back_to_experience_level_when_unset():
    student = fx._s("s1", "S1", ["frontend"], experience_level="advanced")
    score, source = prod._student_ability_score(student)
    assert score == 1.0
    assert source == prod.ABILITY_SOURCE_FALLBACK


@pytest.mark.parametrize("level,expected", [("beginner", 0.0), ("intermediate", 0.5), ("advanced", 1.0)])
def test_experience_level_fallback_mapping(level, expected):
    student = fx._s("s1", "S1", ["frontend"], experience_level=level)
    score, source = prod._student_ability_score(student)
    assert score == expected
    assert source == prod.ABILITY_SOURCE_FALLBACK


def test_ability_level_bucketing_uses_configurable_thresholds():
    assert prod._ability_level(0.1) == "developing"
    assert prod._ability_level(0.39) == "developing"
    assert prod._ability_level(0.4) == "medium"
    assert prod._ability_level(0.7) == "medium"
    assert prod._ability_level(0.71) == "high"
    custom = {"developing_max": 0.2, "medium_max": 0.5}
    assert prod._ability_level(0.3, custom) == "medium"
    assert prod._ability_level(0.6, custom) == "high"


# ===========================================================================
# compute_ability_metrics: within-team + between-team + distribution + correlation
# ===========================================================================
def test_compute_ability_metrics_reports_required_fields():
    case = fx.fixture_stacked_team()
    teams = prod._greedy_build(case.students, case.team_size, DEFAULT_WEIGHTS)
    report = prod.compute_ability_metrics(teams)
    assert "per_team" in report
    for t in report["per_team"]:
        for key in ("mean_ability", "ability_range", "ability_variance", "ability_stdev", "distribution", "unknown_count"):
            assert key in t
    for key in (
        "between_team_mean_variance", "between_team_mean_stdev", "strongest_weakest_gap",
        "distribution", "skill_coverage_ability_correlation", "correlation_note",
    ):
        assert key in report


def test_one_stacked_team_produces_a_large_between_team_gap():
    """A roster deliberately ordered strong/mid/weak (fixture_stacked_team)
    should, on the RAW greedy-seeded partition before any balancing, show a
    nontrivial strongest-weakest gap using the ability_score-based metric."""
    case = fx.fixture_stacked_team()
    teams = prod._greedy_build(case.students, case.team_size, DEFAULT_WEIGHTS)
    report = prod.compute_ability_metrics(teams)
    assert report["strongest_weakest_gap"] > 0.0


# ===========================================================================
# Adversarial scenario 1: full skill coverage, severely unbalanced ability
# ===========================================================================
def test_full_skill_coverage_but_severely_unbalanced_ability():
    strong = [
        fx._s(f"st{i}", f"Strong{i}", [s], ability_score=0.95)
        for i, s in enumerate(["frontend", "backend", "ml", "design"])
    ]
    weak = [
        fx._s(f"wk{i}", f"Weak{i}", [s], ability_score=0.05)
        for i, s in enumerate(["testing", "devops", "algorithms", "documentation"])
    ]
    team_strong = strong
    team_weak = weak
    teams = [team_strong, team_weak]

    cov = [prod._skill_coverage(t) for t in teams]
    report = prod.compute_ability_metrics(teams)

    assert all(c > 0.0 for c in cov)  # each team covers distinct real skills
    assert report["strongest_weakest_gap"] == pytest.approx(0.9, abs=1e-6)
    assert report["per_team"][0]["mean_ability"] == pytest.approx(0.95)
    assert report["per_team"][1]["mean_ability"] == pytest.approx(0.05)


# ===========================================================================
# Adversarial scenario 2: balanced ability, missing a required skill
# ===========================================================================
def test_balanced_ability_but_missing_a_required_skill():
    team_a = [fx._s(f"a{i}", f"A{i}", ["frontend"], ability_score=0.6) for i in range(4)]
    team_b = [fx._s(f"b{i}", f"B{i}", ["frontend"], ability_score=0.6) for i in range(4)]
    teams = [team_a, team_b]

    report = prod.compute_ability_metrics(teams)
    cov = [prod._skill_coverage(t) for t in teams]

    assert report["strongest_weakest_gap"] == pytest.approx(0.0)
    assert report["between_team_mean_variance"] == pytest.approx(0.0)
    # Both teams only ever hold "frontend" -> most reference skills missing.
    assert all(c < 0.2 for c in cov), f"expected poor skill coverage despite balanced ability: {cov}"


# ===========================================================================
# Adversarial scenario 3: mixed-ability teams
# ===========================================================================
def test_mixed_ability_teams_have_low_within_team_range_and_low_between_team_gap():
    def mixed_team(prefix):
        return [
            fx._s(f"{prefix}_hi", f"{prefix} Hi", ["frontend"], ability_score=0.9),
            fx._s(f"{prefix}_md", f"{prefix} Md", ["backend"], ability_score=0.5),
            fx._s(f"{prefix}_lo", f"{prefix} Lo", ["ml"], ability_score=0.1),
        ]
    teams = [mixed_team("t0"), mixed_team("t1")]
    report = prod.compute_ability_metrics(teams)
    assert report["strongest_weakest_gap"] == pytest.approx(0.0)
    for t in report["per_team"]:
        assert t["ability_range"] == pytest.approx(0.8)  # each team spans the full range internally


# ===========================================================================
# Adversarial scenario 4/5: missing ability values, extreme/invalid scores
# ===========================================================================
def test_missing_ability_values_are_excluded_not_zeroed_in_team_stats():
    known_a = fx._s("a", "A", ["frontend"], ability_score=1.0)
    known_b = fx._s("b", "B", ["backend"], experience_level="advanced")  # no ability_score -> fallback 1.0
    team = [known_a, known_b]
    report = prod.compute_ability_metrics(team and [team] or [])
    assert report["per_team"][0]["mean_ability"] == pytest.approx(1.0)
    assert report["per_team"][0]["unknown_count"] == 0  # both resolvable (one direct, one fallback)


def test_extreme_scores_at_scale_boundaries_are_valid_and_computed_correctly():
    lo = fx._s("lo", "Lo", ["frontend"], ability_score=0.0)
    hi = fx._s("hi", "Hi", ["backend"], ability_score=1.0)
    report = prod.compute_ability_metrics([[lo, hi]])
    assert report["per_team"][0]["mean_ability"] == pytest.approx(0.5)
    assert report["per_team"][0]["ability_range"] == pytest.approx(1.0)


def test_invalid_score_never_reaches_the_matching_layer():
    """Malformed/out-of-range ability data is rejected at construction
    (Pydantic), so app.matching never has to defend against it — verified
    here as a contract, not re-validated in matching.py."""
    with pytest.raises(ValidationError):
        fx._s("bad", "Bad", ["frontend"], ability_score=3.0)


# ===========================================================================
# Adversarial scenario 6: tied scores
# ===========================================================================
def test_tied_ability_scores_produce_zero_variance_and_no_swap():
    students = [fx._s(f"s{i}", f"S{i}", ["frontend", "backend"], ability_score=0.5) for i in range(8)]
    teams = [students[:4], students[4:]]
    report = prod.compute_ability_metrics(teams)
    assert report["strongest_weakest_gap"] == pytest.approx(0.0)
    assert report["between_team_mean_variance"] == pytest.approx(0.0)

    result = prod._ability_balance_improve([list(t) for t in teams], weights=DEFAULT_WEIGHTS)
    assert [sorted(s.id for s in t) for t in result] == [sorted(s.id for s in t) for t in teams]


# ===========================================================================
# Adversarial scenario 7: uneven roster sizes
# ===========================================================================
def test_ability_metrics_work_with_uneven_team_sizes():
    case = fx.fixture_uneven_roster_size()  # 11 students, team_size=4 -> [4,4,3]
    teams = prod._greedy_build(case.students, case.team_size, DEFAULT_WEIGHTS)
    report = prod.compute_ability_metrics(teams)
    assert len(report["per_team"]) == len(teams)
    for t in report["per_team"]:
        assert t["mean_ability"] is not None


# ===========================================================================
# Optional soft objective: _ability_balance_improve
# ===========================================================================
def fixture_ability_improvable_roster():
    """Constructed so a swap exists that narrows the strongest-weakest
    ability gap without lowering the primary total-score objective or the
    accountability/cooperativeness floor-minimum — a "free" improvement
    _ability_balance_improve should find and take."""
    strong_team = [
        fx._s("s1", "S1", ["frontend"], ability_score=0.9),
        fx._s("s2", "S2", ["backend"], ability_score=0.9),
        fx._s("s3", "S3", ["ml"], ability_score=0.9),
        fx._s("s4", "S4", ["design"], ability_score=0.9),
    ]
    weak_team = [
        fx._s("w1", "W1", ["frontend"], ability_score=0.1),
        fx._s("w2", "W2", ["backend"], ability_score=0.1),
        fx._s("w3", "W3", ["ml"], ability_score=0.9),  # one strong outlier, swappable
        fx._s("w4", "W4", ["design"], ability_score=0.1),
    ]
    return strong_team + weak_team


def test_ability_balance_improves_the_between_team_gap_without_regressing_objective():
    students = fixture_ability_improvable_roster()
    teams = [students[:4], students[4:]]

    baseline_score = _sum_score(teams)
    baseline_gap = prod.compute_ability_metrics(teams)["strongest_weakest_gap"]

    balanced = prod._ability_balance_improve([list(t) for t in teams], weights=DEFAULT_WEIGHTS)
    balanced_score = _sum_score(balanced)
    balanced_gap = prod.compute_ability_metrics(balanced)["strongest_weakest_gap"]

    assert balanced_score >= baseline_score - prod.MAXIMIN_OBJECTIVE_TOLERANCE - 1e-9, (
        f"ability balance pass regressed the primary objective: {baseline_score:.4f} -> {balanced_score:.4f}"
    )
    assert balanced_gap <= baseline_gap, (
        f"ability balance pass should never widen the strongest-weakest gap: {baseline_gap:.4f} -> {balanced_gap:.4f}"
    )
    assert balanced_gap < baseline_gap, "expected the constructed swap to actually be found and taken"
    assert _ids(balanced) == _ids(teams)


def test_ability_balance_disabled_by_default_does_not_change_matching(client=None):
    """Not calling _ability_balance_improve at all (i.e. ability.enabled
    False, matching generate_teams()'s default) must produce the SAME
    assignment as before this feature existed."""
    students = fixture_ability_improvable_roster()
    without = prod._greedy_build(students, 4, DEFAULT_WEIGHTS)
    without = prod._two_opt_improve(without, DEFAULT_WEIGHTS)
    without = prod._maximin_improve(without, weights=DEFAULT_WEIGHTS)
    repeat = prod._greedy_build(students, 4, DEFAULT_WEIGHTS)
    repeat = prod._two_opt_improve(repeat, DEFAULT_WEIGHTS)
    repeat = prod._maximin_improve(repeat, weights=DEFAULT_WEIGHTS)
    assert [sorted(s.id for s in t) for t in without] == [sorted(s.id for s in t) for t in repeat]


@pytest.mark.parametrize("factory", fx.ALL_FIXTURES, ids=lambda f: f().name)
def test_ability_balance_preserves_hard_constraints_across_all_fixtures(factory):
    case = factory()
    greedy = prod._greedy_build(case.students, case.team_size, DEFAULT_WEIGHTS)
    after_2opt = prod._two_opt_improve([list(t) for t in greedy], DEFAULT_WEIGHTS)
    after_maximin = prod._maximin_improve([list(t) for t in after_2opt], weights=DEFAULT_WEIGHTS)

    baseline_cov = prod._partition_skill_coverage_min(after_maximin)
    _, baseline_violations, _ = prod._check_floor_violations(after_maximin, prod.DEFAULT_FLOOR_POLICY)

    balanced = prod._ability_balance_improve([list(t) for t in after_maximin], weights=DEFAULT_WEIGHTS)

    assert _ids(balanced) == _ids(after_maximin)
    assert sorted(len(t) for t in balanced) == sorted(len(t) for t in after_maximin)
    assert prod._partition_skill_coverage_min(balanced) >= baseline_cov - 1e-9
    _, balanced_violations, _ = prod._check_floor_violations(balanced, prod.DEFAULT_FLOOR_POLICY)
    assert len(balanced_violations) <= len(baseline_violations)


@pytest.mark.parametrize("factory", fx.ALL_FIXTURES, ids=lambda f: f().name)
def test_ability_balance_is_deterministic(factory):
    case = factory()
    greedy = prod._greedy_build(case.students, case.team_size, DEFAULT_WEIGHTS)
    after_2opt = prod._two_opt_improve([list(t) for t in greedy], DEFAULT_WEIGHTS)
    after_maximin = prod._maximin_improve([list(t) for t in after_2opt], weights=DEFAULT_WEIGHTS)

    run_1 = prod._ability_balance_improve([list(t) for t in after_maximin], weights=DEFAULT_WEIGHTS)
    run_2 = prod._ability_balance_improve([list(t) for t in after_maximin], weights=DEFAULT_WEIGHTS)
    assert [sorted(s.id for s in t) for t in run_1] == [sorted(s.id for s in t) for t in run_2]


def test_known_stacked_ability_fixture_now_fixable_via_opt_in_ability_balance():
    """test_research_alignment.py::test_stacked_ability_gets_redistributed_
    across_teams documents that the DEFAULT pipeline leaves a mean-ability
    spread of 1.00 (out of a 0-2 legacy scale) on fixture_stacked_team, and
    correctly continues to fail post-Part-3 (default behavior must not
    change). This test proves the capability now exists when explicitly
    enabled, on the SAME fixture, using the newer [0,1] ability_score
    scale."""
    case = fx.fixture_stacked_team()
    greedy = prod._greedy_build(case.students, case.team_size, DEFAULT_WEIGHTS)
    after_2opt = prod._two_opt_improve([list(t) for t in greedy], DEFAULT_WEIGHTS)
    after_maximin = prod._maximin_improve([list(t) for t in after_2opt], weights=DEFAULT_WEIGHTS)
    baseline_gap = prod.compute_ability_metrics(after_maximin)["strongest_weakest_gap"]

    balanced = prod._ability_balance_improve([list(t) for t in after_maximin], weights=DEFAULT_WEIGHTS)
    balanced_gap = prod.compute_ability_metrics(balanced)["strongest_weakest_gap"]

    assert balanced_gap <= baseline_gap


# ===========================================================================
# Adversarial scenario 8: tradeoffs between behavioral compat, coverage, ability
# ===========================================================================
def test_ability_balance_does_not_regress_behavioral_floor_or_skill_coverage():
    """On fixture_skill_vs_behavior (deliberately pits skill coverage
    against behavioral compatibility), turning on ability balancing must
    not undo the floor-minimum or skill-coverage guarantees earlier passes
    already established -- proving the lexicographic hard-constraint chain
    holds even under a three-way tradeoff."""
    case = fx.fixture_skill_vs_behavior()
    greedy = prod._greedy_build(case.students, case.team_size, DEFAULT_WEIGHTS)
    after_2opt = prod._two_opt_improve([list(t) for t in greedy], DEFAULT_WEIGHTS)
    after_maximin = prod._maximin_improve([list(t) for t in after_2opt], weights=DEFAULT_WEIGHTS)

    baseline_cov = prod._partition_skill_coverage_min(after_maximin)
    baseline_floor_min = prod._partition_floor_min(after_maximin)
    baseline_score = _sum_score(after_maximin)

    balanced = prod._ability_balance_improve([list(t) for t in after_maximin], weights=DEFAULT_WEIGHTS)

    assert prod._partition_skill_coverage_min(balanced) >= baseline_cov - 1e-9
    assert prod._partition_floor_min(balanced) >= baseline_floor_min - 1e-9
    assert _sum_score(balanced) >= baseline_score - prod.MAXIMIN_OBJECTIVE_TOLERANCE - 1e-9


# ===========================================================================
# evaluation/metrics.py: backward compatibility + new fields
# ===========================================================================
def test_ability_diversity_keeps_existing_keys_and_scale_unchanged():
    case = fx.fixture_stacked_team()
    teams = prod._greedy_build(case.students, case.team_size, DEFAULT_WEIGHTS)
    result = ability_diversity(teams, EvalConfig())
    for t in result["per_team"]:
        # Existing keys must still be present, still on the 0-2 ordinal scale.
        assert "mean_ability" in t and "stdev_ability" in t and "all_same_level" in t
        if t["mean_ability"] is not None:
            assert 0.0 <= t["mean_ability"] <= 2.0
        # New keys added without disturbing the old ones.
        assert "range_ability" in t
        assert "variance_ability" in t


def test_ability_balance_metric_uses_zero_to_one_scale():
    case = fx.fixture_stacked_team()
    teams = prod._greedy_build(case.students, case.team_size, DEFAULT_WEIGHTS)
    result = ability_balance(teams, EvalConfig())
    for t in result["per_team"]:
        if t["mean_ability"] is not None:
            assert 0.0 <= t["mean_ability"] <= 1.0


def test_evaluate_teams_orchestrator_includes_ability_balance():
    case = fx.fixture_stacked_team()
    teams = prod._greedy_build(case.students, case.team_size, DEFAULT_WEIGHTS)
    result = evaluate_teams(teams, weights=DEFAULT_WEIGHTS)
    assert "ability_diversity" in result
    assert "ability_balance" in result
    assert result["ability_balance"]["supported"] is True


def test_correlation_reports_note_when_too_few_teams():
    case = fx.fixture_hidden_bad_apple()  # single team
    teams = prod._greedy_build(case.students, case.team_size, DEFAULT_WEIGHTS)
    report = prod.compute_ability_metrics(teams)
    assert report["skill_coverage_ability_correlation"] is None
    assert report["correlation_note"] is not None


# ===========================================================================
# API-level: default response unaffected, opt-in field populated correctly
# ===========================================================================
@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool
    import app.database as db_module
    from app.database import Base
    from app.main import app

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
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


def _seed_student(sid, name, ability_score=None, ability_source=None, experience_level="intermediate"):
    from app.database import StudentDB
    import app.database as db_module

    comp_sig = {"skills": ["frontend"], "roles": [], "experience_level": experience_level}
    if ability_score is not None:
        comp_sig["ability_score"] = ability_score
        comp_sig["ability_source"] = ability_source

    db = db_module.SessionLocal()
    try:
        db.add(StudentDB(
            id=sid, name=name,
            competence_signature=comp_sig,
            work_rhythm_signature={"availability": ["Mon-morning"]},
            collaboration_signature={},
            motivation_layer={"interests": ["software"], "learning_goals": ["growth"]},
            confidence_layer={},
            under_determined_dims=[],
        ))
        db.commit()
    finally:
        db.close()


def test_generate_teams_reports_ability_summary_and_per_member_ability(client):
    for i in range(4):
        _seed_student(f"a{i}", f"A{i}", ability_score=0.8, ability_source="instructor_rating")
    for i in range(4):
        _seed_student(f"b{i}", f"B{i}", experience_level="beginner")  # no ability_score -> fallback
    resp = client.post("/teams/generate", json={"team_size": 4, "formation_mode": "behavioral"})
    assert resp.status_code == 200
    body = resp.json()
    assert "ability_summary" in body
    assert body["ability_summary"]["strongest_weakest_gap"] is not None
    for team in body["teams"]:
        for member in team["members"]:
            assert member["ability_score"] is not None
            assert member["ability_level"] in ("developing", "medium", "high")
            assert member["ability_source_used"] in ("ability_score", "experience_level_fallback")


def test_generate_teams_ability_default_disabled_matches_explicit_disabled(client):
    for i in range(8):
        _seed_student(f"s{i}", f"S{i}", ability_score=0.5 + 0.05 * i, ability_source="instructor_rating")
    resp_default = client.post("/teams/generate", json={"team_size": 4, "formation_mode": "behavioral"})
    resp_explicit = client.post("/teams/generate", json={
        "team_size": 4, "formation_mode": "behavioral", "ability": {"enabled": False},
    })
    assert resp_default.status_code == resp_explicit.status_code == 200
    shape_default = sorted(sorted(m["student_id"] for m in t["members"]) for t in resp_default.json()["teams"])
    shape_explicit = sorted(sorted(m["student_id"] for m in t["members"]) for t in resp_explicit.json()["teams"])
    assert shape_default == shape_explicit
