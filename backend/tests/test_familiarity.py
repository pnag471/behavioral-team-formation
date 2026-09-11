"""
Part 4 — familiarity / anti-clique framework tests.

Covers the FamiliarityEdge/FamiliarityDB schema (relationship, not a
per-student property), the graph algorithms and metrics in
app/matching.py (compute_familiarity_metrics, connected components,
band-violation), the optional _familiarity_balance_improve() soft
objective, and the app/familiarity.py API surface.

This suite is ADDITIVE — it does not modify or weaken any existing test.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app import matching as prod
from app.familiarity import canonical_pair
from app.models import FamiliarityEdge, FamiliarityConfig
from evaluation import fixtures as fx
from evaluation.metrics import EvalConfig, familiarity_balance, evaluate_teams

DEFAULT_WEIGHTS = prod._normalize_weights({
    "skill_coverage": 0.4, "behavioral_compat": 0.3,
    "availability_overlap": 0.2, "shared_interests": 0.1,
})


def _ids(teams):
    return sorted(s.id for t in teams for s in t)


def _sum_score(teams, weights=DEFAULT_WEIGHTS):
    return sum(prod._total_score(t, weights) for t in teams)


def _roster(n, prefix="s"):
    return [fx._s(f"{prefix}{i}", f"S{prefix}{i}", ["frontend"]) for i in range(n)]


def _edge(a, b, strength, source=None):
    return FamiliarityEdge(student_a_id=a, student_b_id=b, strength=strength, source=source)


# ===========================================================================
# Schema: stable IDs, symmetry, scale, self-relationship, duplicates
# ===========================================================================
def test_self_relationship_is_rejected():
    with pytest.raises(ValidationError):
        FamiliarityEdge(student_a_id="s1", student_b_id="s1", strength=0.5)


@pytest.mark.parametrize("bad", [-0.1, 1.1, 5.0])
def test_out_of_range_strength_is_rejected(bad):
    with pytest.raises(ValidationError):
        FamiliarityEdge(student_a_id="s1", student_b_id="s2", strength=bad)


def test_malformed_strength_is_rejected():
    with pytest.raises(ValidationError):
        FamiliarityEdge(student_a_id="s1", student_b_id="s2", strength="close friends")


def test_boundary_strengths_are_accepted():
    FamiliarityEdge(student_a_id="s1", student_b_id="s2", strength=0.0)
    FamiliarityEdge(student_a_id="s1", student_b_id="s2", strength=1.0)


def test_canonical_pair_is_order_independent():
    assert canonical_pair("s2", "s1") == canonical_pair("s1", "s2")
    a, b = canonical_pair("s2", "s1")
    assert a < b or a == b


def test_familiarity_is_symmetric_in_the_graph():
    """A -> B and B -> A must read the same strength: the edge is stored
    once but the adjacency built from it exposes both directions."""
    roster = _roster(2)
    graph = prod.build_familiarity_graph(roster, [_edge("s0", "s1", 0.7)])
    assert graph["s0"]["s1"] == 0.7
    assert graph["s1"]["s0"] == 0.7


def test_familiarity_config_rejects_inverted_band():
    with pytest.raises(ValidationError):
        FamiliarityConfig(lower_bound=0.9, upper_bound=0.1)


# ===========================================================================
# Unknown vs. confirmed zero
# ===========================================================================
def test_missing_edge_is_unknown_not_zero():
    roster = _roster(2)
    graph = prod.build_familiarity_graph(roster, [])  # no edges recorded at all
    assert prod._known_familiarity(graph, "s0", "s1") is None


def test_confirmed_zero_edge_is_a_known_value_of_zero():
    roster = _roster(2)
    graph = prod.build_familiarity_graph(roster, [_edge("s0", "s1", 0.0)])
    val = prod._known_familiarity(graph, "s0", "s1")
    assert val == 0.0
    assert val is not None


def test_team_density_excludes_unknown_pairs_rather_than_zeroing_them():
    """3 students: one KNOWN pair at 1.0, two UNKNOWN pairs. Density must
    be 1.0 (mean of the one known pair), NOT averaged down by treating the
    unknown pairs as 0.0 (which would give 1/3)."""
    roster = _roster(3)
    graph = prod.build_familiarity_graph(roster, [_edge("s0", "s1", 1.0)])
    density, known, unknown = prod._team_familiarity_density(roster, graph)
    assert known == 1
    assert unknown == 2
    assert density == pytest.approx(1.0)


def test_zero_known_pairs_gives_none_density_not_zero():
    roster = _roster(3)
    graph = prod.build_familiarity_graph(roster, [])
    density, known, unknown = prod._team_familiarity_density(roster, graph)
    assert known == 0
    assert unknown == 3
    assert density is None


# ===========================================================================
# Adversarial scenario 1: a completely unfamiliar roster
# ===========================================================================
def test_completely_unfamiliar_roster():
    roster = _roster(8)
    teams = [roster[:4], roster[4:]]
    report = prod.compute_familiarity_metrics(teams, edges=[])
    for t in report["per_team"]:
        assert t["density"] is None
        assert t["known_pair_count"] == 0
        # zero known data with possible pairs to judge -> insufficient_data,
        # not a silent None (Part 4.5 weakness 2)
        assert t["band_status"] == "insufficient_data"
        assert t["largest_connected_cluster"]["size"] == 1
    assert report["total_known_pairs"] == 0
    assert report["teams_below_band"] == 0
    assert report["teams_within_band"] == 0
    assert report["teams_above_band"] == 0
    assert report["teams_insufficient_data"] == 2


# ===========================================================================
# Adversarial scenario 2: one isolated student
# ===========================================================================
def test_one_isolated_student():
    known = _roster(3, prefix="k")
    isolated = fx._s("iso", "Isolated", ["frontend"])
    team = known + [isolated]
    edges = [
        _edge(a.id, b.id, 0.8)
        for i, a in enumerate(known) for b in known[i + 1:]
    ]
    report = prod.compute_familiarity_metrics([team], edges=edges)
    assert "Isolated" in report["per_team"][0]["isolated_members"]
    for s in known:
        assert s.name not in report["per_team"][0]["isolated_members"]


# ===========================================================================
# Adversarial scenario 3: a strong pre-existing friend group
# ===========================================================================
def test_strong_pre_existing_friend_group_flagged_as_a_clique_and_above_band():
    clique = _roster(4, prefix="c")
    edges = [
        _edge(a.id, b.id, 0.9)
        for i, a in enumerate(clique) for b in clique[i + 1:]
    ]
    report = prod.compute_familiarity_metrics([clique], edges=edges, upper_bound=0.6)
    stats = report["per_team"][0]
    assert stats["density"] == pytest.approx(0.9)
    assert stats["band_status"] == "above"
    assert stats["largest_connected_cluster"]["size"] == 4
    assert stats["largest_connected_cluster"]["thresholded_internal_edge_density"] == pytest.approx(1.0)
    assert stats["largest_connected_cluster"]["is_quasi_clique"] is True
    assert stats["isolated_members"] == []


# ===========================================================================
# Adversarial scenario 4: multiple smaller familiarity clusters
# ===========================================================================
def test_multiple_smaller_clusters_split_across_teams():
    """Two separate pairs (AB strong, CD strong) with NO edge between the
    pairs -- if both pairs land on the same team, the largest pre-existing
    subgroup on that team should be 2 (not 4, since A/B and C/D are
    different connected components)."""
    a, b, c, d = _roster(4, prefix="x")
    edges = [_edge(a.id, b.id, 0.9), _edge(c.id, d.id, 0.9)]
    team = [a, b, c, d]
    report = prod.compute_familiarity_metrics([team], edges=edges)
    assert report["per_team"][0]["largest_connected_cluster"]["size"] == 2


# ===========================================================================
# Adversarial scenario 5: enough familiar pairs, no dominant clique
# ===========================================================================
def test_enough_familiar_pairs_but_no_dominant_clique():
    """A 'chain' of familiarity (A-B, B-C, C-D all familiar, but A-C/A-D/
    B-D unknown) still forms ONE connected component graph-theoretically
    once B-C and C-D are both >= clique_edge_threshold, but the density is
    moderate (not every pair known) and no single small tight clique
    exists distinct from the whole team."""
    a, b, c, d = _roster(4, prefix="y")
    edges = [
        _edge(a.id, b.id, 0.6),
        _edge(b.id, c.id, 0.6),
        _edge(c.id, d.id, 0.6),
    ]
    team = [a, b, c, d]
    report = prod.compute_familiarity_metrics([team], edges=edges, lower_bound=0.15, upper_bound=0.6)
    stats = report["per_team"][0]
    assert stats["known_pair_count"] == 3
    assert stats["unknown_pair_count"] == 3
    assert stats["familiar_pair_count"] == 3
    assert stats["band_status"] in ("within", "above")  # density == 0.6 == upper_bound -> "within" per _band_status
    assert stats["isolated_members"] == []  # everyone has at least one familiar neighbor


# ===========================================================================
# Adversarial scenario 6: unknown vs. zero at the ASSIGNMENT level
# ===========================================================================
def test_unknown_vs_zero_do_not_produce_the_same_metrics():
    roster = _roster(2, prefix="z")
    team_unknown = [list(roster)]
    team_zero = [list(roster)]

    report_unknown = prod.compute_familiarity_metrics(team_unknown, edges=[])
    report_zero = prod.compute_familiarity_metrics(team_zero, edges=[_edge("z0", "z1", 0.0)])

    assert report_unknown["per_team"][0]["density"] is None
    assert report_unknown["per_team"][0]["known_pair_count"] == 0
    assert report_zero["per_team"][0]["density"] == pytest.approx(0.0)
    assert report_zero["per_team"][0]["known_pair_count"] == 1
    # Both "isolated" in the sense of no familiar teammate, but for
    # DIFFERENT, distinguishable reasons (no data vs. confirmed non-familiar).
    assert report_unknown["per_team"][0]["isolated_members"] == ["Sz0", "Sz1"]
    assert report_zero["per_team"][0]["isolated_members"] == ["Sz0", "Sz1"]
    assert report_unknown["per_team"][0]["known_pair_count"] != report_zero["per_team"][0]["known_pair_count"]


# ===========================================================================
# Adversarial scenario 7: duplicate and invalid relationships
# ===========================================================================
def test_duplicate_pair_in_opposite_order_resolves_to_one_edge_in_the_graph():
    """(A,B,0.9) and (B,A,0.9) supplied as two separate FamiliarityEdge
    entries (e.g. from two different, order-inconsistent survey rows)
    must still resolve to a single, consistent graph edge -- symmetry
    means order never matters."""
    roster = _roster(2, prefix="d")
    edges = [_edge("d0", "d1", 0.9), _edge("d1", "d0", 0.9)]
    graph = prod.build_familiarity_graph(roster, edges)
    assert graph["d0"]["d1"] == 0.9
    assert graph["d1"]["d0"] == 0.9


def test_invalid_edge_construction_raises_before_reaching_the_graph():
    with pytest.raises(ValidationError):
        _edge("d0", "d0", 0.5)  # self-relationship
    with pytest.raises(ValidationError):
        _edge("d0", "d1", 1.5)  # out of range


def test_canonical_upsert_prevents_a_conflicting_second_row_for_the_same_pair(tmp_path):
    """Exercises app/familiarity.py's upsert path directly against an
    isolated in-memory DB: POSTing the same unordered pair twice with
    different strengths must update the ONE existing row, never create a
    second, conflicting one."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool
    from app.database import Base, FamiliarityDB
    import app.database as db_module
    from app.familiarity import upsert_familiarity

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    original = db_module.SessionLocal
    db_module.SessionLocal = TestSession
    try:
        upsert_familiarity(_edge("p1", "p2", 0.3))
        upsert_familiarity(_edge("p2", "p1", 0.8))  # same pair, opposite order, different strength

        db = TestSession()
        rows = db.query(FamiliarityDB).filter(
            FamiliarityDB.student_a_id.in_(["p1", "p2"]),
            FamiliarityDB.student_b_id.in_(["p1", "p2"]),
        ).all()
        db.close()
        assert len(rows) == 1
        assert rows[0].strength == pytest.approx(0.8)
    finally:
        db_module.SessionLocal = original
        Base.metadata.drop_all(engine)


# ===========================================================================
# Target band classification
# ===========================================================================
def test_band_status_below_within_above():
    assert prod._band_status(0.05, lower=0.15, upper=0.6) == "below"
    assert prod._band_status(0.3, lower=0.15, upper=0.6) == "within"
    assert prod._band_status(0.9, lower=0.15, upper=0.6) == "above"
    assert prod._band_status(None, lower=0.15, upper=0.6) is None


def test_band_violation_is_zero_within_band_and_positive_outside():
    assert prod._band_violation(0.3, 0.15, 0.6) == 0.0
    assert prod._band_violation(0.05, 0.15, 0.6) == pytest.approx(0.1)
    assert prod._band_violation(0.9, 0.15, 0.6) == pytest.approx(0.3)
    assert prod._band_violation(None, 0.15, 0.6) == 0.0


# ===========================================================================
# Optional soft objective: _familiarity_balance_improve
# ===========================================================================
def fixture_familiarity_improvable_roster():
    """A tight 4-person clique (all pairs ~0.9) plus 4 otherwise-unconnected
    students, arranged so that the clique-team's density (well above the
    default upper_bound) can be reduced by swapping one clique member for
    an outsider, without changing skill/behavioral composition (identical
    skills/behavior across all 8) -- an objective-neutral, constraint-safe
    de-clique swap should be found and taken."""
    clique = _roster(4, prefix="c")
    outsiders = _roster(4, prefix="o")
    edges = [
        _edge(a.id, b.id, 0.9)
        for i, a in enumerate(clique) for b in clique[i + 1:]
    ]
    return clique + outsiders, edges


def test_familiarity_balance_reduces_violation_without_regressing_objective():
    students, edges = fixture_familiarity_improvable_roster()
    teams = [students[:4], students[4:]]  # clique entirely on team 0

    baseline_score = _sum_score(teams)
    baseline_violation = prod._partition_familiarity_violation(
        teams, prod.build_familiarity_graph(students, edges),
        prod.DEFAULT_FAMILIARITY_LOWER_BOUND, prod.DEFAULT_FAMILIARITY_UPPER_BOUND,
    )

    balanced = prod._familiarity_balance_improve([list(t) for t in teams], edges=edges, weights=DEFAULT_WEIGHTS)
    balanced_score = _sum_score(balanced)
    balanced_violation = prod._partition_familiarity_violation(
        balanced, prod.build_familiarity_graph(students, edges),
        prod.DEFAULT_FAMILIARITY_LOWER_BOUND, prod.DEFAULT_FAMILIARITY_UPPER_BOUND,
    )

    assert balanced_score >= baseline_score - prod.FAMILIARITY_OBJECTIVE_TOLERANCE - 1e-9
    assert balanced_violation <= baseline_violation
    assert balanced_violation < baseline_violation, "expected the constructed de-clique swap to be found and taken"
    assert _ids(balanced) == _ids(teams)


def test_familiarity_balance_disabled_by_default_does_not_change_matching():
    students, _ = fixture_familiarity_improvable_roster()
    without = prod._greedy_build(students, 4, DEFAULT_WEIGHTS)
    without = prod._two_opt_improve(without, DEFAULT_WEIGHTS)
    without = prod._maximin_improve(without, weights=DEFAULT_WEIGHTS)
    repeat = prod._greedy_build(students, 4, DEFAULT_WEIGHTS)
    repeat = prod._two_opt_improve(repeat, DEFAULT_WEIGHTS)
    repeat = prod._maximin_improve(repeat, weights=DEFAULT_WEIGHTS)
    assert [sorted(s.id for s in t) for t in without] == [sorted(s.id for s in t) for t in repeat]


def test_familiarity_balance_is_a_no_op_with_zero_edges():
    """AssertionError-proofing the documented no-op claim: enabling the
    pass with no recorded familiarity data at all must not change anything
    (every density is None, every violation is 0.0)."""
    students = _roster(8)
    teams = [students[:4], students[4:]]
    result = prod._familiarity_balance_improve([list(t) for t in teams], edges=[], weights=DEFAULT_WEIGHTS)
    assert [sorted(s.id for s in t) for t in result] == [sorted(s.id for s in t) for t in teams]


@pytest.mark.parametrize("factory", fx.ALL_FIXTURES, ids=lambda f: f().name)
def test_familiarity_balance_preserves_hard_constraints_across_all_fixtures(factory):
    case = factory()
    greedy = prod._greedy_build(case.students, case.team_size, DEFAULT_WEIGHTS)
    after_2opt = prod._two_opt_improve([list(t) for t in greedy], DEFAULT_WEIGHTS)
    after_maximin = prod._maximin_improve([list(t) for t in after_2opt], weights=DEFAULT_WEIGHTS)

    baseline_cov = prod._partition_skill_coverage_min(after_maximin)
    _, baseline_violations, _ = prod._check_floor_violations(after_maximin, prod.DEFAULT_FLOOR_POLICY)

    # Synthesize a few edges among this fixture's own students so the pass
    # has something to act on (not just the zero-edges no-op case, which
    # is covered separately above).
    ids = [s.id for t in after_maximin for s in t]
    synth_edges = [_edge(ids[k], ids[k + 1], 0.8) for k in range(0, len(ids) - 1, 2)]

    balanced = prod._familiarity_balance_improve(
        [list(t) for t in after_maximin], edges=synth_edges, weights=DEFAULT_WEIGHTS,
    )

    assert _ids(balanced) == _ids(after_maximin)
    assert sorted(len(t) for t in balanced) == sorted(len(t) for t in after_maximin)
    assert prod._partition_skill_coverage_min(balanced) >= baseline_cov - 1e-9
    _, balanced_violations, _ = prod._check_floor_violations(balanced, prod.DEFAULT_FLOOR_POLICY)
    assert len(balanced_violations) <= len(baseline_violations)


@pytest.mark.parametrize("factory", fx.ALL_FIXTURES, ids=lambda f: f().name)
def test_familiarity_balance_is_deterministic(factory):
    case = factory()
    greedy = prod._greedy_build(case.students, case.team_size, DEFAULT_WEIGHTS)
    after_2opt = prod._two_opt_improve([list(t) for t in greedy], DEFAULT_WEIGHTS)
    after_maximin = prod._maximin_improve([list(t) for t in after_2opt], weights=DEFAULT_WEIGHTS)
    ids = [s.id for t in after_maximin for s in t]
    synth_edges = [_edge(ids[k], ids[k + 1], 0.8) for k in range(0, len(ids) - 1, 2)]

    run_1 = prod._familiarity_balance_improve([list(t) for t in after_maximin], edges=synth_edges, weights=DEFAULT_WEIGHTS)
    run_2 = prod._familiarity_balance_improve([list(t) for t in after_maximin], edges=synth_edges, weights=DEFAULT_WEIGHTS)
    assert [sorted(s.id for s in t) for t in run_1] == [sorted(s.id for s in t) for t in run_2]


# ===========================================================================
# Adversarial scenario 8: familiarity vs. skill/fairness/ability tradeoffs
# ===========================================================================
def test_familiarity_balance_does_not_regress_skill_or_behavioral_floor():
    case = fx.fixture_skill_vs_behavior()
    greedy = prod._greedy_build(case.students, case.team_size, DEFAULT_WEIGHTS)
    after_2opt = prod._two_opt_improve([list(t) for t in greedy], DEFAULT_WEIGHTS)
    after_maximin = prod._maximin_improve([list(t) for t in after_2opt], weights=DEFAULT_WEIGHTS)

    ids = [s.id for t in after_maximin for s in t]
    synth_edges = [_edge(ids[k], ids[k + 1], 0.9) for k in range(0, len(ids) - 1, 2)]

    baseline_cov = prod._partition_skill_coverage_min(after_maximin)
    baseline_floor_min = prod._partition_floor_min(after_maximin)
    baseline_score = _sum_score(after_maximin)

    balanced = prod._familiarity_balance_improve(
        [list(t) for t in after_maximin], edges=synth_edges, weights=DEFAULT_WEIGHTS,
    )

    assert prod._partition_skill_coverage_min(balanced) >= baseline_cov - 1e-9
    assert prod._partition_floor_min(balanced) >= baseline_floor_min - 1e-9
    assert _sum_score(balanced) >= baseline_score - prod.FAMILIARITY_OBJECTIVE_TOLERANCE - 1e-9


def test_full_pipeline_all_four_optional_passes_together_preserve_hard_constraints():
    """ability.enabled + fairness.enabled + familiarity.enabled all at once
    -- documented as NOT guaranteed to jointly optimize every soft
    objective, but hard constraints (membership, sizes, skill-coverage
    floor, behavioral floor) must still hold regardless."""
    case = fx.fixture_utility_vs_fairness()
    greedy = prod._greedy_build(case.students, case.team_size, DEFAULT_WEIGHTS)
    after_2opt = prod._two_opt_improve([list(t) for t in greedy], DEFAULT_WEIGHTS)
    after_maximin = prod._maximin_improve([list(t) for t in after_2opt], weights=DEFAULT_WEIGHTS)
    baseline_cov = prod._partition_skill_coverage_min(after_maximin)

    ids = [s.id for t in after_maximin for s in t]
    synth_edges = [_edge(ids[k], ids[k + 1], 0.7) for k in range(0, len(ids) - 1, 2)]

    after_ability = prod._ability_balance_improve([list(t) for t in after_maximin], weights=DEFAULT_WEIGHTS)
    after_fairness = prod._fairness_improve([list(t) for t in after_ability], weights=DEFAULT_WEIGHTS)
    after_familiarity = prod._familiarity_balance_improve(
        [list(t) for t in after_fairness], edges=synth_edges, weights=DEFAULT_WEIGHTS,
    )

    assert _ids(after_familiarity) == _ids(after_maximin)
    assert sorted(len(t) for t in after_familiarity) == sorted(len(t) for t in after_maximin)
    assert prod._partition_skill_coverage_min(after_familiarity) >= baseline_cov - 1e-9


# ===========================================================================
# evaluation/metrics.py: familiarity_balance() + backward compatibility
# ===========================================================================
def test_familiarity_balance_metric_delegates_to_production_function():
    case = fx.fixture_friend_clique()
    teams = prod._greedy_build(case.students, case.team_size, DEFAULT_WEIGHTS)
    result = familiarity_balance(teams, EvalConfig(), edges=[])
    assert result["supported"] is True
    assert "per_team" in result


def test_legacy_familiarity_density_and_clique_concentration_unchanged():
    """Existing side-channel-data functions must keep working exactly as
    before Part 4 -- they are a DIFFERENT (label/dict-based) mechanism,
    kept for backward compatibility with existing adversarial fixtures."""
    case = fx.fixture_isolated_student()
    teams = prod._greedy_build(case.students, case.team_size, DEFAULT_WEIGHTS)
    result = evaluate_teams(
        teams, weights=DEFAULT_WEIGHTS,
        familiarity=case.familiarity, friend_groups=case.friend_groups,
    )
    assert result["familiarity_density"]["supported"] is True
    assert result["clique_concentration"]["supported"] is False  # fixture supplies no friend_groups
    assert "familiarity_balance" in result  # new Part 4 metric present alongside the legacy ones


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


def _seed_student(sid, name):
    from app.database import StudentDB
    import app.database as db_module

    db = db_module.SessionLocal()
    try:
        db.add(StudentDB(
            id=sid, name=name,
            competence_signature={"skills": ["frontend"], "roles": [], "experience_level": "intermediate"},
            work_rhythm_signature={"availability": ["Mon-morning"]},
            collaboration_signature={},
            motivation_layer={"interests": ["software"], "learning_goals": ["growth"]},
            confidence_layer={},
            under_determined_dims=[],
        ))
        db.commit()
    finally:
        db.close()


def test_post_familiarity_then_get_roundtrips(client):
    resp = client.post("/familiarity", json={"student_a_id": "s1", "student_b_id": "s2", "strength": 0.7, "source": "roster_survey"})
    assert resp.status_code == 200
    listed = client.get("/familiarity").json()
    assert len(listed) == 1
    assert listed[0]["strength"] == pytest.approx(0.7)


def test_post_familiarity_rejects_self_relationship(client):
    resp = client.post("/familiarity", json={"student_a_id": "s1", "student_b_id": "s1", "strength": 0.5})
    assert resp.status_code == 422


def test_post_familiarity_rejects_out_of_range_strength(client):
    resp = client.post("/familiarity", json={"student_a_id": "s1", "student_b_id": "s2", "strength": 2.0})
    assert resp.status_code == 422


def test_generate_teams_reports_familiarity_summary(client):
    # total_known_pairs is a WITHIN-team aggregate, so a single edge is only
    # counted if its two students land on the same team -- team placement
    # for 8 identical students isn't something this test should predict.
    # Recording every pairwise edge sidesteps that: however greedy splits
    # them, every possible pair on every team is known.
    for i in range(8):
        _seed_student(f"s{i}", f"S{i}")
    for i in range(8):
        for j in range(i + 1, 8):
            client.post("/familiarity", json={"student_a_id": f"s{i}", "student_b_id": f"s{j}", "strength": 0.5})
    resp = client.post("/teams/generate", json={"team_size": 4, "formation_mode": "behavioral"})
    assert resp.status_code == 200
    body = resp.json()
    assert "familiarity_summary" in body
    # 2 teams of 4 -> C(4,2)*2 = 12 within-team pairs, all known.
    assert body["familiarity_summary"]["total_known_pairs"] == 12
    assert body["familiarity_summary"]["mean_density"] == pytest.approx(0.5)


def test_generate_teams_familiarity_default_disabled_matches_explicit_disabled(client):
    for i in range(8):
        _seed_student(f"s{i}", f"S{i}")
    for i in range(0, 4):
        client.post("/familiarity", json={"student_a_id": f"s{i}", "student_b_id": f"s{i+4}", "strength": 0.9})
    resp_default = client.post("/teams/generate", json={"team_size": 4, "formation_mode": "behavioral"})
    resp_explicit = client.post("/teams/generate", json={
        "team_size": 4, "formation_mode": "behavioral", "familiarity": {"enabled": False},
    })
    assert resp_default.status_code == resp_explicit.status_code == 200
    shape_default = sorted(sorted(m["student_id"] for m in t["members"]) for t in resp_default.json()["teams"])
    shape_explicit = sorted(sorted(m["student_id"] for m in t["members"]) for t in resp_explicit.json()["teams"])
    assert shape_default == shape_explicit


# ===========================================================================
# Part 4.5 — weakness 1: team-induced connected_familiarity_cluster
# (replaces the Part 4 global-component-intersection computation, which
# could count two teammates as "clustered" via a path through a student on
# a DIFFERENT team; see app/matching.py's familiarity-section module
# docstring for the full explanation).
# ===========================================================================
def test_team_induced_cluster_ignores_a_path_through_an_outside_student():
    """THE Part 4 bug, reproduced directly: A-C and C-B are both strong,
    A-B has NO edge at all, and C sits on a DIFFERENT team. The global
    (whole-roster) connected component containing A and B also contains C,
    so a component-intersected-with-team-membership approach would
    wrongly report A and B as clustered. The team-induced subgraph cannot
    make that mistake -- C, and any edge through C, is structurally
    excluded from team 0's own subgraph."""
    a, b, c = _roster(3, prefix="g")
    outsiders = _roster(2, prefix="h")  # pad team 1 to a valid size
    edges = [_edge(a.id, c.id, 0.9), _edge(c.id, b.id, 0.9)]  # NO a-b edge
    team0 = [a, b]
    team1 = [c] + outsiders
    report = prod.compute_familiarity_metrics([team0, team1], edges=edges)
    cluster0 = report["per_team"][0]["largest_connected_cluster"]
    assert cluster0["size"] == 1, (
        "A and B share no team-internal edge -- they must NOT be reported as "
        "clustered just because both happen to be strongly connected to C, "
        "who is on a different team entirely."
    )
    # Confirm the underlying graph function used for optimization violation
    # agrees: team 0's clique-concentration violation is 0 (no cluster > 1).
    assert prod._team_clique_violation(team0, prod.build_familiarity_graph([a, b, c] + outsiders, edges),
                                        prod.DEFAULT_CLIQUE_EDGE_THRESHOLD, prod.DEFAULT_MAX_CLIQUE_FRACTION) == 0.0


def test_chain_forms_one_cluster_but_is_not_reported_as_a_clique():
    """A-B, B-C, C-D all strong; A-C, A-D, B-D UNKNOWN (never a confirmed
    0.0 -- just no data). Graph-theoretically this is ONE connected
    component of size 4 (a path exists between any two members through
    the chain), but it is very much NOT a clique (only 3 of 6 possible
    pairs are directly connected) -- the cohesion fields must make that
    distinction visible rather than this module ever asserting it IS a
    clique."""
    a, b, c, d = _roster(4, prefix="ch")
    edges = [_edge(a.id, b.id, 0.9), _edge(b.id, c.id, 0.9), _edge(c.id, d.id, 0.9)]
    team = [a, b, c, d]
    cluster = prod._largest_team_cluster(team, prod.build_familiarity_graph(team, edges),
                                          prod.DEFAULT_CLIQUE_EDGE_THRESHOLD,
                                          quasi_clique_density_threshold=0.75)
    assert cluster["size"] == 4
    assert cluster["total_internal_pairs"] == 6
    assert cluster["thresholded_internal_edge_density"] == pytest.approx(3 / 6)
    assert cluster["is_quasi_clique"] is False
    # Unknown edges (A-C, A-D, B-D) must not be treated as zero: the mean
    # known internal strength is the mean of the 3 KNOWN edges only.
    assert cluster["known_internal_pairs"] == 3
    assert cluster["unknown_internal_pairs"] == 3
    assert cluster["known_internal_edge_coverage"] == pytest.approx(0.5)
    assert cluster["mean_known_internal_strength"] == pytest.approx(0.9)


def test_star_forms_one_cluster_but_is_not_reported_as_a_clique():
    """One hub strongly connected to 3 spokes; spoke-spoke pairs are all
    UNKNOWN. Same size-4 cluster as the chain/clique cases, but an even
    sparser internal structure (only 3 of 6 pairs known, all through the
    hub) -- again must not be conflated with a clique."""
    hub, s1, s2, s3 = _roster(4, prefix="st")
    edges = [_edge(hub.id, s1.id, 0.9), _edge(hub.id, s2.id, 0.9), _edge(hub.id, s3.id, 0.9)]
    team = [hub, s1, s2, s3]
    cluster = prod._largest_team_cluster(team, prod.build_familiarity_graph(team, edges),
                                          prod.DEFAULT_CLIQUE_EDGE_THRESHOLD,
                                          quasi_clique_density_threshold=0.75)
    assert cluster["size"] == 4
    assert cluster["thresholded_internal_edge_density"] == pytest.approx(3 / 6)
    assert cluster["is_quasi_clique"] is False


def test_complete_subgraph_is_recognized_as_highly_cohesive():
    """Contrast case for the chain/star tests above: EVERY pair strong and
    known (a genuine clique). Same cluster SIZE (4) as the chain/star
    cases, but full cohesion -- this is what actually earns
    is_quasi_clique=True."""
    a, b, c, d = _roster(4, prefix="cq")
    edges = [_edge(x.id, y.id, 0.9) for i, x in enumerate([a, b, c, d]) for y in [a, b, c, d][i + 1:]]
    team = [a, b, c, d]
    cluster = prod._largest_team_cluster(team, prod.build_familiarity_graph(team, edges),
                                          prod.DEFAULT_CLIQUE_EDGE_THRESHOLD,
                                          quasi_clique_density_threshold=0.75)
    assert cluster["size"] == 4
    assert cluster["known_internal_edge_coverage"] == pytest.approx(1.0)
    assert cluster["thresholded_internal_edge_density"] == pytest.approx(1.0)
    assert cluster["mean_known_internal_strength"] == pytest.approx(0.9)
    assert cluster["is_quasi_clique"] is True


def test_reversing_edge_input_order_produces_identical_metrics():
    """Determinism: the SAME set of edges, supplied in reverse order,
    must produce a byte-for-byte identical compute_familiarity_metrics()
    report -- including largest_connected_cluster, which depends on BFS
    traversal order over the team-induced subgraph."""
    a, b, c, d, e = _roster(5, prefix="rv")
    team = [a, b, c, d, e]
    edges = [
        _edge(a.id, b.id, 0.9), _edge(b.id, c.id, 0.9), _edge(c.id, d.id, 0.9),
        _edge(a.id, d.id, 0.6), _edge(d.id, e.id, 0.4),
    ]
    forward = prod.compute_familiarity_metrics([team], edges=edges)
    backward = prod.compute_familiarity_metrics([team], edges=list(reversed(edges)))
    assert forward == backward


# ===========================================================================
# Part 4.5 — weakness 2: known-pair coverage and insufficient-data handling
# ===========================================================================
def test_coverage_complete_familiarity_data():
    a, b, c, d = _roster(4, prefix="cd")
    edges = [_edge(x.id, y.id, 0.4) for i, x in enumerate([a, b, c, d]) for y in [a, b, c, d][i + 1:]]
    report = prod.compute_familiarity_metrics([[a, b, c, d]], edges=edges)
    stats = report["per_team"][0]
    assert stats["total_possible_pairs"] == 6
    assert stats["known_pair_count"] == 6
    assert stats["known_pair_coverage"] == pytest.approx(1.0)
    assert stats["band_status"] in ("below", "within", "above")


def test_coverage_one_known_pair_among_many_unknown():
    roster = _roster(5, prefix="ok")  # 10 possible pairs
    edges = [_edge(roster[0].id, roster[1].id, 0.9)]
    report = prod.compute_familiarity_metrics([roster], edges=edges)
    stats = report["per_team"][0]
    assert stats["total_possible_pairs"] == 10
    assert stats["known_pair_count"] == 1
    assert stats["known_pair_coverage"] == pytest.approx(0.1)
    assert stats["band_status"] == "insufficient_data"


def test_coverage_all_confirmed_zero_relationships():
    """Full coverage, but every known value is a CONFIRMED 0.0 -- distinct
    from 'no data' (insufficient_data must NOT fire; coverage is 1.0)."""
    a, b, c = _roster(3, prefix="az")
    edges = [_edge(a.id, b.id, 0.0), _edge(b.id, c.id, 0.0), _edge(a.id, c.id, 0.0)]
    report = prod.compute_familiarity_metrics([[a, b, c]], edges=edges, lower_bound=0.15)
    stats = report["per_team"][0]
    assert stats["known_pair_coverage"] == pytest.approx(1.0)
    assert stats["confirmed_zero_pair_count"] == 3
    assert stats["density"] == pytest.approx(0.0)
    assert stats["band_status"] == "below"


def test_coverage_all_unknown_relationships():
    roster = _roster(4, prefix="au")
    report = prod.compute_familiarity_metrics([roster], edges=[])
    stats = report["per_team"][0]
    assert stats["total_possible_pairs"] == 6
    assert stats["known_pair_count"] == 0
    assert stats["known_pair_coverage"] == pytest.approx(0.0)
    assert stats["density"] is None
    assert stats["band_status"] == "insufficient_data"


def test_coverage_mixed_unknown_and_zero_relationships():
    a, b, c, d = _roster(4, prefix="mz")
    edges = [
        _edge(a.id, b.id, 0.0),   # confirmed zero
        _edge(c.id, d.id, 0.0),   # confirmed zero
        _edge(a.id, c.id, 0.8),   # known, nonzero
        _edge(b.id, d.id, 0.6),   # known, nonzero
        # a-d and b-c: left unknown
    ]
    report = prod.compute_familiarity_metrics([[a, b, c, d]], edges=edges)
    stats = report["per_team"][0]
    assert stats["total_possible_pairs"] == 6
    assert stats["known_pair_count"] == 4
    assert stats["unknown_pair_count"] == 2
    assert stats["confirmed_zero_pair_count"] == 2
    assert stats["known_pair_coverage"] == pytest.approx(4 / 6)
    assert stats["density"] == pytest.approx((0.0 + 0.0 + 0.8 + 0.6) / 4)


def test_coverage_exactly_at_threshold_is_sufficient():
    """3 of 6 possible pairs known -> coverage == 0.5 == default
    minimum_known_pair_coverage. '>=' semantics: exactly-at-threshold
    counts as SUFFICIENT, not insufficient."""
    a, b, c, d = _roster(4, prefix="ex")
    edges = [_edge(a.id, b.id, 0.9), _edge(b.id, c.id, 0.9), _edge(c.id, d.id, 0.9)]
    report = prod.compute_familiarity_metrics([[a, b, c, d]], edges=edges, minimum_known_pair_coverage=0.5)
    stats = report["per_team"][0]
    assert stats["known_pair_coverage"] == pytest.approx(0.5)
    assert stats["band_status"] != "insufficient_data"


def test_coverage_just_below_threshold_is_insufficient():
    """2 of 6 possible pairs known -> coverage ~= 0.333, below the default
    0.5 minimum -- must be classified insufficient_data even though a
    density value technically exists."""
    a, b, c, d = _roster(4, prefix="be")
    edges = [_edge(a.id, b.id, 0.9), _edge(c.id, d.id, 0.9)]
    report = prod.compute_familiarity_metrics([[a, b, c, d]], edges=edges, minimum_known_pair_coverage=0.5)
    stats = report["per_team"][0]
    assert stats["known_pair_coverage"] == pytest.approx(2 / 6)
    assert stats["density"] is not None
    assert stats["band_status"] == "insufficient_data"


def test_coverage_single_member_team_is_not_applicable():
    lone = _roster(1, prefix="lo")
    report = prod.compute_familiarity_metrics([lone], edges=[])
    stats = report["per_team"][0]
    assert stats["total_possible_pairs"] == 0
    assert stats["known_pair_count"] == 0
    assert stats["known_pair_coverage"] is None
    assert stats["density"] is None
    assert stats["band_status"] == "not_applicable"
    assert stats["isolated_members"] == []
    assert stats["largest_connected_cluster"]["size"] == 1
    assert report["teams_not_applicable"] == 1
    assert report["teams_insufficient_data"] == 0


# ===========================================================================
# Part 4.5 — item 3: optimizer behavior verification
# ===========================================================================
def test_familiarity_balance_does_not_penalize_insufficient_data_team_by_default():
    """A team with density technically 'above' the band, but coverage far
    below minimum_known_pair_coverage, must NOT be treated as a violation
    by default (penalize_insufficient_data=False) -- the optimizer should
    not confidently chase a target it lacks the evidence to judge. With
    penalize_insufficient_data=True (an explicit opt-in), the same team's
    violation is no longer suppressed."""
    a, b, c, d, e, f = _roster(6, prefix="ins")
    # Only ONE known pair on a 4-person team (a-b), far below 50% coverage,
    # but that one pair is intensely strong -- density would read "above".
    edges = [_edge(a.id, b.id, 0.95)]
    team = [a, b, c, d]
    graph = prod.build_familiarity_graph([a, b, c, d, e, f], edges)

    default_violation = prod._team_familiarity_violation(
        team, graph, prod.DEFAULT_FAMILIARITY_LOWER_BOUND, prod.DEFAULT_FAMILIARITY_UPPER_BOUND,
        prod.DEFAULT_CLIQUE_EDGE_THRESHOLD, prod.DEFAULT_MAX_CLIQUE_FRACTION,
        minimum_known_pair_coverage=0.5, penalize_insufficient_data=False,
    )
    assert default_violation == 0.0

    opted_in_violation = prod._team_familiarity_violation(
        team, graph, prod.DEFAULT_FAMILIARITY_LOWER_BOUND, prod.DEFAULT_FAMILIARITY_UPPER_BOUND,
        prod.DEFAULT_CLIQUE_EDGE_THRESHOLD, prod.DEFAULT_MAX_CLIQUE_FRACTION,
        minimum_known_pair_coverage=0.5, penalize_insufficient_data=True,
    )
    assert opted_in_violation > 0.0


def test_familiarity_balance_fixes_clique_without_chasing_insufficient_data_team():
    """End-to-end: team 0 is a genuine, fixable clique; team 1 has almost
    no recorded data (coverage far below the minimum) but its one known
    pair reads as intensely strong. The optimizer must still find and
    take the clique-reducing swap (team 0's violation strictly decreases)
    while team 1's OWN violation contribution stays 0.0 throughout --
    never 'confidently' chased on the strength of one unrepresentative
    known pair."""
    clique = _roster(4, prefix="cl2")
    clique_edges = [
        _edge(x.id, y.id, 0.9)
        for i, x in enumerate(clique) for y in clique[i + 1:]
    ]
    a, b, c, d = _roster(4, prefix="ins2")
    sparse_edges = [_edge(a.id, b.id, 0.95)]  # 1/6 possible pairs known

    teams = [clique, [a, b, c, d]]
    edges = clique_edges + sparse_edges
    graph = prod.build_familiarity_graph([s for t in teams for s in t], edges)

    baseline_clique_violation = prod._team_familiarity_violation(
        teams[0], graph, prod.DEFAULT_FAMILIARITY_LOWER_BOUND, prod.DEFAULT_FAMILIARITY_UPPER_BOUND,
        prod.DEFAULT_CLIQUE_EDGE_THRESHOLD, prod.DEFAULT_MAX_CLIQUE_FRACTION,
        minimum_known_pair_coverage=0.5, penalize_insufficient_data=False,
    )

    balanced = prod._familiarity_balance_improve(
        [list(t) for t in teams], edges=edges, weights=DEFAULT_WEIGHTS,
        minimum_known_pair_coverage=0.5, penalize_insufficient_data=False,
    )
    balanced_graph = prod.build_familiarity_graph([s for t in balanced for s in t], edges)
    balanced_violations = [
        prod._team_familiarity_violation(
            t, balanced_graph, prod.DEFAULT_FAMILIARITY_LOWER_BOUND, prod.DEFAULT_FAMILIARITY_UPPER_BOUND,
            prod.DEFAULT_CLIQUE_EDGE_THRESHOLD, prod.DEFAULT_MAX_CLIQUE_FRACTION,
            minimum_known_pair_coverage=0.5, penalize_insufficient_data=False,
        )
        for t in balanced
    ]
    assert min(balanced_violations) < baseline_clique_violation, "expected the clique-reducing swap to be found and taken"


def test_familiarity_balance_treats_all_unknown_team_as_zero_violation_not_low_density():
    """A team with ZERO known pairs must contribute exactly 0.0 violation
    (nothing to report), never the (lower_bound - 0.0) violation that
    would result from treating an unknown pair as a confirmed 0.0."""
    roster = _roster(4, prefix="zv")
    graph = prod.build_familiarity_graph(roster, [])
    violation = prod._team_familiarity_violation(
        roster, graph, prod.DEFAULT_FAMILIARITY_LOWER_BOUND, prod.DEFAULT_FAMILIARITY_UPPER_BOUND,
        prod.DEFAULT_CLIQUE_EDGE_THRESHOLD, prod.DEFAULT_MAX_CLIQUE_FRACTION,
        minimum_known_pair_coverage=0.5, penalize_insufficient_data=False,
    )
    assert violation == 0.0


def test_band_status_and_violation_accept_coverage_kwargs_and_stay_backward_compatible():
    """The revised _band_status/_band_violation signatures gained
    keyword-only coverage parameters -- confirm old-style calls (no
    coverage info supplied, as every pre-Part-4.5 call site uses) are
    completely unaffected."""
    assert prod._band_status(0.05, lower=0.15, upper=0.6) == "below"
    assert prod._band_status(0.3, lower=0.15, upper=0.6) == "within"
    assert prod._band_status(0.9, lower=0.15, upper=0.6) == "above"
    assert prod._band_status(None, lower=0.15, upper=0.6) is None

    assert prod._band_violation(0.3, 0.15, 0.6) == 0.0
    assert prod._band_violation(0.05, 0.15, 0.6) == pytest.approx(0.1)
    assert prod._band_violation(0.9, 0.15, 0.6) == pytest.approx(0.3)
    assert prod._band_violation(None, 0.15, 0.6) == 0.0

    # New behavior only activates when coverage info is explicitly passed.
    assert prod._band_status(
        0.9, lower=0.15, upper=0.6,
        known_pair_coverage=0.1, minimum_known_pair_coverage=0.5, total_possible_pairs=6,
    ) == "insufficient_data"
    assert prod._band_violation(
        0.9, 0.15, 0.6, known_pair_coverage=0.1, minimum_known_pair_coverage=0.5,
    ) == 0.0
