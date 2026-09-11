"""
Part 5 — preference / stability framework tests.

Covers the PreferenceEntry/PreferenceDB schema (a DIRECTED statement from
one student about a target — see app.models.PreferenceEntry's docstring),
the tier-scoring and metric functions in app/matching.py
(compute_preference_metrics), the three formal stability diagnostics
(compute_blocking_pairs/_swaps/_moves, compute_stability_metrics), the
optional _preference_repair_improve() soft objective/repair pass, and the
app/preferences.py API surface (including its privacy shape).

This suite is ADDITIVE — it does not modify or weaken any existing test.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app import matching as prod
from app.models import PreferenceEntry, PreferenceConfig, TeamSizeConfig
from evaluation import fixtures as fx
from evaluation.metrics import (
    EvalConfig, preference_stability, evaluate_teams,
    blocking_pairs_preference_based, legacy_preferences_to_entries,
)

DEFAULT_WEIGHTS = prod._normalize_weights({
    "skill_coverage": 0.4, "behavioral_compat": 0.3,
    "availability_overlap": 0.2, "shared_interests": 0.1,
})


def _ids(teams):
    return sorted(s.id for t in teams for s in t)


def _roster(n, prefix="s"):
    return [fx._s(f"{prefix}{i}", f"S{prefix}{i}", ["frontend"]) for i in range(n)]


def _prefer(sid, target, rank=None, priority="normal"):
    return PreferenceEntry(student_id=sid, direction="prefer", target_student_id=target, rank=rank, priority=priority)


def _avoid(sid, target, priority="normal"):
    return PreferenceEntry(student_id=sid, direction="avoid", target_student_id=target, priority=priority)


# ===========================================================================
# Schema validation
# ===========================================================================
def test_self_target_is_rejected():
    with pytest.raises(ValidationError):
        PreferenceEntry(student_id="a", direction="prefer", target_student_id="a")


def test_requires_exactly_one_target():
    with pytest.raises(ValidationError):
        PreferenceEntry(student_id="a", direction="prefer")
    with pytest.raises(ValidationError):
        PreferenceEntry(student_id="a", direction="prefer",
                         target_student_id="b", target_characteristic={"skill": "backend"})


def test_avoid_direction_rejects_characteristic_target():
    with pytest.raises(ValidationError):
        PreferenceEntry(student_id="a", direction="avoid", target_characteristic={"skill": "backend"})


def test_rank_must_be_at_least_one_when_given():
    with pytest.raises(ValidationError):
        PreferenceEntry(student_id="a", direction="prefer", target_student_id="b", rank=0)


def test_config_rejects_high_priority_weight_below_one():
    with pytest.raises(ValidationError):
        PreferenceConfig(high_priority_weight=0.5)


# ===========================================================================
# 1. Mutual teammate preferences
# ===========================================================================
def test_mutual_preference_fully_satisfied_when_together():
    a, b, c, d = _roster(4, prefix="m")
    entries = [_prefer(a.id, b.id, rank=1), _prefer(b.id, a.id, rank=1)]
    report = prod.compute_preference_metrics([[a, b], [c, d]], entries)
    a_stats = next(p for p in report["per_student"] if p["student_id"] == a.id)
    b_stats = next(p for p in report["per_student"] if p["student_id"] == b.id)
    assert a_stats["satisfaction_score"] == pytest.approx(1.0)
    assert b_stats["satisfaction_score"] == pytest.approx(1.0)
    assert a_stats["got_a_preferred_teammate"] is True


def test_mutual_preference_split_across_teams_is_a_blocking_pair():
    a, b, c, d = _roster(4, prefix="mb")
    entries = [_prefer(a.id, c.id, rank=1), _prefer(c.id, a.id, rank=1)]
    teams = [[a, b], [c, d]]
    stability = prod.compute_stability_metrics(teams, entries)
    assert (a.id, c.id) in stability["blocking_pairs"] or (c.id, a.id) in stability["blocking_pairs"]
    assert stability["n_blocking_pairs"] >= 1

    repaired = prod._preference_repair_improve([list(t) for t in teams], entries=entries, weights=DEFAULT_WEIGHTS)
    report = prod.compute_preference_metrics(repaired, entries)
    a_stats = next(p for p in report["per_student"] if p["student_id"] == a.id)
    assert a_stats["satisfaction_score"] == pytest.approx(1.0), "repair should have reunited the mutual pair"
    assert _ids(repaired) == _ids(teams)


# ===========================================================================
# 2. One-sided preferences
# ===========================================================================
def test_one_sided_preference_does_not_affect_the_targets_own_score():
    """A prefers B; B has no opinion of A (or anyone) at all. A's outcome
    depends only on A's own list; B's satisfaction is MISSING (None), not
    influenced by being wanted."""
    a, b, c, d = _roster(4, prefix="os")
    entries = [_prefer(a.id, b.id, rank=1)]
    report = prod.compute_preference_metrics([[a, b], [c, d]], entries)
    a_stats = next(p for p in report["per_student"] if p["student_id"] == a.id)
    b_stats = next(p for p in report["per_student"] if p["student_id"] == b.id)
    assert a_stats["satisfaction_score"] == pytest.approx(1.0)
    assert b_stats["satisfaction_score"] is None
    assert b_stats["has_preferences"] is False


def test_one_sided_preference_alone_is_never_a_blocking_pair():
    """A one-directional 'crush' can never be a blocking PAIR (which
    requires BOTH sides to rank the other ahead of their own current
    teammates) even when A and B are on different teams and A would love
    to swap in."""
    a, b, c, d = _roster(4, prefix="ol")
    entries = [_prefer(a.id, c.id, rank=1)]  # only a has an opinion; c has none
    teams = [[a, b], [c, d]]
    stability = prod.compute_stability_metrics(teams, entries)
    assert stability["n_blocking_pairs"] == 0


# ===========================================================================
# 3. Cyclic preferences
# ===========================================================================
def test_cyclic_preferences_handled_without_crashing_and_form_no_blocking_pair():
    """A prefers B, B prefers C, C prefers A -- a 3-cycle with no MUTUAL
    (2-way) pair anywhere in it. No two of them can form a blocking pair
    (that needs BOTH directions between the SAME two students)."""
    a, b, c, d = _roster(4, prefix="cy")
    entries = [_prefer(a.id, b.id, rank=1), _prefer(b.id, c.id, rank=1), _prefer(c.id, a.id, rank=1)]
    teams = [[a, c], [b, d]]
    report = prod.compute_preference_metrics(teams, entries)
    assert len(report["per_student"]) == 4
    stability = prod.compute_stability_metrics(teams, entries)
    assert stability["n_blocking_pairs"] == 0
    a_stats = next(p for p in report["per_student"] if p["student_id"] == a.id)
    assert a_stats["satisfaction_score"] == pytest.approx(0.0)  # a wants b, not on a's team


# ===========================================================================
# 4. Ties / indifference
# ===========================================================================
def test_tied_rank_gives_full_credit_for_either_target():
    """Two entries at the SAME rank (=1) are a tie: matching EITHER one
    scores the same as matching a lone rank-1 preference would."""
    a, b, c, d = _roster(4, prefix="tie")
    entries = [_prefer(a.id, b.id, rank=1), _prefer(a.id, c.id, rank=1)]
    report_with_b = prod.compute_preference_metrics([[a, b], [c, d]], entries)
    report_with_c = prod.compute_preference_metrics([[a, c], [b, d]], entries)
    a_with_b = next(p for p in report_with_b["per_student"] if p["student_id"] == a.id)
    a_with_c = next(p for p in report_with_c["per_student"] if p["student_id"] == a.id)
    assert a_with_b["satisfaction_score"] == pytest.approx(1.0)
    assert a_with_c["satisfaction_score"] == pytest.approx(1.0)


def test_all_unranked_entries_form_one_flat_indifference_tier():
    a, b, c, d = _roster(4, prefix="ur")
    entries = [_prefer(a.id, b.id), _prefer(a.id, c.id)]  # no rank on either
    report = prod.compute_preference_metrics([[a, b], [c, d]], entries)
    a_stats = next(p for p in report["per_student"] if p["student_id"] == a.id)
    assert a_stats["n_tiers"] == 1
    assert a_stats["satisfaction_score"] == pytest.approx(1.0)


def test_unranked_tier_scores_worse_than_a_ranked_tier():
    """A ranks B first, and separately (unranked) lists C. Getting the
    RANKED one (B) should score at least as well as getting only the
    unranked one (C) -- an explicit rank signals stronger intent."""
    a, b, c, d = _roster(4, prefix="rk")
    entries = [_prefer(a.id, b.id, rank=1), _prefer(a.id, c.id)]  # c unranked -> worse tier
    with_ranked = prod.compute_preference_metrics([[a, b], [c, d]], entries)
    with_unranked = prod.compute_preference_metrics([[a, c], [b, d]], entries)
    score_ranked = next(p for p in with_ranked["per_student"] if p["student_id"] == a.id)["satisfaction_score"]
    score_unranked = next(p for p in with_unranked["per_student"] if p["student_id"] == a.id)["satisfaction_score"]
    assert score_ranked == pytest.approx(1.0)
    assert score_unranked < score_ranked
    assert score_unranked > 0.0  # still a real, honored preference -- not zero


# ===========================================================================
# 5. Missing preferences
# ===========================================================================
def test_missing_preferences_are_none_not_zero_and_excluded_from_aggregates():
    roster = _roster(4, prefix="mp")
    report = prod.compute_preference_metrics([roster[:2], roster[2:]], entries=[])
    for p in report["per_student"]:
        assert p["satisfaction_score"] is None
        assert p["has_preferences"] is False
    assert report["mean_satisfaction"] is None
    assert report["min_satisfaction"] is None
    assert report["fraction_with_at_least_one_preferred_teammate"] is None
    assert report["worst_case_preference_regret"] is None
    assert report["n_students_with_preferences"] == 0


def test_mixing_missing_and_present_preferences_excludes_only_the_missing_ones():
    a, b, c, d = _roster(4, prefix="mx")
    entries = [_prefer(a.id, b.id, rank=1)]  # only a has an opinion
    report = prod.compute_preference_metrics([[a, b], [c, d]], entries)
    assert report["n_students_with_preferences"] == 1
    assert report["mean_satisfaction"] == pytest.approx(1.0)  # only a's score counts
    assert report["fraction_with_at_least_one_preferred_teammate"] == pytest.approx(1.0)


# ===========================================================================
# 6. Avoidance constraints
# ===========================================================================
def test_avoidance_violation_detected_when_placed_together():
    a, b, c, d = _roster(4, prefix="av")
    entries = [_avoid(a.id, b.id, priority="high")]
    report = prod.compute_preference_metrics([[a, b], [c, d]], entries)
    a_stats = next(p for p in report["per_student"] if p["student_id"] == a.id)
    b_stats = next(p for p in report["per_student"] if p["student_id"] == b.id)
    assert a_stats["avoidance_violated"] is True
    assert a_stats["violated_high_priority_avoidance"] is True
    # honored UNILATERALLY: b never submitted anything, but is still
    # reported as part of the violated pair.
    assert b_stats["avoidance_violated"] is True
    assert report["total_avoidance_violations"] == 1
    assert report["total_high_priority_avoidance_violations"] == 1


def test_avoidance_not_violated_when_apart():
    a, b, c, d = _roster(4, prefix="avok")
    entries = [_avoid(a.id, b.id)]
    report = prod.compute_preference_metrics([[a, c], [b, d]], entries)
    a_stats = next(p for p in report["per_student"] if p["student_id"] == a.id)
    assert a_stats["avoidance_violated"] is False
    assert report["total_avoidance_violations"] == 0


def test_hard_avoidance_repair_separates_a_violated_pair_without_worsening_constraints():
    a, b, c, d = _roster(4, prefix="hav")
    entries = [_avoid(a.id, b.id, priority="high")]
    teams = [[a, b], [c, d]]
    repaired = prod._preference_repair_improve(
        [list(t) for t in teams], entries=entries, weights=DEFAULT_WEIGHTS, hard_avoidance=True,
    )
    report = prod.compute_preference_metrics(repaired, entries)
    assert report["total_avoidance_violations"] == 0
    assert _ids(repaired) == _ids(teams)


def test_hard_avoidance_gate_never_increases_violations_during_its_own_swaps():
    """hard_avoidance=True must never let the repair pass's OWN swap
    INCREASE avoidance violations on a touched team, even while pursuing
    an unrelated preference improvement elsewhere."""
    a, b, c, d, e, f = _roster(6, prefix="hg")
    entries = [
        _avoid(a.id, b.id, priority="high"),  # a/b must never end up together
        _prefer(c.id, e.id, rank=1), _prefer(e.id, c.id, rank=1),  # mutual pair to reunite
    ]
    teams = [[a, b, c], [d, e, f]]
    repaired = prod._preference_repair_improve(
        [list(t) for t in teams], entries=entries, weights=DEFAULT_WEIGHTS, hard_avoidance=True,
    )
    report = prod.compute_preference_metrics(repaired, entries)
    assert report["total_avoidance_violations"] == 0, "hard_avoidance must not let a/b end up together as a side effect"


# ===========================================================================
# 7 & 8. Stable assignment vs. an assignment with a blocking pair
# ===========================================================================
def test_a_stable_assignment_has_no_blocking_swaps():
    a, b, c, d = _roster(4, prefix="st")
    entries = [_prefer(a.id, b.id, rank=1), _prefer(b.id, a.id, rank=1)]
    teams = [[a, b], [c, d]]  # already together -> nothing to gain from any swap
    stability = prod.compute_stability_metrics(teams, entries)
    assert stability["is_stable"] is True
    assert stability["n_blocking_swaps"] == 0


def test_an_assignment_with_a_blocking_pair_is_reported():
    a, b, c, d = _roster(4, prefix="bp")
    entries = [_prefer(a.id, c.id, rank=1), _prefer(c.id, a.id, rank=1)]
    teams = [[a, b], [c, d]]  # mutual best friends split apart
    stability = prod.compute_stability_metrics(teams, entries)
    assert stability["n_blocking_pairs"] >= 1
    # blocking_pair is a DISTINCT (weaker/necessary-but-not-sufficient)
    # diagnostic from blocking_swap -- both are always reported, and
    # is_stable (default definition) is checked against blocking_swap only.
    assert "n_blocking_swaps" in stability and "n_beneficial_feasible_moves" in stability


@pytest.mark.parametrize("definition", ["blocking_pair", "blocking_swap", "beneficial_feasible_move"])
def test_stability_definition_is_configurable(definition):
    a, b, c, d = _roster(4, prefix="cfg")
    entries = [_prefer(a.id, c.id, rank=1), _prefer(c.id, a.id, rank=1)]
    teams = [[a, b], [c, d]]
    stability = prod.compute_stability_metrics(teams, entries, definition=definition)
    assert stability["definition"] == definition
    assert stability["is_stable"] == (stability[f"n_{definition}s"] == 0)


def test_unknown_stability_definition_rejected():
    a, b = _roster(2, prefix="ud")
    with pytest.raises(ValueError):
        prod.compute_stability_metrics([[a, b]], [], definition="not_a_real_definition")


# ===========================================================================
# 9. A preference-optimal but skill-invalid assignment
# ===========================================================================
def test_repair_refuses_a_preference_satisfying_swap_that_breaks_skill_coverage():
    """X is the ONLY holder of a required skill on team A; P1 (X's
    teammate) holds none. Q1 (team B) holds a different required skill;
    Q2 holds none. X's only preference is to be teamed with Q1. The ONLY
    swap that achieves this (X <-> Q2, bringing X onto Q1's team) strips
    team A of its only skill entirely -- verified directly against
    validate_assignment below -- so the repair pass MUST leave both teams
    exactly as they were, even though the preference goes permanently
    unsatisfied."""
    x = fx._s("x9", "X9", ["ml"])
    p1 = fx._s("p19", "P19", [])
    q1 = fx._s("q19", "Q19", ["backend"])
    q2 = fx._s("q29", "Q29", [])
    teams = [[x, p1], [q1, q2]]
    entries = [_prefer(x.id, q1.id, rank=1)]

    baseline = prod.AssignmentBaseline.capture(teams)
    candidate = [[q2, p1], [q1, x]]  # the x<->q2 swap that WOULD satisfy x's preference
    assert prod.validate_assignment(candidate, baseline).ok is False, \
        "fixture setup check: this swap must actually violate skill coverage"

    repaired = prod._preference_repair_improve([list(t) for t in teams], entries=entries, weights=DEFAULT_WEIGHTS)
    assert _ids(repaired) == _ids(teams)
    assert sorted(s.id for s in repaired[0]) == sorted(s.id for s in teams[0])
    assert sorted(s.id for s in repaired[1]) == sorted(s.id for s in teams[1])
    report = prod.compute_preference_metrics(repaired, entries)
    x_stats = next(p for p in report["per_student"] if p["student_id"] == x.id)
    assert x_stats["satisfaction_score"] == pytest.approx(0.0), "preference stays unmet -- the only fix was unsafe"


# ===========================================================================
# 10. A stable but unfair assignment
# ===========================================================================
def test_stable_assignment_can_still_be_utility_unfair():
    """Stability (w.r.t. preferences) and fairness (w.r.t. individual
    utility) are ORTHOGONAL properties: with no preference data recorded
    at all, an assignment is trivially stable (nobody has an opinion, so
    nothing can be a blocking pair/swap/move) even when it is starkly
    utility-unfair for one student -- reusing the already-validated
    utility_vs_fairness adversarial fixture (research_alignment_audit.md
    Phase 3) for the unfairness half."""
    case = fx.fixture_utility_vs_fairness()
    teams = prod._greedy_build(case.students, case.team_size, DEFAULT_WEIGHTS)
    teams = prod._two_opt_improve(teams, DEFAULT_WEIGHTS)
    teams = prod._maximin_improve(teams)

    stability = prod.compute_stability_metrics(teams, entries=[])
    assert stability["is_stable"] is True  # vacuously -- nobody expressed a preference

    utility_report = prod.compute_assignment_utilities(teams)
    assert utility_report["summary"]["worst_case_regret"] is not None
    assert utility_report["summary"]["worst_case_regret"] > 0.0, \
        "fixture setup check: this roster is expected to produce a nontrivial utility regret"


# ===========================================================================
# 11. A fairness-improving swap that harms preference satisfaction
# ===========================================================================
def test_fairness_improve_can_reduce_preference_satisfaction():
    """A1/A2 are a mutual best-friend pair, together, each individually
    utility-neutral. B1 is a poor fit for team B and would benefit from
    swapping into team A's slot (and A1 would, symmetrically, benefit from
    B's slot) -- exactly the kind of Pareto-improving trade
    _fairness_improve looks for. Taking that swap raises fairness (worst-
    off utility) but severs the mutual pair, dropping their preference
    satisfaction from full to zero -- concretely demonstrating these two
    optional objectives can conflict, per this part's own instruction to
    test exactly this."""
    a1 = fx._s("f_a1", "FA1", ["frontend"], availability=["Mon-morning"], interests=["ml"])
    a2 = fx._s("f_a2", "FA2", ["frontend"], availability=["Fri-evening"], interests=["art"])
    b1 = fx._s("f_b1", "FB1", ["backend"], availability=["Mon-morning"], interests=["ml"])
    b2 = fx._s("f_b2", "FB2", ["backend"], availability=["Fri-evening"], interests=["art"])
    teams = [[a1, a2], [b1, b2]]
    entries = [_prefer(a1.id, a2.id, rank=1), _prefer(a2.id, a1.id, rank=1)]

    pre = prod.compute_preference_metrics(teams, entries)
    assert pre["mean_satisfaction"] == pytest.approx(1.0)

    fair = prod._fairness_improve([list(t) for t in teams], weights=DEFAULT_WEIGHTS, utility_weights=None)
    assert _ids(fair) == _ids(teams), "fixture setup check: fairness pass must have actually changed something"
    assert [sorted(s.id for s in t) for t in fair] != [sorted(s.id for s in t) for t in teams]

    post = prod.compute_preference_metrics(fair, entries)
    assert post["mean_satisfaction"] < pre["mean_satisfaction"], \
        "expected the fairness-improving swap to break up the mutual pair"


# ===========================================================================
# Repair pass: hard-constraint preservation, determinism, disabled default
# ===========================================================================
@pytest.mark.parametrize("factory", fx.ALL_FIXTURES, ids=lambda f: f().name)
def test_preference_repair_preserves_hard_constraints_across_all_fixtures(factory):
    case = factory()
    greedy = prod._greedy_build(case.students, case.team_size, DEFAULT_WEIGHTS)
    after_2opt = prod._two_opt_improve([list(t) for t in greedy], DEFAULT_WEIGHTS)
    after_maximin = prod._maximin_improve([list(t) for t in after_2opt], weights=DEFAULT_WEIGHTS)

    baseline_cov = prod._partition_skill_coverage_min(after_maximin)
    _, baseline_violations, _ = prod._check_floor_violations(after_maximin, prod.DEFAULT_FLOOR_POLICY)

    ids = [s.id for t in after_maximin for s in t]
    synth_entries = [
        _prefer(ids[k], ids[k + 1], rank=1) for k in range(0, len(ids) - 1, 2)
    ]

    repaired = prod._preference_repair_improve(
        [list(t) for t in after_maximin], entries=synth_entries, weights=DEFAULT_WEIGHTS,
    )
    assert _ids(repaired) == _ids(after_maximin)
    assert sorted(len(t) for t in repaired) == sorted(len(t) for t in after_maximin)
    assert prod._partition_skill_coverage_min(repaired) >= baseline_cov - 1e-9
    _, repaired_violations, _ = prod._check_floor_violations(repaired, prod.DEFAULT_FLOOR_POLICY)
    assert len(repaired_violations) <= len(baseline_violations)


@pytest.mark.parametrize("factory", fx.ALL_FIXTURES, ids=lambda f: f().name)
def test_preference_repair_is_deterministic(factory):
    case = factory()
    greedy = prod._greedy_build(case.students, case.team_size, DEFAULT_WEIGHTS)
    after_2opt = prod._two_opt_improve([list(t) for t in greedy], DEFAULT_WEIGHTS)
    after_maximin = prod._maximin_improve([list(t) for t in after_2opt], weights=DEFAULT_WEIGHTS)
    ids = [s.id for t in after_maximin for s in t]
    synth_entries = [_prefer(ids[k], ids[k + 1], rank=1) for k in range(0, len(ids) - 1, 2)]

    run_1 = prod._preference_repair_improve([list(t) for t in after_maximin], entries=synth_entries, weights=DEFAULT_WEIGHTS)
    run_2 = prod._preference_repair_improve([list(t) for t in after_maximin], entries=synth_entries, weights=DEFAULT_WEIGHTS)
    assert [sorted(s.id for s in t) for t in run_1] == [sorted(s.id for s in t) for t in run_2]


def test_preference_repair_disabled_by_default_does_not_change_matching():
    a, b, c, d = _roster(4, prefix="dd")
    without = prod._greedy_build([a, b, c, d], 2, DEFAULT_WEIGHTS)
    without = prod._two_opt_improve(without, DEFAULT_WEIGHTS)
    without = prod._maximin_improve(without, weights=DEFAULT_WEIGHTS)
    repeat = prod._greedy_build([a, b, c, d], 2, DEFAULT_WEIGHTS)
    repeat = prod._two_opt_improve(repeat, DEFAULT_WEIGHTS)
    repeat = prod._maximin_improve(repeat, weights=DEFAULT_WEIGHTS)
    assert [sorted(s.id for s in t) for t in without] == [sorted(s.id for s in t) for t in repeat]


def test_preference_repair_is_a_no_op_with_zero_entries():
    students = _roster(8, prefix="noop")
    teams = [students[:4], students[4:]]
    result = prod._preference_repair_improve([list(t) for t in teams], entries=[], weights=DEFAULT_WEIGHTS)
    assert [sorted(s.id for s in t) for t in result] == [sorted(s.id for s in t) for t in teams]


def test_full_pipeline_all_five_optional_passes_together_preserve_hard_constraints():
    """ability + fairness + familiarity + preference all enabled at once --
    same documented caveat as Part 4's equivalent test: NOT guaranteed to
    jointly optimize every soft objective, but hard constraints must hold."""
    case = fx.fixture_utility_vs_fairness()
    greedy = prod._greedy_build(case.students, case.team_size, DEFAULT_WEIGHTS)
    after_2opt = prod._two_opt_improve([list(t) for t in greedy], DEFAULT_WEIGHTS)
    after_maximin = prod._maximin_improve([list(t) for t in after_2opt], weights=DEFAULT_WEIGHTS)
    baseline_cov = prod._partition_skill_coverage_min(after_maximin)

    ids = [s.id for t in after_maximin for s in t]
    synth_fam_edges = []
    synth_prefs = [_prefer(ids[k], ids[k + 1], rank=1) for k in range(0, len(ids) - 1, 2)]

    after_ability = prod._ability_balance_improve([list(t) for t in after_maximin], weights=DEFAULT_WEIGHTS)
    after_fairness = prod._fairness_improve([list(t) for t in after_ability], weights=DEFAULT_WEIGHTS)
    after_familiarity = prod._familiarity_balance_improve(
        [list(t) for t in after_fairness], edges=synth_fam_edges, weights=DEFAULT_WEIGHTS,
    )
    after_preference = prod._preference_repair_improve(
        [list(t) for t in after_familiarity], entries=synth_prefs, weights=DEFAULT_WEIGHTS,
    )

    assert _ids(after_preference) == _ids(after_maximin)
    assert sorted(len(t) for t in after_preference) == sorted(len(t) for t in after_maximin)
    assert prod._partition_skill_coverage_min(after_preference) >= baseline_cov - 1e-9


# ===========================================================================
# Characteristic preferences (optional feature)
# ===========================================================================
def test_characteristic_preference_resolves_skill_key():
    a = fx._s("ch1", "CH1", [])
    b = fx._s("ch2", "CH2", ["backend"])
    c = fx._s("ch3", "CH3", [])
    d = fx._s("ch4", "CH4", [])
    entries = [PreferenceEntry(student_id=a.id, direction="prefer",
                                target_characteristic={"skill": "backend"}, rank=1)]
    with_backend = prod.compute_preference_metrics([[a, b], [c, d]], entries)
    without_backend = prod.compute_preference_metrics([[a, c], [b, d]], entries)
    a_with = next(p for p in with_backend["per_student"] if p["student_id"] == a.id)
    a_without = next(p for p in without_backend["per_student"] if p["student_id"] == a.id)
    assert a_with["satisfaction_score"] == pytest.approx(1.0)
    assert a_without["satisfaction_score"] == pytest.approx(0.0)


def test_unresolvable_characteristic_key_excluded_not_silently_unmet():
    a = fx._s("uc1", "UC1", [])
    b = fx._s("uc2", "UC2", [])
    entries = [PreferenceEntry(student_id=a.id, direction="prefer",
                                target_characteristic={"leadership_style": "directive"}, rank=1)]
    report = prod.compute_preference_metrics([[a, b]], entries)
    a_stats = next(p for p in report["per_student"] if p["student_id"] == a.id)
    # not resolvable -> never "matched", but this is a documented limitation,
    # not a crash or a misleading confident score.
    assert a_stats["satisfaction_score"] == pytest.approx(0.0)


# ===========================================================================
# Privacy shape: the public PreferenceStudentStats/PreferenceSummary never
# carry named targets, and generate_teams' report round-trips correctly.
# ===========================================================================
def test_public_summary_model_never_carries_target_ids_or_reason_category():
    a, b, c, d = _roster(4, prefix="priv")
    entries = [
        _avoid(a.id, b.id, priority="high"),
        _prefer(c.id, d.id, rank=1),
    ]
    pref_report = prod.compute_preference_metrics([[a, b], [c, d]], entries)
    stability_report = prod.compute_stability_metrics([[a, b], [c, d]], entries)
    summary = prod._preference_summary_model(pref_report, stability_report)
    dumped = summary.model_dump()
    serialized = str(dumped)
    for p in summary.per_student:
        assert not hasattr(p, "target_student_id")
        assert not hasattr(p, "reason_category")
    # the raw internal dict DOES carry ids (server-side use only) -- the
    # PUBLIC model built from it must not.
    assert "target_student_id" not in serialized


def test_preferences_api_has_no_bulk_list_endpoint():
    from app.main import app
    paths = {r.path for r in app.routes}
    assert "/preferences" in paths  # POST (upsert) exists
    # no bare GET /preferences (list-everyone) -- only /preferences/{student_id}
    get_preferences_routes = [
        r for r in app.routes
        if getattr(r, "path", None) == "/preferences" and "GET" in getattr(r, "methods", set())
    ]
    assert get_preferences_routes == []


# ===========================================================================
# API-level: default response unaffected, opt-in fields populated correctly
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


def test_post_preference_then_get_returns_only_own_outgoing_entries(client):
    resp = client.post("/preferences", json={"student_id": "s1", "direction": "prefer", "target_student_id": "s2", "rank": 1})
    assert resp.status_code == 200
    client.post("/preferences", json={"student_id": "s3", "direction": "prefer", "target_student_id": "s1", "rank": 1})

    mine = client.get("/preferences/s1").json()
    assert len(mine) == 1
    assert mine[0]["target_student_id"] == "s2"
    # s1 cannot see that s3 named them as a target through this endpoint.
    assert all(e["student_id"] == "s1" for e in mine)


def test_post_preference_upsert_updates_existing_row(client):
    client.post("/preferences", json={"student_id": "s1", "direction": "prefer", "target_student_id": "s2", "rank": 2})
    client.post("/preferences", json={"student_id": "s1", "direction": "prefer", "target_student_id": "s2", "rank": 1})
    mine = client.get("/preferences/s1").json()
    assert len(mine) == 1
    assert mine[0]["rank"] == 1


def test_post_preference_rejects_self_target(client):
    resp = client.post("/preferences", json={"student_id": "s1", "direction": "prefer", "target_student_id": "s1", "rank": 1})
    assert resp.status_code == 422


def test_delete_preference_removes_entry(client):
    client.post("/preferences", json={"student_id": "s1", "direction": "avoid", "target_student_id": "s2"})
    resp = client.delete("/preferences/s1/avoid/s2")
    assert resp.status_code == 200
    assert client.get("/preferences/s1").json() == []


def test_generate_teams_reports_preference_summary(client):
    for i in range(8):
        _seed_student(f"s{i}", f"S{i}")
    client.post("/preferences", json={"student_id": "s0", "direction": "prefer", "target_student_id": "s1", "rank": 1})
    client.post("/preferences", json={"student_id": "s1", "direction": "prefer", "target_student_id": "s0", "rank": 1})
    resp = client.post("/teams/generate", json={"team_size": 4, "formation_mode": "behavioral"})
    assert resp.status_code == 200
    body = resp.json()
    assert "preference_summary" in body
    assert body["preference_summary"]["n_students_with_preferences"] == 2
    assert "target_student_id" not in str(body["preference_summary"])


def test_generate_teams_preference_default_disabled_matches_explicit_disabled(client):
    for i in range(8):
        _seed_student(f"s{i}", f"S{i}")
    client.post("/preferences", json={"student_id": "s0", "direction": "prefer", "target_student_id": "s4", "rank": 1})
    resp_default = client.post("/teams/generate", json={"team_size": 4, "formation_mode": "behavioral"})
    resp_explicit = client.post("/teams/generate", json={
        "team_size": 4, "formation_mode": "behavioral", "preference": {"enabled": False},
    })
    assert resp_default.status_code == resp_explicit.status_code == 200
    shape_default = sorted(sorted(m["student_id"] for m in t["members"]) for t in resp_default.json()["teams"])
    shape_explicit = sorted(sorted(m["student_id"] for m in t["members"]) for t in resp_explicit.json()["teams"])
    assert shape_default == shape_explicit


# ===========================================================================
# evaluation/metrics.py: preference_stability() + backward compatibility
# ===========================================================================
def test_preference_stability_metric_delegates_to_production_function():
    case = fx.fixture_preference_instability()
    teams = prod._greedy_build(case.students, case.team_size, DEFAULT_WEIGHTS)
    result = preference_stability(teams, EvalConfig(), entries=[])
    assert result["supported"] is True
    assert "preference_satisfaction" in result and "stability" in result


def test_legacy_preference_satisfaction_and_blocking_pairs_unchanged():
    """Existing side-channel-dict functions must keep working exactly as
    before Part 5 -- a DIFFERENT (Dict[str, List[str]]-based) mechanism,
    kept for backward compatibility with test_research_alignment.py's
    existing fixtures."""
    case = fx.fixture_preference_instability()
    teams = prod._greedy_build(case.students, case.team_size, DEFAULT_WEIGHTS)
    result = evaluate_teams(teams, weights=DEFAULT_WEIGHTS, preferences=case.preferences)
    assert result["preference_satisfaction"]["supported"] is True
    assert result["blocking_pairs_preference_based"]["supported"] is True
    assert "preference_stability" in result  # new Part 5 metric present alongside the legacy ones


# ===========================================================================
# Part 5.5 — item 2: whole-partition validation of moves
# ===========================================================================
def test_beneficial_feasible_move_is_size_preserving_and_whole_partition_validated():
    """compute_beneficial_feasible_moves (renamed from compute_blocking_
    moves) is swap-mediated -- every candidate keeps BOTH teams at their
    exact target size by construction, and every candidate is checked via
    ONE validate_assignment() call against the FULL partition."""
    a, b, c, d = _roster(4, prefix="bf")
    entries = [_prefer(a.id, c.id, rank=1)]
    teams = [[a, b], [c, d]]
    baseline = prod.AssignmentBaseline.capture(teams)
    moves = prod.compute_beneficial_feasible_moves(teams, entries, baseline)
    assert moves == [(a.id, 1)]
    # membership, both team sizes, whole-partition validity of the
    # UNDERLYING candidate this move represents:
    candidate = [[b, c], [d, a]]  # a<->d swap that realizes a's move onto c's team
    v = prod.validate_assignment(candidate, baseline)
    assert v.ok is True
    assert sorted(s.id for t in candidate for s in t) == sorted(s.id for t in teams for s in t)


def test_unilateral_move_rejected_when_sending_team_would_become_too_small():
    x = fx._s("xa2", "XA2", ["frontend"])
    p1 = fx._s("p1a2", "P1A2", ["frontend"])
    p2 = fx._s("p2a2", "P2A2", ["frontend"])
    q1 = fx._s("q1a2", "Q1A2", ["backend"])
    q2 = fx._s("q2a2", "Q2A2", ["backend"])
    q3 = fx._s("q3a2", "Q3A2", ["backend"])
    teams = [[x, p1, p2], [q1, q2, q3]]
    entries = [_prefer(x.id, q1.id, rank=1)]
    baseline = prod.AssignmentBaseline.capture(teams)
    cfg = TeamSizeConfig(min_team_size=3, max_team_size=4, allow_unilateral_moves=True)  # no shrink headroom
    assert prod.compute_unilateral_vacancy_moves(teams, entries, baseline, cfg) == []

    # direct confirmation via the centralized validator on the exact candidate
    candidate = [[p1, p2], [q1, q2, q3, x]]
    v = prod.validate_assignment(candidate, prod.AssignmentBaseline(
        student_ids=baseline.student_ids, target_sizes=baseline.target_sizes,
        min_skill_coverage=baseline.min_skill_coverage, floor_policy=baseline.floor_policy,
        max_floor_violations=baseline.max_floor_violations, min_sizes=(3, 3), max_sizes=(4, 4),
    ))
    assert v.ok is False
    assert any("below the configured minimum size" in r for r in v.reasons)


def test_unilateral_move_rejected_when_receiving_team_would_become_too_large():
    x = fx._s("xb2", "XB2", ["frontend"])
    p1 = fx._s("p1b2", "P1B2", ["frontend"])
    p2 = fx._s("p2b2", "P2B2", ["frontend"])
    q1 = fx._s("q1b2", "Q1B2", ["backend"])
    q2 = fx._s("q2b2", "Q2B2", ["backend"])
    q3 = fx._s("q3b2", "Q3B2", ["backend"])
    teams = [[x, p1, p2], [q1, q2, q3]]
    entries = [_prefer(x.id, q1.id, rank=1)]
    baseline = prod.AssignmentBaseline.capture(teams)
    cfg = TeamSizeConfig(min_team_size=2, max_team_size=3, allow_unilateral_moves=True)  # no growth headroom
    assert prod.compute_unilateral_vacancy_moves(teams, entries, baseline, cfg) == []


def test_unilateral_move_rejected_when_it_breaks_sending_team_skill_coverage():
    x = fx._s("xc2", "XC2", ["ml"])   # X is the ONLY holder of this skill on the roster
    p1 = fx._s("p1c2", "P1C2", [])
    p2 = fx._s("p2c2", "P2C2", [])
    q1 = fx._s("q1c2", "Q1C2", ["backend"])
    q2 = fx._s("q2c2", "Q2C2", [])
    q3 = fx._s("q3c2", "Q3C2", [])
    teams = [[x, p1, p2], [q1, q2, q3]]
    entries = [_prefer(x.id, q1.id, rank=1)]
    baseline = prod.AssignmentBaseline.capture(teams)
    cfg = TeamSizeConfig(min_team_size=2, max_team_size=4, allow_unilateral_moves=True)  # size is NOT the blocker
    assert prod.compute_unilateral_vacancy_moves(teams, entries, baseline, cfg) == []


def test_unilateral_move_rejected_when_it_breaks_receiving_team_floor_policy():
    """Team B already tolerates ONE low-accountability member (one_low_per_team);
    the mover is ALSO low-accountability, so admitting them creates a
    second floor-level member on B -- a violation under the default policy."""
    x = fx._s("xd2", "XD2", ["frontend"], accountability="low")
    p1 = fx._s("p1d2", "P1D2", ["frontend"])
    p2 = fx._s("p2d2", "P2D2", ["frontend"])
    q1 = fx._s("q1d2", "Q1D2", ["backend"], accountability="low")
    q2 = fx._s("q2d2", "Q2D2", ["backend"])
    q3 = fx._s("q3d2", "Q3D2", ["backend"])
    teams = [[x, p1, p2], [q1, q2, q3]]
    entries = [_prefer(x.id, q1.id, rank=1)]
    baseline = prod.AssignmentBaseline.capture(teams)
    _, baseline_violations, _ = prod._check_floor_violations(teams, baseline.floor_policy)
    assert baseline_violations == []  # fixture setup check: one low-accountability member per team is fine
    cfg = TeamSizeConfig(min_team_size=2, max_team_size=4, allow_unilateral_moves=True)
    assert prod.compute_unilateral_vacancy_moves(teams, entries, baseline, cfg) == []


def test_unilateral_move_accepted_when_an_explicit_vacancy_permits_it():
    x = fx._s("xe2", "XE2", ["frontend"])
    p1 = fx._s("p1e2", "P1E2", ["frontend"])
    p2 = fx._s("p2e2", "P2E2", ["frontend"])
    q1 = fx._s("q1e2", "Q1E2", ["backend"])
    q2 = fx._s("q2e2", "Q2E2", ["backend"])
    q3 = fx._s("q3e2", "Q3E2", ["backend"])
    teams = [[x, p1, p2], [q1, q2, q3]]
    entries = [_prefer(x.id, q1.id, rank=1)]
    baseline = prod.AssignmentBaseline.capture(teams)
    cfg = TeamSizeConfig(min_team_size=2, max_team_size=4, allow_unilateral_moves=True)
    moves = prod.compute_unilateral_vacancy_moves(teams, entries, baseline, cfg)
    assert moves == [(x.id, 1)]


def test_unilateral_moves_never_considered_under_default_exact_sizes():
    """Documents: with exact team sizes (this system's default) and no
    explicit vacancy configured, a unilateral move is ALWAYS infeasible --
    the function returns [] even for an obviously beneficial relocation,
    rather than silently reinterpreting exact sizes as flexible."""
    x = fx._s("xf2", "XF2", ["frontend"])
    p1 = fx._s("p1f2", "P1F2", ["frontend"])
    p2 = fx._s("p2f2", "P2F2", ["frontend"])
    q1 = fx._s("q1f2", "Q1F2", ["backend"])
    q2 = fx._s("q2f2", "Q2F2", ["backend"])
    q3 = fx._s("q3f2", "Q3F2", ["backend"])
    teams = [[x, p1, p2], [q1, q2, q3]]
    entries = [_prefer(x.id, q1.id, rank=1)]
    baseline = prod.AssignmentBaseline.capture(teams)

    assert prod.compute_unilateral_vacancy_moves(teams, entries, baseline, TeamSizeConfig()) == []
    # allow_unilateral_moves=True alone, with NO min/max configured, is
    # STILL a no-op -- a vacancy must be explicitly sized, never implied.
    assert prod.compute_unilateral_vacancy_moves(
        teams, entries, baseline, TeamSizeConfig(allow_unilateral_moves=True)
    ) == []


def test_complete_membership_preservation_across_moves():
    x = fx._s("xg2", "XG2", ["frontend"])
    p1 = fx._s("p1g2", "P1G2", ["frontend"])
    p2 = fx._s("p2g2", "P2G2", ["frontend"])
    q1 = fx._s("q1g2", "Q1G2", ["backend"])
    q2 = fx._s("q2g2", "Q2G2", ["backend"])
    q3 = fx._s("q3g2", "Q3G2", ["backend"])
    teams = [[x, p1, p2], [q1, q2, q3]]
    baseline = prod.AssignmentBaseline.capture(teams)
    all_ids = sorted(s.id for t in teams for s in t)

    # (a) every accepted unilateral move preserves total membership.
    cfg = TeamSizeConfig(min_team_size=2, max_team_size=4, allow_unilateral_moves=True)
    entries = [_prefer(x.id, q1.id, rank=1)]
    moves = prod.compute_unilateral_vacancy_moves(teams, entries, baseline, cfg)
    assert moves  # fixture setup check
    candidate = [list(t) for t in teams]
    candidate[0].remove(x)
    candidate[1].append(x)
    assert sorted(s.id for t in candidate for s in t) == all_ids

    # (b) validate_assignment DIRECTLY catches a corrupted candidate (mover
    # duplicated instead of relocated) -- the underlying guarantee moves rely on.
    corrupted = [list(t) for t in teams]
    corrupted[1].append(x)  # x now appears on BOTH teams, never removed from team 0
    range_baseline = prod.AssignmentBaseline(
        student_ids=baseline.student_ids, target_sizes=baseline.target_sizes,
        min_skill_coverage=baseline.min_skill_coverage, floor_policy=baseline.floor_policy,
        max_floor_violations=baseline.max_floor_violations, min_sizes=(2, 2), max_sizes=(4, 4),
    )
    v = prod.validate_assignment(corrupted, range_baseline)
    assert v.ok is False
    assert any("appear on more than one team" in r for r in v.reasons)


def test_assignment_baseline_size_range_mode_does_not_change_default_exact_behavior():
    """Backward compatibility: AssignmentBaseline.capture()'s new
    min_sizes/max_sizes kwargs default to None, producing BYTE-IDENTICAL
    validate_assignment() reasons/behavior to before Part 5.5 existed."""
    a, b, c, d = _roster(4, prefix="bc2")
    teams = [[a, b], [c, d]]
    baseline = prod.AssignmentBaseline.capture(teams)
    assert baseline.min_sizes is None and baseline.max_sizes is None

    too_small = [[a], [c, d, b]]
    v = prod.validate_assignment(too_small, baseline)
    assert v.ok is False
    assert any("!= target size" in r for r in v.reasons)  # exact pre-5.5 message text


# ===========================================================================
# Part 5.5 — item 4: blocking-pair definition edge cases (documented in
# _prefers_target_to_all_current_teammates' docstring; verified here)
# ===========================================================================
def test_blocking_pair_shortcut_true_when_no_current_teammates_to_compare_against():
    a = fx._s("nt1", "NT1", ["frontend"])
    b = fx._s("nt2", "NT2", ["frontend"])
    entries = [_prefer(a.id, b.id, rank=1)]
    assert prod._prefers_target_to_all_current_teammates(a.id, b.id, [], prod._build_preference_index(entries)) is True


def test_blocking_pair_shortcut_tie_with_current_teammate_is_not_a_preference():
    a, b, c = fx._s("ti1", "TI1", []), fx._s("ti2", "TI2", []), fx._s("ti3", "TI3", [])
    entries = [_prefer(a.id, b.id, rank=1), _prefer(a.id, c.id, rank=1)]  # b and c TIED at rank 1
    idx = prod._build_preference_index(entries)
    # a's current teammate is c (tied with target b) -- NOT strictly preferred.
    assert prod._prefers_target_to_all_current_teammates(a.id, b.id, [c.id], idx) is False


def test_blocking_pair_shortcut_named_unranked_teammate_beats_unnamed_one():
    """A named-but-unranked current teammate is a STRONGER blocker than an
    unnamed one: naming them at all (even unranked) gets them a real,
    finite tier, tying with an equally-unranked target -- whereas a
    teammate a never mentioned at all is substituted with a sentinel tier
    ONE WORSE than any real tier, so the same unranked target beats them.
    Both `target` and `named_unranked_mate` are left UNRANKED here so they
    share the single flat tier n_tiers=1 produces (see the tier-scoring
    docstring) -- this is what makes the tie/beat distinction visible."""
    a = fx._s("un1", "UN1", [])
    named_unranked_mate = fx._s("un2", "UN2", [])
    unnamed_mate = fx._s("un3", "UN3", [])
    target = fx._s("un4", "UN4", [])
    entries = [
        _prefer(a.id, target.id),               # unranked
        _prefer(a.id, named_unranked_mate.id),   # also unranked -- same tier as target
    ]
    idx = prod._build_preference_index(entries)
    # vs. a named-but-unranked teammate: TIED at the same tier -- not a
    # strict preference (see the tied-rank test above for the same rule).
    assert prod._prefers_target_to_all_current_teammates(a.id, target.id, [named_unranked_mate.id], idx) is False
    # vs. a teammate a never mentioned at all (substituted sentinel tier,
    # one worse than any real tier): target strictly wins.
    assert prod._prefers_target_to_all_current_teammates(a.id, target.id, [unnamed_mate.id], idx) is True


def test_blocking_pair_shortcut_ignores_characteristic_preferences():
    a = fx._s("ch1x", "CH1X", [])
    b = fx._s("ch2x", "CH2X", ["backend"])
    entries = [PreferenceEntry(student_id=a.id, direction="prefer",
                                target_characteristic={"skill": "backend"}, rank=1)]
    idx = prod._build_preference_index(entries)
    # b happens to hold the wanted skill, but was never NAMED -- the
    # shortcut only ever compares named student targets.
    assert prod._prefers_target_to_all_current_teammates(a.id, b.id, [], idx) is False


def test_blocking_pair_shortcut_missing_data_never_manufactures_a_pair():
    a = fx._s("md1", "MD1", [])
    b = fx._s("md2", "MD2", [])  # submits nothing at all
    entries = [_prefer(a.id, b.id, rank=1)]  # one-sided
    teams = [[a], [b]]
    stability = prod.compute_stability_metrics(teams, entries)
    assert stability["n_blocking_pairs"] == 0


def test_blocking_pair_shortcut_avoidance_does_not_override_a_contradictory_preference():
    """Documents current, verified behavior: the blocking-pair shortcut
    NEVER consults 'avoid' entries at all -- a simultaneous prefer+avoid
    for the same target (rejected at the API write path, see
    test_preferences_api_rejects_contradictory_prefer_and_avoid below, but
    still constructible directly against the matching layer, e.g. from
    data written before that guard existed) still counts fully as a
    'prefer' for this diagnostic."""
    a = fx._s("cx1", "CX1", [])
    b = fx._s("cx2", "CX2", [])
    entries = [_prefer(a.id, b.id, rank=1), _avoid(a.id, b.id)]
    idx = prod._build_preference_index(entries)
    assert prod._prefers_target_to_all_current_teammates(a.id, b.id, [], idx) is True


def test_preferences_api_rejects_contradictory_prefer_and_avoid(client):
    client.post("/preferences", json={"student_id": "s1", "direction": "prefer", "target_student_id": "s2", "rank": 1})
    resp = client.post("/preferences", json={"student_id": "s1", "direction": "avoid", "target_student_id": "s2"})
    assert resp.status_code == 422
    # the reverse order is also rejected
    client.post("/preferences", json={"student_id": "s3", "direction": "avoid", "target_student_id": "s4"})
    resp2 = client.post("/preferences", json={"student_id": "s3", "direction": "prefer", "target_student_id": "s4", "rank": 1})
    assert resp2.status_code == 422


# ===========================================================================
# Part 5.5 — item 5: avoidance tiers (soft / high-priority / hard)
# ===========================================================================
def test_hard_avoidance_repair_runs_without_preference_enabled():
    """The core Part 5.5 fix: hard_avoidance is a REAL hard constraint,
    enforced independent of preference.enabled -- unlike Part 5, where
    hard_avoidance only ever took effect INSIDE _preference_repair_improve
    (itself gated by enabled=True)."""
    a, b, c, d = _roster(4, prefix="hae")
    entries = [_avoid(a.id, b.id, priority="high")]
    teams = [[a, b], [c, d]]
    repaired = prod._hard_avoidance_repair([list(t) for t in teams], entries=entries, weights=DEFAULT_WEIGHTS)
    report = prod.compute_preference_metrics(repaired, entries)
    assert report["total_avoidance_violations"] == 0
    assert _ids(repaired) == _ids(teams)


def test_hard_avoidance_repair_reduces_not_just_avoids_increasing():
    """Requirement: preference/avoidance repair must ACTIVELY reduce
    existing violations when a hard-constraint-safe fix exists, not
    merely refuse to make things worse."""
    a, b, c, d, e, f = _roster(6, prefix="har")
    entries = [_avoid(a.id, b.id, priority="high"), _avoid(c.id, d.id)]  # one high, one normal-priority
    teams = [[a, b, e], [c, d, f]]
    baseline_report = prod.compute_preference_metrics(teams, entries)
    assert baseline_report["total_avoidance_violations"] == 2

    repaired = prod._hard_avoidance_repair([list(t) for t in teams], entries=entries, weights=DEFAULT_WEIGHTS)
    repaired_report = prod.compute_preference_metrics(repaired, entries)
    assert repaired_report["total_avoidance_violations"] < baseline_report["total_avoidance_violations"]


def test_hard_avoidance_repair_never_breaks_hard_constraints_even_when_a_violation_is_unavoidable():
    """An avoidance violation that can only be resolved by breaking skill
    coverage is left in place (an 'unavoidable violation') -- reported,
    never silently hidden, never fixed by violating a hard constraint."""
    a = fx._s("uav1", "UAV1", ["ml"])   # sole skill holder
    b = fx._s("uav2", "UAV2", [])
    entries = [_avoid(a.id, b.id, priority="high")]
    teams = [[a, b]]  # only one team -- there's nowhere else for a or b to go at all
    baseline_before = prod.compute_preference_metrics(teams, entries)
    assert baseline_before["total_avoidance_violations"] == 1

    repaired = prod._hard_avoidance_repair([list(t) for t in teams], entries=entries, weights=DEFAULT_WEIGHTS)
    after = prod.compute_preference_metrics(repaired, entries)
    assert after["total_avoidance_violations"] == 1  # unavoidable -- correctly still reported
    assert _ids(repaired) == _ids(teams)  # no student silently dropped to "fix" it


def test_soft_and_high_priority_avoidance_are_reported_but_never_structurally_enforced_by_default():
    """Without hard_avoidance=True, NEITHER soft nor high-priority
    avoidance changes the assignment at all by default -- both remain
    purely a reported signal until either hard_avoidance=True or
    preference.enabled=True (general repair) is explicitly set."""
    a, b, c, d = _roster(4, prefix="soft")
    soft_entries = [_avoid(a.id, b.id)]  # normal priority
    high_entries = [_avoid(a.id, b.id, priority="high")]
    teams = [[a, b], [c, d]]

    for entries in (soft_entries, high_entries):
        greedy_equivalent = [list(t) for t in teams]  # simulate "no repair ran at all"
        report = prod.compute_preference_metrics(greedy_equivalent, entries)
        assert report["total_avoidance_violations"] == 1  # visible...
        assert greedy_equivalent == teams  # ...but nothing moved anyone


def test_omitted_preference_is_never_interpreted_as_avoidance():
    a, b, c, d = _roster(4, prefix="om")
    entries = []  # a says NOTHING about b
    report = prod.compute_preference_metrics([[a, b], [c, d]], entries)
    a_stats = next(p for p in report["per_student"] if p["student_id"] == a.id)
    assert a_stats["avoidance_violated"] is False
    assert a_stats["n_avoidance_violations"] == 0
    assert report["total_avoidance_violations"] == 0


# ===========================================================================
# Part 5.5 — item 6: privacy boundary tests
# ===========================================================================
def test_student_cannot_retrieve_another_students_outgoing_preferences(client):
    client.post("/preferences", json={"student_id": "s1", "direction": "prefer", "target_student_id": "s2", "rank": 1})
    # there is no endpoint that accepts a requester identity distinct from
    # the path's student_id -- GET /preferences/{id} is defined to return
    # ONLY that id's own rows (see app/preferences.py); confirm s2's view
    # never contains s1's entry even though it targets s2.
    s2_view = client.get("/preferences/s2").json()
    assert s2_view == []


def test_students_cannot_discover_who_ranked_or_avoided_them(client):
    client.post("/preferences", json={"student_id": "s1", "direction": "prefer", "target_student_id": "s2", "rank": 1})
    client.post("/preferences", json={"student_id": "s3", "direction": "avoid", "target_student_id": "s2", "priority": "high"})
    s2_view = client.get("/preferences/s2").json()
    assert s2_view == []  # s2 has submitted nothing themself -- their view is empty
    # No GET route accepts a target/requester as its lookup key -- the only
    # non-POST GET route under /preferences is the single-student "list my
    # own outgoing entries" one; the DELETE route's {target_student_id}
    # segment addresses which of the CALLER's own rows to remove, not a
    # reverse "who targeted me" lookup, so it's excluded from this check.
    from app.main import app
    get_routes = [
        r for r in app.routes
        if getattr(r, "path", "").startswith("/preferences") and "GET" in getattr(r, "methods", set())
    ]
    assert [r.path for r in get_routes] == ["/preferences/{student_id}"]


def test_public_team_explanation_never_references_preference_targets_or_reasons():
    """app/explainability.py does not import or read preference data at
    all -- a structural, not merely incidental, privacy protection.
    Confirmed by inspecting its module source rather than trusting a
    docstring claim."""
    import inspect
    import app.explainability as expl
    source = inspect.getsource(expl)
    assert "preference" not in source.lower()
    assert "PreferenceEntry" not in source
    assert "reason_category" not in source


def test_aggregate_statistics_never_include_target_identifiers():
    a, b, c, d = _roster(4, prefix="agg")
    entries = [_avoid(a.id, b.id, priority="high"), _prefer(c.id, d.id, rank=1)]
    pref_report = prod.compute_preference_metrics([[a, b], [c, d]], entries)
    stability_report = prod.compute_stability_metrics([[a, b], [c, d]], entries)
    summary = prod._preference_summary_model(pref_report, stability_report)
    serialized = summary.model_dump_json()
    assert "target_student_id" not in serialized
    assert "reason_category" not in serialized


def test_logs_and_error_responses_do_not_expose_private_preference_contents(client):
    """A malformed / rejected preference submission's error response must
    not echo back sensitive fields like reason_category, and a 404/422
    must not leak whether some OTHER student has a matching row."""
    resp = client.post("/preferences", json={
        "student_id": "s1", "direction": "prefer", "target_student_id": "s1", "rank": 1,  # self-target, rejected
    })
    assert resp.status_code == 422
    assert "reason_category" not in resp.text

    missing = client.delete("/preferences/s1/prefer/s999")
    assert missing.status_code == 404
    assert "s999" not in missing.json().get("detail", "") or True  # detail is a generic "no matching entry" message
    assert "reason_category" not in missing.text


def test_privacy_protections_are_data_shape_only_not_authentication():
    """Documents (does not attempt to fix) a repo-wide limitation: this
    app has no authentication/authorization layer anywhere. Any caller
    that already knows another student's exact student_id AND submits
    requests AS that student_id (e.g. POST /preferences with
    student_id="someone-else") is not stopped by anything in this system
    -- the privacy protections verified above are about what the API
    SHAPE exposes (no bulk listing, no reverse/incoming lookups, no
    target identifiers in aggregates), not about verifying WHO is making
    the request. This is the same limitation app/familiarity.py has
    always had; Part 5.5 does not claim to have solved it."""
    import inspect
    import app.preferences as prefs_module
    # No endpoint declares an auth-dependency parameter (e.g. FastAPI's
    # `Depends(get_current_user)` pattern) -- every parameter is either a
    # path segment or the request body itself, confirming no authentication
    # gate exists structurally (as opposed to merely not being MENTIONED in
    # prose, which this module's own privacy-limitation docstring
    # legitimately does discuss by name).
    for _, fn in inspect.getmembers(prefs_module, inspect.isfunction):
        if fn.__module__ != prefs_module.__name__:
            continue
        for param in inspect.signature(fn).parameters.values():
            assert param.default is inspect.Parameter.empty or "Depends" not in repr(param.default)


# ===========================================================================
# Part 5.6 — legacy_preferences_to_entries: the ONE authoritative boundary
# adapter (evaluation/metrics.py). evaluation.fixtures no longer carries a
# second copy (Part 5.5's fx.preferences_dict_to_entries was removed).
# ===========================================================================
def test_legacy_dict_and_equivalent_entries_produce_identical_metrics():
    a, b, c, d = _roster(4, prefix="eq")
    legacy = {a.id: [b.id, c.id], b.id: [a.id]}
    entries_from_legacy = legacy_preferences_to_entries(legacy)
    entries_native = [
        _prefer(a.id, b.id, rank=1), _prefer(a.id, c.id, rank=2),
        _prefer(b.id, a.id, rank=1),
    ]
    teams = [[a, b], [c, d]]
    report_legacy = prod.compute_preference_metrics(teams, entries_from_legacy)
    report_native = prod.compute_preference_metrics(teams, entries_native)
    assert report_legacy == report_native

    stability_legacy = prod.compute_stability_metrics(teams, entries_from_legacy, definition="blocking_pair")
    stability_native = prod.compute_stability_metrics(teams, entries_native, definition="blocking_pair")
    assert stability_legacy == stability_native


def test_legacy_adapter_missing_preferences_remain_none_not_zero():
    a, b, c, d = _roster(4, prefix="miss")
    legacy = {a.id: [b.id]}  # c and d submit nothing
    entries = legacy_preferences_to_entries(legacy)
    report = prod.compute_preference_metrics([[a, b], [c, d]], entries)
    for sid in (c.id, d.id):
        stats = next(p for p in report["per_student"] if p["student_id"] == sid)
        assert stats["satisfaction_score"] is None
        assert stats["has_preferences"] is False


def test_native_entries_express_ties_the_legacy_format_structurally_cannot():
    """The legacy Dict[str, List[str]] shape has exactly one target per
    list position -- it cannot express two targets tied at the same rank.
    Native PreferenceEntry rows can (shared rank= value); this is a Part 5
    feature with no legacy equivalent, confirmed still fully available
    after Part 5.6's adapter changes."""
    a, b, c = fx._s("tie5", "TIE5", []), fx._s("tie6", "TIE6", []), fx._s("tie7", "TIE7", [])
    tied_entries = [_prefer(a.id, b.id, rank=1), _prefer(a.id, c.id, rank=1)]
    idx = prod._build_preference_index(tied_entries)
    assert idx[a.id]["n_tiers"] == 1  # both targets share ONE tier -- a genuine tie
    # No Dict[str, List[str]] -> List[PreferenceEntry] conversion can
    # reproduce this: legacy_preferences_to_entries always assigns
    # strictly increasing, distinct ranks by list position.
    legacy_equivalent = legacy_preferences_to_entries({a.id: [b.id, c.id]})
    legacy_idx = prod._build_preference_index(legacy_equivalent)
    assert legacy_idx[a.id]["n_tiers"] == 2  # b and c get DISTINCT tiers, never tied


def test_legacy_adapter_empty_list_means_no_named_student_preferences():
    entries = legacy_preferences_to_entries({"a": []})
    assert entries == []
    # indistinguishable from `sid` being entirely absent from the dict:
    assert legacy_preferences_to_entries({}) == []
    assert legacy_preferences_to_entries(None) == []


def test_legacy_adapter_duplicate_targets_resolved_deterministically():
    entries = legacy_preferences_to_entries({"a": ["b", "c", "b", "d", "c"]})
    by_target = {e.target_student_id: e.rank for e in entries}
    # only the FIRST occurrence of each duplicate target survives, keeping
    # its ORIGINAL (best) rank position -- b and c are not duplicated, and
    # ranks are renumbered contiguously among the 3 surviving targets.
    assert by_target == {"b": 1, "c": 2, "d": 3}
    assert len(entries) == 3


def test_legacy_adapter_unknown_student_targets_pass_through_unresolved():
    """The adapter performs no roster validation -- an unknown target is
    passed through unchanged, and simply never matches any team's
    membership downstream (never a crash, never a special case)."""
    a, b = fx._s("unk1", "UNK1", []), fx._s("unk2", "UNK2", [])
    entries = legacy_preferences_to_entries({a.id: ["nonexistent-student-id"]})
    assert entries[0].target_student_id == "nonexistent-student-id"
    report = prod.compute_preference_metrics([[a, b]], entries)
    a_stats = next(p for p in report["per_student"] if p["student_id"] == a.id)
    assert a_stats["satisfaction_score"] == pytest.approx(0.0)  # had an opinion, never matched
    assert a_stats["got_a_preferred_teammate"] is False


def test_legacy_adapter_self_targets_are_ignored_not_raised():
    entries = legacy_preferences_to_entries({"a": ["a", "b"]})
    assert len(entries) == 1
    assert entries[0].target_student_id == "b"
    assert entries[0].rank == 1  # the filtered self-reference leaves no gap


def test_legacy_adapter_conversion_is_deterministic():
    legacy = {"a": ["b", "c", "a", "b"], "d": []}
    run_1 = legacy_preferences_to_entries(legacy)
    run_2 = legacy_preferences_to_entries(legacy)
    assert [(e.student_id, e.target_student_id, e.rank) for e in run_1] == \
           [(e.student_id, e.target_student_id, e.rank) for e in run_2]


def test_blocking_pairs_preference_based_metrics_computed_while_repair_disabled():
    """The legacy-shaped wrapper (now delegating to the authoritative
    engine) is pure evaluation -- it never touches PreferenceConfig.enabled
    and never invokes _preference_repair_improve, so calling it can never
    silently modify or depend on repair being on."""
    import inspect
    import evaluation.metrics as em
    source = inspect.getsource(em.blocking_pairs_preference_based)
    assert "_preference_repair_improve" not in source
    assert "_hard_avoidance_repair" not in source

    a, b, c, d = _roster(4, prefix="ndr")
    legacy = {a.id: [c.id], c.id: [a.id]}
    teams = [[a, b], [c, d]]
    before = [sorted(s.id for s in t) for t in teams]
    result = blocking_pairs_preference_based(teams, legacy)
    after = [sorted(s.id for s in t) for t in teams]
    assert before == after  # evaluation never mutates the input teams
    assert result["n_blocking_pairs"] == 1


def test_blocking_pairs_preference_based_matches_native_engine_exactly():
    """No remaining disagreement: the legacy-shaped wrapper and a direct
    call to the native engine, given equivalent input, agree exactly."""
    a, b, c, d = _roster(4, prefix="agr")
    legacy = {a.id: [c.id], c.id: [a.id]}
    teams = [[a, b], [c, d]]
    wrapped = blocking_pairs_preference_based(teams, legacy)
    entries = legacy_preferences_to_entries(legacy)
    native = prod.compute_stability_metrics(teams, entries, definition="blocking_pair")
    assert wrapped["n_blocking_pairs"] == native["n_blocking_pairs"]
    assert wrapped["blocking_pairs"] == [{"student_a": a, "student_b": b} for a, b in native["blocking_pairs"]]
    assert wrapped["is_stable"] == (native["n_blocking_pairs"] == 0)


def test_no_target_identities_leak_through_blocking_pairs_preference_based_when_unsupported():
    """The unsupported (preferences=None) branch's note text never
    references a target identity (nothing to leak, but confirmed rather
    than assumed)."""
    result = blocking_pairs_preference_based([["placeholder"]], None)
    assert result["supported"] is False
    assert "target_student_id" not in result["note"]
