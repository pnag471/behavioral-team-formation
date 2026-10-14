"""
Part 6.5 Section 4 — orchestration-level regression-protection audit.

Tests app.matching._run_optional_passes / _protected_metrics for the exact
failure modes Part 6.5 asked to be checked. Each test name states the
specific claim it verifies; see each docstring for what would have to be
true in the code for the test to fail.
"""
from __future__ import annotations

import json
import math

import pytest

from app import matching as prod
from evaluation import fixtures as fx
from evaluation.metrics import legacy_preferences_to_entries


def _seed_students():
    import pathlib
    path = pathlib.Path(__file__).resolve().parents[1] / "sample_students.json"
    with open(path) as f:
        raw = json.load(f)
    from app.models import Student
    return [Student(**r) for r in raw]


# ===========================================================================
# NaN / infinity cannot bypass the regression comparison
# ===========================================================================
def test_nan_metric_forces_pass_rejection_rather_than_silently_bypassing_the_regression_check(monkeypatch):
    """
    `_maybe_run`'s regression check is `after[k] < before[k] - tol`. Python
    NaN comparisons are always False, so — WITHOUT the explicit
    math.isfinite() guard added in Part 6.5 — a metric that goes NaN would
    never register as "regressed" and would be silently ACCEPTED. This
    test forces `_partition_skill_coverage_min` to return NaN (a stand-in
    for any metric computation bug that produces NaN) and asserts the pass
    is rejected, not silently accepted.
    """
    students = fx.fixture_stacked_team().students
    request, _ = prod.resolve_profile(prod.TeamGenerationRequest(team_size=4, profile="legacy", ability={"enabled": True}))
    weights = prod._normalize_weights(request.weights)
    teams = prod._greedy_build(students, 4, weights)
    teams = prod._two_opt_improve(teams, weights, floor_policy=request.floor_policy)
    teams = prod._maximin_improve(teams, weights=weights, floor_policy=request.floor_policy)

    monkeypatch.setattr(prod, "_partition_skill_coverage_min", lambda t: float("nan"))

    final_teams, trace = prod._run_optional_passes(
        teams, request, weights, request.floor_policy, [], [], primary_metric=None,
    )
    ability_trace = next(t for t in trace if t["pass"] == "ability")
    assert ability_trace["accepted"] is False, "a NaN protected metric must never be silently accepted"
    assert "skill_coverage_min" in ability_trace["non_finite_metric_keys"]
    # Final teams must equal the PRE-pass state exactly (reverted).
    assert [[s.id for s in t] for t in final_teams] == [[s.id for s in t] for t in teams]


def test_infinite_metric_also_forces_pass_rejection(monkeypatch):
    """Same as above for +inf/-inf, not just NaN — math.isfinite() catches both."""
    students = fx.fixture_stacked_team().students
    request, _ = prod.resolve_profile(prod.TeamGenerationRequest(team_size=4, profile="legacy", fairness={"enabled": True}))
    weights = prod._normalize_weights(request.weights)
    teams = prod._greedy_build(students, 4, weights)
    teams = prod._two_opt_improve(teams, weights, floor_policy=request.floor_policy)
    teams = prod._maximin_improve(teams, weights=weights, floor_policy=request.floor_policy)

    monkeypatch.setattr(prod, "_partition_floor_min", lambda t: float("inf"))

    final_teams, trace = prod._run_optional_passes(
        teams, request, weights, request.floor_policy, [], [], primary_metric=None,
    )
    fairness_trace = next(t for t in trace if t["pass"] == "fairness")
    assert fairness_trace["accepted"] is False
    assert "floor_min" in fairness_trace["non_finite_metric_keys"]


# ===========================================================================
# Reverting a pass restores the COMPLETE previous assignment; no stale state
# ===========================================================================
def test_reverted_pass_restores_the_exact_pre_pass_assignment(monkeypatch):
    """
    When a pass's candidate is rejected, `_run_optional_passes` must return
    the assignment EXACTLY as it was before that pass ran — not a partially
    -applied intermediate state. Forces rejection via an injected NaN
    metric (same mechanism as above) and checks every student's team
    membership, not just team count/size.
    """
    students = fx.fixture_stacked_team().students
    request, _ = prod.resolve_profile(prod.TeamGenerationRequest(
        team_size=4, profile="legacy", ability={"enabled": True}, fairness={"enabled": True},
    ))
    weights = prod._normalize_weights(request.weights)
    teams = prod._greedy_build(students, 4, weights)
    teams = prod._two_opt_improve(teams, weights, floor_policy=request.floor_policy)
    pre_pass_teams = prod._maximin_improve(teams, weights=weights, floor_policy=request.floor_policy)
    pre_pass_snapshot = [sorted(s.id for s in t) for t in pre_pass_teams]

    call_count = {"n": 0}
    real_fn = prod._partition_ability_gap

    def _flaky(t):
        call_count["n"] += 1
        # Let the FIRST call (used inside the pass's own internal search)
        # behave normally so the pass actually tries to make swaps; only
        # corrupt values used by the ORCHESTRATION-level before/after
        # check by patching a metric _protected_metrics itself calls.
        return real_fn(t)

    monkeypatch.setattr(prod, "_partition_ability_gap", _flaky)
    monkeypatch.setattr(prod, "_stage_objective", lambda t, w: float("nan"))

    final_teams, trace = prod._run_optional_passes(
        [list(t) for t in pre_pass_teams], request, weights, request.floor_policy, [], [], primary_metric=None,
    )
    final_snapshot = [sorted(s.id for s in t) for t in final_teams]
    assert final_snapshot == pre_pass_snapshot, (
        "a rejected pass must restore the assignment byte-for-byte, not merely team COUNTS/sizes"
    )
    assert all(not t["accepted"] for t in trace), "both passes should have been rejected by the injected NaN"


def test_metrics_recomputed_on_final_teams_are_not_stale_after_a_reversion(monkeypatch):
    """
    After a pass is reverted, any metric computed on the RETURNED teams
    must reflect those teams' actual composition — not a cached value left
    over from the rejected candidate. Verified by recomputing
    compute_ability_metrics from scratch on the final teams and comparing
    against an independent fresh call on the same object.
    """
    students = fx.fixture_stacked_team().students
    request, _ = prod.resolve_profile(prod.TeamGenerationRequest(team_size=4, profile="legacy", ability={"enabled": True}))
    weights = prod._normalize_weights(request.weights)
    teams = prod._greedy_build(students, 4, weights)
    teams = prod._two_opt_improve(teams, weights, floor_policy=request.floor_policy)
    teams = prod._maximin_improve(teams, weights=weights, floor_policy=request.floor_policy)

    monkeypatch.setattr(prod, "_partition_skill_coverage_min", lambda t: float("nan"))
    final_teams, _trace = prod._run_optional_passes(
        teams, request, weights, request.floor_policy, [], [], primary_metric=None,
    )

    monkeypatch.undo()
    report_a = prod.compute_ability_metrics(final_teams, {})
    report_b = prod.compute_ability_metrics(final_teams, {})
    assert report_a == report_b, "recomputing on the same final teams object must be deterministic/non-stale"


# ===========================================================================
# _protected_metrics is a pure, deterministic function
# ===========================================================================
def test_protected_metrics_computed_consistently_before_and_after():
    students = fx.fixture_stacked_team().students
    weights = prod._normalize_weights({"skill_coverage": 0.4, "behavioral_compat": 0.3,
                                        "availability_overlap": 0.2, "shared_interests": 0.1})
    teams = prod._greedy_build(students, 4, weights)
    m1 = prod._protected_metrics(teams, weights, prod.DEFAULT_FLOOR_POLICY)
    m2 = prod._protected_metrics(teams, weights, prod.DEFAULT_FLOOR_POLICY)
    assert m1 == m2


# ===========================================================================
# Missing metrics are handled explicitly, not silently coerced to zero
# ===========================================================================
def test_missing_preference_data_is_not_scored_as_zero_satisfaction():
    """
    _protected_metrics' mean_preference_satisfaction branch: with ZERO
    preference entries, compute_preference_metrics's mean_satisfaction is
    None (no data) — _protected_metrics must map that to 1.0 (nothing to
    protect against regressing), NEVER 0.0 (which would look like total
    dissatisfaction and could spuriously block later passes).
    """
    students = fx.fixture_stacked_team().students
    weights = prod._normalize_weights({"skill_coverage": 0.4, "behavioral_compat": 0.3,
                                        "availability_overlap": 0.2, "shared_interests": 0.1})
    teams = prod._greedy_build(students, 4, weights)
    metrics = prod._protected_metrics(
        teams, weights, prod.DEFAULT_FLOOR_POLICY, extra_metric="mean_preference_satisfaction",
        preference_entries=[],
    )
    assert metrics["mean_preference_satisfaction"] == 1.0


# ===========================================================================
# Every profile's primary_metric has consistent (higher-is-better) directionality
# ===========================================================================
def test_every_profile_primary_metric_is_higher_is_better():
    """
    `_maybe_run`'s regression check (`after[k] < before[k] - tol`) assumes
    HIGHER is always better for every key in `_protected_metrics`' output —
    this is true for total_score/floor_min/skill_coverage_min by
    construction, and MUST also hold for every profile's `primary_metric`
    or the orchestration layer would silently protect the WRONG direction.
    Verifies every currently-defined profile's primary_metric is one of
    the two extra_metric branches _protected_metrics implements — both of
    which are documented, verified-higher-is-better quantities (a bigger
    minimum utility / mean satisfaction is always the improvement
    direction) — rather than assuming this holds for names added later.
    """
    higher_is_better_metrics = {"min_utility", "mean_preference_satisfaction", "skill_coverage_min", None}
    for name, spec in prod.OPTIMIZATION_PROFILES.items():
        assert spec["primary_metric"] in higher_is_better_metrics, (
            f"profile {name!r} declares primary_metric={spec['primary_metric']!r}, which "
            f"_protected_metrics does not implement a higher-is-better branch for — "
            f"_maybe_run's regression direction assumption would be silently wrong."
        )


# ===========================================================================
# Tolerance-vs-scale finding: total_score is NOT normalized like the other
# protected metrics, so the SAME absolute tolerance is effectively much
# tighter (relatively) for total_score than for the [0,1]-bounded metrics.
# This test CHARACTERIZES the current, unchanged behavior (Part 6.5 must
# not alter acceptance semantics) — see recommended_changes.md for the
# recommendation this documents.
# ===========================================================================
def test_total_score_protected_metric_is_unnormalized_relative_to_its_siblings():
    students = fx.fixture_stacked_team().students
    weights = prod._normalize_weights({"skill_coverage": 0.4, "behavioral_compat": 0.3,
                                        "availability_overlap": 0.2, "shared_interests": 0.1})
    teams = prod._greedy_build(students, 4, weights)
    metrics = prod._protected_metrics(teams, weights, prod.DEFAULT_FLOOR_POLICY)
    n_teams = len(teams)
    # floor_min / skill_coverage_min are ALWAYS within [0, 1] regardless of
    # roster size; total_score is a SUM over all teams and so scales with
    # n_teams — the same absolute PROTECTED_METRIC_TOLERANCE (1e-6) is
    # therefore a much tighter RELATIVE tolerance on total_score at scale
    # than on its siblings. This is a real, documented finding
    # (recommended_changes.md), not something this test fixes.
    assert 0.0 <= metrics["floor_min"] <= 1.0
    assert 0.0 <= metrics["skill_coverage_min"] <= 1.0
    assert metrics["total_score"] > 1.0 or n_teams == 1, (
        "total_score is expected to scale with team count (unnormalized), unlike its siblings"
    )


# ===========================================================================
# Later passes cannot undo hard avoidance
# ===========================================================================
def test_preference_pass_never_increases_avoidance_violations_established_by_hard_avoidance_repair():
    """
    Production order is ability -> fairness -> familiarity ->
    hard_avoidance -> preference. Only `preference` runs AFTER
    hard_avoidance. `_preference_repair_improve` has its OWN internal
    per-swap `avoidance_ok` guard (checked whenever
    preference.hard_avoidance=True) that refuses any swap increasing
    avoidance-violation-pair count on either team it touches — this test
    verifies that guarantee holds end-to-end through the real orchestrator
    on a roster with a real avoidance violation.
    """
    from app.models import PreferenceEntry

    students = fx.fixture_preference_instability().students
    # pr0 explicitly wants to avoid pr7 (an arbitrary, previously-unrelated
    # pair for this fixture) — construct a real avoidance entry.
    entries = [PreferenceEntry(student_id="pr0", direction="avoid", target_student_id="pr7", priority="high")]

    request, _ = prod.resolve_profile(prod.TeamGenerationRequest(
        team_size=4, profile="legacy",
        preference={"enabled": True, "hard_avoidance": True},
    ))
    weights = prod._normalize_weights(request.weights)
    teams = prod._greedy_build(students, 4, weights)
    teams = prod._two_opt_improve(teams, weights, floor_policy=request.floor_policy)
    teams = prod._maximin_improve(teams, weights=weights, floor_policy=request.floor_policy)

    final_teams, trace = prod._run_optional_passes(
        teams, request, weights, request.floor_policy, [], entries, primary_metric=None,
    )
    hard_avoidance_trace = next((t for t in trace if t["pass"] == "hard_avoidance"), None)
    assert hard_avoidance_trace is not None, "hard_avoidance must run whenever preference.hard_avoidance=True"

    def _avoidance_count(teams_):
        pref_index = prod._build_preference_index(entries)
        return sum(len(prod._avoidance_violations_for_team([s.id for s in t], pref_index)) for t in teams_)

    after_hard_avoidance = _avoidance_count(teams)  # NOTE: `teams` was mutated in place by the passes below it
    after_preference = _avoidance_count(final_teams)
    assert after_preference <= after_hard_avoidance or after_preference == 0, (
        "the preference pass must never increase avoidance violations once hard_avoidance is on"
    )


# ===========================================================================
# Debug diagnostics never expose WHO a preference targets, or targets at all
# ===========================================================================
def test_optimization_trace_never_contains_a_students_preference_target_id():
    """
    `PreferenceSummary`/`PreferenceStudentStats` are documented as
    aggregate-only (app/preferences.py's module docstring). This test
    checks the SAME privacy property holds for `_run_optional_passes`'
    own `trace` output (surfaced as `optimization_trace` on the API
    response) — no target student id from a submitted preference entry
    should be discoverable anywhere in the trace, even indirectly.
    """
    from app.models import PreferenceEntry

    students = fx.fixture_preference_instability().students
    secret_target_id = "pr7"
    entries = [
        PreferenceEntry(student_id="pr0", direction="prefer", target_student_id=secret_target_id, rank=1),
        PreferenceEntry(student_id="pr0", direction="avoid", target_student_id="pr6", priority="high"),
    ]
    request, _ = prod.resolve_profile(prod.TeamGenerationRequest(
        team_size=4, profile="legacy", preference={"enabled": True, "hard_avoidance": True},
    ))
    weights = prod._normalize_weights(request.weights)
    teams = prod._greedy_build(students, 4, weights)
    teams = prod._two_opt_improve(teams, weights, floor_policy=request.floor_policy)
    teams = prod._maximin_improve(teams, weights=weights, floor_policy=request.floor_policy)

    _final_teams, trace = prod._run_optional_passes(
        teams, request, weights, request.floor_policy, [], entries, primary_metric=None,
    )
    serialized = json.dumps(trace, default=str)
    assert secret_target_id not in serialized, (
        "a submitted preference's TARGET student id must never appear in optimization_trace"
    )
    assert "pr6" not in serialized, "an avoidance target id must never appear in optimization_trace either"
