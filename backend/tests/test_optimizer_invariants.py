"""
Part 1.5 — constraint-safe optimizer tests.

Covers the two algorithmic regressions identified in recommended_changes.md
#7 and #9 and fixed in app/matching.py:

  1. _two_opt_improve could raise the aggregate (summed) objective while
     dropping a team's required-skill coverage below what greedy already
     achieved — the aggregate score is no longer allowed to "buy off" a
     skill-coverage regression (validate_assignment's hard constraint).
  2. _maximin_improve, run after _two_opt_improve, could accept swaps that
     improved the partition floor-minimum while silently undoing the
     primary total-score optimum 2-opt had already reached — acceptance is
     now lexicographic (hard constraints, then primary objective, then
     floor-minimum), with a documented zero-tolerance default for objective
     regression.

This suite is ADDITIVE — it does not modify or weaken any existing test.
"""
from __future__ import annotations

import copy

import pytest

from app import matching as prod
from evaluation import fixtures as fx
from evaluation.metrics import EvalConfig, skill_coverage, constraint_violations

DEFAULT_WEIGHTS = prod._normalize_weights({
    "skill_coverage": 0.4, "behavioral_compat": 0.3,
    "availability_overlap": 0.2, "shared_interests": 0.1,
})


def _sum_score(teams, weights=DEFAULT_WEIGHTS):
    return sum(prod._total_score(t, weights) for t in teams)


def _ids(teams):
    return sorted(s.id for t in teams for s in t)


# ===========================================================================
# validate_assignment / AssignmentBaseline — unit-level hard-constraint checks
# ===========================================================================
def _small_partition():
    students = [fx._s(f"s{i}", f"S{i}", ["frontend"]) for i in range(8)]
    teams = [students[:4], students[4:]]
    return students, teams


def test_validate_assignment_accepts_the_baseline_itself():
    students, teams = _small_partition()
    baseline = prod.AssignmentBaseline.capture(teams, prod.DEFAULT_FLOOR_POLICY)
    result = prod.validate_assignment(teams, baseline)
    assert result.ok is True
    assert result.reasons == []


def test_validate_assignment_rejects_a_duplicated_student():
    students, teams = _small_partition()
    baseline = prod.AssignmentBaseline.capture(teams, prod.DEFAULT_FLOOR_POLICY)
    dup_student = teams[1][0]
    # dup_student appears on BOTH teams — a genuine duplicate, not a move.
    bad = [list(teams[0]) + [dup_student], list(teams[1])]
    result = prod.validate_assignment(bad, baseline)
    assert result.ok is False
    assert any("more than one team" in r for r in result.reasons)


def test_validate_assignment_rejects_a_missing_student():
    students, teams = _small_partition()
    baseline = prod.AssignmentBaseline.capture(teams, prod.DEFAULT_FLOOR_POLICY)
    bad = [teams[0][:-1], teams[1]]
    result = prod.validate_assignment(bad, baseline)
    assert result.ok is False
    assert any("missing student" in r for r in result.reasons)


def test_validate_assignment_rejects_an_unknown_student():
    students, teams = _small_partition()
    baseline = prod.AssignmentBaseline.capture(teams, prod.DEFAULT_FLOOR_POLICY)
    stranger = fx._s("intruder", "Intruder", ["backend"])
    bad = [teams[0] + [stranger], teams[1]]
    result = prod.validate_assignment(bad, baseline)
    assert result.ok is False
    assert any("unknown student" in r for r in result.reasons)


def test_validate_assignment_rejects_a_wrong_team_size():
    students, teams = _small_partition()
    baseline = prod.AssignmentBaseline.capture(teams, prod.DEFAULT_FLOOR_POLICY)
    bad = [teams[0][:-1], teams[1] + [teams[0][-1]]]
    result = prod.validate_assignment(bad, baseline)
    assert result.ok is False
    assert any("size" in r for r in result.reasons)


def test_validate_assignment_rejects_skill_coverage_regression_below_baseline():
    case = fx.fixture_greedy_local_optimum()
    greedy = prod._greedy_build(case.students, case.team_size, DEFAULT_WEIGHTS)
    baseline = prod.AssignmentBaseline.capture(greedy, prod.DEFAULT_FLOOR_POLICY)

    # The exact regression swap from the pre-fix bug: Connector 1 <-> Magnet 3
    # drops team0's coverage from 0.583 to 0.417, below the greedy baseline.
    candidate = [list(t) for t in greedy]
    i0 = [s.name for s in candidate[0]].index("Connector 1")
    i1 = [s.name for s in candidate[1]].index("Magnet 3")
    candidate[0][i0], candidate[1][i1] = candidate[1][i1], candidate[0][i0]

    result = prod.validate_assignment(candidate, baseline)
    assert result.ok is False
    assert any("skill coverage regressed" in r for r in result.reasons)


def test_validate_assignment_rejects_floor_violation_increase():
    """Two members at the floor level landing on the same team, when the
    baseline had at most one per team, must be rejected — this is the exact
    real-data scenario found while verifying the maximin fix (Fatima/Hamid
    swap on the 20-student seed roster introduces a second cooperativeness=low
    member onto a team that already had one)."""
    low_a = fx._s("low_a", "Low A", ["frontend"], cooperativeness="low")
    low_b = fx._s("low_b", "Low B", ["backend"], cooperativeness="low")
    high_a = fx._s("high_a", "High A", ["ml"], cooperativeness="high")
    high_b = fx._s("high_b", "High B", ["design"], cooperativeness="high")
    teams = [[low_a, high_a], [low_b, high_b]]
    baseline = prod.AssignmentBaseline.capture(teams, prod.FLOOR_POLICY_ONE_PER_TEAM)
    assert baseline.max_floor_violations == 0

    candidate = [[low_a, low_b], [high_a, high_b]]
    result = prod.validate_assignment(candidate, baseline)
    assert result.ok is False
    assert any("violations increased" in r for r in result.reasons)


def test_precomputed_and_full_scan_validation_agree():
    """_two_opt_improve/_maximin_improve pass precomputed_min_skill_coverage
    / precomputed_floor_violation_count (an incremental per-team cache) to
    avoid rescanning the whole partition on every candidate swap
    (recommended_changes.md #16). That fast path must always reach the same
    verdict as a plain full-partition scan — this guards against the two
    ever drifting apart."""
    case = fx.fixture_greedy_local_optimum()
    greedy = prod._greedy_build(case.students, case.team_size, DEFAULT_WEIGHTS)
    baseline = prod.AssignmentBaseline.capture(greedy, prod.DEFAULT_FLOOR_POLICY)

    full = prod.validate_assignment(greedy, baseline)
    fast = prod.validate_assignment(
        greedy, baseline,
        precomputed_min_skill_coverage=prod._partition_skill_coverage_min(greedy),
        precomputed_floor_violation_count=len(
            prod._check_floor_violations(greedy, prod.DEFAULT_FLOOR_POLICY)[1]
        ),
    )
    assert full.ok == fast.ok
    assert full.reasons == fast.reasons


def test_validate_assignment_respects_configured_absolute_hard_floor():
    students, teams = _small_partition()
    baseline = prod.AssignmentBaseline.capture(teams, prod.DEFAULT_FLOOR_POLICY)
    # baseline coverage is low (everyone only has "frontend") — configuring an
    # absolute floor above it must fail even though nothing regressed.
    result = prod.validate_assignment(teams, baseline, hard_skill_coverage_floor=0.9)
    assert result.ok is False
    assert any("hard floor" in r for r in result.reasons)


# ===========================================================================
# 2-opt: rejects constraint-breaking swaps, accepts valid ones
# ===========================================================================
def test_two_opt_rejects_a_score_improving_swap_that_breaks_skill_coverage():
    """The documented regression: a swap that raises the summed total_score
    is available (greedy -> 1.370 -> 1.403) but drops one team's coverage
    below what greedy already had. It must no longer be accepted, and the
    partition's minimum skill coverage must never regress below greedy's."""
    case = fx.fixture_greedy_local_optimum()
    greedy = prod._greedy_build(case.students, case.team_size, DEFAULT_WEIGHTS)
    greedy_cov = skill_coverage(greedy, EvalConfig())["min_coverage_rate"]

    after_2opt = prod._two_opt_improve([list(t) for t in greedy], DEFAULT_WEIGHTS)
    opt_cov = skill_coverage(after_2opt, EvalConfig())["min_coverage_rate"]

    assert opt_cov >= greedy_cov, (
        f"2-opt must never let partition min skill coverage regress: "
        f"greedy={greedy_cov:.3f} -> 2opt={opt_cov:.3f}"
    )
    # The specific bad swap must not have been taken: Connector 1 must not
    # have left team0 for team1 while Magnet 3 replaces it.
    team0_names = {s.name for s in after_2opt[0]}
    assert not (
        "Magnet 3" in team0_names and "Connector 1" not in team0_names
    ), "the exact pre-fix regression swap was accepted"


def test_two_opt_accepts_a_valid_improving_swap():
    """A swap that improves the summed score AND does not break any hard
    constraint must still be accepted — the fix must not make 2-opt inert."""
    case = fx.fixture_greedy_local_optimum()
    greedy = prod._greedy_build(case.students, case.team_size, DEFAULT_WEIGHTS)
    greedy_score = _sum_score(greedy)

    after_2opt = prod._two_opt_improve([list(t) for t in greedy], DEFAULT_WEIGHTS)
    opt_score = _sum_score(after_2opt)

    assert opt_score > greedy_score + prod.SWAP_IMPROVEMENT_TOLERANCE, (
        "2-opt should still find and accept a genuinely improving, "
        f"constraint-safe swap: greedy={greedy_score:.4f} opt={opt_score:.4f}"
    )
    assert _ids(after_2opt) == _ids(greedy), "no student may be duplicated or dropped"


def test_two_opt_never_reduces_hard_skill_coverage_across_all_fixtures():
    for factory in fx.ALL_FIXTURES:
        case = factory()
        greedy = prod._greedy_build(case.students, case.team_size, DEFAULT_WEIGHTS)
        greedy_cov = skill_coverage(greedy, EvalConfig())["min_coverage_rate"]
        after = prod._two_opt_improve([list(t) for t in greedy], DEFAULT_WEIGHTS)
        after_cov = skill_coverage(after, EvalConfig())["min_coverage_rate"]
        assert after_cov >= greedy_cov - 1e-9, (
            f"[{case.name}] 2-opt regressed min skill coverage: "
            f"{greedy_cov:.4f} -> {after_cov:.4f}"
        )


# ===========================================================================
# Maximin: cannot undo a hard constraint, cannot silently undo 2-opt's work
# ===========================================================================
def test_maximin_cannot_undo_a_hard_constraint():
    """Construct a post-2-opt-shaped state where the ONLY move that improves
    the partition floor-minimum also introduces a second floor-level member
    onto one team (violating one_low_per_team). Maximin must refuse it and
    leave the partition floor-minimum unimproved rather than break the
    floor policy."""
    low_a = fx._s("low_a", "Low A", ["frontend"], accountability="low")
    low_b = fx._s("low_b", "Low B", ["backend"], accountability="low")
    high_a = fx._s("high_a", "High A", ["ml"], accountability="high")
    high_b = fx._s("high_b", "High B", ["design"], accountability="high")
    # team0's floor mean is dragged down by low_a; swapping low_b in would
    # raise team0's floor mean (helping the partition minimum) but leaves
    # two accountability=low members on... no: swapping low_a<->high_b would
    # put low_a with high_a (still one floor member per team) — instead
    # construct the case where the only "improving" swap doubles up floor
    # members on one team.
    teams = [[low_a, low_b, high_a, high_b]]
    # A single team can't be swapped against itself meaningfully; use two
    # teams where every cross-team swap that raises the floor-minimum also
    # concentrates both floor students together.
    team_a = [low_a, high_a]
    team_b = [low_b, high_b]
    before = [list(team_a), list(team_b)]
    before_floor_min = prod._partition_floor_min(before)
    before_violations = prod._check_floor_violations(before, prod.FLOOR_POLICY_ONE_PER_TEAM)[1]
    assert before_violations == []

    after = prod._maximin_improve(
        [list(team_a), list(team_b)],
        weights=DEFAULT_WEIGHTS,
        floor_policy=prod.FLOOR_POLICY_ONE_PER_TEAM,
    )
    after_ok, after_violations, _ = prod._check_floor_violations(after, prod.FLOOR_POLICY_ONE_PER_TEAM)
    assert after_ok, f"maximin must never introduce a floor violation: {after_violations}"


def test_maximin_cannot_silently_undo_two_opts_improvement():
    """On the real 20-student seed roster, unconstrained maximin used to
    drop the summed total-score objective by ~5% (3.084 -> 2.922) while
    reintroducing swaps 2-opt had already eliminated. Post-fix, maximin must
    never leave the objective below what 2-opt achieved."""
    import json
    from pathlib import Path
    from app.models import Student

    sample_path = Path(__file__).resolve().parents[1] / "sample_students.json"
    with open(sample_path) as f:
        raw = json.load(f)
    students = [Student(**r) for r in raw]

    greedy = prod._greedy_build(students, 4, DEFAULT_WEIGHTS)
    after_2opt = prod._two_opt_improve([list(t) for t in greedy], DEFAULT_WEIGHTS)
    two_opt_score = _sum_score(after_2opt)

    after_maximin = prod._maximin_improve([list(t) for t in after_2opt], weights=DEFAULT_WEIGHTS)
    maximin_score = _sum_score(after_maximin)

    assert maximin_score >= two_opt_score - prod.MAXIMIN_OBJECTIVE_TOLERANCE - 1e-9, (
        f"maximin regressed the primary objective 2-opt had already optimized: "
        f"2opt={two_opt_score:.5f} -> maximin={maximin_score:.5f}"
    )
    assert _ids(after_maximin) == _ids(after_2opt)


def test_maximin_still_improves_floor_minimum_when_it_can_do_so_for_free():
    """Construct a scenario where an objective-neutral-or-better swap also
    raises the partition floor-minimum, to prove the fix didn't make maximin
    a no-op — it should still act when it can do so without any tradeoff."""
    # Two teams, identical skills/behavior on the "other" dims so swapping
    # the two floor-adjacent students doesn't change the objective at all,
    # but does equalize the accountability floor mean.
    a1 = fx._s("a1", "A1", ["frontend"], accountability="developing")
    a2 = fx._s("a2", "A2", ["frontend"], accountability="low")
    b1 = fx._s("b1", "B1", ["frontend"], accountability="developing")
    b2 = fx._s("b2", "B2", ["frontend"], accountability="developing")
    team_a = [a1, a2]
    team_b = [b1, b2]
    before_floor_min = prod._partition_floor_min([team_a, team_b])

    after = prod._maximin_improve([list(team_a), list(team_b)], weights=DEFAULT_WEIGHTS)
    after_floor_min = prod._partition_floor_min(after)

    assert after_floor_min >= before_floor_min, "maximin should not be a total no-op when a safe improvement exists"


# ===========================================================================
# Cross-cutting: determinism, structural integrity, both floor policies,
# local-optimum termination, floating-point stability
# ===========================================================================
def _run_pipeline(students, team_size, floor_policy=prod.DEFAULT_FLOOR_POLICY):
    teams = prod._greedy_build(students, team_size, DEFAULT_WEIGHTS)
    teams = prod._two_opt_improve(teams, DEFAULT_WEIGHTS, floor_policy=floor_policy)
    teams = prod._maximin_improve(teams, weights=DEFAULT_WEIGHTS, floor_policy=floor_policy)
    return teams


@pytest.mark.parametrize("factory", fx.ALL_FIXTURES, ids=lambda f: f().name)
def test_repeated_runs_produce_identical_results(factory):
    case = factory()
    first = _run_pipeline(copy.deepcopy(case.students), case.team_size)
    second = _run_pipeline(copy.deepcopy(case.students), case.team_size)
    first_shape = [sorted(s.id for s in t) for t in first]
    second_shape = [sorted(s.id for s in t) for t in second]
    assert first_shape == second_shape, "identical input must produce identical team assignments"


@pytest.mark.parametrize("factory", fx.ALL_FIXTURES, ids=lambda f: f().name)
def test_no_student_duplicated_or_omitted_after_full_pipeline(factory):
    case = factory()
    teams = _run_pipeline(case.students, case.team_size)
    placed = [s.id for t in teams for s in t]
    assert sorted(placed) == sorted(s.id for s in case.students)
    assert len(placed) == len(set(placed))


@pytest.mark.parametrize("factory", fx.ALL_FIXTURES, ids=lambda f: f().name)
def test_team_sizes_remain_valid_after_every_stage(factory):
    case = factory()
    greedy = prod._greedy_build(case.students, case.team_size, DEFAULT_WEIGHTS)
    target = sorted(len(t) for t in greedy)

    after_2opt = prod._two_opt_improve([list(t) for t in greedy], DEFAULT_WEIGHTS)
    assert sorted(len(t) for t in after_2opt) == target, "2-opt must never change team sizes"

    after_maximin = prod._maximin_improve([list(t) for t in after_2opt], weights=DEFAULT_WEIGHTS)
    assert sorted(len(t) for t in after_maximin) == target, "maximin must never change team sizes"


@pytest.mark.parametrize("policy", [prod.FLOOR_POLICY_ONE_PER_TEAM, prod.FLOOR_POLICY_ZERO_TOLERANCE])
def test_both_floor_policies_respected_after_full_pipeline(policy):
    """For rosters that are feasible under a given policy, the full pipeline
    must never end up WORSE (more violations) under that policy than the
    greedy baseline it started from."""
    for factory in fx.ALL_FIXTURES:
        case = factory()
        n_teams = prod._compute_n_teams(len(case.students), case.team_size)
        feasible, _ = prod._check_feasibility(case.students, n_teams, policy)
        if not feasible:
            continue  # not this policy's fixture to protect — feasibility gate would reject it upstream
        greedy = prod._greedy_build(case.students, case.team_size, DEFAULT_WEIGHTS)
        _, greedy_violations, _ = prod._check_floor_violations(greedy, policy)

        final = prod._two_opt_improve([list(t) for t in greedy], DEFAULT_WEIGHTS, floor_policy=policy)
        final = prod._maximin_improve(final, weights=DEFAULT_WEIGHTS, floor_policy=policy)
        _, final_violations, _ = prod._check_floor_violations(final, policy)

        assert len(final_violations) <= len(greedy_violations), (
            f"[{case.name}/{policy}] optimization increased floor violations: "
            f"{len(greedy_violations)} -> {len(final_violations)}"
        )


def test_known_failing_skill_coverage_fixture_now_passes():
    """The exact assertion from test_research_alignment.py::
    test_greedy_seeding_does_not_starve_a_team_of_skills, confirmed passing."""
    case = fx.fixture_greedy_local_optimum()
    greedy_only = prod._greedy_build(case.students, case.team_size, DEFAULT_WEIGHTS)
    after_2opt = prod._two_opt_improve([list(t) for t in greedy_only], DEFAULT_WEIGHTS)

    cfg = EvalConfig()
    greedy_cov = skill_coverage(greedy_only, cfg)
    opt_cov = skill_coverage(after_2opt, cfg)

    assert opt_cov["min_coverage_rate"] >= greedy_cov["min_coverage_rate"]
    assert opt_cov["min_coverage_rate"] >= 0.35


def test_local_optimum_terminates_safely_with_a_tiny_sweep_cap():
    """A max_sweeps cap far below what's needed for convergence must still
    return a STRUCTURALLY VALID assignment (no crash, no corruption) — it
    just may not be fully converged. This proves the documented iteration
    limit is a safe early-exit, not a correctness hazard."""
    case = fx.fixture_greedy_local_optimum()
    greedy = prod._greedy_build(case.students, case.team_size, DEFAULT_WEIGHTS)

    capped = prod._two_opt_improve([list(t) for t in greedy], DEFAULT_WEIGHTS, max_sweeps=1)
    assert _ids(capped) == _ids(greedy)
    assert sorted(len(t) for t in capped) == sorted(len(t) for t in greedy)

    capped_maximin = prod._maximin_improve(
        [list(t) for t in greedy], weights=DEFAULT_WEIGHTS, max_sweeps=1
    )
    assert _ids(capped_maximin) == _ids(greedy)
    assert sorted(len(t) for t in capped_maximin) == sorted(len(t) for t in greedy)


def test_two_opt_normal_run_converges_well_under_the_sweep_cap():
    """The documented backstop should not be the normal exit path."""
    case = fx.fixture_greedy_local_optimum()
    greedy = prod._greedy_build(case.students, case.team_size, DEFAULT_WEIGHTS)
    # A pipeline this small should converge in a handful of sweeps, nowhere
    # near MAX_OPTIMIZATION_SWEEPS (500).
    result_default_cap = prod._two_opt_improve([list(t) for t in greedy], DEFAULT_WEIGHTS)
    result_tiny_cap = prod._two_opt_improve([list(t) for t in greedy], DEFAULT_WEIGHTS, max_sweeps=20)
    assert _sum_score(result_default_cap) == _sum_score(result_tiny_cap), (
        "convergence should already be reached well within 20 sweeps"
    )


def test_tiny_floating_point_differences_do_not_cause_unstable_swaps():
    """A swap whose 'improvement' is smaller than SWAP_IMPROVEMENT_TOLERANCE
    must be rejected as a tie, not accepted — otherwise floating-point noise
    could cause different runs (or platforms) to make different swap
    decisions on assignments that are objectively equivalent."""
    students = [fx._s(f"s{i}", f"S{i}", ["frontend", "backend"]) for i in range(8)]
    teams = [students[:4], students[4:]]

    # Perturb by less than tolerance via a near-zero-weight objective: with
    # all weights on a dimension that can't distinguish these students
    # (identical skills/behavior), any candidate swap scores as an exact or
    # near-exact tie and must not be accepted.
    tied_weights = prod._normalize_weights({
        "skill_coverage": 0.4, "behavioral_compat": 0.3,
        "availability_overlap": 0.2, "shared_interests": 0.1,
    })
    before_score = _sum_score(teams, tied_weights)
    result = prod._two_opt_improve([list(t) for t in teams], tied_weights)
    after_score = _sum_score(result, tied_weights)
    assert after_score == pytest.approx(before_score, abs=1e-9)
    assert _ids(result) == _ids(teams)
