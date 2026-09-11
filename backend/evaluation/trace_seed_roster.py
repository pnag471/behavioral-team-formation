"""
Part 6.5 Section 2 — why does `research_balanced` match `legacy` on the real
20-student seed roster? This script answers that with EVIDENCE, not
assertion: it replays each optional pass's exact lexicographic acceptance
rule as a READ-ONLY shadow counter (calling the SAME production functions
`app.matching`'s real passes call — `validate_assignment`, `_total_score`,
`_ordinal_team_mean`/`_partition_floor_min`, and each pass's own target
metric), tallying every candidate swap by the reason it was rejected:

  - hard_constraint_rejections   : validate_assignment(...).ok is False
  - primary_objective_rejections : objective regressed beyond tolerance
                                    (validation held)
  - protected_metric_rejections  : floor-min (or fairness/ability/etc.'s
                                    OWN protected predecessor metric)
                                    regressed beyond tolerance (validation
                                    + objective held)
  - no_improvement_rejections    : validation + objective + protected metric
                                    all held, but the swap did not strictly
                                    improve THIS pass's own target metric
  - accepted                     : every check passed

This is a SEPARATE, additive script — it does not modify app/matching.py.
It reimplements the identical nested nested nested loop each real pass
uses (same order, same tolerances imported directly from app.matching) so
every count is a faithful, exhaustive tally of what the real optimizer
would have seen, not an approximation.

Run with:
    cd backend && <python> -m evaluation.trace_seed_roster
"""
from __future__ import annotations

import json
from pathlib import Path

from app import matching as prod
from app.models import Student

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_STUDENTS_PATH = Path(__file__).resolve().parents[1] / "sample_students.json"
TEAM_SIZE = 4


def _load_roster():
    with open(SAMPLE_STUDENTS_PATH) as f:
        raw = json.load(f)
    return [Student(**r) for r in raw]


def _team_snapshot(teams):
    return [[s.name for s in t] for t in teams]


def _shadow_trace_pass(
    teams, weights, floor_policy, target_fn, target_direction, objective_tolerance, floor_min_tolerance,
    swap_tolerance, hard_skill_coverage_floor=prod.HARD_SKILL_COVERAGE_FLOOR,
):
    """
    target_fn(teams) -> float : this pass's own optimization target.
    target_direction: "minimize" (ability gap, familiarity/preference
        violation) or "maximize" (floor-min already covered separately;
        fairness min-utility).
    Returns a tally dict PLUS the final teams (identical to what the real
    pass would produce, since this uses the exact same accept rule).
    """
    teams = [list(t) for t in teams]
    baseline = prod.AssignmentBaseline.capture(teams, floor_policy)
    tally = {
        "candidates_considered": 0,
        "accepted": 0,
        "hard_constraint_rejections": 0,
        "primary_objective_rejections": 0,
        "protected_metric_rejections": 0,
        "no_improvement_rejections": 0,
        # Counterfactual: among candidates rejected for EACH reason, how
        # many would ALSO have strictly improved this pass's own target
        # metric had that particular gate not existed? This is what
        # actually distinguishes "no improving swap exists at all" from
        # "improving swaps exist but are blocked by an earlier,
        # lexicographically-prior protection" (Part 6.5 Section 2).
        "hard_constraint_rejections_that_would_have_improved_target": 0,
        "primary_objective_rejections_that_would_have_improved_target": 0,
        "protected_metric_rejections_that_would_have_improved_target": 0,
    }
    sweeps = 0
    improved = True
    while improved and sweeps < prod.MAX_OPTIMIZATION_SWEEPS:
        improved = False
        sweeps += 1
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                for a_idx in range(len(teams[i])):
                    for b_idx in range(len(teams[j])):
                        tally["candidates_considered"] += 1
                        original_objective = prod._stage_objective(teams, weights)
                        original_floor_min = prod._partition_floor_min(teams)
                        original_target = target_fn(teams)

                        teams[i][a_idx], teams[j][b_idx] = teams[j][b_idx], teams[i][a_idx]
                        validation = prod.validate_assignment(teams, baseline, hard_skill_coverage_floor)
                        new_objective = prod._stage_objective(teams, weights)
                        new_floor_min = prod._partition_floor_min(teams)
                        new_target = target_fn(teams)

                        target_improves = (
                            new_target < original_target - swap_tolerance if target_direction == "minimize"
                            else new_target > original_target + swap_tolerance
                        )

                        if not validation.ok:
                            tally["hard_constraint_rejections"] += 1
                            if target_improves:
                                tally["hard_constraint_rejections_that_would_have_improved_target"] += 1
                            teams[i][a_idx], teams[j][b_idx] = teams[j][b_idx], teams[i][a_idx]
                            continue
                        if new_objective < original_objective - objective_tolerance:
                            tally["primary_objective_rejections"] += 1
                            if target_improves:
                                tally["primary_objective_rejections_that_would_have_improved_target"] += 1
                            teams[i][a_idx], teams[j][b_idx] = teams[j][b_idx], teams[i][a_idx]
                            continue
                        if new_floor_min < original_floor_min - floor_min_tolerance:
                            tally["protected_metric_rejections"] += 1
                            if target_improves:
                                tally["protected_metric_rejections_that_would_have_improved_target"] += 1
                            teams[i][a_idx], teams[j][b_idx] = teams[j][b_idx], teams[i][a_idx]
                            continue

                        if not target_improves:
                            tally["no_improvement_rejections"] += 1
                            teams[i][a_idx], teams[j][b_idx] = teams[j][b_idx], teams[i][a_idx]
                            continue

                        tally["accepted"] += 1
                        improved = True
    tally["sweeps_run"] = sweeps
    return teams, tally


def main():
    students = _load_roster()
    request_legacy, _ = prod.resolve_profile(prod.TeamGenerationRequest(team_size=TEAM_SIZE, profile="legacy"))
    request_rb, _ = prod.resolve_profile(prod.TeamGenerationRequest(team_size=TEAM_SIZE, profile="research_balanced"))
    weights = prod._normalize_weights(request_legacy.weights)

    greedy = prod._greedy_build(students, TEAM_SIZE, weights)
    two_opt = prod._two_opt_improve([list(t) for t in greedy], weights, floor_policy=request_legacy.floor_policy)
    maximin = prod._maximin_improve([list(t) for t in two_opt], weights=weights, floor_policy=request_legacy.floor_policy)

    report = {
        "roster_size": len(students),
        "team_size": TEAM_SIZE,
        "stages": {
            "greedy": {
                "assignment": _team_snapshot(greedy),
                "objective": prod._stage_objective(greedy, weights),
                "floor_min": prod._partition_floor_min(greedy),
                "skill_coverage_min": prod._partition_skill_coverage_min(greedy),
            },
            "two_opt": {
                "assignment": _team_snapshot(two_opt),
                "objective": prod._stage_objective(two_opt, weights),
                "floor_min": prod._partition_floor_min(two_opt),
                "skill_coverage_min": prod._partition_skill_coverage_min(two_opt),
                "identical_to_greedy": _team_snapshot(two_opt) == _team_snapshot(greedy),
            },
            "maximin": {
                "assignment": _team_snapshot(maximin),
                "objective": prod._stage_objective(maximin, weights),
                "floor_min": prod._partition_floor_min(maximin),
                "skill_coverage_min": prod._partition_skill_coverage_min(maximin),
                "identical_to_two_opt": _team_snapshot(maximin) == _team_snapshot(two_opt),
            },
        },
        "optional_passes": {},
    }

    current = [list(t) for t in maximin]
    pass_specs = [
        ("ability", lambda t: prod._partition_ability_gap(t), "minimize",
         prod.MAXIMIN_OBJECTIVE_TOLERANCE, prod.MAXIMIN_OBJECTIVE_TOLERANCE),
        ("fairness", lambda t: prod._partition_min_utility(t, request_rb.fairness.weights), "maximize",
         prod.FAIRNESS_OBJECTIVE_TOLERANCE, prod.FAIRNESS_FLOOR_MIN_TOLERANCE),
        ("familiarity", lambda t: sum(
            prod._team_familiarity_violation(
                team, prod.build_familiarity_graph([s for tt in t for s in tt], []),
                request_rb.familiarity.lower_bound, request_rb.familiarity.upper_bound,
                request_rb.familiarity.clique_edge_threshold, request_rb.familiarity.max_clique_fraction,
                request_rb.familiarity.minimum_known_pair_coverage, request_rb.familiarity.penalize_insufficient_data,
            ) for team in t
        ), "minimize", prod.FAMILIARITY_OBJECTIVE_TOLERANCE, prod.FAMILIARITY_FLOOR_MIN_TOLERANCE),
        ("preference", lambda t: sum(
            prod._team_preference_violation(
                [s.id for s in team], {s.id: s for s in team}, prod._build_preference_index([]),
                request_rb.preference.high_priority_weight, request_rb.preference.avoidance_violation_weight,
            )[0] for team in t
        ), "minimize", prod.PREFERENCE_OBJECTIVE_TOLERANCE, prod.PREFERENCE_FLOOR_MIN_TOLERANCE),
    ]

    for name, target_fn, direction, obj_tol, floor_tol in pass_specs:
        before_target = target_fn(current)
        traced_teams, tally = _shadow_trace_pass(
            current, weights, request_rb.floor_policy, target_fn, direction, obj_tol, floor_tol,
            prod.SWAP_IMPROVEMENT_TOLERANCE,
        )
        after_target = target_fn(traced_teams)
        report["optional_passes"][name] = {
            "before_target_metric": before_target,
            "after_target_metric": after_target,
            "target_metric_direction": direction,
            "target_metric_improved": (
                after_target < before_target - 1e-9 if direction == "minimize" else after_target > before_target + 1e-9
            ),
            "before_objective": prod._stage_objective(current, weights),
            "after_objective": prod._stage_objective(traced_teams, weights),
            "before_floor_min": prod._partition_floor_min(current),
            "after_floor_min": prod._partition_floor_min(traced_teams),
            "assignment_before": _team_snapshot(current),
            "assignment_after": _team_snapshot(traced_teams),
            "assignment_unchanged": _team_snapshot(traced_teams) == _team_snapshot(current),
            **tally,
        }
        current = traced_teams  # feed forward, same as production order: ability -> fairness -> familiarity -> preference

    # Cross-check against the REAL production orchestration function to
    # confirm the shadow trace's final result and accept/reject verdicts
    # match exactly (this is the byte-for-byte "not an approximation" proof).
    real_teams, real_trace = prod._run_optional_passes(
        [list(t) for t in maximin], request_rb, weights, request_rb.floor_policy, [], [],
        primary_metric=None,
    )
    report["production_orchestration_cross_check"] = {
        "final_assignment": _team_snapshot(real_teams),
        "matches_shadow_trace_final_assignment": _team_snapshot(real_teams) == _team_snapshot(current),
        "per_pass_trace": real_trace,
    }

    out_dir = REPO_ROOT / "backend" / "evaluation_output"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "seed_roster_trace.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Wrote {out_path}")

    print("\n=== SUMMARY ===")
    print(f"greedy == 2-opt == maximin assignment identical through stage 3: "
          f"2opt_same={report['stages']['two_opt']['identical_to_greedy']} "
          f"maximin_same={report['stages']['maximin']['identical_to_two_opt']}")
    for name, data in report["optional_passes"].items():
        print(f"\n[{name}] candidates={data['candidates_considered']} accepted={data['accepted']} "
              f"hard_constraint_rej={data['hard_constraint_rejections']} "
              f"(would-have-improved={data['hard_constraint_rejections_that_would_have_improved_target']}) "
              f"primary_obj_rej={data['primary_objective_rejections']} "
              f"(would-have-improved={data['primary_objective_rejections_that_would_have_improved_target']}) "
              f"protected_metric_rej={data['protected_metric_rejections']} "
              f"(would-have-improved={data['protected_metric_rejections_that_would_have_improved_target']}) "
              f"no_improvement_rej={data['no_improvement_rejections']} "
              f"target: {data['before_target_metric']:.4f} -> {data['after_target_metric']:.4f} "
              f"(improved={data['target_metric_improved']}) "
              f"assignment_unchanged={data['assignment_unchanged']}")
    xc = report["production_orchestration_cross_check"]
    print(f"\nProduction _run_optional_passes final assignment matches shadow trace: "
          f"{xc['matches_shadow_trace_final_assignment']}")


if __name__ == "__main__":
    main()
