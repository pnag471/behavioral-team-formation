"""
Generates research_alignment_results.json (Phase 5 deliverable).

Runs the production greedy+2opt+maximin pipeline (app/matching.py) against:
  1. The real seed roster (sample_students.json, 20 students)
  2. Each of the 12 adversarial fixtures (evaluation/fixtures.py)
and records the full evaluation.metrics.evaluate_teams() output for each,
plus a runtime/scalability sweep and the pytest pass/fail tally for
tests/test_research_alignment.py.

This script performs no writes to the application database and does not
import anything from app.main (so it never triggers DB seeding). It is
read-only w.r.t. production code.

Run with:
    cd backend && <python> -m evaluation.run_report
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from app import matching as prod
from app.models import FamiliarityEdge, Student, TeamGenerationRequest
from evaluation import fixtures as fx
from evaluation.metrics import EvalConfig, evaluate_teams, measure_runtime, legacy_preferences_to_entries

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_STUDENTS_PATH = Path(__file__).resolve().parents[1] / "sample_students.json"

DEFAULT_WEIGHTS = prod._normalize_weights({
    "skill_coverage": 0.4, "behavioral_compat": 0.3,
    "availability_overlap": 0.2, "shared_interests": 0.1,
})


def _run_behavioral_pipeline(students, team_size):
    teams = prod._greedy_build(students, team_size, DEFAULT_WEIGHTS)
    teams = prod._two_opt_improve(teams, DEFAULT_WEIGHTS)
    teams = prod._maximin_improve(teams)
    return teams


def _team_summary(teams):
    return [
        {"team_index": i, "members": [s.name for s in team], "size": len(team)}
        for i, team in enumerate(teams)
    ]


def run_sample_students() -> dict:
    with open(SAMPLE_STUDENTS_PATH) as f:
        raw = json.load(f)
    students = [Student(**r) for r in raw]
    team_size = 4
    t0 = time.perf_counter()
    teams = _run_behavioral_pipeline(students, team_size)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
    result = evaluate_teams(teams, weights=DEFAULT_WEIGHTS)
    return {
        "n_students": len(students),
        "team_size": team_size,
        "teams": _team_summary(teams),
        "pipeline_wall_clock_ms": elapsed_ms,
        "metrics": result,
    }


def run_fixture(factory) -> dict:
    case = factory()
    teams = _run_behavioral_pipeline(case.students, case.team_size)
    result = evaluate_teams(
        teams,
        weights=DEFAULT_WEIGHTS,
        familiarity=case.familiarity,
        friend_groups=case.friend_groups,
        preferences=case.preferences,
    )
    return {
        "name": case.name,
        "description": case.description,
        "n_students": len(case.students),
        "team_size": case.team_size,
        "teams": _team_summary(teams),
        "metrics": result,
    }


def run_scalability_sweep() -> list:
    from evaluation.fixtures import _s
    sizes = [8, 16, 32, 64]
    out = []
    for n in sizes:
        students = [_s(f"scale{i}", f"Scale {i}", ["frontend"]) for i in range(n)]
        out.append(measure_runtime(students, 4, DEFAULT_WEIGHTS))
    return out


def run_pytest_tally() -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_research_alignment.py", "-v", "--tb=no"],
        cwd=str(REPO_ROOT / "backend"),
        capture_output=True,
        text=True,
    )
    lines = proc.stdout.splitlines()
    per_test = []
    for line in lines:
        line = line.strip()
        if line.startswith("tests/test_research_alignment.py::"):
            parts = line.rsplit(" ", 1)
            if len(parts) == 2:
                name, status = parts
                per_test.append({"test": name.split("::", 1)[1], "status": status.strip("[]% 0123456789")})
    summary_line = next((l for l in lines if "passed" in l or "failed" in l or "error" in l), "")
    return {
        "command": "python -m pytest tests/test_research_alignment.py -v --tb=no",
        "exit_code": proc.returncode,
        "summary_line": summary_line.strip(),
        "per_test": per_test,
    }


def run_existing_suite_tally() -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_assessment.py", "tests/test_dimension_scorer.py", "-v", "--tb=no"],
        cwd=str(REPO_ROOT / "backend"),
        capture_output=True,
        text=True,
    )
    lines = proc.stdout.splitlines()
    summary_line = next((l for l in lines if "passed" in l or "failed" in l or "error" in l), "")
    return {
        "command": "python -m pytest tests/test_assessment.py tests/test_dimension_scorer.py -v --tb=no",
        "exit_code": proc.returncode,
        "summary_line": summary_line.strip(),
    }


# ---------------------------------------------------------------------------
# Part 6 — empirical profile comparison (Section 9 of the Part 6 spec).
# Runs every named profile (app.matching.OPTIMIZATION_PROFILES) against
# the real seed roster, every adversarial fixture, and several synthetic
# rosters, reporting the exact metric set the spec requires per run. This
# performs no writes to the application database and imports nothing from
# app.main.
# ---------------------------------------------------------------------------

def _synthetic_roster(n: int):
    return [
        fx._s(f"syn{i}", f"Syn{i}", [["frontend", "backend", "design", "testing", "ml"][i % 5]],
              experience_level=["beginner", "intermediate", "advanced"][i % 3])
        for i in range(n)
    ]


def _run_profile_and_summarize(students, team_size, profile_name, familiarity_edges=None, preference_entries=None):
    familiarity_edges = familiarity_edges or []
    preference_entries = preference_entries or []

    t0 = time.perf_counter()
    request, resolution = prod.resolve_profile(TeamGenerationRequest(team_size=team_size, profile=profile_name))
    weights = prod._normalize_weights(request.weights)
    teams = prod._greedy_build(students, team_size, weights)
    teams = prod._two_opt_improve(teams, weights, floor_policy=request.floor_policy)
    teams = prod._maximin_improve(teams, weights=weights, floor_policy=request.floor_policy)
    teams, trace = prod._run_optional_passes(
        teams, request, weights, request.floor_policy, familiarity_edges, preference_entries,
        primary_metric=prod.OPTIMIZATION_PROFILES[profile_name]["primary_metric"],
    )
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)

    baseline = prod.AssignmentBaseline.capture(
        prod._greedy_build(students, team_size, weights), request.floor_policy,
    )
    validation = prod.validate_assignment(teams, baseline)

    skill_report = prod._partition_skill_coverage_min(teams)
    _, floor_violations, _ = prod._check_floor_violations(teams, request.floor_policy)
    ability_report = prod.compute_ability_metrics(teams, request.ability.ability_level_thresholds)
    fam_report = prod.compute_familiarity_metrics(
        teams, familiarity_edges,
        familiar_pair_threshold=request.familiarity.familiar_pair_threshold,
        clique_edge_threshold=request.familiarity.clique_edge_threshold,
        lower_bound=request.familiarity.lower_bound, upper_bound=request.familiarity.upper_bound,
        minimum_known_pair_coverage=request.familiarity.minimum_known_pair_coverage,
        quasi_clique_density_threshold=request.familiarity.quasi_clique_density_threshold,
    )
    utility_report = prod.compute_assignment_utilities(teams, utility_weights=request.fairness.weights)
    pref_report = prod.compute_preference_metrics(teams, preference_entries)
    stability_report = prod.compute_stability_metrics(teams, preference_entries, baseline=baseline)
    objective = prod.compute_normalized_objective(
        teams, familiarity_edges=familiarity_edges, preference_entries=preference_entries,
        ability_level_thresholds=request.ability.ability_level_thresholds,
        familiarity_config=request.familiarity, utility_weights=request.fairness.weights,
    )
    largest_cluster = max(
        (t["largest_connected_cluster"]["fraction_of_team"] for t in fam_report["per_team"] if t["total_possible_pairs"] > 0),
        default=None,
    )

    return {
        "profile": profile_name,
        "is_experimental": resolution.is_experimental,
        "hard_constraint_violations": validation.reasons,
        "hard_constraints_ok": validation.ok,
        "overall_objective": objective["total_weighted_objective"],
        "skill_coverage_min": skill_report,
        "behavioral_floor_violations": len(floor_violations),
        "ability_gap": ability_report["strongest_weakest_gap"],
        "familiarity_teams_within_band": fam_report["teams_within_band"],
        "familiarity_teams_insufficient_data": fam_report["teams_insufficient_data"],
        "familiarity_evidence_coverage": (
            (fam_report["teams_within_band"] + fam_report["teams_below_band"] + fam_report["teams_above_band"]) / len(teams)
            if teams else None
        ),
        "largest_connected_familiarity_cluster_fraction": largest_cluster,
        "min_utility": utility_report["summary"]["min_utility"],
        "mean_utility": utility_report["summary"]["mean_utility"],
        "utility_variance": utility_report["summary"]["utility_variance"],
        "worst_case_regret": utility_report["summary"]["worst_case_regret"],
        "mean_preference_satisfaction": pref_report["mean_satisfaction"],
        "total_avoidance_violations": pref_report["total_avoidance_violations"],
        "n_blocking_pairs": stability_report["n_blocking_pairs"],
        "n_blocking_swaps": stability_report["n_blocking_swaps"],
        "runtime_ms": elapsed_ms,
        "optimization_trace_length": len(trace),
    }


def run_profile_comparison() -> dict:
    profiles = list(prod.OPTIMIZATION_PROFILES.keys())
    results = {"seed_roster": {}, "adversarial_fixtures": {}, "synthetic_rosters": {}}

    with open(SAMPLE_STUDENTS_PATH) as f:
        raw = json.load(f)
    seed_students = [Student(**r) for r in raw]
    results["seed_roster"] = {p: _run_profile_and_summarize(seed_students, 4, p) for p in profiles}

    for factory in fx.ALL_FIXTURES:
        case = factory()
        fam_edges = [FamiliarityEdge(student_a_id=a, student_b_id=b, strength=s) for (a, b), s in (case.familiarity or {}).items()]
        pref_entries = legacy_preferences_to_entries(case.preferences)
        results["adversarial_fixtures"][case.name] = {
            p: _run_profile_and_summarize(case.students, case.team_size, p, fam_edges, pref_entries) for p in profiles
        }

    for n in (20, 40, 80):
        students = _synthetic_roster(n)
        results["synthetic_rosters"][f"n={n}"] = {p: _run_profile_and_summarize(students, 4, p) for p in profiles}

    return results


def main():
    report = {
        "generated_by": "backend/evaluation/run_report.py",
        "note": (
            "All 'teams' fields below were produced by calling the UNMODIFIED production "
            "functions app.matching._greedy_build -> _two_opt_improve -> _maximin_improve "
            "(the same sequence app/matching.py:601-603 uses for POST /teams/generate in "
            "'behavioral' mode) against each dataset."
        ),
        "existing_test_suite": run_existing_suite_tally(),
        "research_alignment_test_suite": run_pytest_tally(),
        "sample_students_evaluation": run_sample_students(),
        "adversarial_fixtures": [run_fixture(f) for f in fx.ALL_FIXTURES],
        "scalability_sweep": run_scalability_sweep(),
        "profile_comparison": run_profile_comparison(),
    }
    out_path = REPO_ROOT / "research_alignment_results.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
