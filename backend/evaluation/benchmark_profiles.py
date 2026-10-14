"""
Part 6.5 Section 6 — deterministic, repeated-timing performance benchmark.

Measures every named optimization profile at roster sizes 20/40/60/80/100
(deterministic synthetic data, same generator as evaluation/run_report.py's
profile_comparison), 3 repeats each, reporting median wall-clock (NOT
cProfile-instrumented — profiling overhead is not representative of real
runtime). Writes evaluation_output/performance_benchmark.json for
docs/performance.md and research_alignment_results.json to cite.

Run with:
    cd backend && <python> -m evaluation.benchmark_profiles
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from app import matching as prod
from evaluation import fixtures as fx

REPO_ROOT = Path(__file__).resolve().parents[2]
SIZES = [20, 40, 60, 80, 100]
TEAM_SIZE = 4
REPEATS = 3
PROFILES = ["legacy", "research_balanced", "fairness_priority", "skill_priority", "preference_priority"]


def _synthetic_roster(n: int):
    return [
        fx._s(f"bench{i}", f"Bench{i}", [["frontend", "backend", "design", "testing", "ml"][i % 5]],
              experience_level=["beginner", "intermediate", "advanced"][i % 3])
        for i in range(n)
    ]


def _run_once(students, profile_name: str) -> float:
    request, _ = prod.resolve_profile(prod.TeamGenerationRequest(team_size=TEAM_SIZE, profile=profile_name))
    weights = prod._normalize_weights(request.weights)
    t0 = time.perf_counter()
    teams = prod._greedy_build(students, TEAM_SIZE, weights)
    teams = prod._two_opt_improve(teams, weights, floor_policy=request.floor_policy)
    teams = prod._maximin_improve(teams, weights=weights, floor_policy=request.floor_policy)
    teams, _trace = prod._run_optional_passes(
        teams, request, weights, request.floor_policy, [], [],
        primary_metric=prod.OPTIMIZATION_PROFILES[profile_name]["primary_metric"],
    )
    return time.perf_counter() - t0


def main():
    results = {}
    for n in SIZES:
        students = _synthetic_roster(n)
        results[n] = {}
        for profile_name in PROFILES:
            timings = [round(_run_once(students, profile_name) * 1000, 3) for _ in range(REPEATS)]
            results[n][profile_name] = {
                "repeats_ms": timings,
                "median_ms": round(statistics.median(timings), 3),
                "mean_ms": round(statistics.fmean(timings), 3),
            }
            print(f"n={n:>3} profile={profile_name:<20} median={results[n][profile_name]['median_ms']:>10.1f}ms "
                  f"repeats={timings}")

    out_dir = REPO_ROOT / "backend" / "evaluation_output"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "performance_benchmark.json"
    with open(out_path, "w") as f:
        json.dump({
            "note": (
                "Median of 3 repeats per (roster size, profile), deterministic synthetic "
                "rosters (evaluation.benchmark_profiles._synthetic_roster), wall-clock via "
                "time.perf_counter() around _greedy_build -> _two_opt_improve -> "
                "_maximin_improve -> _run_optional_passes (the exact sequence generate_teams() "
                "calls). NOT cProfile-instrumented (profiling adds its own overhead)."
            ),
            "sizes": SIZES,
            "team_size": TEAM_SIZE,
            "repeats": REPEATS,
            "results": results,
        }, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
