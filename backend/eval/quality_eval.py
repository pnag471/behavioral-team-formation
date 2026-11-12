"""
Team-quality evaluation (offline, deterministic, no LLM).

Compares team-formation methods on the same synthetic cohorts and reports
balance and constraint metrics, mean and 95% bootstrap CI over seeds.

Same arguments give byte-identical output files, regardless of --workers.

Usage (from backend/):
    python3 -m eval.quality_eval --sizes 20 22 40 43 80 --seeds 30 --out eval/results
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from app import matching as prod
from app.models import Student, TeamGenerationRequest
from eval.cohort import record_to_student
from eval.generate_cohort import generate_cohort
from evaluation import metrics as em
from scoring.dimension_scorer import get_rubric

TEAM_SIZE = 4
PROFILE = "legacy"
FLOOR_POLICY = "one_low_per_team"
LOW_MODE = "partial"
PRIMARY_FRACTION = 0.15
SECONDARY_FRACTIONS = (0.0, 0.25)
FLOOR_TRAITS = ("accountability", "cooperativeness")
BOOTSTRAP_SEED = 20261112
BOOTSTRAP_RESAMPLES = 2000
DECIMALS = 6

BEHAVIOR_ONLY_WEIGHTS = {
    "skill_coverage": 0.0,
    "behavioral_compat": 0.3,
    "availability_overlap": 0.2,
    "shared_interests": 0.1,
}

METHODS = (
    "ours",
    "ours_no_floor",
    "ours_no_maximin",
    "random",
    "skills_only",
    "behavior_only",
    "sort_and_chunk",
)

METHOD_DESCRIPTIONS = {
    "ours": "form_teams as is (legacy profile, one_low_per_team)",
    "ours_no_floor": "form_teams with the floor check disabled (no floor spreading or one-per-team rule; maximin still runs)",
    "ours_no_maximin": "form_teams sequence without the _maximin_improve pass",
    "random": "seeded shuffle, chunked into the same balanced team sizes as ours",
    "skills_only": "production formation_mode 'skill' (skill_based_assignment)",
    "behavior_only": "form_teams with the skill_coverage weight set to 0",
    "sort_and_chunk": "sort by behavior score (sum of floor-trait levels), cut into consecutive teams",
}

METRICS = (
    "teams_with_floor_student",
    "max_floor_students_on_one_team",
    "max_same_trait_floor_on_one_team",
    "accountability_min_team_mean",
    "accountability_mean_of_team_means",
    "cooperativeness_min_team_mean",
    "cooperativeness_mean_of_team_means",
    "team_score_std",
    "worst_team_score",
    "skill_coverage_rate",
    "mean_coverage",
    "worst_off_utility",
)

TABLE_COLUMNS = (
    ("teams_with_floor_student", "teams w/ floor student"),
    ("max_floor_students_on_one_team", "max floor/team"),
    ("max_same_trait_floor_on_one_team", "max same-trait floor/team"),
    ("accountability_min_team_mean", "acc min team mean"),
    ("cooperativeness_min_team_mean", "coop min team mean"),
    ("accountability_mean_of_team_means", "acc mean of means"),
    ("cooperativeness_mean_of_team_means", "coop mean of means"),
    ("worst_team_score", "worst team score"),
    ("team_score_std", "team score std"),
    ("mean_coverage", "mean skill coverage"),
    ("skill_coverage_rate", "full-coverage teams"),
    ("worst_off_utility", "worst-off utility"),
)

HEADER_NOTES = (
    "Ordinal trait levels are mapped to 0, 1, 2 (low, developing, high), evenly spaced.",
    f"Team size {TEAM_SIZE}, profile '{PROFILE}', floor_policy '{FLOOR_POLICY}', cohort low_mode '{LOW_MODE}'.",
    "Team score is app.matching._total_score with the default legacy weights, applied to every method.",
    "Behavior score for sort_and_chunk is the sum of a student's accountability and cooperativeness levels.",
    "A floor student is at the lowest level on accountability or cooperativeness. "
    "one_low_per_team limits each trait separately, so a team can hold two floor students "
    "(one per trait); 'max same-trait floor/team' is the per-trait count the policy constrains.",
    "Cohorts infeasible under one_low_per_team (more floor students on a dimension than teams) are skipped and counted.",
    "Values are the mean over cohorts with a 95% percentile bootstrap CI "
    f"({BOOTSTRAP_RESAMPLES} resamples, seed {BOOTSTRAP_SEED}).",
)


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

def _request(**kwargs) -> TeamGenerationRequest:
    return TeamGenerationRequest(team_size=TEAM_SIZE, profile=PROFILE, floor_policy=FLOOR_POLICY, **kwargs)


def _by_id(students: List[Student]) -> List[Student]:
    return sorted(students, key=lambda s: s.id)


def _chunk(ordered: List[Student]) -> List[List[Student]]:
    n_teams = prod._compute_n_teams(len(ordered), TEAM_SIZE)
    teams, i = [], 0
    for size in prod._target_team_sizes(len(ordered), n_teams):
        teams.append(ordered[i:i + size])
        i += size
    return teams


@contextmanager
def _floor_disabled():
    original = prod._is_floor_level
    prod._is_floor_level = lambda student, dim_name, dim_def: False
    try:
        yield
    finally:
        prod._is_floor_level = original


def form_ours(students: List[Student], key: str) -> List[List[Student]]:
    groups, _ = prod.form_teams(students, _request(), [], [])
    return groups


def form_no_floor(students: List[Student], key: str) -> List[List[Student]]:
    with _floor_disabled():
        return form_ours(students, key)


def form_no_maximin(students: List[Student], key: str) -> List[List[Student]]:
    request, _ = prod.resolve_profile(_request())
    weights = prod._normalize_weights(request.weights)
    groups = prod._greedy_build(_by_id(students), request.team_size, weights)
    groups = prod._two_opt_improve(groups, weights, floor_policy=FLOOR_POLICY)
    groups, _ = prod._run_optional_passes(
        groups, request, weights, FLOOR_POLICY, [], [],
        primary_metric=prod.OPTIMIZATION_PROFILES[request.profile]["primary_metric"],
    )
    return groups


def form_random(students: List[Student], key: str) -> List[List[Student]]:
    ordered = _by_id(students)
    random.Random(f"random-baseline:{key}").shuffle(ordered)
    return _chunk(ordered)


def form_skills_only(students: List[Student], key: str) -> List[List[Student]]:
    return prod.skill_based_assignment(_by_id(students), TEAM_SIZE)


def form_behavior_only(students: List[Student], key: str) -> List[List[Student]]:
    groups, _ = prod.form_teams(students, _request(weights=dict(BEHAVIOR_ONLY_WEIGHTS)), [], [])
    return groups


def behavior_score(student: Student) -> int:
    return sum(_level(getattr(student.collaboration_signature, t), t) for t in FLOOR_TRAITS)


def form_sort_and_chunk(students: List[Student], key: str) -> List[List[Student]]:
    return _chunk(sorted(students, key=lambda s: (behavior_score(s), s.id)))


FORMERS: Dict[str, Callable[[List[Student], str], List[List[Student]]]] = {
    "ours": form_ours,
    "ours_no_floor": form_no_floor,
    "ours_no_maximin": form_no_maximin,
    "random": form_random,
    "skills_only": form_skills_only,
    "behavior_only": form_behavior_only,
    "sort_and_chunk": form_sort_and_chunk,
}


def effective_weights(method: str) -> Dict[str, float]:
    weights = BEHAVIOR_ONLY_WEIGHTS if method == "behavior_only" else None
    request, _ = prod.resolve_profile(_request(weights=dict(weights)) if weights else _request())
    return prod._normalize_weights(request.weights)


# ---------------------------------------------------------------------------
# Metrics (reuses evaluation/metrics.py)
# ---------------------------------------------------------------------------

def _level(value: str, trait: str) -> int:
    return list(get_rubric()["dimensions"][trait]["allowed_values"]).index(value)


def _check_partition(students: List[Student], teams: List[List[Student]]) -> None:
    placed = sorted(s.id for t in teams for s in t)
    if placed != sorted(s.id for s in students):
        raise AssertionError("every student must be placed exactly once")
    sizes = [len(t) for t in teams]
    if max(sizes) - min(sizes) > 1:
        raise AssertionError(f"unbalanced team sizes {sizes}")


def team_metrics(teams: List[List[Student]]) -> Dict[str, Optional[float]]:
    cfg = em.EvalConfig(max_exact_regret_roster_size=0)
    con = em.conscientiousness(teams, cfg)
    agr = em.agreeableness(teams, cfg)

    floor_counts = [
        len(set(a["members_at_or_below_floor"]) | set(b["members_at_or_below_floor"]))
        for a, b in zip(con["per_team"], agr["per_team"])
    ]

    def min_mean(report: dict) -> Optional[float]:
        means = [t["mean_level"] for t in report["per_team"] if t["mean_level"] is not None]
        return min(means) if means else None

    scores = [prod._total_score(t, prod._DEFAULT_WEIGHTS) for t in teams]
    sc = em.skill_coverage(teams, cfg)
    util = em.individual_utility(teams, prod._DEFAULT_WEIGHTS, cfg)

    return {
        "teams_with_floor_student": sum(1 for c in floor_counts if c > 0),
        "max_floor_students_on_one_team": max(floor_counts),
        "max_same_trait_floor_on_one_team": max(
            len(t["members_at_or_below_floor"]) for r in (con, agr) for t in r["per_team"]
        ),
        "accountability_min_team_mean": min_mean(con),
        "accountability_mean_of_team_means": con["global_mean_of_team_means"],
        "cooperativeness_min_team_mean": min_mean(agr),
        "cooperativeness_mean_of_team_means": agr["global_mean_of_team_means"],
        "team_score_std": statistics.pstdev(scores),
        "worst_team_score": min(scores),
        "skill_coverage_rate": sum(1 for t in sc["per_team"] if not t["missing_skills"]) / len(teams),
        "mean_coverage": sc["mean_coverage_rate"],
        "worst_off_utility": util["min_utility"],
    }


def _round(x):
    return None if x is None else round(float(x), DECIMALS)


# ---------------------------------------------------------------------------
# One cohort
# ---------------------------------------------------------------------------

def run_cohort(job: Tuple[float, int, int]) -> dict:
    fraction, size, seed = job
    cohort = generate_cohort(size, seed, low_scorer_fraction=fraction, low_mode=LOW_MODE)
    students = [record_to_student(r) for r in cohort["students"]]
    n_teams = prod._compute_n_teams(len(students), TEAM_SIZE)
    feasible, infeasible_dims = prod._check_feasibility(students, n_teams, FLOOR_POLICY)
    out = {"low_scorer_fraction": fraction, "size": size, "seed": seed,
           "n_teams": n_teams, "feasible": feasible}
    if not feasible:
        out["infeasible_dims"] = dict(sorted(infeasible_dims.items()))
        return out

    key = f"{fraction}:{size}:{seed}"
    out["methods"] = {}
    for method in METHODS:
        teams = FORMERS[method](students, key)
        _check_partition(students, teams)
        out["methods"][method] = {k: _round(v) for k, v in team_metrics(teams).items()}
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def bootstrap_ci(values: List[float], key: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not values:
        return None, None, None
    mean = statistics.fmean(values)
    rng = random.Random(f"{BOOTSTRAP_SEED}:{key}")
    n = len(values)
    resampled = sorted(
        statistics.fmean(values[rng.randrange(n)] for _ in range(n))
        for _ in range(BOOTSTRAP_RESAMPLES)
    )
    lo = resampled[int(0.025 * BOOTSTRAP_RESAMPLES)]
    hi = resampled[int(0.975 * BOOTSTRAP_RESAMPLES) - 1]
    return _round(mean), _round(lo), _round(hi)


def aggregate(cohorts: List[dict], fraction: float, sizes: List[int]) -> dict:
    by_size = {}
    for size in sizes:
        rows = [c for c in cohorts if c["low_scorer_fraction"] == fraction and c["size"] == size]
        kept = [c for c in rows if c["feasible"]]
        methods = {}
        for method in METHODS:
            methods[method] = {}
            for metric in METRICS:
                vals = [c["methods"][method][metric] for c in kept
                        if c["methods"][method][metric] is not None]
                mean, lo, hi = bootstrap_ci(vals, f"{fraction}:{size}:{method}:{metric}")
                methods[method][metric] = {"mean": mean, "ci_low": lo, "ci_high": hi, "n": len(vals)}
        by_size[str(size)] = {
            "n_cohorts": len(kept),
            "n_skipped_infeasible": len(rows) - len(kept),
            "methods": methods,
        }
    return by_size


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _fmt(cell: dict) -> str:
    if cell["mean"] is None:
        return "n/a"
    return f"{cell['mean']:.3f} [{cell['ci_low']:.3f}, {cell['ci_high']:.3f}]"


def render_markdown(results: dict) -> str:
    lines = ["# Team-quality evaluation", ""]
    lines += [f"- {n}" for n in results["header"]["notes"]]
    lines += ["", "Methods:", ""]
    lines += [f"- `{m}`: {METHOD_DESCRIPTIONS[m]}" for m in METHODS]
    lines += ["", f"`behavior_only` effective weights: `{json.dumps(results['header']['behavior_only_effective_weights'], sort_keys=True)}`", ""]

    for table_name, fraction in results["header"]["tables"]:
        table = results["tables"][table_name]
        lines += [f"## {table_name.capitalize()}: low-scorer fraction {fraction}", ""]
        for size, block in table.items():
            lines += [f"### {size} students ({block['n_cohorts']} cohorts, "
                      f"{block['n_skipped_infeasible']} skipped as infeasible)", ""]
            lines.append("| method | " + " | ".join(label for _, label in TABLE_COLUMNS) + " |")
            lines.append("|---" * (len(TABLE_COLUMNS) + 1) + "|")
            for method in METHODS:
                cells = [_fmt(block["methods"][method][metric]) for metric, _ in TABLE_COLUMNS]
                lines.append(f"| {method} | " + " | ".join(cells) + " |")
            lines.append("")
    return "\n".join(lines)


def render_csv(results: dict) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["table", "low_scorer_fraction", "size", "method", "metric",
                "mean", "ci_low", "ci_high", "n", "n_cohorts", "n_skipped_infeasible"])
    for table_name, fraction in results["header"]["tables"]:
        for size, block in results["tables"][table_name].items():
            for method in METHODS:
                for metric in METRICS:
                    c = block["methods"][method][metric]
                    w.writerow([table_name, fraction, size, method, metric, c["mean"], c["ci_low"],
                                c["ci_high"], c["n"], block["n_cohorts"], block["n_skipped_infeasible"]])
    return buf.getvalue()


def evaluate(sizes: List[int], seeds: int, workers: int = 1) -> dict:
    fractions = [PRIMARY_FRACTION, *SECONDARY_FRACTIONS]
    jobs = [(f, size, seed) for f in fractions for size in sizes for seed in range(seeds)]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            cohorts = list(pool.map(run_cohort, jobs, chunksize=1))
    else:
        cohorts = [run_cohort(j) for j in jobs]

    weights = effective_weights("behavior_only")
    if weights.get("skill_coverage") != 0.0:
        raise AssertionError(f"behavior_only skill weight is not 0: {weights}")

    tables = [("primary", PRIMARY_FRACTION)] + [(f"secondary_{f}", f) for f in SECONDARY_FRACTIONS]
    return {
        "header": {
            "notes": list(HEADER_NOTES),
            "team_size": TEAM_SIZE,
            "profile": PROFILE,
            "floor_policy": FLOOR_POLICY,
            "low_mode": LOW_MODE,
            "sizes": sizes,
            "seeds": seeds,
            "methods": dict(METHOD_DESCRIPTIONS),
            "behavior_only_effective_weights": {k: _round(v) for k, v in sorted(weights.items())},
            "tables": tables,
        },
        "tables": {name: aggregate(cohorts, f, sizes) for name, f in tables},
        "cohorts": cohorts,
    }


def write_outputs(results: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "quality_results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    (out_dir / "quality_table.csv").write_text(render_csv(results))
    (out_dir / "quality_table.md").write_text(render_markdown(results) + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Compare team-formation methods on synthetic cohorts.")
    p.add_argument("--sizes", type=int, nargs="+", default=[20, 22, 40, 43, 80])
    p.add_argument("--seeds", type=int, default=30, help="cohort seeds per size (default 30)")
    p.add_argument("--out", default="eval/results")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                   help="parallel processes; output is identical for any value")
    args = p.parse_args(argv)

    results = evaluate(args.sizes, args.seeds, args.workers)
    write_outputs(results, Path(args.out))
    print(f"wrote {args.out}/quality_results.json, quality_table.csv, quality_table.md", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
