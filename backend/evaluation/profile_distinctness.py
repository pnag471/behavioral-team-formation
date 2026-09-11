"""
Part 6.5 Section 3 — are optimization profiles experimentally distinguishable?

For every pair of profiles, on the real seed roster and every adversarial
fixture, computes team-label-invariant partition-similarity measures:

  - percent_reassigned        : fraction of students on a DIFFERENT team
                                 (after solving the label-correspondence via
                                 maximum team-overlap matching, since team
                                 indices are arbitrary)
  - pairwise_comembership_distance : fraction of student PAIRS whose
                                 "same team?" bit differs between the two
                                 partitions — this is inherently
                                 label-invariant (no matching needed)
  - adjusted_rand_index       : sklearn.metrics.adjusted_rand_score,
                                 the standard permutation-invariant
                                 partition-similarity measure (1.0 =
                                 identical partition, ~0.0 = no better than
                                 chance, can go negative)
  - identical_partition       : True iff pairwise_comembership_distance==0
                                 (a stronger, exact-match notion than a
                                 high ARI, since ARI can be 1.0 without
                                 partitions being LABEL-identical in a
                                 pathological tie case — this script checks
                                 both explicitly rather than assuming)

Team labels never affect any of these (the whole point) — see
_partition_to_labels()'s docstring.

Run with:
    cd backend && <python> -m evaluation.profile_distinctness
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Dict, List

from sklearn.metrics import adjusted_rand_score

from app import matching as prod
from app.models import FamiliarityEdge, Student
from evaluation import fixtures as fx
from evaluation.metrics import EvalConfig, evaluate_teams, legacy_preferences_to_entries

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_STUDENTS_PATH = Path(__file__).resolve().parents[1] / "sample_students.json"
PROFILES = ["legacy", "research_balanced", "fairness_priority", "skill_priority", "preference_priority"]


def _partition_to_labels(teams: List[List[Student]]) -> Dict[str, int]:
    """student_id -> team INDEX in `teams`'s own order. This index is
    arbitrary/not meaningful across two DIFFERENT partitions on its own —
    every metric below either explicitly re-solves a label correspondence
    (percent_reassigned) or is mathematically invariant to relabeling
    (pairwise co-membership distance, ARI) rather than comparing these
    raw indices directly."""
    return {s.id: i for i, team in enumerate(teams) for s in team}


def _percent_reassigned(teams_a: List[List[Student]], teams_b: List[List[Student]]) -> float:
    """
    Label-invariant: solves the best team-index correspondence between A
    and B by greedy maximum-overlap matching (each team in A claims the
    team in B it shares the most members with, highest-overlap pairs
    first), then reports the fraction of students whose team under that
    correspondence differs. This is a natural-language-meaningful
    ("how many students actually moved") measure, distinct from ARI
    (a purely combinatorial statistic).
    """
    overlaps = []
    for ia, ta in enumerate(teams_a):
        ids_a = {s.id for s in ta}
        for ib, tb in enumerate(teams_b):
            ids_b = {s.id for s in tb}
            overlaps.append((len(ids_a & ids_b), ia, ib))
    overlaps.sort(reverse=True)
    correspondence: Dict[int, int] = {}
    used_b = set()
    for _, ia, ib in overlaps:
        if ia in correspondence or ib in used_b:
            continue
        correspondence[ia] = ib
        used_b.add(ib)

    labels_a = _partition_to_labels(teams_a)
    labels_b = _partition_to_labels(teams_b)
    all_ids = set(labels_a) | set(labels_b)
    if not all_ids:
        return 0.0
    reassigned = 0
    for sid in all_ids:
        team_a = labels_a.get(sid)
        team_b = labels_b.get(sid)
        expected_b = correspondence.get(team_a) if team_a is not None else None
        if expected_b != team_b:
            reassigned += 1
    return reassigned / len(all_ids)


def _pairwise_comembership_distance(teams_a: List[List[Student]], teams_b: List[List[Student]]) -> float:
    """Fraction of student PAIRS whose 'on the same team?' truth value
    differs between A and B. Inherently label-invariant — no correspondence
    solving needed, since "same team as X" never refers to a team INDEX."""
    labels_a = _partition_to_labels(teams_a)
    labels_b = _partition_to_labels(teams_b)
    ids = sorted(set(labels_a) & set(labels_b))
    if len(ids) < 2:
        return 0.0
    disagreements = 0
    total = 0
    for x, y in itertools.combinations(ids, 2):
        total += 1
        same_a = labels_a[x] == labels_a[y]
        same_b = labels_b[x] == labels_b[y]
        if same_a != same_b:
            disagreements += 1
    return disagreements / total if total else 0.0


def _ari(teams_a: List[List[Student]], teams_b: List[List[Student]]) -> float:
    labels_a = _partition_to_labels(teams_a)
    labels_b = _partition_to_labels(teams_b)
    ids = sorted(set(labels_a) & set(labels_b))
    if len(ids) < 2:
        return 1.0
    a = [labels_a[i] for i in ids]
    b = [labels_b[i] for i in ids]
    return float(adjusted_rand_score(a, b))


def _run_profile(students: List[Student], team_size: int, profile_name: str,
                  familiarity_edges=None, preference_entries=None) -> List[List[Student]]:
    request, _ = prod.resolve_profile(prod.TeamGenerationRequest(team_size=team_size, profile=profile_name))
    weights = prod._normalize_weights(request.weights)
    teams = prod._greedy_build(students, team_size, weights)
    teams = prod._two_opt_improve(teams, weights, floor_policy=request.floor_policy)
    teams = prod._maximin_improve(teams, weights=weights, floor_policy=request.floor_policy)
    teams, _trace = prod._run_optional_passes(
        teams, request, weights, request.floor_policy,
        familiarity_edges or [], preference_entries or [],
        primary_metric=prod.OPTIMIZATION_PROFILES[profile_name]["primary_metric"],
    )
    return teams


def _legacy_familiarity_to_edges(familiarity) -> List[FamiliarityEdge]:
    """Converts evaluation.fixtures.FixtureCase.familiarity's
    Dict[(id, id), float] shape into real FamiliarityEdge rows, the same
    boundary role legacy_preferences_to_entries() plays for preferences."""
    if not familiarity:
        return []
    seen = set()
    edges: List[FamiliarityEdge] = []
    for (a, b), strength in familiarity.items():
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        edges.append(FamiliarityEdge(student_a_id=key[0], student_b_id=key[1], strength=strength))
    return edges


def _compare_dataset(name: str, students: List[Student], team_size: int,
                      familiarity=None, preferences=None) -> dict:
    familiarity_edges = _legacy_familiarity_to_edges(familiarity)
    preference_entries = legacy_preferences_to_entries(preferences)
    partitions = {
        p: _run_profile(students, team_size, p, familiarity_edges, preference_entries)
        for p in PROFILES
    }
    metrics = {p: evaluate_teams(partitions[p], weights=prod._normalize_weights({
        "skill_coverage": 0.4, "behavioral_compat": 0.3,
        "availability_overlap": 0.2, "shared_interests": 0.1,
    })) for p in PROFILES}

    pairwise = {}
    for a, b in itertools.combinations(PROFILES, 2):
        pct = _percent_reassigned(partitions[a], partitions[b])
        dist = _pairwise_comembership_distance(partitions[a], partitions[b])
        ari = _ari(partitions[a], partitions[b])
        metric_diffs = {}
        for metric_key in ("skill_coverage", "ability_diversity", "conscientiousness", "agreeableness"):
            ma = metrics[a][metric_key]
            mb = metrics[b][metric_key]
            if metric_key == "skill_coverage":
                va, vb = ma.get("min_coverage_rate"), mb.get("min_coverage_rate")
            elif metric_key == "ability_diversity":
                va = min((t["mean_ability"] for t in ma["per_team"] if t["mean_ability"] is not None), default=None)
                vb = min((t["mean_ability"] for t in mb["per_team"] if t["mean_ability"] is not None), default=None)
            else:
                va, vb = ma.get("global_min_of_team_mins"), mb.get("global_min_of_team_mins")
            metric_diffs[metric_key] = {"a": va, "b": vb, "diff": (vb - va) if (va is not None and vb is not None) else None}

        pairwise[f"{a}__vs__{b}"] = {
            "percent_students_reassigned": round(pct, 4),
            "pairwise_comembership_distance": round(dist, 4),
            "adjusted_rand_index": round(ari, 4),
            "identical_partition": dist == 0.0,
            "metric_diffs": metric_diffs,
        }

    n_distinct_partitions = len({
        tuple(sorted(tuple(sorted(s.id for s in t)) for t in partitions[p])) for p in PROFILES
    })

    return {
        "dataset": name,
        "n_students": len(students),
        "n_distinct_partitions_among_5_profiles": n_distinct_partitions,
        "pairwise": pairwise,
    }


def main():
    with open(SAMPLE_STUDENTS_PATH) as f:
        raw = json.load(f)
    seed_students = [Student(**r) for r in raw]

    reports = [_compare_dataset("seed_roster_20", seed_students, 4)]

    for factory in fx.ALL_FIXTURES:
        case = factory()
        reports.append(_compare_dataset(
            case.name, case.students, case.team_size,
            familiarity=case.familiarity, preferences=case.preferences,
        ))

    out_dir = REPO_ROOT / "backend" / "evaluation_output"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "profile_distinctness.json"
    with open(out_path, "w") as f:
        json.dump({"datasets": reports}, f, indent=2, default=str)
    print(f"Wrote {out_path}")

    print("\n=== SUMMARY: identical partitions among the 5 profiles, per dataset ===")
    for r in reports:
        collapsed = [k for k, v in r["pairwise"].items() if v["identical_partition"]]
        print(f"{r['dataset']:<28} n_distinct_partitions={r['n_distinct_partitions_among_5_profiles']} "
              f"identical_pairs={len(collapsed)}/10")


if __name__ == "__main__":
    main()
