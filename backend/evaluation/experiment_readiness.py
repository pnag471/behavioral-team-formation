"""
Part 6.5 Section 7 — experiment-readiness check.

Given a roster and two selected profiles, reports whether they constitute
usable, distinguishable experimental conditions for a real study — WITHOUT
claiming a universal "different enough" cutoff (Part 6.5's own instruction:
any warning threshold here is a configurable, labeled STUDY-DESIGN CHOICE,
not a validated statistical criterion).

Run with:
    cd backend && <python> -m evaluation.experiment_readiness
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app import matching as prod
from app.models import FamiliarityEdge, PreferenceEntry, Student
from evaluation.profile_distinctness import _ari, _pairwise_comembership_distance, _percent_reassigned


@dataclass
class ReadinessConfig:
    """Every threshold below is a STUDY-DESIGN CHOICE this repo does not
    claim is universally correct — see module docstring."""
    # Below this pairwise co-membership distance, the two conditions are
    # flagged as "may not be distinguishable" for a between-subjects design.
    # Chosen to be a round, conservative, clearly-labeled default — not
    # derived from a power analysis or any validated source.
    min_comembership_distance_for_distinct_conditions: float = 0.10
    min_known_ability_fraction_for_sufficient_data: float = 0.7
    min_known_pair_coverage_for_sufficient_familiarity_data: float = 0.5
    min_submission_fraction_for_sufficient_preference_data: float = 0.5


def _run_profile(students, team_size, profile_name, familiarity_edges, preference_entries):
    request, resolution = prod.resolve_profile(prod.TeamGenerationRequest(team_size=team_size, profile=profile_name))
    weights = prod._normalize_weights(request.weights)
    teams = prod._greedy_build(students, team_size, weights)
    teams = prod._two_opt_improve(teams, weights, floor_policy=request.floor_policy)
    teams = prod._maximin_improve(teams, weights=weights, floor_policy=request.floor_policy)
    teams, _trace = prod._run_optional_passes(
        teams, request, weights, request.floor_policy, familiarity_edges, preference_entries,
        primary_metric=prod.OPTIMIZATION_PROFILES[profile_name]["primary_metric"],
    )
    return teams, request, resolution


def assess_experiment_readiness(
    students: List[Student],
    team_size: int,
    profile_a: str,
    profile_b: str,
    familiarity_edges: Optional[List[FamiliarityEdge]] = None,
    preference_entries: Optional[List[PreferenceEntry]] = None,
    config: Optional[ReadinessConfig] = None,
) -> Dict[str, object]:
    config = config or ReadinessConfig()
    familiarity_edges = familiarity_edges or []
    preference_entries = preference_entries or []

    import time
    t0 = time.perf_counter()
    teams_a, request_a, resolution_a = _run_profile(students, team_size, profile_a, familiarity_edges, preference_entries)
    teams_b, request_b, resolution_b = _run_profile(students, team_size, profile_b, familiarity_edges, preference_entries)
    runtime_s = time.perf_counter() - t0

    pct_reassigned = _percent_reassigned(teams_a, teams_b)
    comembership_dist = _pairwise_comembership_distance(teams_a, teams_b)
    ari = _ari(teams_a, teams_b)
    assignments_differ = comembership_dist > 0.0

    # Data sufficiency, per criterion this pair of profiles actually enables.
    n_students = len(students)
    ability_known = sum(1 for s in students if prod._student_ability_score(s)[0] is not None)
    ability_direct = sum(1 for s in students if prod._student_ability_score(s)[1] == prod.ABILITY_SOURCE_DIRECT)
    ability_fallback = sum(1 for s in students if prod._student_ability_score(s)[1] == prod.ABILITY_SOURCE_FALLBACK)
    ability_missing = n_students - ability_known
    ability_known_fraction = (ability_known / n_students) if n_students else None

    all_pairs = n_students * (n_students - 1) // 2
    graph = prod.build_familiarity_graph(students, familiarity_edges)
    known_pairs = sum(
        1 for i, a in enumerate(students) for b in students[i + 1:]
        if prod._known_familiarity(graph, a.id, b.id) is not None
    )
    familiarity_pair_coverage = (known_pairs / all_pairs) if all_pairs else None

    students_with_preferences = len({e.student_id for e in preference_entries})
    preference_submission_fraction = (students_with_preferences / n_students) if n_students else None

    enabled_a = {
        "ability": request_a.ability.enabled, "fairness": request_a.fairness.enabled,
        "familiarity": request_a.familiarity.enabled, "preference": request_a.preference.enabled,
    }
    enabled_b = {
        "ability": request_b.ability.enabled, "fairness": request_b.fairness.enabled,
        "familiarity": request_b.familiarity.enabled, "preference": request_b.preference.enabled,
    }
    any_criterion_enabled = {k: enabled_a[k] or enabled_b[k] for k in enabled_a}

    data_sufficiency = {
        "ability": {
            "enabled_in_either_profile": any_criterion_enabled["ability"],
            "known_fraction": ability_known_fraction,
            "direct_count": ability_direct, "fallback_count": ability_fallback, "missing_count": ability_missing,
            "sufficient": (
                ability_known_fraction is not None
                and ability_known_fraction >= config.min_known_ability_fraction_for_sufficient_data
            ) if any_criterion_enabled["ability"] else None,
        },
        "familiarity": {
            "enabled_in_either_profile": any_criterion_enabled["familiarity"],
            "known_pair_coverage": familiarity_pair_coverage,
            "known_pairs": known_pairs, "all_possible_pairs": all_pairs,
            "sufficient": (
                familiarity_pair_coverage is not None
                and familiarity_pair_coverage >= config.min_known_pair_coverage_for_sufficient_familiarity_data
            ) if any_criterion_enabled["familiarity"] else None,
        },
        "preference": {
            "enabled_in_either_profile": any_criterion_enabled["preference"],
            "submission_fraction": preference_submission_fraction,
            "students_with_submissions": students_with_preferences,
            "sufficient": (
                preference_submission_fraction is not None
                and preference_submission_fraction >= config.min_submission_fraction_for_sufficient_preference_data
            ) if any_criterion_enabled["preference"] else None,
        },
    }
    excluded_for_missing_data = [
        crit for crit, d in data_sufficiency.items()
        if d["enabled_in_either_profile"] and d["sufficient"] is False
    ]

    distinct_enough = comembership_dist >= config.min_comembership_distance_for_distinct_conditions

    return {
        "profile_a": profile_a, "profile_b": profile_b,
        "n_students": n_students, "team_size": team_size,
        "assignments_differ": assignments_differ,
        "partition_similarity": {
            "adjusted_rand_index": round(ari, 4),
            "pairwise_comembership_distance": round(comembership_dist, 4),
            "percent_students_reassigned": round(pct_reassigned, 4),
        },
        "data_sufficiency": data_sufficiency,
        "excluded_for_missing_data": excluded_for_missing_data,
        "distinct_enough_for_experimental_conditions": distinct_enough,
        "distinct_enough_threshold_used": config.min_comembership_distance_for_distinct_conditions,
        "distinct_enough_caveat": (
            "This threshold is a CONFIGURABLE STUDY-DESIGN CHOICE (ReadinessConfig."
            "min_comembership_distance_for_distinct_conditions), not a validated statistical "
            "cutoff for 'different enough' — no such universal cutoff is claimed."
        ),
        "runtime_seconds_observed": round(runtime_s, 4),
        "profile_resolutions": {
            profile_a: {"is_experimental": resolution_a.is_experimental, "enabled_passes": enabled_a},
            profile_b: {"is_experimental": resolution_b.is_experimental, "enabled_passes": enabled_b},
        },
    }


if __name__ == "__main__":
    import json
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    with open(repo_root / "backend" / "sample_students.json") as f:
        raw = json.load(f)
    students = [Student(**r) for r in raw]

    report = assess_experiment_readiness(students, 4, "legacy", "research_balanced")
    print(json.dumps(report, indent=2, default=str))
