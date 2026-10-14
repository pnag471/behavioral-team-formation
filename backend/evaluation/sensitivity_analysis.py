"""
Part 6.5 Section 5 — controlled sensitivity analysis over configuration
weights/thresholds. For each knob, holds every other setting at its class
default, varies the ONE knob across a small documented grid, and records:

  - the resulting partition's pairwise co-membership distance from the
    knob's OWN baseline value (first grid point) — 0.0 means "this
    setting change did not move the assignment at all"
  - the target/primary metric's value at each grid point
  - whether a discontinuity was observed (a small step in the knob
    producing a large jump in comembership distance between adjacent
    grid points)

This is explicitly NOT a search for "best" settings (Part 6.5's own
instruction) — it characterizes SENSITIVITY only: which knobs matter, and
under what data, for this synthetic/seed data. No weight is recommended
here for real deployment.

Run with:
    cd backend && <python> -m evaluation.sensitivity_analysis
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from app import matching as prod
from app.models import FamiliarityEdge, Student
from evaluation import fixtures as fx
from evaluation.metrics import legacy_preferences_to_entries
from evaluation.profile_distinctness import _legacy_familiarity_to_edges, _pairwise_comembership_distance

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_STUDENTS_PATH = Path(__file__).resolve().parents[1] / "sample_students.json"


def _run(students: List[Student], team_size: int, request: prod.TeamGenerationRequest,
         familiarity_edges=None, preference_entries=None):
    resolved, _ = prod.resolve_profile(request)
    weights = prod._normalize_weights(resolved.weights)
    teams = prod._greedy_build(students, team_size, weights)
    teams = prod._two_opt_improve(teams, weights, floor_policy=resolved.floor_policy)
    teams = prod._maximin_improve(teams, weights=weights, floor_policy=resolved.floor_policy)
    teams, _trace = prod._run_optional_passes(
        teams, resolved, weights, resolved.floor_policy,
        familiarity_edges or [], preference_entries or [],
        primary_metric=prod.OPTIMIZATION_PROFILES.get(resolved.profile, {}).get("primary_metric"),
    )
    return teams


def _sweep(
    name: str, students, team_size: int, base_overrides: dict, knob_path: List[str], grid: List[float],
    target_fn, familiarity_edges=None, preference_entries=None,
) -> dict:
    """knob_path e.g. ["weights", "skill_coverage"] or ["familiarity", "lower_bound"]."""
    baseline_partition = None
    points = []
    for value in grid:
        overrides = json.loads(json.dumps(base_overrides))  # deep copy
        node = overrides
        for key in knob_path[:-1]:
            node = node.setdefault(key, {})
        node[knob_path[-1]] = value

        request_data = prod.TeamGenerationRequest().model_dump()
        request_data = prod._deep_update(request_data, overrides)
        request = prod.TeamGenerationRequest(**request_data)

        teams = _run(students, team_size, request, familiarity_edges, preference_entries)
        if baseline_partition is None:
            baseline_partition = teams
        dist_from_baseline = _pairwise_comembership_distance(baseline_partition, teams)
        points.append({
            "knob_value": value,
            "comembership_distance_from_first_grid_point": round(dist_from_baseline, 4),
            "target_metric": target_fn(teams),
        })

    discontinuities = []
    for a, b in zip(points, points[1:]):
        jump = abs(b["comembership_distance_from_first_grid_point"] - a["comembership_distance_from_first_grid_point"])
        if jump >= 0.15:  # documented threshold for "large step" in this sweep, not a universal cutoff
            discontinuities.append({"from": a["knob_value"], "to": b["knob_value"], "comembership_jump": round(jump, 4)})

    ever_moves = any(p["comembership_distance_from_first_grid_point"] > 0.0 for p in points)

    return {
        "sweep": name,
        "knob_path": ".".join(knob_path),
        "grid": grid,
        "points": points,
        "never_affects_assignment": not ever_moves,
        "discontinuities": discontinuities,
    }


def main():
    with open(SAMPLE_STUDENTS_PATH) as f:
        raw = json.load(f)
    seed_students = [Student(**r) for r in raw]

    stacked = fx.fixture_stacked_team()
    friend_clique = fx.fixture_friend_clique()
    pref_instab = fx.fixture_preference_instability()

    sweeps = []

    # 1. Primary objective reweighting toward skill_coverage — expect the
    #    assignment to move (like skill_priority) and, per
    #    docs/optimization_profiles.md's own empirical finding, ability_gap
    #    to worsen as skill_coverage's weight rises.
    sweeps.append(_sweep(
        "skill_coverage_weight_on_seed_roster", seed_students, 4,
        {"profile": "legacy"}, ["weights", "skill_coverage"], [0.1, 0.4, 0.7, 0.9],
        target_fn=lambda t: prod._partition_ability_gap(t),
    ))

    # 2. AbilityConfig.ability_level_thresholds — REPORTING-only per its own
    #    docstring (bucketing labels, never read by _ability_balance_
    #    improve's continuous-gap accept rule). Expect ZERO assignment
    #    movement — a "never affects results" knob, verified rather than
    #    assumed.
    sweeps.append(_sweep(
        "ability_level_thresholds_developing_max_on_stacked_team", stacked.students, stacked.team_size,
        {"ability": {"enabled": True}}, ["ability", "ability_level_thresholds"],
        [{"developing_max": 0.2, "medium_max": 0.5}, {"developing_max": 0.4, "medium_max": 0.7},
         {"developing_max": 0.6, "medium_max": 0.9}],
        target_fn=lambda t: prod._partition_ability_gap(t),
    ))

    # 3. FairnessConfig.acceptable_utility_floor — REPORTING-only per its
    #    own docstring (who counts as "below floor" for display). Expect
    #    ZERO assignment movement.
    sweeps.append(_sweep(
        "fairness_acceptable_utility_floor_on_seed_roster", seed_students, 4,
        {"fairness": {"enabled": True}}, ["fairness", "acceptable_utility_floor"], [0.3, 0.5, 0.7, 0.9],
        target_fn=lambda t: prod._partition_min_utility(t, prod.DEFAULT_UTILITY_WEIGHTS),
    ))

    # 4. FamiliarityConfig.lower_bound/upper_bound band — expect assignment
    #    movement once real familiarity data exists (friend_clique fixture).
    fam_edges = _legacy_familiarity_to_edges(friend_clique.familiarity)
    sweeps.append(_sweep(
        "familiarity_upper_bound_on_friend_clique", friend_clique.students, friend_clique.team_size,
        {"familiarity": {"enabled": True}}, ["familiarity", "upper_bound"], [0.3, 0.5, 0.6, 0.8, 0.95],
        # NOTE: target_fn here reports density-band violation at the FIXED
        # default clique_edge_threshold (0.5) for display only — the
        # SWEPT upper_bound is correctly applied inside the real pass via
        # `request` (see _run/_sweep), so comembership_distance (the
        # actual sensitivity signal this sweep exists to measure) is
        # accurate regardless; only this one display column holds the
        # bound fixed rather than reflecting each grid point's value.
        target_fn=lambda t: sum(
            prod._team_clique_violation(
                team, prod.build_familiarity_graph([s for tt in t for s in tt], fam_edges),
                0.5, 0.5,
            ) for team in t
        ),
        familiarity_edges=fam_edges,
    ))

    # 5. PreferenceConfig.high_priority_weight — expect movement on the
    #    preference_instability fixture (real preference data present).
    pref_entries = legacy_preferences_to_entries(pref_instab.preferences)
    sweeps.append(_sweep(
        "preference_high_priority_weight_on_preference_instability",
        pref_instab.students, pref_instab.team_size,
        {"preference": {"enabled": True}}, ["preference", "high_priority_weight"], [1.0, 3.0, 6.0, 10.0],
        target_fn=lambda t: sum(
            prod._team_preference_violation(
                [s.id for s in team], {s.id: s for s in team}, prod._build_preference_index(pref_entries),
                3.0, 5.0,
            )[0] for team in t
        ),
        preference_entries=pref_entries,
    ))

    # 6. Does research_balanced ever COLLAPSE into legacy's exact assignment
    #    as every optional-pass weight/threshold is pushed to its most
    #    conservative extreme? Explicit check requested by Part 6.5 Section 5.
    conservative_request = prod.TeamGenerationRequest(**prod._deep_update(
        prod.TeamGenerationRequest().model_dump(),
        {"ability": {"enabled": True}, "fairness": {"enabled": True},
         "familiarity": {"enabled": True}, "preference": {"enabled": True}},
    ))
    legacy_teams = _run(seed_students, 4, prod.TeamGenerationRequest(profile="legacy"))
    conservative_teams = _run(seed_students, 4, conservative_request)
    collapse_distance = _pairwise_comembership_distance(legacy_teams, conservative_teams)

    out = {
        "sweeps": sweeps,
        "collapse_to_legacy_check": {
            "description": (
                "research_balanced's own default settings against legacy, on the seed roster — "
                "checks whether enabling every pass with conservative defaults ever produces a "
                "DIFFERENT result from simply not running them."
            ),
            "comembership_distance_from_legacy": round(collapse_distance, 4),
            "collapses_to_legacy": collapse_distance == 0.0,
        },
    }

    out_dir = REPO_ROOT / "backend" / "evaluation_output"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "sensitivity_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Wrote {out_path}")

    print("\n=== SUMMARY ===")
    for s in sweeps:
        print(f"{s['sweep']:<55} never_affects_assignment={s['never_affects_assignment']} "
              f"discontinuities={len(s['discontinuities'])}")
    print(f"\ncollapse_to_legacy (seed roster, all passes on, defaults): "
          f"distance={collapse_distance:.4f} collapses={collapse_distance == 0.0}")


if __name__ == "__main__":
    main()
