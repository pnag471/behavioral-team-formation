"""
Determinism harness (offline). Runs the REAL team-formation pipeline N times
on one cohort and checks every run yields the identical partition.

Two kinds of repetition, because they catch different bugs:
  * in-process runs  — same interpreter, repeated calls. Catches hidden state
                       (caches, globals, unseeded randomness).
  * fresh-process runs — each run in a new Python process with a DIFFERENT
                       PYTHONHASHSEED. Catches dependence on set/dict-of-str
                       iteration order, which is randomized per process and
                       is invisible to in-process repetition.

Partitions are compared order-independently: sort member ids within each
team, then sort the teams.

A random-shuffle baseline former runs through the same check so the report
shows the contrast: the real system should PASS, the baseline should FAIL.

Usage (from backend/):
    python3 -m eval.determinism_harness eval/cohort_seed7.json --runs 10
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from app import matching
from app.models import Student, TeamGenerationRequest
from eval.cohort import load_cohort

Partition = Tuple[Tuple[str, ...], ...]
Former = Callable[[List[Student], int], List[List[str]]]

BACKEND_DIR = Path(__file__).resolve().parents[1]

# TODO(team-decision): default team size. 4 matches TeamGenerationRequest's
# own default (team_size=4, allowed 2..8). Override with --team-size.
DEFAULT_TEAM_SIZE = 4


# ---------------------------------------------------------------------------
# Adapter to the real team-formation code
# ---------------------------------------------------------------------------

def real_former(students: List[Student], team_size: int) -> List[List[str]]:
    """
    Thin adapter: (students, team_size) -> list of teams of student ids.

    There is no single pure team-formation function. The production entry
    point is app.matching.generate_teams (POST /teams/generate), which reads
    students from SQLite and DELETES/REWRITES the teams table, so it cannot be
    called here. This adapter replays the exact formation sequence from that
    function's `formation_mode == "behavioral"` branch, with the same calls
    and arguments, minus the DB I/O and the post-formation reporting (which
    never changes the assignment):

        resolve_profile -> _normalize_weights -> _greedy_build
        -> _two_opt_improve -> _maximin_improve -> _run_optional_passes

    TODO(team-decision): this mirrors generate_teams by copy, so it can drift
    if that branch changes. Cleaner fix (needs a change to matching.py, out of
    scope here): extract that branch into a pure
    `form_teams(students, request, familiarity_edges, preference_entries)`
    that both the endpoint and this harness call.

    TODO(team-decision): optimizer config. Uses TeamGenerationRequest defaults
    -> profile "legacy" (every optional pass OFF), floor_policy
    "one_low_per_team", default weights, and NO familiarity edges or
    preference entries. Other profiles are not exercised by default; pass
    --profile to check one.
    """
    return _form(students, team_size, profile=None)


def _form(students: List[Student], team_size: int, profile: Optional[str]) -> List[List[str]]:
    req_kwargs = {"team_size": team_size}
    if profile is not None:
        req_kwargs["profile"] = profile
    request, _ = matching.resolve_profile(TeamGenerationRequest(**req_kwargs))
    weights = matching._normalize_weights(request.weights)
    floor_policy = request.floor_policy

    groups = matching._greedy_build(students, request.team_size, weights)
    groups = matching._two_opt_improve(groups, weights, floor_policy=floor_policy)
    groups = matching._maximin_improve(groups, weights=weights, floor_policy=floor_policy)
    groups, _trace = matching._run_optional_passes(
        groups, request, weights, floor_policy,
        [], [],  # familiarity_edges, preference_entries: none in a synthetic cohort
        primary_metric=matching.OPTIMIZATION_PROFILES[request.profile]["primary_metric"],
    )
    return [[s.id for s in g] for g in groups]


def feasibility_note(students: List[Student], team_size: int, profile: Optional[str]) -> Optional[str]:
    """generate_teams refuses (HTTP 422) infeasible cohorts before forming teams. Report, don't raise."""
    req_kwargs = {"team_size": team_size}
    if profile is not None:
        req_kwargs["profile"] = profile
    request, _ = matching.resolve_profile(TeamGenerationRequest(**req_kwargs))
    n_teams = matching._compute_n_teams(len(students), team_size)
    ok, bad = matching._check_feasibility(students, n_teams, request.floor_policy)
    if ok:
        return None
    return (f"cohort is INFEASIBLE under floor_policy={request.floor_policy!r} "
            f"({bad}); production would refuse it with HTTP 422")


# ---------------------------------------------------------------------------
# Random-shuffle baseline (expected to FAIL)
# ---------------------------------------------------------------------------

def random_shuffle_former(students: List[Student], team_size: int) -> List[List[str]]:
    """Shuffle with fresh OS entropy each call, then chunk into the same team sizes as production."""
    ids = [s.id for s in students]
    random.Random(os.urandom(16)).shuffle(ids)
    n_teams = matching._compute_n_teams(len(ids), team_size)
    teams, i = [], 0
    for size in matching._target_team_sizes(len(ids), n_teams):
        teams.append(ids[i:i + size])
        i += size
    return teams


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def canonical(teams: List[List[str]]) -> Partition:
    return tuple(sorted(tuple(sorted(t)) for t in teams))


def _fresh_process_partition(cohort_path: str, team_size: int, profile: Optional[str], hashseed: int) -> Partition:
    cmd = [sys.executable, "-m", "eval.determinism_harness", cohort_path,
           "--team-size", str(team_size), "--emit-partition"]
    if profile is not None:
        cmd += ["--profile", profile]
    env = dict(os.environ, PYTHONHASHSEED=str(hashseed))
    out = subprocess.run(cmd, cwd=BACKEND_DIR, env=env, capture_output=True, text=True, check=True)
    return tuple(tuple(t) for t in json.loads(out.stdout))


def check(label: str, partitions: List[Partition]) -> bool:
    distinct = sorted(set(partitions))
    passed = len(distinct) == 1
    print(f"  {label:<44} runs={len(partitions):<3} distinct partitions={len(distinct):<3} "
          f"{'PASS' if passed else 'FAIL'}")
    return passed


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Check that real team formation is deterministic.")
    p.add_argument("cohort", help="cohort JSON from eval.generate_cohort")
    p.add_argument("--runs", type=int, default=10, help="in-process runs per former (default 10)")
    p.add_argument("--processes", type=int, default=3,
                   help="extra fresh-process runs of the real system, each with a different PYTHONHASHSEED (default 3; 0 to skip)")
    p.add_argument("--team-size", type=int, default=DEFAULT_TEAM_SIZE)
    p.add_argument("--profile", default=None, help="optimization profile (default: request default, 'legacy')")
    p.add_argument("--shuffle-input", type=int, default=0, metavar="K",
                   help="also run the real system on K shuffled input orders (informational; default 0 = off)")
    p.add_argument("--emit-partition", action="store_true", help=argparse.SUPPRESS)  # used by fresh-process runs
    args = p.parse_args(argv)

    students = load_cohort(args.cohort)

    if args.emit_partition:
        print(json.dumps(canonical(_form(students, args.team_size, args.profile))))
        return 0

    print(f"cohort: {args.cohort}  students={len(students)}  team_size={args.team_size}  "
          f"profile={args.profile or 'legacy (default)'}")
    note = feasibility_note(students, args.team_size, args.profile)
    if note:
        print(f"  NOTE: {note}")
    print()

    real_runs = [canonical(_form(students, args.team_size, args.profile)) for _ in range(args.runs)]
    real_ok = check("real system (in-process)", real_runs)
    if args.processes > 0:
        proc_runs = [_fresh_process_partition(args.cohort, args.team_size, args.profile, hs)
                     for hs in range(args.processes)]
        real_ok &= check(f"real system (fresh processes, PYTHONHASHSEED 0..{args.processes - 1})",
                         [real_runs[0]] + proc_runs)

    if args.shuffle_input > 0:
        # Informational, does not affect the exit code: the spec's check is
        # "same cohort, same order". Production loads students with
        # db.query(StudentDB).all() (no ORDER BY), so input order is not
        # guaranteed there; this shows whether the result depends on it.
        # TODO(team-decision): is input-order invariance part of the
        # determinism claim? If yes, either sort students by id before
        # formation in matching.py, or add ORDER BY to the DB query.
        shuffled = []
        for k in range(args.shuffle_input):
            order = list(students)
            random.Random(k).shuffle(order)
            shuffled.append(canonical(_form(order, args.team_size, args.profile)))
        order_ok = check(f"real system, input order shuffled (seeds 0..{args.shuffle_input - 1})",
                         [real_runs[0]] + shuffled)
        if not order_ok:
            print("    ^ informational: result depends on the ORDER students are passed in")

    base_runs = [canonical(random_shuffle_former(students, args.team_size)) for _ in range(args.runs)]
    base_ok = check("random-shuffle baseline (expected FAIL)", base_runs)

    print()
    print(f"RESULT: real system {'PASS' if real_ok else 'FAIL'} | baseline "
          f"{'FAIL (expected)' if not base_ok else 'PASS — contrast NOT shown (cohort too small?)'}")
    if real_ok:
        print("teams (real system):")
        for i, t in enumerate(real_runs[0], 1):
            print(f"  team {i}: {', '.join(t)}")
    return 0 if real_ok else 1


if __name__ == "__main__":
    sys.exit(main())
