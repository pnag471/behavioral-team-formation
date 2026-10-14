"""
Verifies validate.py's own correctness, not a team-formation method.

Passes the oracle (partitioning.brute_force_best) in as the "method"
under test. If validate_method is implemented correctly, validating the
oracle against itself must show it beating random on effectively every
cohort, hitting the ceiling on every brute-feasible cohort, and closing
100% of the gap on every non-easy cohort, because the method being
tested IS the thing gap_closed and hits_ceiling are measured against.
Anything less than that means the gap or ceiling logic in validate.py is
wrong, not that the oracle failed.

Run with:
    cd team_formation_eval && python3 selfcheck.py

Exits 0 and prints "selfcheck: OK" if every check holds, exits 1 and
prints what failed otherwise.
"""
from __future__ import annotations

from partitioning import brute_force_best
from validate import validate_method

SKILL_SET = frozenset({"frontend", "backend", "design", "testing", "ml", "devops"})
SIZES = (3, 3, 3, 3)
N_SAMPLES = 150
BASE_SEED = 5000
TEAM_METRIC = "total"
N_COHORTS = 12


def oracle_as_method(cohort, sizes, skill_set, weights, threshold):
    result = brute_force_best(cohort, sizes, skill_set, weights, threshold)
    if not result.feasible:
        raise RuntimeError(
            "selfcheck cohort is not brute-force feasible; shrink SIZES or raise the cap"
        )
    return result.best_partition


def main():
    failures = []

    records, summary = validate_method(
        oracle_as_method,
        n_cohorts=N_COHORTS, cohort_factory=None, sizes=SIZES, skill_set=SKILL_SET,
        weights=None, n_samples=N_SAMPLES, base_seed=BASE_SEED,
        team_metric=TEAM_METRIC, threshold=None,
    )

    if summary["n_oracle_infeasible"] != 0:
        failures.append(f"expected every cohort brute-feasible, got {summary['n_oracle_infeasible']} infeasible")

    if summary["beats_random_fraction"] != 1.0:
        failures.append(f"expected beats_random_fraction == 1.0, got {summary['beats_random_fraction']}")

    if summary["hits_ceiling_fraction"] != 1.0:
        failures.append(f"expected hits_ceiling_fraction == 1.0, got {summary['hits_ceiling_fraction']}")

    for r in records:
        if r.easy is False and r.gap_closed_mean is not None and abs(r.gap_closed_mean - 1.0) > 1e-9:
            failures.append(f"seed {r.seed}: expected gap_closed_mean == 1.0 on a non-easy cohort, got {r.gap_closed_mean}")
        if r.brute_feasible and not r.hits_ceiling:
            failures.append(f"seed {r.seed}: oracle-as-method did not hit its own ceiling")
        if r.brute_feasible and not r.exact_match:
            # The oracle's own best_partition is picked by score_partition's
            # own tie-break (first partition found with the max score during
            # enumeration), so this SHOULD always match itself exactly, not
            # merely tie on score. A mismatch here would mean
            # canonical_partition or the enumeration order is not
            # deterministic, which is a real bug worth catching.
            failures.append(f"seed {r.seed}: oracle-as-method did not exactly match its own best partition")

    print(f"ran {N_COHORTS} cohorts, {summary['n_easy']} easy, {summary['n_oracle_infeasible']} oracle-infeasible")
    print(f"beats_random_fraction = {summary['beats_random_fraction']}")
    print(f"hits_ceiling_fraction = {summary['hits_ceiling_fraction']}")
    print(f"gap_closed_mean stats = {summary['gap_closed_mean_stats']}")

    if failures:
        print()
        print("selfcheck: FAILED")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)

    print()
    print("selfcheck: OK")


if __name__ == "__main__":
    main()
