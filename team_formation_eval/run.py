"""
Entry point. Run with:

    cd team_formation_eval && python3 run.py

(Must be run from inside this directory, or as a direct script path like
`python3 team_formation_eval/run.py` from one level up. Python puts the
running script's own directory on sys.path automatically, which is what
lets the sibling imports below resolve; this is a flat script collection,
not an installed package.)

Prints, in order:
  1. a single-cohort report (harness.run_single_cohort), the kind of
     check you'd run by hand on one roster
  2. cross-cohort validation of current_logic_placeholder (the snake
     draft) over many synthetic cohorts
  3. the same validation, same cohorts, for greedy_coverage_logic

Reports 2 and 3 use the same sizes/skill_set/weights/seeds, so they are
a direct, like-for-like comparison.
"""
from __future__ import annotations

from harness import print_single_cohort_report, run_single_cohort
from partitioning import current_logic_placeholder, greedy_coverage_logic
from seed_data import make_synthetic_cohort
from validate import print_validation_report, validate_method

SKILL_SET = frozenset({"frontend", "backend", "design", "testing", "ml", "devops"})
SIZES = (3, 3, 3, 3)          # 4 teams of 3, 12 students per cohort
WEIGHTS = None                 # metrics.DEFAULT_WEIGHTS
N_SAMPLES = 300
BASE_SEED = 1000
TEAM_METRIC = "total"
THRESHOLD = None               # metrics.DEFAULT_THRESHOLD
N_COHORTS = 25


def main():
    print("############################################")
    print("# 1. single-cohort report (greedy_coverage) #")
    print("############################################")
    cohort = make_synthetic_cohort(seed=BASE_SEED, n_students=sum(SIZES), skill_set=SKILL_SET)
    report = run_single_cohort(
        greedy_coverage_logic, "greedy_coverage_logic", cohort, SIZES, SKILL_SET, WEIGHTS,
        N_SAMPLES, BASE_SEED, TEAM_METRIC, THRESHOLD,
    )
    print_single_cohort_report(report, TEAM_METRIC)
    print()

    print("###################################################")
    print("# 2. cross-cohort validation: current_logic_placeholder (snake draft) #")
    print("###################################################")
    records, summary = validate_method(
        current_logic_placeholder,
        n_cohorts=N_COHORTS, cohort_factory=None, sizes=SIZES, skill_set=SKILL_SET,
        weights=WEIGHTS, n_samples=N_SAMPLES, base_seed=BASE_SEED,
        team_metric=TEAM_METRIC, threshold=THRESHOLD,
    )
    print_validation_report("current_logic_placeholder", records, summary, TEAM_METRIC)
    print()

    print("###################################################")
    print("# 3. cross-cohort validation: greedy_coverage_logic #")
    print("###################################################")
    records, summary = validate_method(
        greedy_coverage_logic,
        n_cohorts=N_COHORTS, cohort_factory=None, sizes=SIZES, skill_set=SKILL_SET,
        weights=WEIGHTS, n_samples=N_SAMPLES, base_seed=BASE_SEED,
        team_metric=TEAM_METRIC, threshold=THRESHOLD,
    )
    print_validation_report("greedy_coverage_logic", records, summary, TEAM_METRIC)


if __name__ == "__main__":
    main()
