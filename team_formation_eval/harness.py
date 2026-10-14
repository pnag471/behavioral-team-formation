"""
Runs one method against one cohort and reports where it lands relative
to a random partition and, when feasible, the brute-force best partition.

This is the single-cohort report. validate.py reuses the same pieces
(score_partition, random_baseline, brute_force_best) to do the same
comparison across many cohorts and summarize it; it does not call into
this module, to keep the per-cohort math in exactly one place
(metrics.score_partition / partitioning.random_baseline /
partitioning.brute_force_best) rather than two.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

from metrics import score_partition
from partitioning import brute_force_best, canonical_partition, random_baseline, DEFAULT_ORACLE_CAP


@dataclass(frozen=True)
class SingleCohortReport:
    method_name: str
    seed: int
    method_score: object          # metrics.PartitionScore
    random_floor: float
    random_mean: float
    random_std: float
    n_samples: int
    brute_feasible: bool
    n_partitions: int
    oracle_score: object = None   # metrics.PartitionScore, or None if infeasible
    hits_ceiling: object = None   # bool, or None if infeasible
    exact_match: object = None    # bool, or None if infeasible


def run_single_cohort(
    method, method_name, cohort, sizes, skill_set, weights, n_samples, seed,
    team_metric, threshold, cap=DEFAULT_ORACLE_CAP,
):
    partition = method(cohort, sizes, skill_set, weights, threshold)
    method_score = score_partition(partition, skill_set, weights, threshold)
    method_key = method_score.objective_key(team_metric)

    samples = random_baseline(cohort, sizes, n_samples, seed, skill_set, weights, threshold)
    sample_keys = [s.objective_key(team_metric) for s in samples]
    random_floor = min(sample_keys)
    random_mean = statistics.fmean(sample_keys)
    random_std = statistics.pstdev(sample_keys) if len(sample_keys) > 1 else 0.0

    oracle = brute_force_best(cohort, sizes, skill_set, weights, threshold, cap=cap)
    oracle_score = None
    hits_ceiling = None
    exact_match = None
    if oracle.feasible:
        oracle_score = oracle.best_score
        oracle_key = oracle_score.objective_key(team_metric)
        hits_ceiling = abs(method_key - oracle_key) < 1e-9
        exact_match = canonical_partition(partition) == canonical_partition(oracle.best_partition)

    return SingleCohortReport(
        method_name=method_name,
        seed=seed,
        method_score=method_score,
        random_floor=random_floor,
        random_mean=random_mean,
        random_std=random_std,
        n_samples=n_samples,
        brute_feasible=oracle.feasible,
        n_partitions=oracle.n_partitions,
        oracle_score=oracle_score,
        hits_ceiling=hits_ceiling,
        exact_match=exact_match,
    )


def print_single_cohort_report(report, team_metric):
    method_key = report.method_score.objective_key(team_metric)
    print(f"-- single cohort report: {report.method_name} (seed={report.seed}) --")
    print(f"team_metric = {team_metric}")
    print(f"method score       : {method_key:.4f}")
    print(f"random floor/mean   : {report.random_floor:.4f} / {report.random_mean:.4f} "
          f"(std {report.random_std:.4f}, n={report.n_samples})")
    if report.brute_feasible:
        oracle_key = report.oracle_score.objective_key(team_metric)
        print(f"brute force best    : {oracle_key:.4f} ({report.n_partitions} partitions checked)")
        print(f"hits ceiling        : {report.hits_ceiling}")
        print(f"exact partition match: {report.exact_match}")
    else:
        print(f"brute force         : skipped, {report.n_partitions} partitions exceeds the cap")
