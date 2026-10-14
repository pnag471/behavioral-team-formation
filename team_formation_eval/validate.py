"""
Cross-cohort validation of one team-formation method against a random
floor and a brute-force ceiling.

For each cohort this computes four things (see validate_method's
docstring for exact definitions), then summarizes all of them across
many cohorts into one verdict: VALIDATED, INCONSISTENT, or WEAK.

Reuses metrics.score_partition (through PartitionScore.objective_key),
partitioning.random_baseline, partitioning.brute_force_best, and
partitioning.canonical_partition. Does not reimplement any of them.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

from metrics import score_partition
from partitioning import DEFAULT_ORACLE_CAP, brute_force_best, canonical_partition, count_partitions, random_baseline
from seed_data import make_synthetic_cohort

VALIDATED = "VALIDATED"
INCONSISTENT = "INCONSISTENT"
WEAK = "WEAK"

# TODO(owner): these verdict cutoffs are judgment calls, not derived from
# anything. They decide what counts as "consistently beats random" and
# "clustered well above zero" in plain language below. Revisit once real
# cohorts are in use and the numbers mean something to a human reader.
VERDICT_BEATS_RANDOM_FRACTION_FOR_VALIDATED = 0.9
VERDICT_MEDIAN_GAP_FOR_VALIDATED = 0.5
VERDICT_HITS_CEILING_FRACTION_FOR_VALIDATED = 0.9
VERDICT_NEGATIVE_GAP_FOR_INCONSISTENT = -0.05
VERDICT_BEATS_RANDOM_FRACTION_FOR_WEAK = 0.5
VERDICT_HITS_CEILING_FRACTION_FOR_WEAK = 0.5


@dataclass(frozen=True)
class CohortResult:
    seed: int
    method_key: float
    random_floor: float
    random_mean: float
    random_std: float
    n_samples: int
    # Item 1: percentile evidence. percentile_mean is the fraction of
    # random samples the method meets or beats; this is the number the
    # "beats random" verdict is actually based on. percentile_floor only
    # checks whether the method clears the single worst random sample,
    # which almost anything will do; it is included for completeness
    # next to gap_closed_floor below, not as strong evidence on its own.
    percentile_mean: float
    percentile_floor: float
    # Item 2: gap closed, floor- and mean-anchored. None when the
    # corresponding denominator is ~0 (no gap to close) or when brute
    # force was infeasible (no optimal_key to anchor against).
    gap_closed_mean: object
    gap_closed_floor: object
    # Item 3: is there a gap worth closing on this cohort at all. None
    # when brute force was infeasible, since "easy" is defined relative
    # to optimal_key.
    easy: object
    # Item 4: does the method reach the brute-force ceiling. Both None
    # when brute force was infeasible for this cohort.
    brute_feasible: bool
    n_partitions: int
    optimal_key: object = None
    hits_ceiling: object = None
    exact_match: object = None


def _percentile(sample_keys, value):
    """Fraction of sample_keys that value meets or beats."""
    if not sample_keys:
        return None
    return sum(1 for k in sample_keys if k <= value) / len(sample_keys)


def _gap_closed(method_key, baseline_key, optimal_key):
    denom = optimal_key - baseline_key
    if abs(denom) < 1e-9:
        return None
    return (method_key - baseline_key) / denom


def validate_method(
    current_logic, *, n_cohorts, cohort_factory=None, sizes, skill_set, weights,
    n_samples, base_seed, team_metric, threshold,
    beats_random_pct=0.95, easy_std_mult=0.5, oracle_cap=DEFAULT_ORACLE_CAP,
):
    """
    Runs current_logic over n_cohorts cohorts and validates it against a
    random floor and a brute-force ceiling on each one.

    current_logic(cohort, sizes, skill_set, weights, threshold) -> partition
        the method under test. Same call signature as
        partitioning.current_logic_placeholder / greedy_coverage_logic.

    cohort_factory(seed) -> list[Student]
        defaults to seed_data.make_synthetic_cohort bound to
        n_students=sum(sizes) and the given skill_set.
        TODO(prisha): replace with real cohorts (real L0-L5 pulls, or
        resampled real data), one per seed.

    team_metric: which PartitionScore field to validate on (passed to
    PartitionScore.objective_key), e.g. "coverage_min" or "total".

    threshold: forwarded to score_partition for coverage_adequate_fraction.
    See metrics.score_partition's docstring; not used by the validation
    math here directly.

    Determinism: cohort i uses seed base_seed + i for both cohort
    generation and the random baseline's own sampling. Two calls with the
    same arguments produce byte-identical records and summary.

    Returns (records, summary):
      records: list[CohortResult], one per cohort, in seed order.
      summary: dict, see _summarize's docstring for its exact keys.
    """
    if cohort_factory is None:
        def cohort_factory(seed):
            return make_synthetic_cohort(seed, n_students=sum(sizes), skill_set=skill_set)

    records = []
    for i in range(n_cohorts):
        seed = base_seed + i
        cohort = cohort_factory(seed)

        partition = current_logic(cohort, sizes, skill_set, weights, threshold)
        method_score = score_partition(partition, skill_set, weights, threshold)
        method_key = method_score.objective_key(team_metric)

        samples = random_baseline(cohort, sizes, n_samples, seed, skill_set, weights, threshold)
        sample_keys = [s.objective_key(team_metric) for s in samples]
        random_floor = min(sample_keys)
        random_mean = statistics.fmean(sample_keys)
        random_std = statistics.pstdev(sample_keys) if len(sample_keys) > 1 else 0.0

        percentile_mean = _percentile(sample_keys, method_key)
        percentile_floor = 1.0 if method_key >= random_floor - 1e-9 else 0.0

        n_partitions = count_partitions(sizes)
        oracle = brute_force_best(cohort, sizes, skill_set, weights, threshold, cap=oracle_cap)

        optimal_key = None
        gap_closed_mean = None
        gap_closed_floor = None
        easy = None
        hits_ceiling = None
        exact_match = None

        if oracle.feasible:
            optimal_key = oracle.best_score.objective_key(team_metric)
            gap_closed_mean = _gap_closed(method_key, random_mean, optimal_key)
            gap_closed_floor = _gap_closed(method_key, random_floor, optimal_key)
            # TODO(owner): boundary case, not fixed silently. When a cohort
            # has zero gap AND zero variance (every partition scores the
            # same, e.g. every student already has every skill), this is
            # `0 < easy_std_mult * 0`, which is False under strict "<".
            # That cohort is intuitively as "easy" as it gets, but this
            # formula (given, not invented here) does not flag it. It does
            # not corrupt gap_closed_mean_stats either way, since
            # gap_closed_mean is already None whenever the denominator is
            # ~0 (see _gap_closed), and None values are filtered out of the
            # stats regardless of which bucket (easy vs evaluated_for_gap)
            # the cohort lands in. Only the n_easy / n_evaluated_for_gap
            # counts are affected at this exact boundary.
            easy = (optimal_key - random_mean) < easy_std_mult * random_std
            hits_ceiling = abs(method_key - optimal_key) < 1e-9
            exact_match = canonical_partition(partition) == canonical_partition(oracle.best_partition)

        records.append(CohortResult(
            seed=seed,
            method_key=method_key,
            random_floor=random_floor,
            random_mean=random_mean,
            random_std=random_std,
            n_samples=n_samples,
            percentile_mean=percentile_mean,
            percentile_floor=percentile_floor,
            gap_closed_mean=gap_closed_mean,
            gap_closed_floor=gap_closed_floor,
            easy=easy,
            brute_feasible=oracle.feasible,
            n_partitions=n_partitions,
            optimal_key=optimal_key,
            hits_ceiling=hits_ceiling,
            exact_match=exact_match,
        ))

    summary = _summarize(records, beats_random_pct)
    return records, summary


def _quantiles_or_none(values):
    if not values:
        return {"min": None, "p25": None, "median": None, "p75": None, "max": None, "mean": None}
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        v = sorted_values[0]
        return {"min": v, "p25": v, "median": v, "p75": v, "max": v, "mean": v}
    quantile_points = statistics.quantiles(sorted_values, n=4, method="inclusive")
    return {
        "min": sorted_values[0],
        "p25": quantile_points[0],
        "median": statistics.median(sorted_values),
        "p75": quantile_points[2],
        "max": sorted_values[-1],
        "mean": statistics.fmean(sorted_values),
    }


def _summarize(records, beats_random_pct):
    """
    summary keys:
      n_cohorts, n_easy, n_evaluated_for_gap, n_oracle_infeasible
      gap_closed_mean_stats   : dict from _quantiles_or_none, over
                                 non-easy, brute-feasible cohorts only
      beats_random_fraction   : fraction of ALL cohorts with
                                 percentile_mean >= beats_random_pct
      hits_ceiling_fraction   : fraction of brute-feasible cohorts with
                                 hits_ceiling True
      exact_match_fraction    : fraction of brute-feasible cohorts with
                                 exact_match True
      n_brute_feasible        : how many cohorts that ceiling fraction is over
      verdict, verdict_reason
    """
    n_cohorts = len(records)
    oracle_infeasible = [r for r in records if not r.brute_feasible]
    easy = [r for r in records if r.easy is True]
    evaluated_for_gap = [r for r in records if r.easy is False]

    gap_closed_mean_values = [r.gap_closed_mean for r in evaluated_for_gap if r.gap_closed_mean is not None]
    gap_closed_mean_stats = _quantiles_or_none(gap_closed_mean_values)

    beats_random_count = sum(1 for r in records if r.percentile_mean is not None and r.percentile_mean >= beats_random_pct)
    beats_random_fraction = beats_random_count / n_cohorts if n_cohorts else None

    brute_feasible = [r for r in records if r.brute_feasible]
    n_brute_feasible = len(brute_feasible)
    hits_ceiling_fraction = (
        sum(1 for r in brute_feasible if r.hits_ceiling) / n_brute_feasible if n_brute_feasible else None
    )
    exact_match_fraction = (
        sum(1 for r in brute_feasible if r.exact_match) / n_brute_feasible if n_brute_feasible else None
    )

    verdict, verdict_reason = _decide_verdict(
        beats_random_fraction, hits_ceiling_fraction, n_brute_feasible,
        gap_closed_mean_stats, gap_closed_mean_values,
    )

    return {
        "n_cohorts": n_cohorts,
        "n_easy": len(easy),
        "n_evaluated_for_gap": len(evaluated_for_gap),
        "n_oracle_infeasible": len(oracle_infeasible),
        "gap_closed_mean_stats": gap_closed_mean_stats,
        "beats_random_fraction": beats_random_fraction,
        "n_brute_feasible": n_brute_feasible,
        "hits_ceiling_fraction": hits_ceiling_fraction,
        "exact_match_fraction": exact_match_fraction,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
    }


def _decide_verdict(beats_random_fraction, hits_ceiling_fraction, n_brute_feasible, gap_stats, gap_values):
    # No ceiling evidence at all (every cohort was oracle-infeasible)
    # counts as "no ceiling evidence" rather than silently passing that
    # bar; VALIDATED still requires either real ceiling evidence or an
    # explicit absence of brute-feasible cohorts to check it against.
    ceiling_ok = (n_brute_feasible == 0) or (
        hits_ceiling_fraction is not None and hits_ceiling_fraction >= VERDICT_HITS_CEILING_FRACTION_FOR_VALIDATED
    )
    beats_random_ok = beats_random_fraction is not None and beats_random_fraction >= VERDICT_BEATS_RANDOM_FRACTION_FOR_VALIDATED
    median_gap_ok = gap_stats["median"] is not None and gap_stats["median"] >= VERDICT_MEDIAN_GAP_FOR_VALIDATED

    if beats_random_ok and median_gap_ok and ceiling_ok:
        return VALIDATED, (
            f"beats random on {beats_random_fraction:.0%} of cohorts, "
            f"median gap closed {gap_stats['median']:.2f}, "
            f"hits ceiling on {hits_ceiling_fraction if n_brute_feasible else 'no brute-feasible cohorts'}"
        )

    if any(g < VERDICT_NEGATIVE_GAP_FOR_INCONSISTENT for g in gap_values):
        return INCONSISTENT, (
            f"gap closed goes negative on at least one cohort (min {min(gap_values):.2f}) "
            f"while other cohorts look fine; the method is not reliable cohort to cohort"
        )

    if (beats_random_fraction is not None and beats_random_fraction < VERDICT_BEATS_RANDOM_FRACTION_FOR_WEAK) or (
        hits_ceiling_fraction is not None and hits_ceiling_fraction < VERDICT_HITS_CEILING_FRACTION_FOR_WEAK
    ):
        return WEAK, (
            f"beats random on only {beats_random_fraction:.0%} of cohorts and/or "
            f"hits ceiling on only {hits_ceiling_fraction if hits_ceiling_fraction is not None else 'n/a'} "
            f"of brute-feasible cohorts"
        )

    return INCONSISTENT, (
        "does not clearly meet the VALIDATED bar or the WEAK bar; "
        f"beats_random_fraction={beats_random_fraction}, median_gap={gap_stats['median']}, "
        f"hits_ceiling_fraction={hits_ceiling_fraction}"
    )


def print_validation_report(method_name, records, summary, team_metric):
    print(f"== cross-cohort validation: {method_name} ==")
    print(f"team_metric = {team_metric}, n_cohorts = {summary['n_cohorts']}")
    print(f"easy cohorts (no real gap to close): {summary['n_easy']}")
    print(f"oracle infeasible (too many partitions to brute force): {summary['n_oracle_infeasible']}")
    print(f"evaluated for gap_closed: {summary['n_evaluated_for_gap']}")
    print()

    stats = summary["gap_closed_mean_stats"]
    if stats["mean"] is not None:
        print("gap_closed_mean distribution over non-easy cohorts:")
        print(f"  min={stats['min']:.3f} p25={stats['p25']:.3f} median={stats['median']:.3f} "
              f"p75={stats['p75']:.3f} max={stats['max']:.3f} mean={stats['mean']:.3f}")
    else:
        print("gap_closed_mean distribution: no non-easy cohorts to summarize")
    print()

    print(f"beats random (percentile_mean >= threshold): {summary['beats_random_fraction']:.0%} of cohorts")
    if summary["n_brute_feasible"]:
        print(f"hits ceiling: {summary['hits_ceiling_fraction']:.0%} of {summary['n_brute_feasible']} brute-feasible cohorts")
        print(f"exact partition match: {summary['exact_match_fraction']:.0%} of {summary['n_brute_feasible']} brute-feasible cohorts")
    else:
        print("hits ceiling: no brute-feasible cohorts to check")
    print()

    print(f"VERDICT: {summary['verdict']}")
    print(f"  {summary['verdict_reason']}")
    print()

    print(f"{'seed':>6}  {'pct_mean':>9}  {'gap_mean':>9}  {'easy':>6}  {'hits_ceiling':>12}")
    for r in records:
        gap_str = f"{r.gap_closed_mean:.3f}" if r.gap_closed_mean is not None else "n/a"
        pct_str = f"{r.percentile_mean:.3f}" if r.percentile_mean is not None else "n/a"
        print(f"{r.seed:>6}  {pct_str:>9}  {gap_str:>9}  {str(r.easy):>6}  {str(r.hits_ceiling):>12}")
