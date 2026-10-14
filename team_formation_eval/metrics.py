"""
Scoring for one partition of students into teams.

A partition is a tuple of teams, each team a tuple of Student. Order of
teams and order of students within a team never affects the score (see
partitioning.canonical_partition for the order-invariant form used to
compare two partitions for exact equality).
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

DEFAULT_WEIGHTS = {"mean": 0.6, "min": 0.3, "balance": 0.1}

# TODO(prisha): no published threshold for "adequately covered" exists for
# this domain. 0.5 (at least half the required skill set present on a
# team) is a placeholder, not a validated cutoff.
DEFAULT_THRESHOLD = 0.5


def _normalize_weights(weights):
    total = sum(weights.values())
    if total <= 0:
        return dict(weights)
    return {k: v / total for k, v in weights.items()}


@dataclass(frozen=True)
class PartitionScore:
    coverage_mean: float
    coverage_min: float
    balance: float
    coverage_adequate_fraction: float
    total: float

    def objective_key(self, team_metric):
        """Returns the scalar named by team_metric (one of this
        dataclass's own field names), for comparing two PartitionScores
        on the same basis. Raises AttributeError on an unknown name,
        same as plain attribute access would."""
        return getattr(self, team_metric)


def _team_coverage(team, skill_set):
    if not skill_set:
        return 1.0
    team_skills = frozenset().union(*(s.skills for s in team)) if team else frozenset()
    return len(team_skills & skill_set) / len(skill_set)


def score_partition(partition, skill_set, weights=None, threshold=None):
    """
    Scores one partition. partition is a sequence of teams (each a
    sequence of Student). skill_set is the set of required skill names.

    weights combine three raw signals into `total`:
      mean    : average per-team skill coverage (0 to 1)
      min     : worst team's skill coverage (0 to 1), the "weakest link"
      balance : 1 minus the spread (population stdev) of per-team
                coverage across the partition, clamped to [0, 1]. 1.0
                means every team covers the same fraction of skill_set;
                0.0 means teams are maximally uneven.
    Missing keys fall back to DEFAULT_WEIGHTS; weights are normalized to
    sum to 1 so partial overrides do not silently shrink `total`.

    threshold feeds coverage_adequate_fraction: the fraction of teams
    whose own coverage meets or exceeds `threshold`. Defaults to
    DEFAULT_THRESHOLD if None. coverage_adequate_fraction is reported on
    PartitionScore but is not part of `total` by default; it exists for a
    team_metric that specifically wants a pass/fail notion of coverage
    rather than a continuous one.
    """
    weights = _normalize_weights({**DEFAULT_WEIGHTS, **(weights or {})})
    threshold = DEFAULT_THRESHOLD if threshold is None else threshold

    per_team_coverage = [_team_coverage(team, skill_set) for team in partition]
    if not per_team_coverage:
        return PartitionScore(0.0, 0.0, 1.0, 0.0, 0.0)

    coverage_mean = statistics.fmean(per_team_coverage)
    coverage_min = min(per_team_coverage)
    spread = statistics.pstdev(per_team_coverage) if len(per_team_coverage) > 1 else 0.0
    balance = max(0.0, 1.0 - min(1.0, spread))
    coverage_adequate_fraction = sum(1 for c in per_team_coverage if c >= threshold) / len(per_team_coverage)

    total = (
        weights.get("mean", 0.0) * coverage_mean
        + weights.get("min", 0.0) * coverage_min
        + weights.get("balance", 0.0) * balance
    )

    return PartitionScore(
        coverage_mean=coverage_mean,
        coverage_min=coverage_min,
        balance=balance,
        coverage_adequate_fraction=coverage_adequate_fraction,
        total=total,
    )
