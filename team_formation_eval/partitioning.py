"""
Ways to produce a partition of students into teams, plus the two
reference points everything else validates against:

  random_baseline   : the floor. What do you get by luck alone.
  brute_force_best  : the ceiling. What is the best possible partition,
                       found by exhaustive search (small cohorts only).

And two concrete team-building methods to validate against those two
reference points:

  current_logic_placeholder : a naive snake draft. Does not look at
                               which specific skills are missing.
  greedy_coverage_logic     : greedy seed-and-fill by skill coverage.

All functions here take students/sizes/skill_set/weights/threshold and
return either a partition (tuple of tuples of Student) or, for
random_baseline, a list of PartitionScore.
"""
from __future__ import annotations

import itertools
import random
from collections import Counter
from dataclasses import dataclass
from math import factorial

from metrics import score_partition

# TODO(prisha): no principled cap exists yet for "small enough to brute
# force in a reasonable time on a laptop." 200,000 partitions keeps each
# oracle run under a few seconds for the scoring function above; revisit
# if score_partition gets more expensive or cohorts get bigger.
DEFAULT_ORACLE_CAP = 200_000


def canonical_partition(partition):
    """Order-invariant form of a partition, for exact_match comparisons.
    Two partitions that assign the same students to the same groups
    canonicalize to the same value regardless of team order or
    within-team student order."""
    return frozenset(frozenset(s.id for s in team) for team in partition)


def _validate_sizes(students, sizes):
    if sum(sizes) != len(students):
        raise ValueError(
            f"sizes {sizes} sum to {sum(sizes)}, but got {len(students)} students"
        )


def count_partitions(sizes):
    """
    Number of distinct ways to split len(sizes) distinguishable students
    into unlabeled groups of the given sizes (groups are unlabeled: two
    groups of the same size are interchangeable, since nothing downstream
    cares which group is "team 1" versus "team 2").

    n! / (prod(size_i!) * prod(count_of_each_distinct_size!))
    """
    n = sum(sizes)
    total = factorial(n)
    for size in sizes:
        total //= factorial(size)
    for count in Counter(sizes).values():
        total //= factorial(count)
    return total


def _enumerate_partitions(students, sizes):
    """
    Yields every distinct partition of students into groups of the given
    sizes, as tuples of tuples of Student, with groups of equal size
    deduplicated (see count_partitions's docstring: groups are unlabeled).

    Dedup trick: when filling slot k, if an earlier slot j < k has the
    same size as slot k, slot k's chosen group (compared as a sorted
    tuple of indices) must be strictly greater than slot j's. This is
    transitive across a whole run of equal-size slots, so comparing only
    against the most recent same-size slot is sufficient.
    """
    _validate_sizes(students, sizes)
    n = len(students)
    all_idx = tuple(range(n))

    def recurse(used, slot, chosen):
        if slot == len(sizes):
            yield tuple(chosen)
            return
        size = sizes[slot]
        remaining = tuple(i for i in all_idx if i not in used)

        prev_same_size = None
        for j in range(slot - 1, -1, -1):
            if sizes[j] == size:
                prev_same_size = chosen[j]
                break

        for combo in itertools.combinations(remaining, size):
            if prev_same_size is not None and combo <= prev_same_size:
                continue
            yield from recurse(used | frozenset(combo), slot + 1, chosen + [combo])

    for idx_partition in recurse(frozenset(), 0, []):
        yield tuple(tuple(students[i] for i in group) for group in idx_partition)


@dataclass(frozen=True)
class BruteForceResult:
    feasible: bool
    n_partitions: int
    best_partition: object = None  # tuple of tuples of Student, or None if infeasible
    best_score: object = None      # metrics.PartitionScore, or None if infeasible


def brute_force_best(students, sizes, skill_set, weights=None, threshold=None, cap=DEFAULT_ORACLE_CAP):
    """
    Exhaustive search for the best-scoring partition, by total score
    (metrics.PartitionScore.total). Returns a BruteForceResult; feasible
    is False (best_partition/best_score left None) when count_partitions
    exceeds cap, so a caller can skip or annotate instead of hanging.
    """
    _validate_sizes(students, sizes)
    n_partitions = count_partitions(sizes)
    if n_partitions > cap:
        return BruteForceResult(feasible=False, n_partitions=n_partitions)

    best_partition = None
    best_score = None
    for partition in _enumerate_partitions(students, sizes):
        score = score_partition(partition, skill_set, weights, threshold)
        if best_score is None or score.total > best_score.total:
            best_score = score
            best_partition = partition

    return BruteForceResult(
        feasible=True, n_partitions=n_partitions,
        best_partition=best_partition, best_score=best_score,
    )


def random_baseline(students, sizes, n_samples, seed, skill_set, weights=None, threshold=None):
    """
    Scores n_samples random partitions of students into the given sizes.
    Deterministic for a fixed seed: same students (by id, in the same
    input order), same sizes, same seed always produces the same list of
    scores in the same order.

    Returns a list of metrics.PartitionScore, one per sample.
    """
    _validate_sizes(students, sizes)
    rng = random.Random(seed)
    scores = []
    for _ in range(n_samples):
        shuffled = list(students)
        rng.shuffle(shuffled)
        partition = []
        start = 0
        for size in sizes:
            partition.append(tuple(shuffled[start:start + size]))
            start += size
        scores.append(score_partition(partition, skill_set, weights, threshold))
    return scores


def current_logic_placeholder(students, sizes, skill_set, weights=None, threshold=None):
    """
    Naive snake draft. Ranks students by their raw skill count
    (descending, ties broken by id for determinism), then deals them to
    teams in snake order (0,1,...,k-1,k-1,...,1,0,0,1,...), filling each
    team up to its target size as its turn comes around.

    Does not look at skill_set at all while assigning: a student with 3
    overlapping skills counts the same as a student with 3 skills that
    cover 3 different gaps. This is the point: it is a plausible-looking
    but coverage-blind placeholder, kept around as the thing
    greedy_coverage_logic should clearly beat.
    """
    _validate_sizes(students, sizes)
    ranked = sorted(students, key=lambda s: (-len(s.skills), s.id))

    n_teams = len(sizes)
    teams = [[] for _ in range(n_teams)]
    remaining_capacity = list(sizes)

    order = list(range(n_teams)) + list(range(n_teams - 1, -1, -1))
    pos = 0
    for student in ranked:
        # advance to the next team in snake order that still has room
        for _ in range(2 * n_teams):
            team_idx = order[pos % len(order)]
            pos += 1
            if remaining_capacity[team_idx] > 0:
                teams[team_idx].append(student)
                remaining_capacity[team_idx] -= 1
                break
        else:
            raise RuntimeError("no team had room; sizes do not match student count")

    return tuple(tuple(team) for team in teams)


def greedy_coverage_logic(students, sizes, skill_set, weights=None, threshold=None):
    """
    Greedy seed-and-fill by skill coverage, the same shape as this
    project's own production heuristic: seed each team with the student
    who alone covers the most distinct skills in skill_set, then
    repeatedly add whichever (team, remaining student) pair raises that
    team's score_partition total the most, respecting each team's target
    size. Deterministic: ties are broken by student id, in a fixed scan
    order.
    """
    _validate_sizes(students, sizes)
    n_teams = len(sizes)
    teams = [[] for _ in range(n_teams)]
    remaining = sorted(students, key=lambda s: s.id)

    def skill_score(s):
        return len(s.skills & skill_set)

    for team_idx in range(n_teams):
        if not remaining:
            break
        seed = max(remaining, key=lambda s: (skill_score(s), s.id))
        teams[team_idx].append(seed)
        remaining.remove(seed)

    while remaining:
        best_team_idx = None
        best_student = None
        best_total = None
        for team_idx in range(n_teams):
            if len(teams[team_idx]) >= sizes[team_idx]:
                continue
            for candidate in remaining:
                trial = score_partition([tuple(teams[team_idx] + [candidate])], skill_set, weights, threshold).total
                if best_total is None or trial > best_total:
                    best_total = trial
                    best_team_idx = team_idx
                    best_student = candidate
        teams[best_team_idx].append(best_student)
        remaining.remove(best_student)

    return tuple(tuple(team) for team in teams)
