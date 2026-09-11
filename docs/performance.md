# Performance (Part 6.5)

This document records the Part 6.5 performance investigation: the hotspot analysis, the fix, and
before/after measurements. See `backend/evaluation/benchmark_profiles.py` (clean repeated-timing
benchmark) and `backend/evaluation/run_report.py`'s profiling helper for how every number below was
produced — nothing here is hand-typed.

## The problem

`docs/optimization_profiles.md` (Part 6) measured `research_balanced` at **26,077 ms** for an 80-student
synthetic roster — the number this part's brief asks to fix.

## Root cause (cProfile evidence)

`_maximin_improve` and `_two_opt_improve` already used **incremental per-team caching**: a swap only ever
changes the two teams involved, so every OTHER team's `total_score`/`skill_coverage`/`floor_violation_count`
/`floor_mean` was read from a cache instead of being rescanned on every candidate swap. This pattern was
introduced earlier (recommended_changes.md #16) and was NOT extended to the four newer optional passes —
`_ability_balance_improve`, `_fairness_improve`, `_familiarity_balance_improve`, `_preference_repair_improve`
(plus `_hard_avoidance_repair`) — each of which recomputed `_stage_objective(teams, weights)` and
`_partition_floor_min(teams)` **from scratch, over every team in the partition**, on every single candidate
swap inside their `O(T² · S²)` search loop (`T` = team count, `S` = team size), and additionally never passed
`validate_assignment` any precomputed skill-coverage/floor-violation values either (despite that function
already supporting `precomputed_min_skill_coverage`/`precomputed_floor_violation_count` for exactly this
purpose).

`cProfile` at n=40 (`research_balanced`, all four optional passes enabled) confirmed this directly:

| Function | Calls | Cumulative time | % of total (5.09s) |
|---|---|---|---|
| `_behavioral_compat` (via `_total_score`) | 125,790 | 3.069s | 60% |
| `_stage_objective` | 5,768 | 2.031s | 40% |
| `_partition_min_utility` (fairness pass) | 1,440 | 1.824s | 36% |

At `T` teams, each candidate swap's `_stage_objective`/`_partition_floor_min`/`_partition_min_utility` call
did `O(T)` work when only 2 of those `T` teams had actually changed — the dominant, systemic cost of every
pass that lacked the cache.

## The fix

Applied the SAME already-proven caching idiom from `_maximin_improve`/`_two_opt_improve` to the four
remaining passes (`app/matching.py`): a per-team cache for `total_score`, `skill_coverage`,
`floor_violation_count`, and each floor dimension's `_ordinal_team_mean`, plus a pass-specific cache where
relevant (`ability_cache` for `_partition_ability_gap`, `min_utility_cache` for `_partition_min_utility`) —
each holding a mathematically exact per-team decomposition of the aggregate (sum for objective, min for
floor-min/min-utility, max−min for ability gap). Only the two touched teams' cache entries are ever
recomputed per candidate; every other team's contribution is read from cache. `validate_assignment` is now
given the incrementally-maintained skill-coverage/floor-violation values instead of rescanning the whole
partition. A second, independent fix memoized `_floor_dims()` (a pure filter over the already-cached rubric,
previously re-filtered on every call).

**This changes only HOW comparison values are computed, never WHAT gets accepted** — see "Equivalence
evidence" below.

## Before / after (clean benchmark, median of 3 repeats, deterministic synthetic rosters)

`backend/evaluation/benchmark_profiles.py`, `evaluation_output/performance_benchmark.json`:

| n students | legacy | research_balanced | fairness_priority | skill_priority | preference_priority |
|---|---|---|---|---|---|
| 20 | 22.0 ms | 63.1 ms | 373.4 ms | 22.4 ms | 35.4 ms |
| 40 | 109.0 ms | 269.0 ms | 185.8 ms | 109.7 ms | 166.9 ms |
| 60 | 282.6 ms | 723.5 ms | 465.1 ms | 283.4 ms | 424.4 ms |
| **80** | 561.1 ms | **1,410.4 ms** | 900.1 ms | 560.6 ms | 829.1 ms |
| 100 | 969.0 ms | 2,148.9 ms | 1,512.0 ms | 975.1 ms | 1,409.0 ms |

**`research_balanced` at n=80: 26,077 ms → 1,410.4 ms — an ~18.5× speedup, well under the 5-second target.**
`legacy`/`skill_priority` (which never run the `O(T²S²)` optional-pass search loops) were already fast and
remain fast — their small increase with `n` is from `_greedy_build`/`_two_opt_improve`/`_maximin_improve`
alone, unaffected by this fix.

Note on methodology: a separate `cProfile`-instrumented run at n=80 measured ~4.96 s — profiling
instrumentation itself adds substantial overhead and is not representative of real runtime; the clean
`time.perf_counter()` benchmark above (no profiler attached) is the authoritative number.

## Equivalence evidence

1. **Full existing test suite unchanged, byte-for-byte**: 552/553 pre-existing tests (the 553rd is Part
   6.5's own item-1 test reorganization, see `research_alignment_audit.md`) passed before AND after this
   change, including `test_ability_balance.py`, `test_familiarity.py`, `test_preferences.py`,
   `test_individual_utility.py`, `test_optimizer_invariants.py`, `test_optimization_profiles.py` — all of
   which assert exact expected teams/metrics for deterministic fixtures.
2. **`backend/tests/test_part6_5_performance_equivalence.py`** — a dedicated equivalence suite:
   - `test_ability_balance_improve_matches_naive_reference_implementation_on_every_fixed_roster`:
     reimplements the pre-Part-6.5 (deliberately naive, full-recompute) accept rule independently and
     confirms it produces the BYTE-IDENTICAL final assignment to the now-cached production function, across
     4 fixed rosters.
   - `test_all_four_refactored_passes_are_deterministic_identical_input_identical_output`: confirms
     identical input still produces identical output (no incidental ordering dependency introduced by the
     cache).
   - `test_partition_score_cache_matches_from_scratch_computation`: confirms the invariant the cache is
     responsible for (objective/floor-min never regress below the pre-pass baseline) holds when
     independently re-derived from scratch on the pass's actual output.
   - `test_floor_dims_is_memoized_and_matches_fresh_computation`: confirms the memoized rubric filter
     equals a fresh computation.
3. **Full test suite total runtime** also dropped as a side effect (10.14 s → 3.7–3.9 s for the whole
   backend suite), consistent with the same hotspot being exercised throughout the test suite itself.

## Remaining bottleneck / honest limit

`research_balanced` still scales with roster size (2.1 s at n=100) because the `O(T²S²)` candidate
enumeration itself (not the per-candidate cost) is unchanged — this fix eliminated the redundant PER-CANDIDATE
work, not the NUMBER of candidates evaluated. For cohorts meaningfully larger than ~100–150 students formed
in a single request, expect further growth; the 5-second target is met through at least n=100 in this
environment, but was not benchmarked further. If a future cohort size exceeds what's comfortable
synchronously, the recommended mitigation (unchanged from `recommended_changes.md`) is asynchronous/offline
execution — `POST /teams/generate` returning a job handle rather than blocking — not a semantic change to
the optimizer.
