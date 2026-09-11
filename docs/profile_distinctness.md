# Profile distinctness (Part 6.5)

Are the 5 named optimization profiles (`legacy`, `research_balanced`, `fairness_priority`,
`skill_priority`, `preference_priority`) experimentally distinguishable — i.e. do they actually produce
different team assignments, on data a real study might use? Measured, not assumed:
`backend/evaluation/profile_distinctness.py` → `evaluation_output/profile_distinctness.json`. All
similarity measures are **team-label-invariant** (team indices are never compared directly — see that
module's docstrings for the label-correspondence solving and the inherently label-invariant pairwise
co-membership measure).

## Method

For every pair of the 5 profiles, on the real 20-student seed roster and all 12 adversarial fixtures
(feeding each fixture's own familiarity/preference side-channel data through the real
`app.matching.build_familiarity_graph`/`PreferenceEntry` machinery, not a shortcut):

- **percent_students_reassigned** — after solving the best team-index correspondence via greedy
  maximum-overlap matching, the fraction of students on a different team under that correspondence.
- **pairwise_comembership_distance** — fraction of student PAIRS whose "same team?" bit differs. Needs no
  correspondence-solving; inherently label-invariant.
- **adjusted_rand_index (ARI)** — the standard permutation-invariant partition-similarity statistic
  (`sklearn.metrics.adjusted_rand_score`; 1.0 = identical, ~0 = chance-level, can go negative).
- **identical_partition** — `True` iff `pairwise_comembership_distance == 0.0` (checked directly, not
  inferred from a high ARI, since ARI can equal 1.0 in degenerate cases without label-for-label identity).

## Results

| Dataset | Distinct partitions among the 5 profiles | Identical-partition pairs (of 10) |
|---|---|---|
| seed_roster_20 | 2 | 6/10 |
| skill_vs_behavior | 1 | 10/10 |
| hidden_bad_apple | 1 | 10/10 |
| **stacked_team** | 2 | 6/10 |
| isolated_student | 1 | 10/10 |
| friend_clique | 1 | 10/10 |
| utility_vs_fairness | 1 | 10/10 |
| **preference_instability** | 2 | 4/10 |
| uneven_ability | 1 | 10/10 |
| uneven_roster_size | 1 | 10/10 |
| missing_tied_extreme | 1 | 10/10 |
| greedy_local_optimum | 1 | 10/10 |
| **2opt_harms_fairness** | 2 | 4/10 |

**9 of 13 datasets collapse to a single partition across ALL 5 profiles.** Only 4 datasets — the real seed
roster, and the `stacked_team`, `preference_instability`, and `2opt_harms_fairness` adversarial fixtures —
produce more than one distinct assignment at all.

### Where profiles actually diverge

- **seed_roster_20**: `skill_priority` is the ONLY profile that differs from the other four (which are all
  mutually identical to `legacy`). `legacy` vs `skill_priority`: 40% of students reassigned, ARI=0.367,
  co-membership distance=0.168 — a real, moderate-sized difference, consistent with `skill_priority`
  reweighting the PRIMARY objective (not an optional pass) and therefore changing greedy construction's
  search path from the first swap onward.
- **stacked_team**: `research_balanced` is the only profile that differs (25% reassigned, ARI=0.312,
  co-membership distance=0.273) — it is the only profile with `ability.enabled=True` by default among the
  five, and this fixture exists specifically to expose ability stacking.
- **preference_instability**: `research_balanced` and `preference_priority` both differ from `legacy`/
  `fairness_priority`/`skill_priority` (25% reassigned, ARI=0.125, co-membership distance=0.429) — the two
  profiles with `preference.enabled=True`.
- **2opt_harms_fairness**: `research_balanced` and `fairness_priority` both differ from the rest (25%
  reassigned, ARI=0.125, co-membership distance=0.429).

### Where profiles collapse to identical results — and why

- **friend_clique**: even `research_balanced` (familiarity enabled) matches `legacy` exactly. Per
  `docs/optimization_profiles.md`'s existing finding: this fixture's clique fraction (0.5) sits exactly AT
  `max_clique_fraction`'s default threshold (0.5), so `_team_clique_violation` reports zero violation —
  there is nothing for the optimizer to act on, not a bug.
- **utility_vs_fairness**: all 5 profiles match, including `fairness_priority` — no swap-reachable fairness
  improvement exists for this specific construction beyond what `_maximin_improve` already reached (same
  mechanism documented in detail for the seed roster in `research_alignment_audit.md`'s Part 6.5 update).
- **seed_roster_20** (4 of the 5 profiles): `legacy`/`research_balanced`/`fairness_priority`/
  `preference_priority` are all mutually identical — this is the SAME "improving swaps exist but are
  blocked by lexicographic protections / data absent" finding documented in detail via
  `backend/evaluation/trace_seed_roster.py`.
- 8 other adversarial fixtures (`skill_vs_behavior`, `hidden_bad_apple`, `isolated_student`,
  `uneven_ability`, `uneven_roster_size`, `missing_tied_extreme`, `greedy_local_optimum`) show NO
  distinguishing effect from ANY profile change on this dataset — most of these fixtures were designed to
  test a DIFFERENT property (skill coverage, floor violations, roster-size edge cases) than the
  ability/fairness/familiarity/preference criteria the profiles vary, so this is expected, not a finding
  about profile power.

## Study-design implication

**If a real study wants to compare `legacy` against an experimental profile and observe a different
assignment, roster and profile choice both matter enormously.** On 9 of 13 datasets tested here — including
most adversarial fixtures NOT specifically designed to expose ability/fairness/preference differences, and
one real classroom-sized roster missing familiarity/preference data entirely — every profile collapses to
`legacy`'s exact assignment. A study naively running `research_balanced` on an arbitrary roster without
first collecting familiarity/preference data (see `docs/pipeline.md`'s "Missing-data behavior" notes) or
without an ability imbalance already present in the roster should NOT expect to observe a different
assignment purely from selecting a different profile — see `backend/evaluation/experiment_readiness.py`
(item 7) for a per-roster check of exactly this before running an actual study.

This is stated here as a **study-design limitation**, not a code defect: every profile's own passes
correctly enforce their documented non-regression guarantees; the underlying reason so many datasets
collapse is that (a) several fixtures/rosters genuinely have no swap-reachable improvement for the
criterion in question, and (b) for familiarity/preference specifically, the data those passes act on is
frequently simply absent (see `research_alignment_audit.md`'s Part 6.5 update and
`recommended_changes.md`).
