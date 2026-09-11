# Recommended Changes

Ranked gaps from [`research_alignment_audit.md`](research_alignment_audit.md) /
[`research_traceability_matrix.md`](research_traceability_matrix.md). Evidence for every item is either a
file:line citation or an executed result in
[`research_alignment_results.json`](research_alignment_results.json) — nothing here is asserted without a
pointer to where it was checked.

Legend for the last column: **Hard constraint** = must never be violated by a returned team; **Soft
constraint** = penalize in the objective but allow if infeasible; **Metric** = surface for the instructor,
don't auto-optimize; **Config option** = expose as an instructor-facing setting.

---

## Critical

### 1. The "floor" guarantee the rubric documents is not what the code enforces
- **Research criterion affected:** Conscientiousness / agreeableness floor (Bell 2007 bad-apple effect).
- **Evidence from code:** `backend/traits/rubric_v1.yaml:183,238` states *"Floor: no team may include a
  member at the 'low' level"* — zero tolerance. But `matching.py:_check_feasibility()` (L249-264) treats
  the roster as *feasible* whenever floor-level students number **≤ the team count**, and
  `_greedy_build()` (L387-389) deliberately places **one floor-level student per team** to satisfy that.
  `_check_floor_violations()` (L267-295) then applies the *zero-tolerance* rule from the rubric and flags
  every one of those "successfully spread" students as a violation.
- **Evidence from tests:** `constraint_violations.floor_ok == False` on the real 20-student seed roster
  (`research_alignment_results.json → sample_students_evaluation.metrics.constraint_violations`), and in
  6 of 12 adversarial fixtures (`test_no_uncaught_floor_violations_after_full_pipeline`, parametrized).
- **Practical consequence:** `POST /teams/generate` returns `"feasible": true`-shaped success (HTTP 200,
  no 422) for rosters that will *always* carry non-empty `floor_violations` in the response whenever any
  floor-level student exists in the pool at all — the field exists but the situation it flags is close to
  the common case, not the exceptional one. An instructor UI that doesn't specifically surface
  `floor_violations` (current frontend does not, per `docs/mvp-plan.md` scope) will never learn its
  central "no bad apples" claim (`docs/pipeline.md:176-180`) didn't hold.
- **Recommended change:** This is a team decision already tracked as `CHANGES.md` TODO #3, but it's a
  different, unaddressed question: pick ONE semantic and align the other three functions (`_check_
  feasibility`, `_greedy_build`, `_check_floor_violations`) to it. If "one-per-team" is the intended
  policy (defensible — Bell's finding is about *concentration*, not presence), change the rubric wording
  and `_check_floor_violations()` to flag only teams with **≥2** floor-level members on the same floor
  dimension. If true zero-tolerance is intended, `_check_feasibility` must reject any roster with ≥1
  floor-level student unless the instructor accepts an under-strength team explicitly.
- **Difficulty:** Small (rewording + one conditional change) once the semantic is decided; the decision
  itself needs a human call.
- **Type:** Hard constraint (either version) — currently shipped as a **reported-but-unenforced** metric,
  which is the actual bug.

### 2. "Mixed ability" has no representation in the objective at all
- **Research criterion affected:** Mixed ability, within- and between-team.
- **Evidence from code:** No `ability` field exists in `Student`/`CompetenceSignature` (`models.py:6-11`).
  `experience_level` is the only plausible proxy and is never read by `_total_score()`,
  `_behavioral_compat()`, or any `composition_rule` (`grep -n experience_level backend/app/matching.py`
  → no hits outside display fields at L545).
- **Evidence from tests:** `test_stacked_ability_gets_redistributed_across_teams` **FAILED** — mean-ability
  spread of 1.00 (of a possible 2.0) across 3 teams built from a roster order-blocked strong/mid/weak.
  On the real seed roster, per-team mean ability ranges 0.75–1.75
  (`research_alignment_results.json → sample_students_evaluation.metrics.ability_diversity`).
- **Practical consequence:** Any instructor who uploads (or the system ever ingests) a roster where
  ability correlates with upload/section order will get ability-stacked teams with no warning, mirroring
  the exact failure mode the research brief's "mixed ability" criterion exists to prevent.
- **Recommended change:** (a) Add a validated or instructor-supplied ability signal — do not invent one
  from behavioral MCQ answers (that would reintroduce the "checkbox relabeling" problem
  `docs/pipeline.md:32-36` explicitly warns about for behavioral traits). (b) Once it exists, add it as a
  new `high_mean_with_floor`- or dedicated `balanced_distribution`-style `composition_rule` entry, since
  the research brief wants *mixing*, not maximizing — the current `high_mean_with_floor` rule maximizes
  the mean, which is the wrong shape for ability (you don't want every team's ability as high as possible,
  you want it *evenly split*).
- **Difficulty:** Medium — needs a new composition rule type (`app/matching.py` + `rubric_v1.yaml`) plus a
  real data source decision (instructor-entered grade/prior-course tier is the safest starting point).
- **Type:** New optimization goal + hard "no team below floor N ability" constraint (configurable).
- **Status — IMPLEMENTED (Part 3), verified:** Option (a): `CompetenceSignature.ability_score` (Optional
  float, [0.0, 1.0], source-tagged via `ability_source` — NOT hardcoded to GPA, no value invented from
  behavioral MCQ data) plus optional per-skill `skill_proficiency`. Option (b), adapted: rather than routing
  through the rubric's `composition_rule` machinery (built for MCQ-derived behavioral dims, not competence
  data), a dedicated `_ability_balance_improve()` pass targets the *gap* directly — it accepts a swap only
  if it strictly narrows (strongest_team_mean_ability − weakest_team_mean_ability), which is the "mixing,
  not maximizing" shape this recommendation called for, achieved without a new rubric dimension type.
  `experience_level` is still the fallback when `ability_score` is unset (`_student_ability_score()`), so
  every existing record — including 10 of the 20 seed students left deliberately without an `ability_score`
  — continues to produce a resolvable ability value. Off by default (`AbilityConfig.enabled=false`);
  metrics (`compute_ability_metrics()` / `evaluation/metrics.py:ability_balance()`) are always computed and
  reported regardless. On the real 20-student seed roster: `strongest_weakest_gap=0.183`,
  `between_team_mean_variance=0.0046`, distribution `{developing: 2, medium: 11, high: 7}`,
  skill-coverage/mean-ability correlation `r=0.58`. See `backend/tests/test_ability_balance.py` (63 tests),
  including one proving `test_stacked_ability_gets_redistributed_across_teams`'s exact fixture is now
  fixable when `ability.enabled=true`, without that test itself being weakened. **Explicitly still a
  research hypothesis, not a validated finding** — `AbilityConfig`'s docstring and
  `research_alignment_audit.md`'s Part 3 update both flag that no outcome data in this repo supports
  "balanced ability" over "stacked ability" for this population; see the Part 3 report's "research
  assumptions still requiring validation."

### 3. No individual-level utility — "fair distribution of benefit" cannot be represented, let alone measured, by the production algorithm
- **Research criterion affected:** Fair distribution of benefit.
- **Evidence from code:** `TeamScoreBreakdown` (`models.py:103-109`) is a per-**team** structure; every
  member of a team is assigned identically via `_build_teams()` (`matching.py:531-538`). There is no
  function anywhere that computes a *student-specific* number.
- **Evidence from tests:** `evaluation/metrics.py:individual_utility()` had to *invent* a proxy definition
  (utility = the team's score) purely to make the criterion testable at all; even so, on the seed roster
  `utility_variance` is non-zero and `min_utility=0.530` vs. `mean_utility=0.584`
  (`research_alignment_results.json`).
- **Practical consequence:** The system can currently only ever report "team 3 scored lower than team 1,"
  never "student X individually did worse than they needed to" — which is the harder, more actionable
  half of the research criterion (a team can score fine in aggregate while one member's alternative
  assignment would have been meaningfully better for them, per fixture #6 / regret evidence).
- **Recommended change:** This does not require inventing new data — it can be computed today from
  existing signatures via the brute-force alternative-team search already implemented in
  `evaluation/metrics.py:individual_utility()`. Promote a bounded version of that (cap search size, as the
  harness already does via `max_exact_regret_roster_size`) into a **diagnostic-only** field on the
  `POST /teams/generate` response (worst-case regret per student), without changing what gets optimized.
- **Difficulty:** Small–Medium (the algorithm exists; wiring it into the production response and capping
  cost for large rosters is the work).
- **Type:** Metric (surfaced, not auto-optimized, until validated against real outcomes).
- **Status — IMPLEMENTED (Part 2), verified:** A real per-student utility now exists —
  `app.matching.compute_assignment_utilities()` / `_individual_utility()` — replacing the old placeholder
  ("utility = the team's total score, identical for every teammate") that `evaluation/metrics.py:
  individual_utility()` previously used. Utility is a weighted average of 4 components built entirely
  from existing schema data (skill_fit, behavioral_fit, availability_fit, interest_fit — see that file's
  module docstring for exact formulas); missing components (e.g. no teammates to compare availability
  against) are tracked explicitly and excluded via renormalized weights, never silently treated as zero.
  `POST /teams/generate` now ALWAYS returns a `fairness_summary` (mean/min/max/range/variance/stdev/total
  utility, worst-case regret, students-below-floor count+fraction) and a per-member `utility` breakdown —
  purely informational, matching this recommendation's "diagnostic-only" framing exactly. On the real
  20-student seed roster, min_utility=0.476 vs. mean=0.602 (previously min=0.530/mean=0.584 under the old
  identical-per-teammate placeholder) — the new numbers actually differ per student, which the old ones
  structurally could not. A SEPARATE, disabled-by-default `fairness.enabled` flag additionally lets the
  optimizer act on this (a new lexicographic `_fairness_improve` pass, after maximin, raising the worst-off
  student's utility without regressing the total-score objective or the floor-minimum) — default matching
  results are unchanged unless an instructor opts in. See `backend/tests/test_individual_utility.py`.

---

## High

### 4. No familiarity or social-graph data — "avoid clique formation" cannot be enforced
- **Research criterion affected:** Familiarity without clique formation.
- **Evidence from code:** Zero matches for `familiar|friend|clique` across `backend/app`,
  `backend/scoring`, `backend/traits`, `backend/questions` outside doc comments.
- **Evidence from tests:** `test_friend_clique_is_not_broken_up_by_production_algorithm` **FAILED** — a
  4-person synthetic clique was not broken up because nothing in the pipeline can see it.
- **Recommended change:** Add either (a) a `Student.prior_teammates: List[str]` self-report field
  collected at onboarding, or (b) a `FamiliarityDB` edge table `(student_a, student_b, strength, source)`
  populated from LMS co-enrollment history. Add a `composition_rule: bounded_familiarity` dimension to the
  rubric once data exists — reward teams above a minimum density (coordination) while capping the fraction
  from any single prior group (anti-clique). `evaluation/metrics.py:familiarity_density()` /
  `clique_concentration()` already implement both halves of this and can be lifted directly into
  `matching.py` once the data source is decided.
- **Difficulty:** Medium (schema + one new data-collection step) once a data source is chosen; the metric
  logic already exists and is tested.
- **Type:** Soft constraint (reward density above a threshold; penalize/cap concentration above another).
- **Status — IMPLEMENTED (Part 4), verified:** Option (b): a `familiarity` edge table
  `(id, student_a_id, student_b_id, strength, source, created_at)`, unique on the pair, populated via
  `POST /familiarity` — `app.models.FamiliarityEdge` documents the `[0.0, 1.0]` scale and the deliberate
  choice of SYMMETRIC (not directed) familiarity. A missing row is UNKNOWN, not zero — every metric and the
  optional soft objective preserve that distinction rather than defaulting a missing pair to 0.0 the way
  the OLD `_familiarity_lookup()` side-channel dict did. Graph built in pure Python (adjacency dict + BFS
  for connected components) — no networkx dependency, per this part's own instruction. The soft objective
  (`_familiarity_balance_improve`, off by default) targets a TWO-PART violation: a configurable density
  band (below `lower_bound` = may lack coordination support, above `upper_bound` = possible clique
  concentration) PLUS a direct clique-concentration term (largest pre-existing connected subgroup exceeding
  `max_clique_fraction` of a team) — the second term was added after empirical testing showed density alone
  (a mean over KNOWN pairs) is insensitive to shrinking a uniformly-strong clique one member at a time; see
  the Part 4 report's limitations. Confirmed via `backend/tests/test_familiarity.py` (60 tests) that a
  4-person synthetic clique (the exact scenario `test_friend_clique_is_not_broken_up_by_production_algorithm`
  documents as undetectable by default) is now measurably de-concentrated when `familiarity.enabled=true`,
  without regressing the total-score objective or the accountability/cooperativeness floor.
- **Status — CORRECTED (Part 4.5), verified:** the Part 4 "largest pre-existing connected subgroup"
  computation had a real bug: it found connected components over the **global** familiarity graph (every
  student across every team) and intersected each component with a team's membership afterward, which could
  count two teammates as clustered when they were connected only through a **path through a student on a
  different team**. Fixed by recomputing the metric from each team's own **team-induced subgraph** (that
  team's members and only the edges between them) — a path through an outside student is now structurally
  impossible to include. Renamed to `connected_familiarity_cluster` / `largest_connected_cluster` throughout
  (no deprecated alias kept for the old field — this is a pre-1.0 prototype and the old field was a bug, not
  a naming preference) and stopped calling it a "clique": a connected component only needs a *path* between
  members, not every pair directly connected, so a chain or star can form one cluster without the cohesion a
  clique implies — new per-cluster fields (`known_internal_edge_coverage`, `mean_known_internal_strength`,
  `thresholded_internal_edge_density`, `is_quasi_clique`) make that distinction visible. Also added
  `known_pair_coverage` (known / all-possible teammate pairs) and a `minimum_known_pair_coverage` config
  (default 0.5): a team with too little recorded familiarity data now gets `band_status="insufficient_data"`
  instead of a density-band verdict resting on a handful of pairs, and the optimizer does not penalize such a
  team by default (`penalize_insufficient_data=False`). Verified via 17 new tests in
  `backend/tests/test_familiarity.py` (team-induced-cluster correctness, chain/star-vs-clique distinction,
  edge-order determinism, coverage/insufficient-data classification at and around the threshold, optimizer
  non-penalization) — 77/77 tests in that file pass; full backend suite 322 passing with no regression from
  this change (2 pre-existing, unrelated failures in preference/ability areas explicitly out of this part's
  scope). See `research_alignment_audit.md`'s Part 4.5 update for the full before/after graph definition.

### 5. No preference data — "preference satisfaction and stability" cannot be enforced
- **Research criterion affected:** Preference satisfaction and stability (blocking pairs).
- **Evidence from code:** No preference field anywhere in `models.py`; no assessment scenario in
  `bank_v1.yaml` asks about teammate preference.
- **Evidence from tests:** `test_preference_pairs_are_mostly_kept_together` and
  `test_preference_instability_blocking_pairs_detected` both **FAILED** against synthetic preference data
  — 2 blocking pairs found, most students didn't get a top-half preference satisfied.
- **Recommended change:** Add an optional `preferred_teammates: List[str]` / `avoid_teammates: List[str]`
  field, collected with a clear "requests are considered, not guaranteed" framing (over-weighting
  preference can reproduce cliques — this interacts directly with gap #4). `evaluation/metrics.py:
  blocking_pairs_preference_based()` implements a standard pairwise-stability check once ranked
  preferences exist and can be reused as-is for a post-formation stability report.
- **Difficulty:** Medium — data collection + a low-weight preference term in `_total_score()`, gated
  carefully against clique risk.
- **Type:** Soft constraint / metric (start as a diagnostic "N blocking pairs" report before any
  auto-optimization, to avoid fighting gap #4's anti-clique goal).
- **Status — IMPLEMENTED (Part 5), verified:** a `preferences` edge table (`PreferenceDB` / `POST
  /preferences` / `app.models.PreferenceEntry`) — a DIRECTED statement from one student about a target
  (named student, by rank/tie, or optionally a team characteristic like `{"skill": "backend"}`) or an
  avoidance request. Ranked/tied preferences are scored via a TIER system (entries sharing a rank are a
  tie/indifference group; unranked entries form one flat tier after all ranked ones); a student with zero
  submitted entries is MISSING data (`None`), never scored as dissatisfaction — see `app/matching.py`'s
  preference-framework module docstring for the exact formula. Stability is formalized via THREE distinct,
  precisely defined notions for many-to-one team assignment (not simple roommate matching) — blocking pair
  (envy diagnostic), blocking swap (executable, both-sides-improve exchange — the documented DEFAULT
  `is_stable` is checked against), and blocking move (one-sided: only the mover's utility must improve, and
  the RECEIVING team's HARD constraints, not its members' preferences, must hold) — `blocking_pairs_
  preference_based()`'s existing rank-vs-current-teammates logic was the direct basis for the blocking-pair
  definition. The optional soft objective (`_preference_repair_improve`, off by default) minimizes a
  per-team violation (unmet satisfaction + a weighted term per unmet HIGH-priority preference + a weighted
  term per avoidance-violation pair), with the SAME lexicographic hard-constraint-first acceptance rule as
  every other optional pass, plus a `hard_avoidance` gate that additionally refuses any of the pass's own
  swaps that would increase avoidance violations on a touched team. Confirmed via
  `backend/tests/test_preferences.py` (67 tests, including the exact scenario `test_preference_instability_
  blocking_pairs_detected` documents: `fraction_with_at_least_one_preferred_teammate` rises from 0.5 to 1.0
  and blocking pairs drop from 2 to 0 once `preference.enabled=true`) that gap #4's clique-risk concern was
  addressed by keeping this a purely additive, opt-in signal with no default weight in `_total_score()`, not
  by baking preference into the always-on objective. Privacy: `PreferenceSummary`/`PreferenceStudentStats`
  are aggregate-only (no target identities), and `app/preferences.py` deliberately has no bulk "list
  everyone's preferences" endpoint — see that module's docstring and `research_alignment_audit.md`'s Part 5
  update for the full design.
- **Status — CORRECTED (Part 5.5), verified:** a review of Part 5 found the "blocking move" name overclaimed
  a two-sided guarantee (only the mover's utility was ever required to improve) — renamed to
  `beneficial_feasible_move` (no alias kept, same precedent as Part 4.5), and its whole-partition
  `validate_assignment()` check documented explicitly. `PreferenceConfig.hard_avoidance` previously only took
  effect inside the opt-in repair pass; a new `_hard_avoidance_repair` now enforces it as a genuine hard
  constraint independent of `preference.enabled`, actively reducing existing violations (not merely refusing
  to worsen them) and reporting any it cannot resolve. `app.models.TeamSizeConfig` +
  `compute_unilateral_vacancy_moves` were added for genuinely non-swap-mediated moves under flexible team
  sizes — a no-op by default, never silently reinterpreting exact sizes as flexible. The blocking-pair
  shortcut's edge-case behavior (ties, unranked/unnamed teammates, missing data, characteristic preferences,
  avoidance non-interaction) was verified against the implementation and one docstring inaccuracy corrected.
  Also proved — not merely asserted — that `test_preference_instability_blocking_pairs_detected`'s continued
  failure is entirely a LEGACY (Part-4-era) code-path artifact: a new test beside it shows the real Part 5
  engine detects the identical instability without `preference.enabled` ever being set. 27 new tests in
  `backend/tests/test_preferences.py` (94 total) plus 1 in `test_research_alignment.py`; full backend suite
  417 passing, same 2 pre-existing failures as Part 5. See `research_alignment_audit.md`'s Part 5.5 update
  for the full accounting.
- **Status — INTEGRATION GAP CLOSED (Part 5.6), verified:** Part 5.5 proved the still-failing test exercised
  a legacy code path but left `evaluate_teams()` calling that legacy path's OWN independent blocking-pair
  algorithm (`blocking_pairs_preference_based()`), a second implementation of the same concept
  `app.matching.compute_blocking_pairs` already covers. Added `evaluation.metrics.legacy_preferences_to_
  entries()` — THE authoritative boundary adapter (documented rank/duplicate/self-target/unknown-target
  policy) — and made `blocking_pairs_preference_based()` a thin wrapper that converts through it and
  delegates entirely to `compute_stability_metrics(definition="blocking_pair")`. There is exactly one
  blocking-pair algorithm left in this codebase. Removed Part 5.5's fixture-only adapter copy
  (`evaluation.fixtures.preferences_dict_to_entries`) rather than leaving two conversions to maintain.
  Corrected `test_preference_instability_blocking_pairs_detected`'s assertion, which had been checking an
  impossible condition (zero blocking pairs from a preference-blind, repair-disabled default pipeline) —
  verified the actual team split genuinely separates 2 real mutual-preference pairs, and the test now
  asserts exactly that (2 named pairs detected), a strictly MORE demanding check, not a weaker one. Full
  backend suite: 429 passing, 1 failing (`test_stacked_ability_gets_redistributed_across_teams`, unrelated,
  unaltered) — the only remaining known failure. See `research_alignment_audit.md`'s Part 5.6 update for the
  full accounting.

### 6. Leftover-student distribution can produce wildly uneven team sizes
- **Research criterion affected:** Comparable team strength (indirectly — team size is a strength
  confound); also a plain correctness issue independent of the research framing.
- **Evidence from code:** `matching.py:_greedy_build()` L423-430 distributes "true leftover" students (when
  `len(students) % team_size != 0`) one at a time to `argmax_i total_score(teams[i] + [s])`, with **no
  capacity check** — nothing stops every leftover from landing on the same team.
- **Evidence from tests:** `test_uneven_roster_size_produces_reasonable_team_sizes` **FAILED**: 11 students
  at `team_size=4` produced teams of size **[7, 4]**, not the expected ~[4, 4, 3]-shaped split
  (`research_alignment_results.json → adversarial_fixtures` for `uneven_roster_size`).
- **Practical consequence:** `team_size` in `TeamGenerationRequest` (`models.py:125`) is effectively a
  soft suggestion, not a bound — a team nearly double the requested size is a materially different
  classroom outcome than what an instructor asked for.
- **Recommended change:** Cap the leftover-assignment loop to teams with `len(team) < team_size + 1`
  (or better, round-robin leftovers across the least-full teams first) so no team exceeds the requested
  size by more than one student.
- **Difficulty:** Small — a bounded-capacity check in one existing loop.
- **Type:** Hard constraint (`max team size ≤ requested size + 1`).

### 7. 2-opt can leave the weakest team's skill coverage below what plain greedy already achieved
- **Research criterion affected:** Skill coverage (interacts with objective-function tradeoffs, Phase 2).
- **Evidence from tests:** `test_greedy_seeding_does_not_starve_a_team_of_skills` **FAILED**: skill
  coverage of the weakest team dropped from 0.500 (post-greedy) to 0.417 (post-2opt) on fixture #11
  (`research_alignment_results.json → adversarial_fixtures` for `greedy_local_optimum`).
- **Practical consequence:** `_two_opt_improve()` (`matching.py:435-452`) accepts any swap that raises the
  **combined** score of the two teams involved — it has no visibility into any single component
  (skill coverage specifically) dropping for one of them, only the weighted total. A swap that trades a
  skill-diverse member for a behaviorally-compatible one can pass this test even though it moves a team
  further from full skill coverage.
- **Recommended change:** Either (a) add skill coverage as a `_maximin_improve`-style protected minimum
  (never let post-swap min-team skill-coverage fall below pre-swap), mirroring how floor dims are already
  protected against `_two_opt_improve`'s blind spot, or (b) accept this as an intentional tradeoff but
  surface it: log/report when a 2-opt swap lowers any single component for either team, even while raising
  the total.
- **Difficulty:** Small (mirrors the existing `_maximin_improve` pattern) for (a); trivial for (b).
- **Type:** Soft constraint or metric, instructor's choice via config.
- **Status — RESOLVED (Part 1.5), verified:** Implemented as option (a), generalized to a hard constraint
  rather than a `_maximin_improve`-only mechanism. `app/matching.py` now has a centralized
  `validate_assignment()` (backed by `AssignmentBaseline.capture()`, taken right after `_greedy_build`)
  that every swap in `_two_opt_improve` — and `_maximin_improve` — must pass before being accepted: the
  partition's minimum per-team skill coverage may never regress below the greedy baseline, regardless of
  aggregate score gain. An optional absolute `hard_skill_coverage_floor` is also available (default 0.0 =
  off, preserving the prior default). On `fixture_greedy_local_optimum`, 2-opt now finds a different,
  equally-improving swap (sum score 1.370 → 1.403, same as before) that raises coverage to 0.583/0.583 on
  both teams instead of dropping one to 0.417. Confirmed via
  `test_greedy_seeding_does_not_starve_a_team_of_skills` (now passing) and the new
  `backend/tests/test_optimizer_invariants.py`.

---

## Medium

### 8. `GET /teams/{id}/explanation` always returns empty `team_norms` in the live (Gemini) path
- **Research criterion affected:** None of the 7 core criteria directly — flagged because the audit brief
  asked how explanations are generated.
- **Evidence from code:** `explainability.py:92` calls `explainer.generate(team, students)` only, never
  `suggest_norms()`. `GeminiExplainer.generate()` (`gemini_explainer.py:103-106,135`) reads and returns
  `team.team_norms`, but `_build_teams()` (`matching.py:550`) always sets `team_norms=[]` at generation
  time, and nothing populates it afterward on that code path. (`MockExplainer.generate()`,
  `mock_explainer.py:208`, is unaffected — it calls `_build_norms()` internally — so this only manifests
  when `GEMINI_API_KEY` is set.)
- **Recommended change:** Call `explainer.suggest_norms(team, students)` first inside
  `get_explanation()` and pass the result into `generate()` (or have `generate()` call it internally),
  matching what `MockExplainer` already does.
- **Difficulty:** Small (a few lines in `explainability.py`).
- **Type:** Bug fix, not a research-alignment configuration.

### 9. `_maximin_improve` runs after `_two_opt_improve` and can reintroduce score-improving swaps
- **Research criterion affected:** Comparable team strength / preference-stability proxy.
- **Evidence from tests:** `blocking_moves_score_based` found **10 unresolved improving swaps** on the
  real seed roster *after* the full greedy→2opt→maximin pipeline
  (`research_alignment_results.json → sample_students_evaluation.metrics.blocking_moves_score_based`).
  Since `_two_opt_improve` converges to a state with no improving swap by construction, and
  `_maximin_improve` runs afterward optimizing a *different* objective (partition floor-min), it can move
  the partition back into a state 2-opt would have rejected.
- **Recommended change:** Either interleave the two passes until a joint fixed point (no swap improves
  total score AND no swap improves partition-min) or re-run `_two_opt_improve` once more after
  `_maximin_improve` and document that the pipeline settles for "close to" rather than strictly
  2-opt-stable.
- **Difficulty:** Small (re-run one existing pass, or wrap both in an outer fixed-point loop).
- **Type:** Metric today (`blocking_moves_score_based` in the harness); could become a soft constraint.
- **Status — RESOLVED (Part 1.5), verified:** Neither of the two options above — `_maximin_improve` is now
  constraint-safe by construction instead: acceptance is lexicographic (hard constraints via
  `validate_assignment()`, then the primary total-score objective may not regress by more than a documented
  `MAXIMIN_OBJECTIVE_TOLERANCE`, default 0.0 — i.e. no regression at all — then the partition floor-minimum
  must strictly improve). 2-opt's optimum is now a floor `_maximin_improve` cannot trade away. On the real
  20-student seed roster this took the summed total-score regression from -5% (3.084 → 2.922, pre-fix) to
  none (3.072 → 3.072 — maximin made zero swaps here since no floor-min improvement was available at zero
  objective cost), and `blocking_moves_score_based` from 10 unresolved swaps down to 1. The 1 remaining
  blocking move (Fatima Al-Hassan / Hamid Karimi) was traced and confirmed to be correctly rejected: taking
  it would introduce a second cooperativeness=low member onto one team, violating the `one_low_per_team`
  floor policy — i.e. it is a hard-constraint violation, not an unresolved soft-objective gap. Confirmed via
  `backend/tests/test_optimizer_invariants.py::test_maximin_cannot_silently_undo_two_opts_improvement` and
  `test_maximin_cannot_undo_a_hard_constraint`.

### 10. `leadership_style` uses `soft_compatibility` (rewards similarity) while its own code comments argue for `complementary_coverage`
- **Research criterion affected:** None of the 7 core criteria directly (this dimension isn't part of the
  audited definition), but it's a live, self-flagged inconsistency worth resolving alongside this audit.
- **Evidence from code:** `rubric_v1.yaml:148-152` and `matching.py:87-91,128-130` both carry
  `TODO(team-decision)` comments questioning whether `leadership_style` should reward *diversity of
  coverage* (directive+facilitative+emergent on one team) instead of *similarity*. Already tracked in
  `CHANGES.md` TODO #2 — restated here because this audit's evidence base (adversarial fixtures) can help
  resolve it empirically rather than by inspection alone.
- **Recommended change:** No code change recommended by this audit alone — needs the team decision
  `CHANGES.md` already asks for. Noting it here so it isn't lost.
- **Difficulty:** N/A (design decision, not implementation).
- **Type:** Config option once decided (the `composition_rule` field already supports switching it).

### 11. `conflict_style` weight is zeroed — as documented, correctly, but it means "conflict handling" is not actually part of matching today
- **Research criterion affected:** Adjacent to (not one of) the 7 core criteria, but affects "constraints
  and weights" the audit brief asked to map.
- **Evidence from code:** `rubric_v1.yaml:112-142`, `matching_weight: 0.0`, with a clear citation
  (De Dreu & Weingart 2003; Jehn 1995) explaining *why* — the MCQ bank can't currently separate task
  conflict from relationship conflict, so the signal is unreliable. This is the one place in the codebase
  where a research citation is used correctly to justify **not** acting on a signal.
  Already tracked as `CHANGES.md` TODO #1.
- **Recommended change:** No change recommended until the question bank (`bank_v1.yaml`) can distinguish
  the two conflict types — re-affirming the existing TODO rather than introducing a new one.
- **Difficulty:** N/A until the bank redesign happens.
- **Type:** Currently correctly a no-op; revisit as a soft constraint after bank redesign.

### 12. `safety_contribution` and `stress_response` are collected by Pipeline B but have no rubric definition and are never used
- **Research criterion affected:** None of the 7 directly, but violates the codebase's own stated
  invariant.
- **Evidence from code:** `models.py:29-30` define both fields on `CollaborationSignature`;
  `conversation.py:226,292-293` actively elicit and populate them from interview transcripts. Neither
  appears anywhere in `rubric_v1.yaml`'s `dimensions:` block, and `grep` for either name in
  `backend/app/matching.py` returns nothing.
- **Practical consequence:** Directly contradicts `docs/pipeline.md:97-100`'s stated rule — *"no dimension
  definitions live anywhere except the L0 rubric... every scoring function and matching rule reads from
  rubric_v1.yaml"* — since these two are scored/elicited but have no L0 entry and no matching effect.
- **Recommended change:** Either add rubric entries (with a `composition_rule` and cited evidence basis,
  per the pattern every other dimension follows) or stop collecting them until they have one.
- **Difficulty:** Small (rubric entries) or trivial (stop collecting).
- **Type:** Config/consistency cleanup.

### 13. `REFERENCE_SKILLS` is a hardcoded 12-item module constant, not configurable per course
- **Evidence from code:** `matching.py:23-27`.
- **Recommended change:** Move to `TeamGenerationRequest` as an optional instructor-supplied list,
  defaulting to the current 12 if omitted, so skill coverage is measured against what a specific course
  actually needs rather than a fixed generic set.
- **Difficulty:** Small.
- **Type:** Config option.

---

## Low

### 14. `requirements.txt` is missing the Gemini SDK dependency the live code imports
- **Evidence:** `backend/ai/gemini_explainer.py:6-7` (`from google import genai`) — `google-genai` is not
  listed in `backend/requirements.txt`, discovered while setting up this audit's test environment (see
  `research_alignment_audit.md` Phase 5 for the exact failure).
- **Recommended change:** Add `google-genai` to `requirements.txt` (both `ai/gemini_analyzer.py` and
  `ai/gemini_explainer.py` depend on it, and both are wired into live routers in `main.py:66-71`).
- **Difficulty:** Trivial.
- **Type:** Packaging fix.

### 15. `docs/architecture.md` describes a pre-rubric-refactor version of the system
- **Evidence:** `docs/architecture.md:79,81,117` describe `conflict_style` values including
  `"confrontational"` (renamed to `"assertive"` per `CHANGES.md` §9) and a hardcoded 4-factor behavioral
  compatibility formula ("conflict style 35%, leadership 30%, planning 20%, accountability 15%") that no
  longer matches the rubric-driven `_behavioral_compat()` implemented in `matching.py:103-138`.
  `docs/pipeline.md` is the accurate, current document.
- **Recommended change:** Either update or delete `docs/architecture.md` in favor of `docs/pipeline.md` to
  avoid a reader trusting stale vocabulary/formulas.
- **Difficulty:** Trivial–Small.
- **Type:** Documentation.

### 16. Pipeline runtime scales worse than cubically with roster size — **PARTIALLY RESOLVED (Part 6.5)**
- **Evidence:** `research_alignment_results.json → scalability_sweep`: 8→64 students (8×) increased total
  pipeline wall-clock from 1.1ms to 322.9ms (~295×) — the original finding. Part 6 later reintroduced the
  same anti-pattern in 4 newer optional passes (`_ability_balance_improve`, `_fairness_improve`,
  `_familiarity_balance_improve`, `_preference_repair_improve`), measured at **26,077 ms** for
  `research_balanced` at n=80 (`docs/optimization_profiles.md`).
- **Part 6.5 fix:** applied the SAME incremental per-team caching `_maximin_improve`/`_two_opt_improve`
  already used to all 4 newer passes — cProfile-confirmed root cause, equivalence-proven fix (see
  `docs/performance.md`). `research_balanced` at n=80: 26,077 ms → **1,410 ms** (~18.5× speedup).
- **What remains unresolved:** the underlying `O(teams² × team_size²)` CANDIDATE ENUMERATION itself is
  unchanged — this fix eliminated the redundant PER-CANDIDATE full-partition recompute, not the NUMBER of
  candidates evaluated. `research_balanced` still measured 2,148.9 ms at n=100 and continues to grow with
  roster size; not benchmarked past n=100.
- **Practical consequence:** Comfortably fast through at least 100 students in this environment now;
  immaterial at typical class sizes. Would still degrade for large-cohort or multi-section batch runs well
  beyond that.
- **Recommended change:** No action needed at current scale (through ~100 students, confirmed). If a much
  larger cohort becomes a target, cap sweep count, move to a randomized/neighborhood-limited local search,
  or run generation asynchronously (job handle rather than a blocking request) rather than attempting
  further synchronous algorithmic speedups.
- **Difficulty:** Done for the measured range; N/A beyond it without a design change.
- **Type:** Performance fix (done) + future scalability note (open, informational).

### 17. Existing `client` pytest fixture doesn't fully isolate the app's `lifespan` from the real database
- **Evidence:** `backend/app/main.py:13` does `from app.database import init_db, SessionLocal, StudentDB` at
  module scope; `lifespan()` (`main.py:19-45`) calls those names directly. `backend/tests/test_assessment.py`'s
  `client` fixture (L54-74) patches `db_module.SessionLocal` / `db_module.init_db` (attributes on the
  `app.database` module) — but `app.main` already bound its own local names to the pre-patch objects the
  first time it was imported (once, cached in `sys.modules`, before any fixture runs). Every
  `with TestClient(app) as c:` therefore still runs the **real** `init_db()` against
  `backend/behavioral_team_formation.db` on startup, not the in-memory test database. Confirmed empirically
  in this session: running the suite added the (locally uncommitted) `evidence_units` table to the real
  `.db` file — schema-only, no test student rows leaked in (verified by row count/content inspection), but
  the isolation the fixture intends is incomplete. Endpoint handlers themselves are fine — `behavioral_
  analysis.py` and `matching.py` re-import `SessionLocal` lazily *inside* each function
  (e.g. `behavioral_analysis.py:48,88,120,152`), so request-time reads/writes correctly hit the patched
  in-memory database; only the `lifespan` startup hook is affected.
- **Recommended change:** Move the `init_db`/`SessionLocal` references inside `lifespan()` to lazy
  (function-local) imports, matching the pattern already used everywhere else in the codebase, or have the
  fixture monkeypatch `app.main.init_db` / `app.main.SessionLocal` directly in addition to `app.database`'s.
- **Difficulty:** Trivial.
- **Type:** Test-infrastructure bug fix, not a research-alignment item — noted because Phase 5 of the audit
  requires disclosing environmental/test-execution issues found while running the suite.

### 18. `match_confidence` is a heuristic never validated against real outcomes
- **Evidence:** `matching.py:_match_confidence()` L210-218 — a hand-tuned formula
  (`0.40 + relative_position * 0.52`, capped `[0.40, 0.92]`) with no citation and no calibration data.
- **Recommended change:** None implementable now — flagged as a criterion needing real student-outcome
  data (see `research_alignment_audit.md`'s closing section) before it can be validated or recalibrated.
- **Difficulty:** N/A (needs external data, not code work).
- **Type:** Needs real-world validation.

### 19. All seven research criteria were addressed independently (Parts 2–5.6), with no coherent, configurable
way to combine them or compare tradeoffs
- **Research criterion affected:** All seven (skill coverage, mixed ability, accountability/cooperativeness,
  familiarity, comparable team quality, individual fairness, preference/stability).
- **Evidence from code (before Part 6):** Each criterion's optional pass (`ability.enabled`,
  `fairness.enabled`, `familiarity.enabled`, `preference.enabled`, `preference.hard_avoidance`) had to be
  flipped independently by a caller who already knew all five existed; nothing classified which criteria
  were hard constraints vs. soft hypotheses in one place; nothing let an instructor compare "what would
  prioritizing fairness cost me on ability" without hand-running the pipeline twice.
- **Status — IMPLEMENTED (Part 6), verified:** a centralized `app.matching.CRITERION_REGISTRY` (15 rows)
  classifies every criterion; a named-profile system (`legacy` — the default, reproduces Parts 1–5.6 exactly
  — plus 4 EXPERIMENTAL profiles: `research_balanced`, `fairness_priority`, `skill_priority`,
  `preference_priority`) resolves via `resolve_profile()` (system defaults → named profile → explicit
  request overrides, using Pydantic's `model_fields_set` for correct nested-override precedence) into one
  complete, immutable `TeamGenerationRequest`, returned on every response as `"profile"`; a normalized
  cross-criterion objective report (`compute_normalized_objective`, 9 documented `[0,1]` components,
  missing data excluded and weights renormalized rather than scored zero) is returned as
  `"normalized_objective"`; and an explicit, documented cross-pass acceptance policy
  (`_run_optional_passes`, Option B — lexicographic, extended one level beyond each pass's own existing
  per-swap rule) prevents a later optional pass from silently undoing an earlier one's protected metrics,
  recorded as `"optimization_trace"`. No profile is claimed optimal — all 4 non-legacy profiles are labeled
  EXPERIMENTAL in both code and `docs/optimization_profiles.md`. Empirically compared all 5 profiles against
  the real seed roster, all 12 fixtures, and synthetic 20/40/80-student rosters
  (`research_alignment_results.json → profile_comparison`) — genuine tradeoffs found and reported (e.g.
  `skill_priority` worsens ability gap while improving worst-case regret on the real roster), and a real
  performance limitation measured (`research_balanced` took 26s at n=80 synthetic students). 122 new tests
  in `backend/tests/test_optimization_profiles.py`. **Update (Part 6.5):** the 26s performance limitation is
  fixed (see #16 above and item 20 below) and the one remaining failing test is resolved (see
  `research_alignment_audit.md`'s Part 6.5 update, item 1) — full backend suite is now **567 passing, 0
  failing**. See `research_alignment_audit.md`'s Part 6 and Part 6.5 updates for the full accounting,
  including decisions still requiring professor/research-team approval before real deployment.

---

## Part 6.5 additions

### 20. `_run_optional_passes`' cross-pass regression check could be silently bypassed by a NaN/infinite metric — **FIXED**
- **Research criterion affected:** None of the 7 directly — an orchestration-layer correctness gap found
  while auditing the Part 6 regression-protection mechanism itself.
- **Evidence from code (before the fix):** `app/matching.py`'s `_maybe_run` (inside `_run_optional_passes`)
  computed `regressed = any(after[k] < before[k] - PROTECTED_METRIC_TOLERANCE for k in before)`. Python
  evaluates `NaN < x` as `False` unconditionally, so a metric computation bug that produced `NaN` for
  `after[k]` (or `before[k]`) would never register as a regression and would be silently ACCEPTED into the
  final assignment.
- **Evidence from tests:** `test_nan_metric_forces_pass_rejection_rather_than_silently_bypassing_the_
  regression_check` and `test_infinite_metric_also_forces_pass_rejection`
  (`backend/tests/test_part6_5_orchestration_audit.py`) — both confirm the fix rejects the pass rather than
  accepting a non-finite metric.
- **Fix implemented:** an explicit `math.isfinite()` guard now treats any non-finite value on either side as
  an unconditional regression; the pass's trace entry additionally reports which metric key(s) were
  non-finite (`non_finite_metric_keys`), an additive field that does not change any existing consumer's
  reading of `pass`/`before`/`after`/`accepted`.
- **Difficulty:** Small (implemented).
- **Type:** Correctness bug fix, done.

### 21. `PROTECTED_METRIC_TOLERANCE` is an absolute tolerance applied uniformly to metrics on different scales
- **Research criterion affected:** None of the 7 directly — an orchestration-layer robustness finding.
- **Evidence from code:** `total_score` (`_stage_objective`) is a SUM over every team and therefore scales
  with team count (unbounded above 1.0 for more than one team); `floor_min`/`skill_coverage_min`/
  `min_utility`/`mean_preference_satisfaction` are all bounded in `[0, 1]` regardless of roster size. The
  SAME absolute `PROTECTED_METRIC_TOLERANCE = 1e-6` is compared against both kinds of quantity.
- **Evidence from tests:**
  `test_total_score_protected_metric_is_unnormalized_relative_to_its_siblings`
  (`backend/tests/test_part6_5_orchestration_audit.py`) characterizes this directly.
- **Practical consequence:** At scale (many teams), a 1e-6 absolute tolerance on `total_score` is a much
  tighter RELATIVE tolerance than the same 1e-6 applied to a `[0,1]`-bounded sibling metric — meaning the
  orchestration layer is effectively stricter about `total_score` drift than about floor/utility/preference
  drift as roster size grows, an inconsistency that was not designed for, only inherited from reusing one
  constant everywhere.
- **Recommended change:** Normalize `total_score` by team count (mean rather than sum) before comparing
  against a shared tolerance, OR give `total_score` its own, separately-scaled tolerance. NOT implemented in
  Part 6.5 — either change would alter which candidate swaps get accepted at the margin, and this part's own
  instruction is not to change research conclusions merely to create different assignments; this is
  reported as a finding for a future part to decide deliberately, with test coverage already in place to
  detect the current (unnormalized) behavior if it changes.
- **Difficulty:** Small, but requires a deliberate decision about acceptable behavior change first.
- **Type:** Metric/config consistency finding, not yet fixed.

### 22. Before Part 7 (a real student study), familiarity and preference data collection — not further algorithm work — is the concrete blocker
- **Research criterion affected:** Familiarity/anti-clique (#4 above) and preference/stability (#5 above),
  specifically their READINESS for a real comparative study.
- **Evidence from code/tests:** `backend/evaluation/experiment_readiness.py` run against `legacy` vs.
  `research_balanced` on the real 20-student seed roster reports `distinct_enough_for_experimental_
  conditions: false` with `excluded_for_missing_data: ["familiarity", "preference"]` —
  `familiarity_pair_coverage=0.0` (0 of 190 possible pairs known) and `preference_submission_
  fraction=0.0` (0 of 20 students submitted). `backend/evaluation/profile_distinctness.py` independently
  confirms 9 of 13 tested datasets collapse to an identical partition across every profile (see
  `docs/profile_distinctness.md`), and the seed-roster shadow trace
  (`backend/evaluation/trace_seed_roster.py`) shows the familiarity/preference passes' own target metrics
  sitting at their zero/no-violation floor specifically BECAUSE no data exists for them to act on, not
  because of any algorithmic limitation.
- **Practical consequence:** Running a real study on `research_balanced`/`preference_priority`/a
  familiarity-focused configuration WITHOUT first collecting familiarity and preference data would very
  likely reproduce this session's own finding — two "different" experimental conditions that turn out to
  produce the identical assignment, for want of data rather than for any flaw in the optimizer.
- **Recommended change:** Before Part 7, collect familiarity data (a roster survey or LMS co-enrollment
  import feeding `POST /familiarity` — see #4 above) and preference data (`POST /preferences`, which already
  exists and is fully tested — see #5 above) from the actual cohort that will be used in the study. Re-run
  `backend/evaluation/experiment_readiness.py` against the intended profile pair on the REAL collected data
  before committing to a study design.
- **Difficulty:** Not a code change — a data-collection and study-design planning item.
- **Type:** Study-design blocker, not a code defect.
- **Difficulty:** Large (this WAS the "combined optimizer integration" every part since 5.5 explicitly
  deferred).
- **Type:** Architecture / configuration (does not itself add a new research criterion — integrates the six
  already added).
