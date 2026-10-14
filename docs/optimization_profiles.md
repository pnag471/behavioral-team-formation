# Optimization profiles (Part 6)

This document describes the named optimization-profile system introduced in Part 6 of the
research-alignment implementation. See `research_alignment_audit.md`'s Part 6 update for the full
audit trail and `app/matching.py`'s Part 6 module docstring for the exact code.

## What a profile is

A profile is a named, complete set of `TeamGenerationRequest` field values. Resolving a request
(`app.matching.resolve_profile`) always follows this precedence, in order:

```text
system defaults  ->  named profile's overrides  ->  explicit request-level field overrides
```

An explicit field the caller actually sends in the request body ALWAYS wins over the profile,
down to individual nested sub-fields (e.g. `{"profile": "research_balanced", "familiarity":
{"enabled": false}}` keeps every other `research_balanced` setting but turns familiarity back
off). This is implemented with Pydantic's own `model_fields_set`, not string/dict guessing, so
"the caller didn't mention this field" and "the caller sent the class default value" are never
confused with each other.

`resolve_profile` returns a fresh `TeamGenerationRequest` (never a shared/mutated object — see
`test_resolve_profile_never_mutates_the_registry_or_shared_state`) plus a `ProfileResolution`
record (`name`, `is_experimental`, `description`, `effective_settings`) that is returned on every
`/teams/generate` response under the `"profile"` key, so a caller can always tell exactly what
configuration actually produced a given assignment.

An unknown profile name is rejected with HTTP 422 — never silently substituted with `legacy`.

## Profiles

### `legacy` (default, NOT experimental)

Reproduces this system's shipped-by-default behavior from Parts 1 through 5.6, exactly: every
optional pass (`ability`, `fairness`, `familiarity`, `preference`) off, `hard_avoidance` off, the
original `weights`. **An existing client that omits `profile` from its request body gets this
profile — nothing about existing behavior changes.**

- Effective settings: `TeamGenerationRequest()`'s own class defaults, unmodified.
- Objective weights: the original `_total_score` weights only (skill_coverage 0.4, behavioral_compat
  0.3, availability_overlap 0.2, shared_interests 0.1).
- Hard constraints: the same ones this system has always enforced (membership, team size, skill-
  coverage non-regression, floor policy).
- Missing-data behavior: unaffected — every optional metric is still computed and reported (Part
  2–5 requirement), only the *assignment* is unaffected by any of it.
- Experimental status: **not experimental** — this is the validated, shipped baseline.
- Known limitations: does not address ability balance, familiarity clustering, individual-utility
  fairness, or preference satisfaction in the *assignment itself* (all four are still measured and
  reported).
- Appropriate use: any deployment that has not made an explicit decision to opt into a research
  criterion, or that needs to reproduce a prior semester's exact algorithmic behavior.

### `research_balanced` (EXPERIMENTAL)

Enables **every** supported research-criterion optimization pass — ability, fairness, familiarity,
preference — each with the conservative defaults its own `*Config` class already ships (no
threshold is independently re-tuned for this profile).

- Effective settings: `ability.enabled=true`, `fairness.enabled=true`, `familiarity.enabled=true`,
  `preference.enabled=true`; every other field at its class default.
- Objective weights: unchanged from `legacy` (`DEFAULT_OBJECTIVE_WEIGHTS` is used only for the
  *reporting-layer* normalized objective, not for reweighting `_total_score`).
- Hard constraints: identical to `legacy` — enabling every optional pass never relaxes any hard
  constraint (verified: `test_every_profile_preserves_membership_and_team_sizing_on_every_fixture`
  and siblings, parametrized over `research_balanced` too).
- Missing-data behavior: unchanged per-pass policy (e.g. familiarity still refuses to act
  confidently on `insufficient_data`; ability still reports `None` gaps rather than a fabricated 0).
- **Experimental status: EXPERIMENTAL. This profile is NOT validated against real student outcomes
  and is NOT claimed to be optimal for any population.** It exists so every documented research
  hypothesis can be tested TOGETHER, not because a citation establishes that pursuing all seven
  criteria simultaneously produces the best real-world teams.
- Known limitations: running every pass together is not guaranteed to reach the jointly-best
  outcome for any single criterion (each pass's own docstring says so); on the real 20-student seed
  roster this session ran, most components landed identically to `legacy` (no swap-reachable
  improvement existed for THIS particular roster beyond ability — see the comparison table below) —
  "enables everything" does not mean "changes everything" every time. Runtime scales the worst of
  any profile (see Performance below) since it runs the most passes.
- Appropriate use: exploring what a fully research-aligned assignment would look like for a given
  roster, before deciding which SPECIFIC criteria (if any) a real deployment should adopt.

### `fairness_priority` (EXPERIMENTAL)

Prioritizes the worst-off individual student's utility and worst-case regret.

- Effective settings: `fairness.enabled=true`, `fairness.acceptable_utility_floor=0.6` (raised from
  the class default 0.5, tightening the reporting-only "below floor" threshold); every other field
  at its class default (ability/familiarity/preference stay off).
- Hard constraints: identical to `legacy`.
- Missing-data behavior: `compute_assignment_utilities` already handles a roster too large for exact
  regret search by reporting `exact_regret_search: false` rather than a fabricated number —
  unchanged here.
- Experimental status: EXPERIMENTAL.
- Known limitations: raising `acceptable_utility_floor` only changes REPORTING (who counts as
  "below floor"); it does not itself change `_fairness_improve`'s target (still: raise the
  minimum). If no swap-reachable improvement exists (see the seed-roster comparison below, where
  this profile matched `legacy` exactly), this profile cannot manufacture one.
- Appropriate use: a deployment whose primary concern is that no single student ends up
  significantly worse off than their peers, even at some cost to average/aggregate quality.

### `skill_priority` (EXPERIMENTAL)

Prioritizes required skill coverage and skill-based complementarity by reweighting the PRIMARY
`_total_score` objective itself (not merely an optional pass).

- Effective settings: `weights = {"skill_coverage": 0.7, "behavioral_compat": 0.15,
  "availability_overlap": 0.1, "shared_interests": 0.05}`; ability/fairness/familiarity/preference
  all stay off.
- Hard constraints: identical to `legacy`.
- Missing-data behavior: unaffected — this profile only changes a weight vector already handled by
  existing normalization (`_normalize_weights`).
- Experimental status: EXPERIMENTAL. Reweighting toward skill coverage is a configuration choice,
  not a validated claim that this weighting produces better teams.
- **Known limitation, empirically observed (see the comparison table below): on this session's
  synthetic and seed-roster runs, `skill_priority` did not change `skill_coverage_min` at all
  (already at its ceiling for these rosters) but DID measurably worsen `ability_gap`** (0.1825 ->
  0.370 on the real seed roster) **while slightly improving `min_utility`/`worst_case_regret`** —
  because reweighting the primary objective changes greedy construction's own search path from the
  first swap onward, not just which optional pass runs. This does not mean "unimportant" for other
  criteria — every other criterion is still measured and reported regardless — but it demonstrates a
  genuine, measurable tradeoff a deployment should expect, not assume away.
- Appropriate use: a roster/course where skill coverage is a hard requirement in spirit (even
  though `hard_skill_coverage_floor` remains an optional, separate hard-constraint knob) and where
  the instructor accepts a possible ability-balance cost.

### `preference_priority` (EXPERIMENTAL)

Prioritizes preference satisfaction and blocking-swap reduction.

- Effective settings: `preference.enabled=true`, `preference.hard_avoidance=true`,
  `preference.high_priority_weight=4.0` (raised from 3.0), `preference.avoidance_violation_weight=6.0`
  (raised from 5.0); ability/fairness/familiarity stay off.
- Hard constraints: identical to `legacy`, PLUS avoidance becomes a genuine hard constraint
  (`hard_avoidance=true`) — a deliberate choice, not an oversight: a profile whose whole purpose is
  preference-aware matching should not silently leave avoidance requests unenforced.
- Missing-data behavior: unchanged (`compute_preference_metrics`'s `None`-not-zero rule for missing
  preferences; `_hard_avoidance_repair` still only ever REDUCES existing violations, never fabricates
  new ones).
- Experimental status: EXPERIMENTAL.
- Known limitations: empirically (comparison table below), this profile resolved the
  `preference_instability` adversarial fixture exactly as well as `research_balanced` did (mean
  satisfaction 0.875 -> 1.0, blocking pairs 2 -> 0) while leaving ability/familiarity untouched —
  a deliberately narrower, faster-to-run alternative to `research_balanced` when preference/
  stability is the ONLY criterion a deployment cares about changing.
- Appropriate use: a deployment that has specifically collected preference data and wants it acted
  on, including treating avoidance requests as non-negotiable.

`ability_priority`/`familiarity_priority` were considered but NOT added — this session's empirical
comparison did not surface a scenario where a single-criterion ability- or familiarity-focused
profile would behave differently from simply enabling that one pass directly via an explicit
override on `legacy` (e.g. `{"profile": "legacy", "ability": {"enabled": true}}`), so a dedicated
named profile did not clear this part's own "genuinely useful and tested" bar. Revisit if a future
part identifies a scenario where reweighting the primary objective specifically toward ability or
familiarity (the way `skill_priority` reweights toward skill) produces a materially different
result from just flipping the one flag.

## The normalized objective (reporting layer)

Every `/teams/generate` response includes `normalized_objective` — see `app.matching.
compute_normalized_objective` and `OBJECTIVE_COMPONENT_SPECS`. This is a REPORTING construct: it
does not itself choose which assignment is returned (the existing per-pass mechanism, described in
`docs/pipeline.md`, still does that) — it exists so every criterion can be compared on one
consistent, documented [0,1] scale, with missing data EXCLUDED (weights renormalized over the
remaining usable components) rather than scored as a fabricated zero.

| Component | Definition | Missing-data behavior |
|---|---|---|
| `skill_quality` | mean per-team skill-coverage fraction | None only if zero teams |
| `behavioral_compatibility` | mean per-team rubric-weighted compatibility | None only if zero teams |
| `team_balance` | 1 − stdev(per-team total score), clamped [0,1] | None only if zero teams |
| `ability_balance` | 1 − strongest/weakest mean-ability gap | None if no team has a resolvable mean ability |
| `familiarity_band_fit` | fraction of teams WITHIN the density band, among teams with a resolvable band_status | None if no team has resolvable band_status |
| `anti_cluster` | 1 − mean cluster fraction, over teams with ≥2 members | None if every team has <2 members |
| `individual_fairness` | minimum per-student utility | None only if zero students |
| `preference_satisfaction` | mean tier-scored satisfaction, over students with ≥1 'prefer' entry | None if no student submitted any preference |
| `stability_penalty` (optional, weight 0.0 by default) | 1 − (blocking-swap count / max(1, n_students)) | None only if zero students |

**Overlap check (explicitly inspected, per this part's own instruction):**
`behavioral_compatibility` (team aggregate) vs. `individual_fairness`'s own `behavioral_fit`
sub-component (per-student) measure related but non-identical things and are kept separate;
`skill_quality` (team coverage) vs. `individual_fairness`'s `skill_fit` sub-component, same
relationship; `familiarity_band_fit`/`anti_cluster` (prior relationships, FamiliarityDB) vs.
`preference_satisfaction` (stated desires, PreferenceDB) are structurally independent data sources
with no double-count possible; `team_balance` (spread of total score) vs. `ability_balance` (spread
of mean ability specifically) measure related but distinct spreads; `stability_penalty` (blocking-
swap COUNT) vs. `preference_satisfaction` (mean SATISFACTION) are kept separate and
`stability_penalty` defaults to weight 0.0 specifically so it never silently double-counts against
`preference_satisfaction` unless a profile deliberately enables it (none currently does).

## Empirical profile comparison

Full data: `research_alignment_results.json`'s `profile_comparison` key (seed roster, all 12
adversarial fixtures, and synthetic rosters of 20/40/80 students), generated by
`backend/evaluation/run_report.py` (the real evaluation command — nothing below is hand-typed).

**Real 20-student seed roster** (`profile_comparison.seed_roster`):

| Profile | Overall objective | Ability gap | Min utility | Worst-case regret | Runtime (ms) |
|---|---|---|---|---|---|
| legacy | 0.789 | 0.183 | 0.476 | 0.200 | 35 |
| research_balanced | 0.789 | 0.183 | 0.476 | 0.200 | 206 |
| fairness_priority | 0.789 | 0.183 | 0.476 | 0.200 | 606 |
| skill_priority | 0.773 | **0.370** | **0.486** | **0.175** | 35 |
| preference_priority | 0.789 | 0.183 | 0.476 | 0.200 | 87 |

**Finding, with the exact tradeoff named:** on this specific roster, `skill_priority` is the ONLY
profile that changes the outcome at all — and it makes ability balance measurably *worse* (gap
0.183 → 0.370) while making minimum utility and worst-case regret measurably *better* (0.476 →
0.486, 0.200 → 0.175). Every other profile matched `legacy` exactly for this roster — not because
those passes didn't run (`optimization_trace` confirms they did, at nonzero runtime cost), but
because no swap existed that improved that pass's own target without regressing the (already
lexicographically protected) total-score/floor-min metrics for THIS specific roster. This is the
single clearest illustration in this dataset of "do not claim one profile is superior without
specifying the metric": skill_priority is better on utility/regret, worse on ability, for this
roster specifically.

**Adversarial fixtures** (`profile_comparison.adversarial_fixtures`), selected findings:

| Fixture | Profile(s) that changed the outcome | Effect |
|---|---|---|
| `stacked_team` | `research_balanced` | ability_gap 0.5 → 0.0; all others unchanged at 0.5 |
| `preference_instability` | `research_balanced`, `preference_priority` | mean_preference_satisfaction 0.875 → 1.0; n_blocking_pairs 2 → 0; `fairness_priority`/`skill_priority` unchanged at 0.875/2 |
| `friend_clique` | none | largest_connected_familiarity_cluster_fraction stays exactly 0.5 under every profile, including research_balanced — **explained, not a bug**: this fixture's clique fraction (0.5) sits exactly AT `max_clique_fraction`'s default threshold (0.5), so `_team_clique_violation` reports zero violation and there is nothing for the optimizer to fix |
| `utility_vs_fairness` | none | every profile matched `legacy` — no swap-reachable fairness improvement existed for this specific construction beyond what `_maximin_improve` already reached |

**Synthetic rosters (20/40/80 students)** — see Performance below for the runtime figures; the
ability-gap finding from `stacked_team` generalizes: `research_balanced` was the only profile that
ever improved `ability_gap` on the synthetic rosters generated for this comparison (0.375 → 0.25 at
n=20; unchanged at n=40; 0.75 → 0.5 at n=80), at a steadily increasing runtime cost.

## Performance impact

Runtime scales with roster size roughly as the `O(T² S²)` (or worse, for preference passes)
complexity each pass's own docstring already documents — measured, not assumed, on synthetic
rosters:

| n students | legacy | research_balanced | fairness_priority | skill_priority | preference_priority |
|---|---|---|---|---|---|
| 20 | 23 ms | 191 ms | 441 ms | 26 ms | 101 ms |
| 40 | 306 ms | 2,972 ms | 1,107 ms | 142 ms | 630 ms |
| 80 | 732 ms | **26,077 ms** | 9,294 ms | 778 ms | 7,865 ms |

**`research_balanced` and `fairness_priority`/`preference_priority` do not scale comfortably past
roughly 40-80 students in this implementation** — `research_balanced` took 26 seconds at n=80 in
this session's measurement. `legacy` and `skill_priority` (which never run the O(T²S²) optional
swap-search passes) stay fast at every size tested. Any real deployment considering
`research_balanced`/`fairness_priority`/`preference_priority` for a roster larger than ~50-60
students should budget for this, or treat it as a known implementation limitation to address before
Part 7's real-world study (see that section in `research_alignment_audit.md`'s Part 6 update).

## Part 6.5 update — performance fix, distinctness, sensitivity

Before preparing a real student study, Part 6.5 validated the profile system rather than extending it:

- **Performance fixed.** `research_balanced` at n=80 measured **26,077 ms** above — cProfile found the
  cause (the 4 optional passes lacking the per-team incremental caching `_maximin_improve`/
  `_two_opt_improve` already used) and fixing it brought that same measurement to **1,410 ms** (an ~18.5×
  speedup), with the identical final assignment on every existing test proven byte-for-byte unchanged. Full
  writeup: [`docs/performance.md`](performance.md).
- **Distinctness measured, not assumed.** Across the seed roster and all 12 adversarial fixtures, **9 of 13
  datasets collapse to an IDENTICAL partition across all 5 profiles** (team-label-invariant ARI = 1.0,
  co-membership distance = 0.0) — only the seed roster and 3 fixtures (`stacked_team`,
  `preference_instability`, `2opt_harms_fairness`) show any profile producing a different assignment at
  all. Full writeup: [`docs/profile_distinctness.md`](profile_distinctness.md).
- **Why `research_balanced` matches `legacy` on the real seed roster, precisely**: a read-only shadow trace
  (`backend/evaluation/trace_seed_roster.py`) replaying every candidate swap each optional pass considers
  found that for `ability`/`fairness`, improving swaps DO exist (37 and 21 of 160 candidates respectively
  would have improved that pass's own target metric) but are blocked — overwhelmingly by the hard
  skill-coverage-non-regression constraint 2-opt already saturated for this tight 20-student/5-team roster
  (139 of 160 candidates hit `skill_coverage_regressed`). For `familiarity`/`preference`, the target metric
  is already exactly 0 regardless of protections, because **the seed roster has zero recorded familiarity
  edges and zero preference submissions in the database** — there is nothing for those two passes to act
  on. Full detail, including the counterfactual breakdown, in `research_alignment_audit.md`'s Part 6.5
  section.
- **Sensitivity, not tuning.** `backend/evaluation/sensitivity_analysis.py` confirmed
  `AbilityConfig.ability_level_thresholds` and `FairnessConfig.acceptable_utility_floor` are, as their own
  docstrings claim, purely reporting-level (never move the assignment); `weights.skill_coverage` does move
  the assignment (with one observed discontinuity between 0.4 and 0.7). `research_balanced`'s own default
  settings were confirmed to collapse to `legacy`'s exact partition on the seed roster
  (co-membership distance 0.0) — the same finding as above, from a different angle. No weight is
  recommended by this analysis; see `evaluation_output/sensitivity_results.json`.
- **Experiment-readiness check added** (`backend/evaluation/experiment_readiness.py`) — given a roster and
  two profiles, reports whether they're likely to produce distinguishable conditions, which criteria have
  sufficient data, and why not when they don't (e.g. run against `legacy` vs `research_balanced` on the
  seed roster: `distinct_enough_for_experimental_conditions: false`, `excluded_for_missing_data:
  ["familiarity", "preference"]`). The "distinct enough" threshold is an explicit, configurable
  `ReadinessConfig` field, not a claimed universal cutoff.
- **The one remaining permanently-failing adversarial test was retired.** `test_stacked_ability_gets_
  redistributed_across_teams` asserted the research requirement against `legacy` and was left deliberately,
  permanently failing to "document a gap." It is renamed to `test_legacy_profile_does_not_redistribute_
  stacked_ability` and now asserts LEGACY's own documented contract (ability balancing stays off) instead —
  the original research requirement is unweakened, asserted in full, and passing, against
  `research_balanced` in the sibling test directly below it. The full backend suite is 0 failures.

## Configuration precedence recap

```text
system defaults (TeamGenerationRequest()'s own class defaults)
        |
        v
named profile's overrides (OPTIMIZATION_PROFILES[profile]["overrides"])
        |
        v
explicit request-level field overrides (highest precedence — Pydantic model_fields_set)
        |
        v
resolved TeamGenerationRequest (immutable, fresh instance) + ProfileResolution record
```
