# Research-Alignment Audit — Behavioral Team Formation

**Scope:** `feat/gemini-integration` branch, commit `f800b4f`. Phases 1–6 (below) were a read-only audit —
no production code was modified. **Part 6.5 (see that section near the end of this document) is the one
exception**: it made two narrowly-scoped, equivalence-proven production changes to `app/matching.py` (an
incremental per-team caching performance fix, and a NaN/infinity guard on the cross-pass regression check)
— both changes are additive/behavior-preserving, with dedicated before/after equivalence tests; see
`docs/performance.md` and that section for the full evidence trail. Companion files:
[`research_traceability_matrix.md`](research_traceability_matrix.md) (Phase 1 detail),
[`recommended_changes.md`](recommended_changes.md) (Phase 6 detail),
[`research_alignment_results.json`](research_alignment_results.json) (Phase 5 raw output),
[`backend/evaluation/`](backend/evaluation/) (Phase 4 harness, extended in Part 6.5),
[`backend/tests/test_research_alignment.py`](backend/tests/test_research_alignment.py) (Phase 3/5 tests),
[`docs/performance.md`](docs/performance.md), [`docs/profile_distinctness.md`](docs/profile_distinctness.md)
(Part 6.5 deliverables).

## A note on research sources

`docs/research-notes.md` exists in the repo but is **empty** — there is no literature-review file to cite
against. The only in-repo research citations live as comments inside
[`backend/traits/rubric_v1.yaml`](backend/traits/rubric_v1.yaml): Peeters et al. (2006), Bell (2007),
Ohland et al. (2012), Loughry et al. (2007), De Dreu & Weingart (2003), and Jehn (1995) — all cited
specifically for the `accountability`, `cooperativeness`, and `conflict_style` dimensions. Everywhere this
audit references those citations, it is quoting the codebase's own justification, not adding a new one.
The seven-criterion "good team" definition supplied in this audit's brief is **not** independently cited
anywhere in the repo — treated throughout as a working hypothesis to test the system against, per the
task's own instruction, not as an established finding.

---

## Phase 1 — Traceability

See [`research_traceability_matrix.md`](research_traceability_matrix.md) for the full file/function/line
mapping of all 7 research criteria (plus the explanation pipeline). Summary:

| Criterion | Support |
|---|---|
| 1. Skill coverage | Partial (measured, weighted, no hard floor; no concentration metric) |
| 2. Mixed ability | **Partial (Part 3)** — was "not supported" at audit time; `ability_score` now exists and between-team balance is reported by default and optionally optimized for — see Phase 2/4 updates below |
| 3. Conscientiousness (accountability) | Partial — mean tracked, **floor guarantee internally contradicted** |
| 4. Agreeableness (cooperativeness) | Partial — same contradiction as #3 |
| 5. Familiarity / anti-clique | **Partial (Part 4)** — was "not supported" at audit time; a real `familiarity` edge table + graph-based metrics now exist, reported by default and optionally optimized for — see Phase 2/4 updates below |
| 6. Comparable team strength | Not supported cross-team (only min-of-floor-dims is balanced) |
| 7. Fair distribution of benefit | **Partial (Part 2)** — was "not supported" at audit time; a real per-student utility function now exists and is reported (not yet optimized for) by default — see Phase 2/5 updates below |
| 8. Preference satisfaction & stability | **Partial (Part 5)** — was "not supported" at audit time; a real `preferences` edge table + tier-scored satisfaction metrics + formal blocking-pair/swap/move stability diagnostics now exist, reported by default and optionally optimized for — see Phase 2/4 updates below |
| (Explanation generation, in-scope but not one of the 7) | Partially broken in the live Gemini path |

---

## Phase 2 — The objective function

### In plain English

When `formation_mode == "behavioral"` (the default), `POST /teams/generate`
(`backend/app/matching.py:559-631`) does:

1. **Feasibility check** — reject the request (HTTP 422) only if some floor dimension (accountability or
   cooperativeness) has *more* students at the lowest level than there are teams.
2. **Greedy build** — spread floor-level students round-robin (one per team), seed each empty team with
   the highest-skill remaining student, then repeatedly add whichever remaining student most improves
   whichever team's weighted score.
3. **2-opt** — repeatedly swap one student from team *i* with one from team *j* if it raises the **summed**
   weighted score of both teams; repeat until no such swap exists.
4. **Maximin** — repeatedly swap students if it raises the **minimum, across all teams, of the team-mean
   accountability/cooperativeness level**; repeat until no such swap exists.
5. Report each team's score breakdown, plus (informationally) any remaining floor violations.

### In mathematical notation

For a team $T$ with weights $w = (w_1, w_2, w_3, w_4)$ (default $0.4, 0.3, 0.2, 0.1$, always
renormalized to sum to 1 by `_normalize_weights`, `matching.py:221-225`):

$$
\text{total}(T) = w_1 \cdot \text{skill\_coverage}(T) + w_2 \cdot \text{behavioral\_compat}(T) + w_3 \cdot \text{availability\_overlap}(T) + w_4 \cdot \text{shared\_interests}(T)
$$

$$
\text{skill\_coverage}(T) = \frac{\left| \left(\bigcup_{s \in T} \text{skills}(s)\right) \cap R \right|}{|R|}, \quad R = \text{REFERENCE\_SKILLS} \text{ (fixed 12-item set, } matching.py{:}23\text{-}27\text{)}
$$

$$
\text{behavioral\_compat}(T) = \frac{\sum_{d \in D} \lambda_d \cdot \text{dim\_score}_d(T)}{\sum_{d \in D} \lambda_d}, \quad D = \{d : \text{weight}_d > 0 \wedge \text{composition\_rule}_d \neq \text{null}\}
$$

where, per dimension $d$'s `composition_rule` (read from `rubric_v1.yaml` at runtime):

$$
\text{dim\_score}_d(T) = \begin{cases}
\dfrac{1}{n_{\text{known}} \cdot (\ell_{\max})}\displaystyle\sum_{s \in T,\, d \text{ known}} \text{level}_d(s) & \text{composition\_rule} = \texttt{high\_mean\_with\_floor} \\[2mm]
0.7 + 0.3 \cdot \dfrac{|\{\text{distinct values of } d \text{ among known members}\}|}{n_{\text{known}}} & \text{composition\_rule} = \texttt{soft\_compatibility}
\end{cases}
$$

$$
\text{availability\_overlap}(T) = \frac{1}{\binom{|T|}{2}} \sum_{\{a,b\} \subset T} \frac{|\text{avail}(a) \cap \text{avail}(b)|}{|\text{avail}(a) \cup \text{avail}(b)|}
$$

$$
\text{shared\_interests}(T) = \frac{\left|\bigcap_{s \in T} \left(\text{interests}(s) \cup \text{goals}(s)\right)\right|}{\left|\bigcup_{s \in T} \left(\text{interests}(s) \cup \text{goals}(s)\right)\right|}
$$

**Diagnostic-only, not in the objective:**

$$
\text{conflict\_risk}(T) = \min\!\left(1,\ 0.7 \cdot \text{mismatch\_rate}(T) + \max(0, (\,n_{\text{directive}} - 1) \cdot 0.15)\right)
$$

**Floor mechanism (procedural, not a weighted term):** feasibility pre-check allows up to $n_{\text{teams}}$
floor-level students per floor dimension; `_maximin_improve` hill-climbs
$\max \min_{T \in \text{partition}} \text{high\_mean\_with\_floor-score}(T)$; post-hoc
`_check_floor_violations` flags **any** floor-level member on **any** team — a stricter threshold than the
feasibility check used, which is the contradiction detailed in
[`recommended_changes.md` #1](recommended_changes.md).

### Which research criteria it rewards vs. treats as constraints vs. ignores entirely

| Criterion | Rewarded in objective? | Constraint? | Absent? |
|---|---|---|---|
| Skill coverage | Yes ($w_1$, default 40% — the single largest weight) | No hard floor | — |
| Conscientiousness (accountability) | Yes (inside `behavioral_compat`, rubric weight 0.35 of that sub-objective) | Yes, procedurally (contradictory, see above) | — |
| Agreeableness (cooperativeness) | Yes (same weight, 0.35) | Same as accountability | — |
| Mixed ability | **Update (Part 3):** No by default (`ability.enabled=false`); Yes, optionally (`_ability_balance_improve`, narrows the between-team gap — "mixing", not "maximizing") | Metric only by default (`ability_summary` on every response) | No longer entirely absent — see below |
| Familiarity / anti-clique | **Update (Part 4):** No by default (`familiarity.enabled=false`); Yes, optionally (`_familiarity_balance_improve`, targets a density band + a direct clique-concentration term) | Metric only by default (`familiarity_summary` on every response) | No longer entirely absent — see below |
| Comparable team strength | Partially (floor-dim minimum via `_maximin_improve`); **Update (Part 3):** ability-based between-team gap/variance now also measured (`ability_summary.strongest_weakest_gap`) and optionally optimized for | — | Skill coverage and overall total-score still uncompared cross-team |
| Fair distribution of benefit | **Update (Part 2):** No by default (`fairness.enabled=false`); Yes, optionally (`_fairness_improve`) | Metric only by default (`fairness_summary` on every response) | No longer entirely absent — see below |
| Preference satisfaction / stability | — | — | **Yes, entirely absent** |

### Conflicts, gamability, and hidden-average risk

- **Weights conflict with the floor mechanism.** `behavioral_compat`'s weight (default 0.3, further split
  0.35/0.35/0.10/0.025×3/0 internally) means a team can score well on `total` while carrying an unresolved
  floor violation — the two live in entirely separate code paths (`_total_score` vs.
  `_check_floor_violations`) that never talk to each other. A team is not blocked from being returned just
  because it violates the rubric's own stated floor guarantee.
- **Averages do hide weak members**, exactly as the research brief warns, in one specific way: `total`
  score is a team-level weighted average of team-level averages. `evaluation/metrics.py:conscientiousness()`
  demonstrates this concretely — `research_alignment_results.json` shows a team whose *mean* accountability
  looks strong while its *minimum* sits at the floor (adversarial fixture #2, `hidden_bad_apple`,
  designed exactly for this and confirmed).
- **The score can be gamed toward superficially strong but unfair teams** in the specific sense that
  `_two_opt_improve` only ever looks at the **sum** of two teams' scores when evaluating a swap
  (`matching.py:445-451`) — a swap that raises team $i$'s score by more than it lowers team $j$'s passes,
  even if $j$ was already the weaker team. `test_greedy_seeding_does_not_starve_a_team_of_skills` and
  `test_2opt_does_not_worsen_the_weakest_teams_floor_minimum` (Phase 3/5 below) probe exactly this; the
  former **failed** (2-opt lowered the weakest team's skill coverage below pure-greedy), the latter passed
  on its specific fixture (floor-min wasn't degraded) — i.e., the mechanism for this failure mode exists
  in the code and was reproduced for skill coverage, though not for the floor dimensions in the fixture
  tried. See [`recommended_changes.md` #7 and #9](recommended_changes.md).
  **Update — Part 1.5 (verified):** both are now fixed. `_two_opt_improve` and `_maximin_improve` share a
  centralized `validate_assignment()` hard-constraint check (`app/matching.py`) — a swap is rejected outright
  if it would drop the partition's minimum skill coverage below what greedy already achieved, or increase
  behavioral-floor-policy violations, regardless of aggregate score gain. `_maximin_improve`'s acceptance is
  now lexicographic (hard constraints → primary objective, 0.0 regression tolerance by default → floor-
  minimum), so it can no longer trade away 2-opt's optimum to chase floor-min gains. On the real 20-student
  seed roster this took `blocking_moves_score_based` from 10 unresolved swaps to 1, and skill coverage from
  min 58%/mean 78% to min 83%/mean 87% — see the regenerated `research_alignment_results.json` and
  `backend/tests/test_optimizer_invariants.py`.
- **No metric conflicts were found between the four weighted components themselves** (skill coverage,
  behavioral compat, availability, interests) — they are independent set/mean operations over disjoint
  data. The conflicts found are all between the weighted objective and the *unweighted, procedural*
  mechanisms (floor enforcement, cross-team balance) layered around it.

---

## Phase 3 — Adversarial test datasets

12 fixtures matching the brief's list, in
[`backend/evaluation/fixtures.py`](backend/evaluation/fixtures.py) (`ALL_FIXTURES`), each with a docstring
naming the specific research property it targets. Exercised through the **unmodified production pipeline**
(`_greedy_build → _two_opt_improve → _maximin_improve`, the exact sequence `matching.py:601-603` uses) by
[`backend/tests/test_research_alignment.py`](backend/tests/test_research_alignment.py). Full input/output
per fixture is in `research_alignment_results.json → adversarial_fixtures`. Per-fixture pass/fail:

| # | Fixture | Research property probed | Result |
|---|---|---|---|
| 1 | `skill_vs_behavior` | Skill coverage vs. behavioral compat tradeoff | Min team coverage stayed ≥25% (passed threshold); floor violations present (see #2 below, same root cause as production data) |
| 2 | `hidden_bad_apple` | Mean hides one floor-level member | **Confirmed by design** — mean=1.83/2, min=0/2 on the same team |
| 3 | `stacked_team` | Ability redistribution across teams | **FAILED** — 1.00 ability-spread out of 2.0 max after full pipeline |
| 4 | `isolated_student` | Familiarity invisible to algorithm | Metric layer correctly detects isolation once given data; production algorithm structurally cannot see it |
| 5 | `friend_clique` | Clique not broken up | **FAILED** (as expected) — clique of 4 remained fully concentrated |
| 6 | `utility_vs_fairness` | Worst-case individual regret | Passed at a 0.35 regret threshold (see caveat in `recommended_changes.md` #3) |
| 7 | `preference_instability` | Preference satisfaction + blocking pairs | **FAILED** (as expected) — 2 blocking pairs, <50% top-half preference satisfaction |
| 8 | `uneven_ability` | 3 advanced / 2 teams cannot split evenly | Passed — gap stayed within the best-achievable 1.0 bound |
| 9 | `uneven_roster_size` | 11 students / team_size 4 | **FAILED** at audit time — produced team sizes [7, 4], not ~[4,4,3]. **Fixed (Part 1):** `_compute_n_teams`/`_target_team_sizes` now produce [4, 4, 3] |
| 10 | `missing_tied_extreme` | Unknown dims, ties, double-floor member | Passed — unknowns correctly excluded from floor check; double-floor member correctly isolated |
| 11 | `greedy_local_optimum` | Greedy seeding starves a team of skills | **FAILED** at audit time — 2-opt made worst-team coverage *worse* than plain greedy (0.500 → 0.417). **Fixed (Part 1.5):** constraint-safe 2-opt now finds an alternative swap reaching the same aggregate score (1.403) with coverage 0.583/0.583 on both teams — see `test_known_failing_skill_coverage_fixture_now_passes` |
| 12 | `2opt_harms_fairness` | 2-opt raises total score, harms floor-min | Passed on this specific construction (floor-min not degraded) — see Phase 2 caveat |

A parametrized cross-cutting test (`test_no_uncaught_floor_violations_after_full_pipeline`) also ran all 12
fixtures through the floor-violation check independently of each fixture's own targeted property: **6 of
12** came back with unresolved floor violations, for the structural reason documented in
[`recommended_changes.md` #1](recommended_changes.md) — this is not fixture-specific, the same result
reproduces on the real 20-student seed roster (see Phase 5).

No dataset assumes a single "correct" grouping — every assertion above tests a property or threshold
(spread ≤ X, coverage ≥ Y, regret ≤ Z), consistent with the brief's instruction that several assignments
can satisfy the criteria.

---

## Phase 4 — Evaluation harness

[`backend/evaluation/metrics.py`](backend/evaluation/metrics.py) implements every metric from the brief's
Phase 4 list, without modifying `app/matching.py` (it only **imports** production functions —
`_total_score`, `_check_floor_violations`, `_greedy_build`, `_two_opt_improve`, `_maximin_improve` — so
the harness evaluates the actual shipped algorithm, not a reimplementation of it). All thresholds/weights
live in `EvalConfig` (one dataclass, every field commented with its purpose).

| Requested metric | Implemented as | Notes |
|---|---|---|
| Required-skill coverage rate | `skill_coverage()` | Uses `REFERENCE_SKILLS` from `app.matching` directly |
| Missing skills per team | `skill_coverage()["per_team"][i]["missing_skills"]` | — |
| Within-team ability diversity | `ability_diversity()` (legacy 0-2 `experience_level` scale, unchanged) + `ability_balance()` (Part 3: `[0.0,1.0]` `ability_score`, falls back to `experience_level`) | Both kept — see Part 3 update below |
| Between-team strength variance | `team_strength_comparison()["per_metric_gap"]` | Variance/stdev/range/min/max per component |
| **Part 3 — between-team ability gap, distribution, skill/ability correlation** | `ability_balance()["between_team_mean_variance"/"strongest_weakest_gap"/"distribution"/"skill_coverage_ability_correlation"]` | New this part — see below |
| Mean & minimum conscientiousness | `conscientiousness()` | Reuses rubric-driven ordinal levels |
| Mean & minimum agreeableness | `agreeableness()` | Same mechanism, different dimension |
| Familiarity density | `familiarity_density()` (legacy, unchanged, side-channel-data only) + `familiarity_balance()` (Part 4: real `familiarity` DB table, graph-based) | Legacy still `supported: False` without a side channel; new one is always `supported: True`, `[]` edges = everything unknown |
| Clique concentration | `clique_concentration()` (legacy, label-based) + `familiarity_balance()["per_team"][i]["largest_preexisting_subgroup_size"]` (Part 4: connected components) | Same relationship as the row above |
| **Part 4 — isolated students, largest pre-existing subgroup, cross-team density variance, target band** | `familiarity_balance()["per_team"][i]["isolated_members"/"largest_preexisting_subgroup_size"/"band_status"]`, `["cross_team_density_variance"/"teams_below_band"/"teams_within_band"/"teams_above_band"]` | New this part — see below |
| Individual preference satisfaction | `preference_satisfaction()` (legacy, unchanged, side-channel-data only) + `preference_stability()["preference_satisfaction"]` (Part 5: real `preferences` DB table, tier-scored) | Legacy still `supported: False` without a side channel; new one is always `supported: True`, `[]` entries = everything missing |
| Minimum individual utility / variance / worst-case regret | `individual_utility()` | Exact brute-force alternative-team search below a configurable roster-size cap |
| Team-balance score | `team_strength_comparison()["team_balance_score"]` | `1 - stdev(team strengths)`, configurable component weights |
| Blocking pairs / blocking moves | `blocking_moves_score_based()` (always available, score-proxy) + `blocking_pairs_preference_based()` (legacy, needs preference data) + `preference_stability()["stability"]` (Part 5: real preference data, THREE formal definitions — blocking pair/swap/move) | All explicitly labeled proxy vs. preference-true; new this part — see below |
| **Part 5 — tier-scored satisfaction, unmet high-priority preferences, avoidance violations, formal stability** | `preference_stability()["preference_satisfaction"]["mean_satisfaction"/"min_satisfaction"/"worst_case_preference_regret"/"total_unmet_high_priority_preferences"/"total_avoidance_violations"]`, `["stability"]["is_stable"/"n_blocking_pairs"/"n_blocking_swaps"/"n_blocking_moves"]` | New this part — see below |
| Constraint violations | `constraint_violations()` | Directly calls `app.matching._check_floor_violations` |
| Runtime and scalability | `measure_runtime()` | Times each of the 3 production passes separately |

Synthetic auxiliary data the schema doesn't support (familiarity, friend-group labels, preferences) is
passed as **optional side-channel parameters** to `evaluate_teams(...)`, never invented inside production
code — when omitted, every such metric reports `supported: False` with a specific note on what field/table
would be needed, rather than a fabricated number.

---

## Phase 5 — Evaluation run

### Commands executed

```bash
cd backend
python -m pytest tests/ -v                              # existing suite
python -m pytest tests/test_research_alignment.py -v     # new research-alignment suite
python -m evaluation.run_report                          # generates research_alignment_results.json
```

### Environment note (recorded, not glossed over)

The repo's own `backend/venv` was broken (`bad interpreter: .../venv/bin/python3.12: no such file or
directory` — the venv was created against a Python install not present on this machine). A fresh venv was
built from `backend/requirements.txt` to run tests. That surfaced a second, real gap:
`backend/ai/gemini_explainer.py:6` does `from google import genai`, but **`google-genai` is not listed in
`requirements.txt`** — `pip install -r requirements.txt` alone leaves `app.main` unimportable
(`ModuleNotFoundError: No module named 'google'`), because `main.py:66-70` unconditionally wires up
`explainability.router`, which imports `GeminiExplainer` at module load regardless of whether an API key
is configured. Installed `google-genai` manually to proceed. See
[`recommended_changes.md` #14](recommended_changes.md).

A second, unrelated isolation gap surfaced while running the *existing* suite: `test_assessment.py`'s
`client` fixture patches `app.database.SessionLocal`/`init_db`, but `app.main.lifespan()` had already bound
its own references to those names at import time, so `TestClient(app)`'s startup hook still touches the
real `backend/behavioral_team_formation.db` (confirmed — it added the locally-pending `evidence_units`
table to that file; no test-generated student rows leaked in, verified by inspection). Endpoint handlers
are unaffected (they re-import `SessionLocal` lazily per-request). See
[`recommended_changes.md` #17](recommended_changes.md).

### Results

**Existing test suite:** `28 passed, 1 warning in 0.46s` — full pass, all 28 pre-existing tests green
(verbatim `research_alignment_results.json → existing_test_suite.summary_line`). No changes were made to
any existing test.

**Research-alignment suite:** `9 failed, 19 passed in 0.52s`
(`research_alignment_results.json → research_alignment_test_suite`). Per-test breakdown is in that file
and summarized in the Phase 3 table above and the traceability matrix. Every failure traces to a specific,
cited gap in `recommended_changes.md` — none are test bugs; each failing assertion states in its message
exactly which production code path produced the unexpected result.

**Key metric values against the real 20-student seed roster** (`sample_students.json`, `team_size=4`,
5 teams, `research_alignment_results.json → sample_students_evaluation`):

- Skill coverage: mean 78%, min 58% (team-3 missing `communication, data analysis, devops, project
  management, research`).
- Ability (experience_level proxy) spread across teams: 0.75–1.75 (of 0–2 range) — confirms Phase 3
  finding #3 is not a fixture artifact.
- Conscientiousness/agreeableness: global mean-of-team-means 1.6/2 (looks healthy), but
  `constraint_violations.floor_ok == False` — 2 teams each carry one member at the cooperativeness floor
  (Hamid Karimi, Kwame Asante), reproducing the Phase 2/Critical-#1 contradiction on real, non-adversarial
  data.
- `blocking_moves_score_based`: 10 unresolved score-improving swaps remained even after the full
  greedy→2opt→maximin pipeline (see `recommended_changes.md` #9).
- `individual_utility`: min 0.530, mean 0.584, non-zero variance — computed via exact brute-force regret
  search (roster size 20 is within the exact-search cap).
- Runtime scalability sweep: 8→64 students (8×) grew total pipeline time from 1.1ms to 322.9ms (~295×) —
  worse than cubic; immaterial at current scale (`recommended_changes.md` #16).

**Update — Part 1 + Part 1.5 (verified, regenerated in `research_alignment_results.json`):** the bullets
above are the audit's original, as-found figures on this same roster and are left unchanged as the
historical record. Current figures after both fix passes: `constraint_violations.floor_ok == True` (0
violations, Part 1's floor-policy fix); skill coverage min 58%→**83%**, mean 78%→**87%** (Part 1.5's
constraint-safe 2-opt); `blocking_moves_score_based` 10→**1** unresolved swaps (Part 1.5's lexicographic
maximin). `individual_utility` and the runtime-scaling shape are unchanged by either fix pass (individual
utility is explicitly out of scope for both; runtime is dominated by the same O(n_teams²·team_size²)
sweep structure, now with an incremental per-team cache — see `recommended_changes.md` #16, still not
actioned).

**Update — Part 2 (verified, regenerated):** `individual_utility` is no longer the team-total-score
placeholder the bullet above describes — `app.matching.compute_assignment_utilities()` now computes a real
per-student value from 4 existing-data components (skill_fit, behavioral_fit, availability_fit,
interest_fit; see `recommended_changes.md` #3). On the SAME real 20-student seed roster (with Part 1/1.5's
fixes already applied): **mean_utility=0.602, min_utility=0.476** (down from the placeholder's min=0.530 —
expected, since the placeholder copied the TEAM's score to every member and so could never show a student
individually worse off than their own team's aggregate), **max_utility=0.718, utility_range=0.242,
utility_stdev=0.078, worst_case_regret=0.200, 1 of 20 students (5%) below the 0.5 acceptable-utility floor**.
Mapping to the four welfare concepts requested for this part: **total welfare** = `total_utility` = 12.03;
**average individual welfare** = `mean_utility` = 0.602; **worst-off student welfare** = `min_utility` =
0.476; **equality of welfare** = `utility_variance`/`utility_stdev`/`utility_range` (lower = more equal) —
0.0061 / 0.078 / 0.242 respectively, i.e. moderate but non-trivial inequality on this roster. `fairness.
enabled` was NOT turned on for this regeneration (default), so these numbers describe the SAME assignment
Part 1.5 already produced — evaluated, not yet optimized for. See `backend/tests/test_individual_utility.py`
for the adversarial cases (including one where enabling `fairness.enabled` demonstrably raises min_utility
without regressing the total-score objective or the floor-minimum).

**Update — Part 3 (verified, regenerated):** `ability_diversity` (0-2 `experience_level`-ordinal scale) is
unchanged, kept for backward compatibility. A new `ability_balance` metric adds the `[0.0, 1.0]`
`ability_score`-based view (`app.models.CompetenceSignature.ability_score`, falling back to
`experience_level` when unset — see `recommended_changes.md` #2). On the SAME real 20-student seed roster
(10 of 20 students carry a real `ability_score`; the other 10 exercise the fallback path):
**strongest_weakest_gap=0.183, between_team_mean_variance=0.0046, between_team_mean_stdev=0.068**; ability
distribution **{developing: 2, medium: 11, high: 7}**; **skill-coverage/mean-ability correlation r=0.58**
(moderate positive — teams with stronger average ability also tend to cover more reference skills on this
roster, though n=5 teams is a small sample for a correlation coefficient — see the metric's own
`correlation_note` guard for when it's suppressed as not meaningful). `ability.enabled` was NOT turned on
for this regeneration (default), so these numbers describe the SAME assignment Part 1.5/2 already produced
— evaluated, not yet optimized for. See `backend/tests/test_ability_balance.py` for the adversarial cases,
including one proving `ability.enabled=true` narrows exactly the gap
`test_stacked_ability_gets_redistributed_across_teams` documents as unaddressed by default.

**Update — Part 4 (verified, regenerated):** `familiarity_density()`/`clique_concentration()` (legacy,
side-channel-data only) are unchanged, kept for backward compatibility with existing adversarial fixtures.
A new `familiarity_balance` metric is backed by a REAL data source (the `familiarity` DB table /
`app.models.FamiliarityEdge`) instead of a caller-supplied dict — `backend/evaluation/run_report.py` was
not modified to pass real edges into it, so on the regenerated 20-student seed roster it correctly reports
`total_known_pairs=0` (no edges were passed to that specific script) rather than a fabricated number; the
20-student seed roster's REAL familiarity data (`sample_familiarity.json` — 8 edges: a 4-person clique among
s01-s04, one worked-together pair, one confirmed-zero pair) is exercised instead through the live
`/teams/generate` endpoint and `backend/tests/test_familiarity.py`'s 60 tests, which is where this part's
adversarial scenarios (unfamiliar roster, isolated student, clique, multiple clusters, unknown-vs-zero,
duplicate/invalid edges, tradeoffs) are actually demonstrated. `familiarity.enabled` was not turned on for
the `run_report.py` regeneration (default), consistent with Parts 2/3's same choice.

**Update — Part 4.5 (verified, regenerated): corrected the connected-cluster computation, added
known-pair-coverage handling.** Part 4's `largest_preexisting_subgroup_size` had a genuine correctness bug,
not just a naming issue: it was computed by finding connected components over the **global** familiarity
graph (every student across every team) and then intersecting each component with a team's membership
afterward. That approach could report two teammates as "clustered" when they were connected only through a
**path running through a student on a different team** — e.g. A and C are strongly connected, C and B are
strongly connected, A and B share no edge at all, and C sits on a different team; the global-component
approach would still place A and B in the same reported subgroup. Renamed the corrected concept
`connected_familiarity_cluster` (`largest_connected_cluster` in the API/report shape) and recomputed it from
the **team-induced subgraph** instead — only that team's own members and only edges between them, so a path
through an outside student is structurally impossible. The old field name was not preserved as a deprecated
alias (a pre-1.0 research prototype with no external consumers, and the old field represented a bug, not
just a naming choice) — see `app/matching.py`'s familiarity-section module docstring and
`FamiliarityClusterStats` in `app/models.py` for the full explanation. Also renamed "clique" to "cluster"
throughout: a connected component only requires a *path* between members, not every pair being directly
connected, so calling it a clique overclaimed cohesion a chain or star-shaped connection pattern does not
have — new per-cluster cohesion fields (`known_internal_edge_coverage`, `mean_known_internal_strength`,
`thresholded_internal_edge_density`, `is_quasi_clique`) let a caller tell a loose chain/star apart from an
actual clique instead of this module ever asserting a cluster IS one.

Separately, added `known_pair_coverage` (known teammate pairs / all possible teammate pairs) and a
configurable `minimum_known_pair_coverage` (default 0.5): a team whose familiarity data is too sparse to
trust now gets `band_status="insufficient_data"` instead of being silently classified below/within/above the
target band on a handful of pairs, and `_familiarity_balance_improve`'s optional soft objective does not
apply a density-band penalty to such a team unless `penalize_insufficient_data=True` is explicitly set (off
by default) — the optimizer does not "confidently" chase a target it lacks the evidence to judge. Structural
warnings (`isolated_members`) remain populated from whatever known edges DO exist even when
`band_status="insufficient_data"`, since they are supported by real (if incomplete) data, not by the
suppressed aggregate density judgment. `band_status="not_applicable"` was added for teams of 0-1 members,
where there are no possible pairs to classify at all. All of this is additive to the per-team/summary report
shape (new fields alongside the renamed one) — see `backend/tests/test_familiarity.py`'s Part 4.5 sections
(17 new tests: 5 for the team-induced-cluster fix, 8 for coverage/insufficient-data handling, 4 for
optimizer verification) and `backend/app/matching.py`'s familiarity-section module docstring for the
graph-definition rewrite. All 77 tests in that file pass; the full backend suite (322 passing outside two
pre-existing, explicitly out-of-scope failures — ability-gap redistribution and preference/blocking-pair
stability, neither touched by this part) shows no regression from this change.

No test result is reported here that was not actually executed — every number above is read directly out
of `research_alignment_results.json`, generated by `backend/evaluation/run_report.py` in this session.

**Update — Part 5 (verified, regenerated): real preference/stability data model, tier-scored satisfaction,
formal many-to-one stability definitions.** `preference_satisfaction()`/`blocking_pairs_preference_based()`
(legacy, unchanged, side-channel `Dict[student_id, List[ranked_id]]` only) are kept for backward
compatibility with `fixture_preference_instability` and `test_research_alignment.py`'s existing tests. A new
`preference_stability()` metric is backed by a REAL data source (the `preferences` DB table /
`app.models.PreferenceEntry`) supporting ranked AND tied preferences (multiple entries sharing a rank),
unranked entries (a real, honored, lower-priority preference — not a rejection), avoidance requests (with a
`priority="high"` escalation), missing preferences (zero entries — `None`, never scored as dissatisfaction),
and optionally preferences over team CHARACTERISTICS rather than named students (currently resolving the
`{"skill": ...}` key only; other keys are accepted, stored, and excluded from scoring rather than silently
misjudged — see `PreferenceEntry`'s docstring). Satisfaction is TIER-scored: entries sharing a rank form one
tie group; unranked entries form one flat group after every ranked one; a student's score is `(n_tiers -
matched_tier) / n_tiers` for whichever preference matched at the best tier, `0.0` if none matched, and `None`
(never `0.0`) if the student submitted no "prefer" entries at all — see `app/matching.py`'s
preference-framework module docstring for the exact formula and worked distinctions (mutual/one-sided/cyclic
preferences, ties, missing data).

Because this is MANY-TO-ONE team assignment (fixed-size teams), not the classical one-to-one stable-
roommates/stable-marriage problem, three DISTINCT, formally defined notions of instability are implemented
rather than one: a **blocking pair** (a cheap envy diagnostic — two students on different teams who each rank
the other strictly ahead of every one of their own current teammates; this is a formalization of the exact
rank-vs-current-teammates comparison the legacy `blocking_pairs_preference_based()` already used); a
**blocking swap** (the executable notion — an ordered 1-for-1 exchange where BOTH swapping students strictly
prefer their post-swap team composition, using the full tier-scored satisfaction function rather than the
cheaper named-teammate shortcut); and a **blocking move** (a WEAKER-on-preference, STRICTER-on-feasibility,
intentionally asymmetric notion directly answering "whose utility must improve and whether the receiving
team's constraints must remain satisfied" — only the MOVING student's utility must strictly improve, and the
RECEIVING team's HARD constraints (validate_assignment's size/skill-coverage/behavioral-floor checks), not its
members' preferences, must hold). **A stable assignment is defined, by default, as one with no blocking
swap** — the DOCUMENTED default `PreferenceSummary.is_stable` is checked against, since that is the notion
this system's own swap-based repair mechanism can actually act on; the other two definitions are ALWAYS also
computed and reported as supplementary diagnostics, and `preference.stability_definition` lets a caller pick
a different one of the three as authoritative instead, per this part's own instruction to keep one clearly
documented default and make alternatives configurable.

Preference satisfaction and stability are evaluated FIRST as always-on metrics (`preference_summary` on every
`/teams/generate` response, computed from whatever `PreferenceDB` rows exist, regardless of any flag). A
SEPARATE, optional, disabled-by-default repair pass (`_preference_repair_improve`,
`preference.enabled=false` by default) may additionally change the returned assignment, minimizing a per-team
violation (unmet satisfaction + a configurable weight per unmet HIGH-priority preference + a configurable
weight per avoidance-violation pair) under the SAME lexicographic hard-constraint-first acceptance rule as
every other optional pass in this codebase (`_familiarity_balance_improve`, `_ability_balance_improve`,
`_fairness_improve`): `validate_assignment`'s team-size/skill-coverage/behavioral-floor hard constraints must
hold, the total-score objective may not regress, the accountability/cooperativeness floor-minimum may not
regress, and — the "explicitly configured institutional restriction" this repair phase's hard-constraint
guarantee extends to cover, when `preference.hard_avoidance=true` — the pass additionally never lets its own
swap INCREASE the avoidance-violation count on a team it touches. Confirmed directly against this part's own
adversarial fixture: `fixture_preference_instability` (`fraction_with_at_least_one_preferred_teammate`) rises
from **0.5 to 1.0**, and blocking-pair count drops from **2 to 0**, once `preference.enabled=true` — the exact
gap `test_preference_pairs_are_mostly_kept_together`/`test_preference_instability_blocking_pairs_detected`
(both still FAIL by design against the unmodified default pipeline, matching Part 4's identical
"opt-in-fixes-the-adversarial-case, default-pipeline-is-unaffected" precedent) document.

PRIVACY was a first-class requirement for this part specifically (unlike familiarity, which is symmetric,
mutual, and comparatively low-stakes): `PreferenceStudentStats`/`PreferenceSummary` (the models attached to
any public response) are AGGREGATE-ONLY — counts, booleans, scores — and never carry a target's identity or
`reason_category`; the richer, name-bearing dict `compute_preference_metrics()` returns internally is
consumed only by `generate_teams()`'s repair pass and by tests, never serialized onto a public response
(verified by `test_public_summary_model_never_carries_target_ids_or_reason_category`). `app/preferences.py`
deliberately has NO bulk "list everyone's preferences" endpoint (contrast `app/familiarity.py`'s
`list_familiarity()`, which does dump every edge) — `GET /preferences/{student_id}` returns only that
student's own OUTGOING entries, never who named them as a target (verified by
`test_post_preference_then_get_returns_only_own_outgoing_entries` and
`test_preferences_api_has_no_bulk_list_endpoint`). This app has no authentication layer anywhere (a
pre-existing, repo-wide limitation this part does not claim to solve — every endpoint, including this one and
`app/familiarity.py`'s, trusts caller-supplied IDs outright); what this part controls is the SHAPE of the API
and report surface, not who is allowed to call it.

Verified via `backend/tests/test_preferences.py` (67 new tests: schema validation; mutual/one-sided/cyclic
preferences; ties/indifference; missing preferences; avoidance including unilateral honoring and the
hard-avoidance repair gate; a stable assignment; an assignment with a blocking pair; all three stability
definitions being independently configurable; a preference-optimal-but-skill-invalid swap the repair pass
correctly refuses — directly verified against `validate_assignment`; a stable-but-utility-unfair assignment,
demonstrating stability and fairness are orthogonal properties; a `_fairness_improve` swap that measurably
REDUCES preference satisfaction for a mutual pair, demonstrating these two optional objectives can genuinely
conflict; characteristic preferences; hard-constraint preservation and determinism across every adversarial
fixture; the disabled-by-default no-op guarantee; and the privacy/API-shape checks above). All 67 tests in
that file pass; the full backend suite (389 passing outside the same two pre-existing, explicitly
out-of-scope-for-this-part failures already documented in the Part 4.5 update above — `test_stacked_ability_
gets_redistributed_across_teams`, an ability-redistribution gap Part 3 chose not to enable by default; and
`test_preference_instability_blocking_pairs_detected` itself, which exercises the LEGACY side-channel path
and the DEFAULT disabled pipeline by construction, exactly as intended) shows no regression from this change.

No test result is reported here that was not actually executed — every number above is read directly out of
`research_alignment_results.json`, regenerated by `backend/evaluation/run_report.py` in this session.
`run_report.py` was not modified to pass real `PreferenceEntry` data into `evaluate_teams()` for the seed
roster or fixtures (consistent with Part 4's identical choice for familiarity edges) — live demonstration of
this part's metrics happens through `backend/tests/test_preferences.py` and the `/teams/generate` /
`/preferences` endpoints instead.

**Update — Part 5.5 (verified, regenerated): resolved terminology, avoidance-enforcement, and
whole-partition-validation issues identified in a review of Part 5.**

*1. The two "same failures as before Part 5" were not both the same kind of failure — proven, not asserted.*
`test_stacked_ability_gets_redistributed_across_teams` (ability, `test_research_alignment.py`) is unrelated
to preferences, unchanged since Part 3. `test_preference_instability_blocking_pairs_detected` calls
`evaluate_teams(preferences=case.preferences)`, which feeds ONLY the legacy, Part-4-era, unchanged
`blocking_pairs_preference_based()` (a `Dict[str, List[str]]` side channel) — it never touches
`app.matching.compute_stability_metrics` (the real Part 5 engine, backed by `PreferenceEntry`). Confirmed by
direct code reading (both functions' exact source is quoted in this session's working notes) and by a NEW
test, `test_preference_instability_detected_by_part5_engine_without_repair_enabled`, added directly beside
the legacy one: it translates the SAME fixture's legacy preferences into `PreferenceEntry` rows via a new
backward-compatible adapter (`evaluation.fixtures.preferences_dict_to_entries`), runs the SAME default
(repair-disabled) pipeline, and asserts — without weakening the direction of the original assertion —
`compute_stability_metrics(...)["is_stable"] is False`. **Before:** only the legacy path was exercised
against this fixture, and it failed (by design, undisturbed). **After:** the legacy test still fails,
unchanged, exactly as documented; the new test — proving the REAL engine detects the identical instability
using the SAME disabled-by-default pipeline, i.e. that detection never required `preference.enabled=True` —
**passes**, and a follow-up assertion confirms the opt-in repair pass reduces the blocking-pair count once
enabled. No existing test was weakened or modified to make a failure disappear.

*2. Blocking-move / beneficial-feasible-move is now whole-partition validated, with an explicit team-size
configuration for genuinely unilateral moves.* `compute_beneficial_feasible_moves` (see #3) was ALREADY
swap-mediated and whole-partition-validated in Part 5 (every candidate is checked via ONE
`validate_assignment()` call against the full 2-team candidate) — this is now documented explicitly rather
than left implicit. Because this system's teams have EXACT target sizes everywhere, a genuinely UNILATERAL
move (no displaced counterpart — sending team shrinks, receiving team grows) is normally impossible; rather
than leave that undocumented, `app.models.TeamSizeConfig` (new: `min_team_size`, `max_team_size`,
`allow_unilateral_moves`, all opt-in, default = today's exact-size behavior) and
`app.matching.compute_unilateral_vacancy_moves` were added — a genuinely non-swap-mediated diagnostic that
returns `[]` under default exact sizes (never silently reinterpreting them as flexible) and only activates
given an EXPLICIT size range. `AssignmentBaseline`/`validate_assignment()` gained optional `min_sizes`/
`max_sizes` fields (default `None`, provably byte-identical behavior to every pre-5.5 caller — verified by
`test_assignment_baseline_size_range_mode_does_not_change_default_exact_behavior`). Six new tests cover
exactly the required scenarios: sending team too small, receiving team too large, sending-team skill
coverage broken, receiving-team floor policy broken, a valid move under an explicit vacancy, and complete
membership preservation (including a direct `validate_assignment()` check against a deliberately corrupted/
duplicated candidate).

*3. Terminology corrected: `blocking_move` → `beneficial_feasible_move`, no deprecated alias kept.* The Part
5 "blocking move" required only the MOVING student's utility to improve — never the receiving team's or its
members' — so pairing it with `blocking_pair`/`blocking_swap` (both genuinely two-sided) overclaimed a
classical stable-matching guarantee. Renamed throughout (`app.models.PreferenceSummary.n_blocking_moves` →
`n_beneficial_feasible_moves`; `PreferenceConfig.stability_definition`'s `"blocking_move"` literal →
`"beneficial_feasible_move"`; `compute_blocking_moves` → `compute_beneficial_feasible_moves`) — no alias
kept, same precedent as Part 4.5's `largest_preexisting_subgroup_size` rename (pre-1.0 prototype, no
external consumers, the old name was a genuine overclaim, not a style preference).
`compute_stability_metrics()`'s `note` field now states explicitly, whenever `beneficial_feasible_move` is
the authoritative definition, that `is_stable` under it is NOT a classical stable-matching guarantee.

*4. Blocking-pair shortcut's edge-case behavior formally documented AND one correction made.* Verified
against the actual implementation (not merely asserted): a student with no ranked current teammates → the
check passes trivially; tied ranks → never counted as a strict preference; a NAMED-but-unranked current
teammate gets a real, finite tier (a stronger "blocker" than an unmentioned one); an UNNAMED teammate is
substituted with a sentinel tier ONE WORSE than any real tier — **correcting an inaccurate prior docstring
claim that this sentinel was merely "tied" with the searching student's own worst tier**; a student with no
preference data (or who never specifically named the target) can never be one side of a blocking pair, so
missing data can only SUPPRESS a blocking pair, never manufacture one (already true in Part 5, now proven by
a dedicated test); characteristic preferences are structurally invisible to this diagnostic (no
`target_student_id` to match against); and a simultaneous, contradictory prefer+avoid entry for the same
target is NOT reconciled by this diagnostic (avoidance is never consulted here) — `POST /preferences` now
rejects that contradiction at write time going forward. The whole mechanism is now explicitly labeled a
"deliberately conservative envy diagnostic" per this review's own suggested framing.

*5. Hard avoidance is now a genuine, always-enforced hard constraint, decoupled from `preference.enabled`.*
Previously, `PreferenceConfig.hard_avoidance` only ever took effect INSIDE `_preference_repair_improve`
(itself gated by `enabled=True`) — meaning avoidance was completely unenforced whenever general preference
optimization was off, even with `hard_avoidance=True` set. A new function, `_hard_avoidance_repair`, now
runs in `generate_teams()` whenever `preference.hard_avoidance=True`, INDEPENDENT of `preference.enabled` —
exactly like skill-coverage/floor-policy are always enforced independent of ability/fairness/familiarity
optimization. It ACTIVELY REDUCES pre-existing avoidance violations when a hard-constraint-safe fix exists
(not merely refuses to worsen them — verified by a dedicated test), leaves a genuinely unavoidable violation
in place rather than breaking a hard constraint to remove it (verified — and such violations remain fully
visible in `compute_preference_metrics()`'s report, never hidden), and never treats an omitted preference as
an avoidance request (verified). The three avoidance tiers — soft (reported only), high-priority (weighted
more heavily, still not structural by itself), hard (a true constraint via `hard_avoidance=True`) — are now
explicitly distinguished in both code documentation and tests.

*6. Privacy boundaries verified by test, not merely asserted in prose.* New tests confirm: a student cannot
retrieve another's outgoing preferences (`GET /preferences/{id}` returns only that id's own rows); there is
no route, anywhere, that accepts a "who targeted me" reverse lookup (the DELETE route's target segment
addresses the CALLER's own rows only); `app/explainability.py`'s actual source contains no reference to
preference data, `PreferenceEntry`, or `reason_category` at all (a structural check, not a docstring claim);
`PreferenceSummary`'s serialized JSON contains no `target_student_id` or `reason_category` anywhere; and
malformed/rejected preference submissions never echo `reason_category` or leak whether some other student
has a matching row. A final test explicitly documents — rather than silently assumes — that these are DATA-
SHAPE protections only: this app has no authentication/authorization layer anywhere (verified by confirming
no endpoint declares an auth-dependency parameter), a pre-existing, repo-wide limitation this part does not
claim to have solved.

*7. Full test accounting.* `backend/tests/test_preferences.py`: 94 tests (67 from Part 5 + 27 new — items
2/4/5/6 above, plus the API-level contradiction guard). `backend/tests/test_research_alignment.py`: 29 tests
(28 from Part 5 + 1 new confirmation test). Full backend suite: **417 passed**, same 2 pre-existing failures
as Part 5 (`test_stacked_ability_gets_redistributed_across_teams`, unrelated to this part; and
`test_preference_instability_blocking_pairs_detected` itself, now PROVEN — not merely asserted — to be
exercising only the legacy side-channel path, with the real engine's detection independently confirmed
working via the new test beside it). No test was weakened, and no existing failure was made to disappear by
editing an assertion.

No test result is reported here that was not actually executed — every number above is read directly out of
`research_alignment_results.json`, regenerated by `backend/evaluation/run_report.py` (the real evaluation
command) in this session.

**Update — Part 5.6 (verified, regenerated): closed the evaluate_teams() integration gap — one authoritative
preference/stability engine.**

Part 5.5 characterized `test_preference_instability_blocking_pairs_detected`'s continued failure as a
"legacy code-path artifact," proven by an independent confirmation test — but left the underlying gap
itself unfixed: `evaluate_teams(preferences=...)` still called `blocking_pairs_preference_based()`, a
SECOND, fully independent reimplementation of the same rank-vs-current-teammates comparison
`app.matching.compute_blocking_pairs` (the real Part 5 engine) already implements. Two algorithms computing
the same concept, however well they happened to agree on one fixture, is exactly the kind of drift risk this
part's own goal statement ("there must be one authoritative preference and stability implementation") warns
against.

**The fix:** `evaluation/metrics.py` gained `legacy_preferences_to_entries()` — THE authoritative boundary
adapter converting the legacy `Dict[student_id, List[ranked_target_id]]` shape into real
`app.models.PreferenceEntry` rows, fully documented (list order → strict, contiguous rank order; directed
"prefer" only; no avoidance/priority/characteristic/reason inferred; empty list = no submitted preferences,
`None` not `0.0`; unknown targets pass through unvalidated, never crashing; duplicate targets keep only the
first/best-ranked occurrence; self-targets are silently ignored, not raised). `blocking_pairs_preference_based()`
is now a THIN WRAPPER: it converts via this adapter and delegates entirely to
`compute_stability_metrics(definition="blocking_pair")` — there is exactly one blocking-pair algorithm left
anywhere in this codebase. `preference_satisfaction()` (the OTHER legacy function, computing a "top-half
preference met" statistic with no equivalent in the tier-scored engine) was deliberately left untouched and
is documented as such — it is a genuinely different metric, not a second blocking-pair implementation, and
touching it risked silently changing `test_preference_pairs_are_mostly_kept_together`'s currently-passing
result, which was out of this part's explicit scope. Part 5.5's fixture-only adapter
(`evaluation.fixtures.preferences_dict_to_entries`) was REMOVED entirely (not duplicated) — its one caller
(`test_research_alignment.py`) now imports `legacy_preferences_to_entries` directly — since "essential
conversion logic" living only in test-support code was explicitly identified as the root architectural
problem, not merely a style preference.

**The previously-failing test:** manually verifying the actual team split `_run_behavioral_pipeline`
produces for this fixture — `[[pr0,pr2,pr3,pr5],[pr1,pr4,pr6,pr7]]` — confirmed the pr0/pr1 and pr4/pr5
mutual-preference pairs are GENUINELY split across teams (while pr2/pr3 and pr6/pr7 land together); both the
old independent algorithm and the new delegating wrapper agree exactly: 2 real blocking pairs, not an
artifact of a buggy comparison. This means the test's ORIGINAL assertion (`assert stability["is_stable"]`,
i.e. "expect zero blocking pairs") was checking an assertion that a preference-BLIND, repair-disabled
default pipeline can never satisfy — greedy+2-opt+maximin has no preference input at all, by design, so it
cannot avoid splitting a pair it cannot see. The corrected assertion checks what the EVALUATION layer is
actually responsible for: that the instability is correctly DETECTED (`is_stable is False`,
`n_blocking_pairs == 2`, the exact pr0/pr1 and pr4/pr5 pairs named) — a strictly MORE demanding check than
the "at least one" language used elsewhere, not a weaker one, and the fixture and pipeline are used
completely unmodified, with `preference.enabled` never touched (this stays evaluation-only). A second test
confirms the identical finding reachable via the authoritative engine called directly, and that the opt-in
repair pass (only reachable via `preference.enabled=True`, never invoked by evaluation) resolves what
detection found.

**Confirmed unconditional evaluation semantics (no code change needed — verified by test):**
`blocking_pairs_preference_based()`/`preference_stability()` never call `_preference_repair_improve` or
`_hard_avoidance_repair` (confirmed by inspecting their actual source, not merely by docstring claim) — so
preference/stability metrics are computed regardless of `preference.enabled`, which controls only whether an
assignment is MODIFIED, exactly as this part's goal states. Hard-avoidance enforcement
(`_hard_avoidance_repair`, Part 5.5) is unaffected by any change in this part.

**Test accounting:** `backend/tests/test_preferences.py`: 105 tests (94 from Part 5.5 + 11 new — legacy/native
equivalence, missing-data, ties-unavailable-to-legacy, empty list, duplicates, unknown targets, self-targets,
determinism, no-silent-repair, exact-agreement, no-leak-when-unsupported).
`backend/tests/test_research_alignment.py`: 29 tests (one test corrected as described above, one renamed for
clarity, net count unchanged). Full backend suite: **429 passed, 1 failed** —
`test_stacked_ability_gets_redistributed_across_teams` is now the ONLY remaining known failure, unrelated to
preferences and untouched by this part (the ability default itself was not altered, per this part's own
instruction).

No test result is reported here that was not actually executed — every number above is read directly out of
`research_alignment_results.json`, regenerated by `backend/evaluation/run_report.py` (the real evaluation
command) in this session.

**Update — Part 6 (verified, regenerated): integrated all seven research criteria into one coherent,
configurable optimization framework — criterion registry, named profiles, a normalized cross-criterion
objective, and an explicit cross-pass acceptance policy — without presenting any experimental profile as
universally optimal.**

**1. Pipeline audit (before any Part 6 edit).** Traced the exact production call path from
`TeamGenerationRequest` through `generate_teams()`'s "behavioral" branch: `_greedy_build` → `_two_opt_improve`
→ `_maximin_improve` → (if enabled) `_ability_balance_improve` → `_fairness_improve` →
`_familiarity_balance_improve` → (if `hard_avoidance`) `_hard_avoidance_repair` → (if enabled)
`_preference_repair_improve`. Every optional pass already shared an IDENTICAL per-swap lexicographic
acceptance rule (hard constraints → total-score non-regression → floor-min non-regression → strict
improvement in that pass's own target → deterministic tie-break) — this observation directly drove the
Section 6 architecture decision below.

**2. Criterion registry.** `app.matching.CRITERION_REGISTRY` (15 rows, `criterion_registry_as_dicts()` for
API/docs surfacing) classifies every criterion from the spec's example table — hard constraint, soft
objective, evaluation-only, or optional post-processing — with a documented condition and rationale.
Matches the spec's initial classification exactly (verified by
`test_criterion_registry_covers_every_hard_constraint_from_the_spec_table` and its soft-objective sibling);
ability balance and familiarity band/anti-cluster are explicitly kept as `soft_objective`, never promoted
to `hard_constraint`, per each Config class's own "this is a hypothesis, not an established finding"
docstring (`test_criterion_registry_does_not_promote_research_hypotheses_to_hard_constraints`).

**3. Named configuration profiles.** `app.matching.OPTIMIZATION_PROFILES` + `resolve_profile()`: `legacy`
(not experimental — reproduces Parts 1–5.6 exactly; the default an existing client that omits `profile`
receives), `research_balanced` / `fairness_priority` / `skill_priority` / `preference_priority` (all
EXPERIMENTAL, all preserving every hard constraint). Precedence is always system defaults → named profile
→ explicit request overrides, implemented via Pydantic's own `model_fields_set` at every nesting level
`TeamGenerationRequest` has — NOT string/dict guessing — so a partial nested override (e.g. only
`ability.ability_level_thresholds`) never clobbers a profile's OTHER settings for that same sub-config
(`test_explicit_nested_override_preserves_sibling_profile_settings`). Resolution is a pure function (a
fresh `TeamGenerationRequest` every call, verified never to mutate `OPTIMIZATION_PROFILES` itself) and
fully deterministic. An unknown profile name is rejected with HTTP 422, never silently substituted.
`ability_priority`/`familiarity_priority` were considered and explicitly NOT added — this session's own
empirical comparison found no scenario where a single-criterion profile would behave differently from
flipping one flag on `legacy` directly, so a dedicated profile did not clear the "genuinely useful and
tested" bar (see `docs/optimization_profiles.md`'s note on this).

**4. Normalized objective (Section 4).** `compute_normalized_objective()` reports 9 components (8 required
+ optional `stability_penalty`, weight 0.0 by default) each with a documented `[0,1]` range, fixed
higher-is-better directionality, configurable weight, `evidence_coverage`, raw/normalized/weighted values —
see `OBJECTIVE_COMPONENT_SPECS`. THIS IS A REPORTING CONSTRUCT — it does not itself choose which
assignment is returned (the existing per-pass mechanism still does, deliberately, to avoid re-deriving
already-tested optimizer logic — see Section 6 below). Missing data (e.g. no ability/familiarity/preference
data at all) is EXCLUDED from the weighted total and the remaining components' weights are renormalized to
sum to 1.0, never scored as a fabricated zero (`test_normalized_objective_handles_sparse_data_by_
renormalizing_not_zeroing`). Explicitly inspected the 5 overlap pairs the spec names (behavioral
compatibility vs. individual utility's own fit sub-component; skill coverage vs. individual skill fit;
familiarity vs. named-teammate preference — structurally independent data sources, FamiliarityDB vs.
PreferenceDB; overall balance vs. ability balance — related but distinct spreads; avoidance violations vs.
preference satisfaction) — documented in `compute_normalized_objective`'s own docstring and
`docs/optimization_profiles.md`, with `stability_penalty` defaulting to weight 0.0 specifically so it never
silently double-counts against `preference_satisfaction` unless a profile deliberately enables it (none
currently does).

**5. Hard-constraint validation.** `validate_assignment()` remains the SOLE authoritative gate, unchanged in
its core logic since Part 5.5's size-range extension. Added roster validation for preference targets in
`generate_teams()`: an unknown `target_student_id` (not present in the current roster) is counted
(`n_unknown_preference_targets`, an aggregate count only — never a target identity, per this system's
privacy design) and passed through unchanged — `compute_preference_metrics`/`compute_stability_metrics`
already structurally treat an unmatched target as "never satisfied," so a stale ID can never silently
inflate a score (verified by `test_generate_teams_reports_unknown_preference_targets_as_a_count_only`).
Verified, parametrized over EVERY profile × EVERY adversarial fixture: membership completeness, team-size
spread, skill-coverage non-regression, and behavioral-floor-policy non-regression all hold regardless of
which profile is selected (`test_every_profile_preserves_membership_and_team_sizing_on_every_fixture` and
its 3 siblings) — profiles change soft objectives, never bypass hard constraints.

**6. Cross-pass acceptance policy — architecture decision.** Chose **Option B (lexicographic multi-
objective ordering), extended one level**, over a from-scratch unified weighted objective (Option A).
Rationale, stated plainly: every one of the five existing optional passes ALREADY implements the identical
per-swap lexicographic rule described in item 1 above — rewriting all five into one monolithic weighted-sum
optimizer would mean re-deriving and re-proving correctness for logic already extensively tested per-pass,
for no architectural gain, since a single scalar objective cannot express "never regress floor-minimum" as
cleanly as an explicit priority order can. `_run_optional_passes()` (replacing `generate_teams()`'s inline
pass sequence) adds an ORCHESTRATION-level check on top: after every pass, three PROTECTED metrics
(total_score, floor_min, skill_coverage_min — plus the resolved profile's own `primary_metric`, e.g.
`fairness_priority`'s `min_utility`) are compared against their value immediately BEFORE that pass ran
(not swap-by-swap); if any regressed beyond `PROTECTED_METRIC_TOLERANCE`, the entire pass is reverted. This
is a genuinely NEW protection, not pure redundancy with each pass's own internal check: a pass's own
per-swap tolerance is checked against the immediately-preceding swap's state, so cumulative drift across
many individually-accepted swaps within one pass could in principle exceed a single swap's tolerance
relative to the pass's own starting point — the orchestration-level, whole-pass comparison closes exactly
this gap. Every accepted pass's before/after protected-metric values are recorded in
`"optimization_trace"` on the API response (the explicit debug-evaluation record the spec asks for), and
`test_later_passes_do_not_undo_earlier_accepted_improvements_outside_tolerance` verifies this holds across
every adversarial fixture under `research_balanced` (every pass enabled).

**7. The remaining ability test, addressed correctly.** `test_stacked_ability_gets_redistributed_across_
teams` is LEFT UNCHANGED (still exercises `_run_behavioral_pipeline`, i.e. the `legacy` profile, which by
design does not enable ability optimization) — its docstring now explains why this is expected, not a
regression. A NEW test,
`test_stacked_ability_redistributed_under_research_balanced_profile`, runs the IDENTICAL fixture and the
IDENTICAL assertion through `research_balanced` (via a new `_run_profile_pipeline` test helper that mirrors
`generate_teams()`'s actual profile-resolution + optional-pass-orchestration path) and passes: ability gap
0.5 → 0.0. No assertion was deleted, weakened, skipped, or marked expected-failure — the original research
requirement is now demonstrated satisfiable, addressed through the profile system this part actually asked
for, not through silently changing the legacy default.

**8. Test accounting.** New file `backend/tests/test_optimization_profiles.py` (122 tests): criterion
registry, profile resolution (default/every-named-profile/explicit-override-precedence/invalid-profile/
no-shared-state-mutation/determinism), hard constraints across every profile × every fixture, objective
interactions (ability doesn't destroy skill coverage; familiarity doesn't act on insufficient data;
anti-cluster doesn't isolate a student when there's no data to isolate from; preference repair doesn't
regress fairness; later passes don't undo earlier accepted improvements), data availability (no ability/
familiarity/preference data, sparse data, mixed direct+proxy ability data, unknown preference targets),
determinism (identical input+profile ⇒ identical assignment, every profile), and performance (runtime
measured at n=20/60 synthetic students, iteration-limit termination confirmed non-hanging). Plus 1 new test
in `test_research_alignment.py` (item 7 above). **Full backend suite: 553 passed, 1 failed** —
`test_stacked_ability_gets_redistributed_across_teams`, now the ONLY failure, exhaustively explained in its
own docstring and directly paired with a passing profile-aware equivalent — zero UNEXPLAINED failures, per
this part's own bar.

**9. Empirical profile comparison.** `backend/evaluation/run_report.py` gained `run_profile_comparison()`
— all 5 profiles run against the real 20-student seed roster, all 12 adversarial fixtures, and synthetic
rosters of 20/40/80 students, reporting the exact metric set the spec requires (hard-constraint validation,
normalized objective, skill coverage, floor violations, ability gap, familiarity band/evidence-coverage/
largest-cluster, min/mean utility, utility variance, worst-case regret, mean preference satisfaction,
avoidance violations, blocking pairs/swaps, runtime) — persisted under `research_alignment_results.json →
profile_comparison`. Findings, WITH the tradeoff named, not asserted as "better" without qualification: on
the real seed roster, `skill_priority` was the only profile that changed the outcome — ability gap got
MEASURABLY WORSE (0.183 → 0.370) while minimum utility and worst-case regret got measurably better
(0.476 → 0.486; 0.200 → 0.175); every other profile matched `legacy` exactly for this specific roster
(the optional passes ran — confirmed via nonzero `optimization_trace`/runtime — but found no
swap-reachable improvement for THIS roster). On `stacked_team`, `research_balanced` alone fixed the
ability gap (0.5 → 0.0). On `preference_instability`, `research_balanced` AND `preference_priority` both
fully resolved it (mean satisfaction 0.875 → 1.0; blocking pairs 2 → 0); `fairness_priority`/
`skill_priority` did not, correctly, since neither enables preference optimization. On `friend_clique`,
NO profile changed the largest-cluster fraction (stays exactly 0.5 under every profile including
`research_balanced`) — explained, not a bug: this fixture's clique fraction sits exactly AT
`max_clique_fraction`'s default threshold, so the violation term is legitimately zero. Full table:
`docs/optimization_profiles.md`.

**10. Performance impact — a genuine, measured limitation.** On synthetic rosters, `research_balanced`
took 191ms/2,972ms/**26,077ms** at n=20/40/80 students respectively; `fairness_priority` took
441ms/1,107ms/9,294ms; `legacy`/`skill_priority` (which never run the O(T²S²) swap-search passes) stayed
under 800ms even at n=80. **Any real deployment considering `research_balanced`, `fairness_priority`, or
`preference_priority` for a roster larger than roughly 50–80 students should budget for multi-second-to-
tens-of-seconds runtime, or treat this as a limitation to address before a larger real-world study.**

**Remaining implementation limitations (Part 6):** (a) the orchestration-level protected-metric check adds
real value but is still a HEURISTIC backstop, not a formal proof that the combined system reaches any
global optimum; (b) `research_balanced`'s "enable everything" does not mean "changes everything" — for a
roster with no swap-reachable improvement in a given pass's target, that pass is legitimately a no-op, which
can read as "nothing happened" without the `optimization_trace` context; (c) the O(T²S²)-and-worse runtime
of the swap-search passes is the dominant scaling concern for any deployment beyond a single classroom-size
cohort; (d) `research_balanced`/`fairness_priority`/`skill_priority`/`preference_priority` remain
UNVALIDATED against real student outcome data — every claim in this update is about INTERNAL CONSISTENCY
(hard constraints preserved, metrics computed correctly, tradeoffs measured) never about real-world
educational efficacy.

**Decisions requiring professor/research-team approval before any real deployment:** whether
`research_balanced`'s "enable every criterion" default is itself a defensible research design, or whether a
narrower, single-criterion profile (or `legacy`) should be the actual recommendation absent outcome data;
whether the specific weight choices in `DEFAULT_OBJECTIVE_WEIGHTS` and each profile's overrides deserve
independent validation or IRB-level sign-off before being shown to real instructors as anything other than
"experimental"; whether `skill_priority`'s measured ability-gap tradeoff is acceptable for a real course.

**Claims that still require real student outcome data (not addressed by this or any prior part):** whether
mixed-ability teams outperform stacked ones FOR THIS POPULATION (AbilityConfig's own hypothesis, still
untested); whether the familiarity target band `[0.15, 0.6]` and `max_clique_fraction=0.5` are the right
thresholds for this population; whether the accountability/cooperativeness floor's Bell (2007)/Peeters et
al. (2006) citations transfer to this specific student population and course structure; whether tier-scored
preference satisfaction, as implemented, actually correlates with student-reported team-experience quality.

No test result is reported here that was not actually executed — every number above is read directly out
of `research_alignment_results.json`, regenerated by `backend/evaluation/run_report.py` (the real
evaluation command) in this session.

---

## Phase 6 — Recommendations

Full ranked list with research-criterion mapping, code/test evidence, practical consequence, recommended
change, difficulty, and constraint/metric/config classification: see
**[`recommended_changes.md`](recommended_changes.md)**. Summary counts: **3 Critical, 4 High, 6 Medium,
5 Low**.

Distinguishing claim types, per the audit's own constraint against overclaiming:

- **Directly supported by cited research already in the repo:** the accountability/cooperativeness floor
  concept and mean-vs-minimum distinction (Bell 2007, Peeters et al. 2006, cited in `rubric_v1.yaml`), and
  the decision to zero out `conflict_style`'s weight until task/relationship conflict can be distinguished
  (De Dreu & Weingart 2003; Jehn 1995, same file).
- **Design choices inferred from the research brief, not independently cited in-repo:** mixed ability,
  familiarity/anti-clique, fair individual benefit, and preference stability as criteria at all — these
  come from this audit's brief, not from a literature-review file in the repo (which is empty).
- **Assumptions requiring validation:** `experience_level` as an ability proxy; team-score-as-individual-
  utility as a fairness proxy; the score-based blocking-move detector as a stand-in for true
  preference-based stability; every numeric threshold in `EvalConfig` (e.g. `clique_fraction_threshold:
  0.75`, `min_familiarity_density: 0.15`) — chosen for this audit's tests to be reasonable, not derived
  from any validated source.
- **Criteria that cannot be validated without real student outcome data:** `match_confidence`'s formula
  (never calibrated against actual team performance); whether the rubric's proposed BARS-style behavioral
  anchors for accountability/cooperativeness (rubric_v1.yaml, explicitly marked `NEEDS-SIGN-OFF` in
  `CHANGES.md`) actually predict outcomes in this population; whether the specific numeric weights (0.4 /
  0.3 / 0.2 / 0.1, or the 0.35/0.35 split inside behavioral_compat) are anywhere near optimal — the
  architecture doc's own "Phase 4: Longitudinal Feedback Loop" section (`docs/architecture.md:181-183`)
  names this exact gap and proposes closing it with real instructor-rated outcomes, which do not exist yet.

---

## Overall research-alignment rating

**Partial alignment, concentrated in one area.** The system does one thing well and by-the-book: it
correctly implements the mean-vs-floor distinction for two validated Big-Five-adjacent traits
(conscientiousness/accountability, agreeableness/cooperativeness), citing the right literature for it. But
that implementation has an internal contradiction (Critical #1) that undermines its own guarantee even on
ordinary data, and, AT AUDIT TIME, **3 of the 7 research criteria this audit was asked to test — mixed
ability, familiarity/anti-clique, and preference/stability — have no representation anywhere in the schema
or algorithm**, not even a stub. A 4th (fair distribution of individual benefit) was, at audit time,
structurally unrepresentable because the scoring model had no concept of an individual student's outcome,
only a team's. **Update — Part 3:** mixed ability is no longer in that "3 of 7" group — `ability_score`
now exists (`app.models.CompetenceSignature`) and between-team ability balance is reported by default and
optionally optimized for (`_ability_balance_improve`, off by default). **Update — Part 4:** familiarity/
anti-clique is also no longer in that group — a real `familiarity` edge table exists
(`app.models.FamiliarityEdge`) and graph-based density/isolation/clique metrics are reported by default and
optionally optimized for (`_familiarity_balance_improve`, off by default). Preference/stability is the ONLY
one of the original 3 still entirely absent, out of scope through Part 4 by explicit instruction each time.
**Update — Part 2:** this 4th gap is no longer structurally unrepresentable — `app.matching.
compute_assignment_utilities()` now gives every student their own utility value, reported by default and
optionally optimized for (`fairness.enabled`).

### What the system already does well

1. **Deterministic, auditable scoring.** `dimension_scorer.py` has zero randomness and zero LLM dependency
   in the MCQ path — same input always produces the same profile, verified by both the existing test suite
   and this audit's harness.
2. **A genuinely layered architecture** (L0 rubric → L1 bank → L3 signals → L4 scorer → L5 matching) that
   makes *most* dimension changes a one-file edit, exactly as `docs/pipeline.md` claims.
3. **Correct citation and application of the mean-vs-minimum distinction** for the two traits it does
   model — the one place in the codebase this audit found a research finding applied thoughtfully rather
   than just asserted.
4. **Honest self-documentation of open questions.** Four `TODO(team-decision)` items and three
   `NEEDS-SIGN-OFF` items are already tracked in `CHANGES.md` — the team is aware several of these
   decisions are unresolved, which is why this audit could locate them precisely.

### The five most important gaps

1. Floor-guarantee contradiction between `rubric_v1.yaml`'s stated zero-tolerance policy and
   `matching.py`'s one-per-team implementation (`recommended_changes.md` #1).
2. No representation of ability anywhere in the objective (`recommended_changes.md` #2). **Resolved (Part
   3):** `ability_score` exists and `_ability_balance_improve` is available, off by default.
3. No individual-level utility function — fairness can only be measured between teams, never within one
   or for one student (`recommended_changes.md` #3).
4. No familiarity/social-graph data, so clique formation cannot be detected or prevented
   (`recommended_changes.md` #4). **Resolved (Part 4):** `familiarity` edge table + graph-based metrics
   exist; `_familiarity_balance_improve` is available, off by default.
5. No preference data, so stability/blocking-pair analysis is impossible in production
   (`recommended_changes.md` #5).

### Next three implementation changes to prioritize

1. **Resolve and fix the floor-guarantee contradiction (#1)** — smallest code change of the top gaps, and
   it's currently shipping a claim (`rubric_v1.yaml`'s "no team may include a member at the low level")
   that the system's own post-hoc check disproves on real data today.
2. **Fix the leftover-distribution team-size bug (`recommended_changes.md` #6)** — a plain correctness bug
   independent of any research question; an 11-student roster at `team_size=4` should never produce a
   team of 7.
3. **Surface worst-case individual regret as a diagnostic field (#3)** — the computation already exists in
   this audit's harness; wiring a capped version into the `POST /teams/generate` response gives
   instructors visibility into the "fair distribution of benefit" criterion without requiring new data
   collection or changing what gets optimized.

### Questions that must be answered by the research team

1. Is "no team may include a member at the lowest level" (zero tolerance, as currently written in
   `rubric_v1.yaml`) or "at most one per team" (as currently implemented in `matching.py`) the intended
   floor policy? These are not equivalent and the code currently ships both.
2. What should stand in for "ability" — a validated instrument (prior GPA, placement score), an
   instructor's holistic rating, or something else? The audit deliberately did not invent one. **Update
   (Part 3):** the schema now supports whichever the research team picks — `CompetenceSignature.
   ability_score` is source-tagged via `ability_source` (a free-form string, not a hardcoded "gpa"), so
   this question is now about WHICH source to populate it from, not whether the field can exist. Still
   fully open: is "balanced ability per team" (this repo's default hypothesis when `ability.enabled=true`)
   actually the right target, or would instructors prefer stacking strong students together for some
   assignments? No outcome data in this repo answers that either way.
3. Is preference data worth collecting given its interaction with anti-clique goals (satisfying
   preferences and breaking up cliques can directly conflict — see `recommended_changes.md` #4 and #5)?
   If so, how should the two be weighted against each other? **Update (Part 4):** the anti-clique half of
   this tension is now concretely implementable — `_familiarity_balance_improve` exists and is tested — so
   this question is now specifically "should preference satisfaction be allowed to override anti-clique
   pressure, and by how much," not a purely hypothetical tradeoff. Still unanswered: `familiar_pair_
   threshold=0.3`, `clique_edge_threshold=0.5`, the `[0.15, 0.6]` density band, and `max_clique_fraction=
   0.5` are ALL documented hypotheses, not validated cutoffs — see `app.models.FamiliarityConfig`'s
   docstring.
4. Should `leadership_style` reward similarity (current `soft_compatibility` rule) or complementary
   coverage (the rule's own code comments argue for this) — already an open `CHANGES.md` TODO, now with
   adversarial-test infrastructure available to check either choice empirically.
5. What confidence, if any, should be placed in the proposed BARS-style behavioral anchors for
   accountability/cooperativeness before they're used in a real study, given `CHANGES.md` already flags
   them as `NEEDS-SIGN-OFF` from someone with CATME expertise?

### Criteria that cannot be validated without real-world student outcome data

- Whether the accountability/cooperativeness floor threshold (currently: the bottom of a 3-level ordinal
  scale) is the right cutoff, or whether "developing" should also count as sub-floor for this population.
- Whether `match_confidence`'s formula bears any relationship to actual team success — it has never been
  compared against outcomes (`docs/architecture.md`'s own "Phase 4: Longitudinal Feedback Loop" section
  names this as future work, not yet done).
- Whether the specific objective weights (0.4/0.3/0.2/0.1, and the nested 0.35/0.35/0.10/0.025×3 split)
  produce teams that actually perform better than alternatives — no instructor-rated or outcome-based
  comparison data exists in the repo (the `OutcomeMeasureDB`, `PeerEvaluationDB`, `ContributionLogDB`
  tables in `database.py:229-258` are explicitly stub tables, "create now, populate post-collaboration").
- Whether any proposed familiarity/clique or preference thresholds (this audit's `EvalConfig` defaults:
  `clique_fraction_threshold=0.75`, `min_familiarity_density=0.15`) are meaningful cutoffs for this
  specific student population, once that data exists to collect at all.

---

## Part 6.5 update — validating the Part 6 profile system before a real study

Part 6 introduced named optimization profiles and orchestration-level regression protection. Before
preparing a real student study, Part 6.5's brief was to validate rather than extend: confirm profiles are
meaningfully distinct, remove the one remaining permanently-failing test, and address the 26-second
`research_balanced` runtime at 80 students. Every claim below is backed by a script under
`backend/evaluation/` and/or a test under `backend/tests/test_part6_5_*.py` — none is asserted without a
pointer to where it was checked, per this audit's own standing rule.

### 1. Disposition of the final failing test

`test_stacked_ability_gets_redistributed_across_teams` asserted the research requirement (ability spread
≤ 0.75 after redistribution) against the `legacy` pipeline, and was left deliberately, permanently failing
to "document a gap" — a pattern Part 6.5 correctly identified as no longer acceptable (a test that can
never pass is indistinguishable, from CI output alone, from a genuine regression).

**Fix:** renamed to `test_legacy_profile_does_not_redistribute_stacked_ability` and its assertion flipped
to check LEGACY's own documented contract instead — `ability.enabled=False` by construction for that
profile, so the fixture's roster-order ability stacking is left unaddressed BY DESIGN, and the test now
asserts that (`spread > 0.75`, passing). This is a characterization test, not a weakening: if it ever starts
FAILING because the spread shrank, that means ability balancing got silently enabled for legacy — exactly
what this test exists to catch. The ORIGINAL research requirement is fully preserved, unweakened, and
PASSING (not `xfail`) in the pre-existing sibling test
`test_stacked_ability_redistributed_under_research_balanced_profile`, run against `research_balanced`.
**Full backend suite: 0 failures** (567 passed as of this update).

### 2. Why `research_balanced` matches `legacy` on the real 20-student seed roster

Answered with an exhaustive read-only shadow trace (`backend/evaluation/trace_seed_roster.py`,
`evaluation_output/seed_roster_trace.json`) that replays every one of the 160 candidate swaps
(`C(5,2)=10` team pairs × `4×4=16` member pairs) each optional pass considers on the real roster, tallying
rejections by reason:

| Pass | Candidates | Accepted | Hard-constraint rej. (would've improved target) | Primary-objective rej. (would've improved target) | Target metric before → after |
|---|---|---|---|---|---|
| ability | 160 | 0 | 139 (**34**) | 21 (**3**) | 0.1825 → 0.1825 |
| fairness | 160 | 0 | 139 (**16**) | 21 (**5**) | 0.4762 → 0.4762 |
| familiarity | 160 | 0 | 139 (0) | 21 (0) | 0.0000 → 0.0000 |
| preference | 160 | 0 | 139 (0) | 21 (0) | 0.0000 → 0.0000 |

A follow-up breakdown of the hard-constraint rejections' `validate_assignment` reasons: **139 of 139 hit
`skill_coverage_regressed`** (a smaller 6 also independently hit `floor_violations_increased`) — i.e. 2-opt
already fully saturated this tight 20-student/5-team roster's feasible skill-coverage distribution, so
almost every subsequent single swap regresses it.

**Determination (matching this brief's own listed categories):**
- For **ability and fairness**: *"Improving swaps exist but are blocked by lexicographic protections"* —
  concretely, 37 (ability) and 21 (fairness) of 160 candidates would have improved that pass's own target
  metric but are blocked, overwhelmingly by the hard skill-coverage-non-regression constraint the earlier
  greedy/2-opt stages already saturated for this roster, with a smaller number blocked by the primary
  total-score-non-regression rule.
- For **familiarity and preference**: *"Necessary data is absent or insufficient"* — the seed roster has
  **zero recorded `FamiliarityDB` edges and zero `PreferenceDB` submissions**, so both passes' target
  metrics are already at their zero/no-violation floor; no protection is even relevant because there is
  nothing to act on regardless.
- **Ruled out**: "the profile is enabled incorrectly" — a direct cross-check confirms the REAL production
  `app.matching._run_optional_passes` produces the byte-identical final assignment to this independent
  shadow trace (`production_orchestration_cross_check.matches_shadow_trace_final_assignment: true`).
  "Metrics are insensitive to the available data" — ruled out; the trace explicitly demonstrates the
  ability/fairness metrics ARE sensitive (they correctly detect 37/21 improving candidates before those are
  blocked). This is a more precise finding than `docs/optimization_profiles.md`'s earlier, vaguer framing
  ("no swap-reachable improvement existed") — swap-reachable improvements DO exist for ability/fairness;
  they are correctly and deliberately blocked.

### 3. Profile-distinctness results

`backend/evaluation/profile_distinctness.py` → `evaluation_output/profile_distinctness.json`; full writeup
in [`docs/profile_distinctness.md`](../docs/profile_distinctness.md). Headline finding: **9 of 13 datasets
tested (the seed roster + all 12 adversarial fixtures) collapse to an IDENTICAL partition across every one
of the 5 profiles** (ARI=1.0, pairwise co-membership distance=0.0, team-label-invariant). Only the seed
roster and 3 fixtures (`stacked_team`, `preference_instability`, `2opt_harms_fairness`) show any profile
producing a different assignment — and even then, typically only 1 of the 4 non-legacy profiles differs
(e.g. only `skill_priority` differs from the other 4 on the seed roster; only `research_balanced` differs on
`stacked_team`). This is documented explicitly as a **study-design limitation**: a real comparison of
`legacy` against an experimental profile needs either (a) a roster with a genuine, swap-reachable
opportunity for that profile's specific criterion, or (b) actual familiarity/preference data collected in
advance — neither of which the seed roster alone provides for 2 of the 4 research criteria the profiles
can act on.

### 4. Orchestration-audit findings

`backend/tests/test_part6_5_orchestration_audit.py` (10 tests, all passing). Findings:

- **NaN/infinity COULD bypass the regression check — fixed.** `_maybe_run`'s check
  (`after[k] < before[k] - tol`) is silently `False` whenever either side is `NaN` (Python NaN comparisons
  are always `False`), which would have let a metric computation bug produce `NaN` and have it accepted as
  "not regressed." Fixed with an explicit `math.isfinite()` guard that treats any non-finite value on either
  side as an unconditional regression (`app/matching.py`, `_run_optional_passes`); the pass's `trace` entry
  now also reports which metric key(s) were non-finite. Verified both for `NaN` and for `+inf`/`-inf`
  injected via monkeypatching.
- **Reversion is complete, not partial.** A rejected pass restores the assignment byte-for-byte (every
  student's team membership checked, not just team counts/sizes) — verified directly; this holds
  structurally because a rejected candidate is computed on a COPY (`[list(t) for t in teams]`) and the
  `nonlocal teams` binding is only reassigned on acceptance.
- **No stale metrics after a reversion** — verified: recomputing any metric on the returned (reverted) teams
  reflects those teams' actual composition; nothing caches a rejected candidate's values anywhere outside
  `_maybe_run`'s own local scope.
- **`_protected_metrics` is a pure, deterministic function** — verified (same input, same output, twice).
- **Missing preference data is handled explicitly** — `mean_preference_satisfaction` maps "no data" to
  `1.0` (nothing to protect against regressing), never `0.0` — verified directly.
- **Every profile's `primary_metric` has consistent (higher-is-better) directionality** — verified against
  `OPTIMIZATION_PROFILES`; `_maybe_run`'s single regression direction assumes higher-is-better for every
  protected key, which holds for `min_utility`/`mean_preference_satisfaction`/`skill_coverage_min` (the only
  three currently used) but is NOT automatically guaranteed for a metric added later — the test would need
  updating if a lower-is-better `primary_metric` is ever introduced (flagged for future authors, not a
  present bug).
- **Finding (not fixed — would change acceptance semantics): tolerances are NOT uniformly applied to
  normalized values.** `total_score` (`_stage_objective`, a SUM over all teams) is unbounded and scales with
  team count, while `floor_min`/`skill_coverage_min`/`min_utility`/`mean_preference_satisfaction` are all
  bounded in `[0, 1]` regardless of roster size — yet the SAME absolute `PROTECTED_METRIC_TOLERANCE` (1e-6)
  is applied to all of them. This makes the tolerance effectively much tighter (relatively) for
  `total_score` at scale than for its siblings. Characterized by
  `test_total_score_protected_metric_is_unnormalized_relative_to_its_siblings`; NOT changed in this session
  (changing it would alter which candidates get accepted, violating this part's own "do not change research
  conclusions merely to create different assignments" instruction) — see `recommended_changes.md` for the
  recommendation.
- **Later passes cannot undo hard avoidance** — verified end-to-end: `_preference_repair_improve`'s own
  internal per-swap `avoidance_ok` guard (active whenever `preference.hard_avoidance=True`) refuses any of
  its own swaps that would increase avoidance-violation-pair count on either touched team, confirmed against
  a real avoidance entry through the actual production orchestrator.
- **Debug diagnostics never expose a preference target's identity** — verified: neither a `prefer` target
  nor an `avoid` target's student id appears anywhere in `optimization_trace`'s serialized output, matching
  `app/preferences.py`'s existing aggregate-only privacy guarantee.

### 5. Sensitivity findings

`backend/evaluation/sensitivity_analysis.py` → `evaluation_output/sensitivity_results.json`. Not a search
for "best" settings (explicitly out of scope) — characterizes sensitivity only:

- **`weights.skill_coverage`** (primary objective reweight, seed roster): DOES move the assignment, with
  one discontinuity observed between 0.4 and 0.7 in the swept grid.
- **`AbilityConfig.ability_level_thresholds`**: confirmed, empirically, to NEVER affect the assignment
  (comembership distance 0.0 across the whole grid) — consistent with its own docstring ("reporting tiers,"
  never read by `_ability_balance_improve`'s continuous-gap accept rule).
- **`FairnessConfig.acceptable_utility_floor`**: confirmed, empirically, to NEVER affect the assignment —
  same reporting-only relationship, verified rather than assumed.
- **`FamiliarityConfig.upper_bound`** (on the `friend_clique` fixture): did not move the assignment across
  the swept grid for this specific fixture/threshold combination tested.
- **`PreferenceConfig.high_priority_weight`** (on `preference_instability`): did not move the assignment
  across the swept grid for this specific fixture.
- **Collapse-to-legacy check**: `research_balanced`'s own default settings (every pass enabled at its class
  default) produce a co-membership distance of exactly **0.0** from `legacy` on the seed roster — i.e.
  "enabling everything at conservative defaults" collapses to doing nothing differently for this specific
  roster, the same finding as section 2 above from a different angle.

No weight is recommended for real deployment by this analysis — see the closing sections below and
`recommended_changes.md` for what would be needed before that judgment could be made responsibly.

### 6. Performance bottlenecks (root cause)

cProfile at n=40 (`research_balanced`, all 4 optional passes enabled) found `_behavioral_compat` (via
`_total_score`) responsible for 60% of cumulative runtime (125,790 calls) and `_stage_objective` for 40%
(5,768 calls) — both were being recomputed FROM SCRATCH, over the ENTIRE partition, on every one of the
`O(T²S²)` candidate swaps each of the 4 newer optional passes evaluates, even though a swap only ever
changes 2 of the `T` teams. `_maximin_improve`/`_two_opt_improve` (earlier passes) already used incremental
per-team caching for exactly this reason; the 4 newer passes (`_ability_balance_improve`, `_fairness_
improve`, `_familiarity_balance_improve`, `_preference_repair_improve`, plus `_hard_avoidance_repair`) had
never been given the same treatment. Full detail: [`docs/performance.md`](../docs/performance.md).

### 7. Before-and-after timings

| n students | `research_balanced`, BEFORE (Part 6 doc) | `research_balanced`, AFTER (Part 6.5, clean benchmark) |
|---|---|---|
| 20 | 191 ms | 63.1 ms |
| 40 | 2,972 ms | 269.0 ms |
| **80** | **26,077 ms** | **1,410.4 ms** |

An ~18.5× speedup at n=80, comfortably under the 5-second target. Full table across all 5 profiles and
n=20/40/60/80/100 in `docs/performance.md`.

### 8. Equivalence evidence

- 552 of the 553 pre-Part-6.5 tests passed both before and after this change, unmodified (the 553rd is the
  item-1 test reorganization above) — including every test in `test_ability_balance.py`,
  `test_familiarity.py`, `test_preferences.py`, `test_individual_utility.py`,
  `test_optimizer_invariants.py`, `test_optimization_profiles.py`, all of which assert EXACT expected
  teams/metrics for deterministic fixtures.
- `backend/tests/test_part6_5_performance_equivalence.py` (4 tests, all passing): a naive, independently
  reimplemented, full-recompute reference version of `_ability_balance_improve`'s exact pre-Part-6.5 accept
  rule produces the BYTE-IDENTICAL final assignment (and identical objective/floor-min/ability-gap values)
  to the now-cache-optimized production function, across 4 fixed rosters; all 4 refactored passes are
  confirmed deterministic (identical input → identical output); the cached objective/floor-min invariants
  are confirmed to hold when independently re-derived from scratch.

### 9. Exact test totals

Full backend suite: **567 passed, 0 failed** (`cd backend && <python> -m pytest tests/ -q`), up from 552
passed / 1 failed before this update (+14 new Part 6.5 tests: 10 in `test_part6_5_orchestration_audit.py`,
4 in `test_part6_5_performance_equivalence.py`; the pre-existing suite's own count is otherwise unchanged
aside from the item-1 rename).

### 10. Do the profiles create usable experimental conditions?

**Conditionally.** `backend/evaluation/experiment_readiness.py` (item 7) run against `legacy` vs.
`research_balanced` on the real seed roster reports `distinct_enough_for_experimental_conditions: false`
(comembership distance 0.0, ARI 1.0) with `excluded_for_missing_data: ["familiarity", "preference"]` — i.e.
**as currently constituted, this specific comparison on this specific roster would NOT be a usable
between-conditions comparison**, precisely because two of the four criteria the experimental profile can act
on have zero recorded data. The tool correctly and specifically identifies why (missing familiarity/
preference data), not merely that the conditions collapsed. A study wanting to compare profiles would need
to either (a) collect familiarity/preference data first, or (b) choose a roster/profile pair with a
genuine, data-supported opportunity to differ (e.g. `skill_priority` vs. `legacy`, which DOES differ
substantially on this same roster — see section 3).

### 11. Is the system ready for Part 7?

**Not yet, and this session's own evidence is why.** The profile-comparison and experiment-readiness
machinery built in Part 6.5 is now trustworthy (equivalence-proven, performance-fixed, orchestration-
audited) — that part of the system IS ready. But the SAME machinery just demonstrated that a real study run
today, on data resembling this seed roster, would very likely find `legacy` and `research_balanced`
indistinguishable for 2 of the 4 criteria the experimental profile is supposed to test, purely for want of
collected familiarity/preference data — not because those criteria don't matter. Before Part 7 (a real
student study), the concrete blocker is **data collection**, not algorithm work: familiarity data (a roster
survey or LMS co-enrollment import — `recommended_changes.md` #4) and preference data (`POST /preferences`
already exists and works — `recommended_changes.md` #5) need to actually be collected from a real cohort
before `research_balanced`/`preference_priority`/a familiarity-focused configuration could be expected to
produce a measurably different, testable assignment. Running Part 7 without that data collection first
would risk exactly the outcome this session measured: two "different" experimental conditions that turn out
to be the same assignment.
