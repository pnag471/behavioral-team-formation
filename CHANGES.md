# CHANGES — Layered Architecture Foundation

Branch: `feat/claude-integration`

## What changed and which file

| # | Change | Files |
|---|--------|-------|
| 1 | **L0 rubric** — `rubric_v1.yaml` is the new single source of truth for all dimension definitions. Replaces inline definitions scattered across models.py, matching.py, explainability.py, mock_analyzer.py, and the LLM prompt. | `backend/traits/rubric_v1.yaml` |
| 2 | **L1 question bank** — `bank_v1.yaml` moves scenarios out of the router and adds trait tags and option→signal mappings. Anonymous option IDs (opt_a–opt_d) prevent trait-value leakage to clients. | `backend/questions/bank_v1.yaml` |
| 3 | **L4 deterministic scorer** — `dimension_scorer.py` replaces the LLM call in the scoring path. Plurality voting for categorical dims, weighted-mean for ordinal. Tracks unknown dims (no MCQ signals). `mcq_adapter.py` bridges the bank to the scorer. | `backend/scoring/dimension_scorer.py`, `backend/scoring/mcq_adapter.py` |
| 4 | **LLM removed from scoring** — `POST /assessment/analyze` no longer calls Claude. Route: bank lookup → SignalUnits → deterministic score → write profile. `ClaudeBehavioralAnalyzer` is retained in `ai/` for future L3 extraction but is not wired to any endpoint. | `backend/app/behavioral_analysis.py` |
| 5 | **`GET /assessment/scenarios`** — now loads from bank_v1.yaml with anonymous option IDs. | `backend/app/behavioral_analysis.py` |
| 6 | **Persistence fixes** — `ResponseDB`: surrogate UUID PK + `UniqueConstraint(assessment_id, scenario_id)`; explicit lookup-or-update replaces `db.merge()`. `BehavioralSignatureDB`: UUID PK, `student_id` indexed non-PK, provenance fields `rubric_version / bank_version / scorer_version` (no `model_version` — scoring is code). `StudentDB` gains `under_determined_dims` column. | `backend/app/database.py` |
| 7 | **Matching refactor** — `_behavioral_compat()` reads `composition_rule` from rubric instead of hardcoded pairwise tables. `high_mean_with_floor` dims scored by team mean (no within-team variance penalty). `soft_compatibility` dims scored by value-diversity reward [0.7, 1.0]. Floor-level students spread round-robin; `_maximin_improve()` equalises team means (maximin). POST /teams/generate returns `feasible`, `floor_violations`, `under_determined_teams`. | `backend/app/matching.py` |
| 8 | **L5 cleanup** — `suggest_norms()` removed from `_build_teams()`; norms generated on-demand in `GET /teams/{id}/explanation` only. | `backend/app/matching.py` |
| 9 | **Vocabulary renames** — `"confrontational"` → `"assertive"` everywhere (conflict_style values). `"medium"` → `"developing"` for accountability ordinal level. | `backend/app/explainability.py`, `backend/ai/mock_analyzer.py`, `backend/ai/mock_explainer.py`, `backend/sample_students.json` |
| 10 | **New dimension: `cooperativeness`** — added to `CollaborationSignature`, rubric, bank signals, seed data, mock analyzer/explainer. | `backend/app/models.py`, seed data, mocks |
| 11 | **`confidence_score` redefined** — no longer a proxy for student self-assurance. Now means scorer measurement quality: mean signal coverage fraction across non-derived dims. | `backend/traits/rubric_v1.yaml`, `backend/scoring/dimension_scorer.py` |
| 12 | **requirements.txt** — added `pyyaml>=6.0`, `alembic>=1.13.0`. | `backend/requirements.txt` |
| 13 | **Tests** — 28 tests total (up from 7). New: 20 scorer/adapter unit tests, 1 determinism integration test. Updated: anonymous option IDs, no analyzer mock, provenance assertions. | `backend/tests/test_dimension_scorer.py`, `backend/tests/test_assessment.py` |

## soft_compatibility: reward-diversity vs penalise-mismatch

The matcher implements **reward-for-diversity**: score = `0.7 + 0.3 * (n_distinct / n_members)`. A fully homogeneous team scores 0.7; a fully diverse team scores 1.0. This is NOT a mismatch penalty — a homogeneous team is simply scored slightly lower, not heavily penalised.

## complementary_coverage status

`complementary_coverage` is **defined AND implemented** — it is applied to skills in `_skill_coverage()` (matching.py), which scores teams by `|team_skills ∩ REFERENCE_SKILLS| / |REFERENCE_SKILLS|`. It is not yet applied to any rubric dimension (leadership_style is a candidate — see TODO below).

## DB migration note

The schema changes to `BehavioralSignatureDB` and `ResponseDB` are breaking.  
**Delete `backend/behavioral_team_formation.db` before running the app** after pulling this branch.  
`init_db()` will recreate all tables from the updated models.

---

## TODO(team-decision) — all four items, with file:line

| # | Question | File:line |
|---|----------|-----------|
| 1 | `conflict_style` weight is 0.0 until the question bank can distinguish task conflict from relationship conflict. Confirm this is correct, or redesign the bank first. | `backend/traits/rubric_v1.yaml` — under `conflict_style.matching_weight`; `backend/app/matching.py` — `_soft_compat_score()` comment |
| 2 | `leadership_style` uses `soft_compatibility` (rewards similar styles). Should it switch to `complementary_coverage` to reward teams that cover directive + facilitative + emergent? | `backend/traits/rubric_v1.yaml` — under `leadership_style.composition_rule`; `backend/app/matching.py` — `_soft_compat_score()` TODO comment |
| 3 | Unknown floor dims: currently flagged as `under_determined` without triggering a floor violation. Alternative: treat-as-worst-case (count unknown as floor-level). Confirm which is correct. | `backend/app/matching.py` — `_is_floor_level()` docstring; `_check_floor_violations()` |
| 4 | Greedy seeding is skills-coverage-first for non-floor students. Consider floor-dim-score-first to better balance high-quality members before the maximin pass. | `backend/app/matching.py` — `_greedy_build()` TODO comment |

---

## NEEDS-SIGN-OFF — proposed research design items

These involve judgment calls that belong in a team/advisor discussion:

**Behavioral anchors (BARS)** for `accountability` and `cooperativeness` are proposed in the rubric. The low/developing/high descriptions are inspired by CATME but are NOT identical to the validated CATME instrument. A researcher familiar with the CATME literature should review and edit them before they are used in any study.

**`cooperativeness` assignments in seed data** (`sample_students.json`) were inferred from existing collaboration_signature values. These are placeholder values for demo purposes only — they do not represent validated assessments.

**MCQ signal mappings** in `bank_v1.yaml` are the author's judgment calls (which option signals which trait at which value). A subject-matter expert should validate these before the bank is used in any study.
