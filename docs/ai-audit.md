# AI layer and team-formation audit

Date: 2026-11-06 · Branch: `feat/gemini-integration` @ `9803acc` · Read-only pass, nothing changed.

Design being checked against (from `backend/ai/README.md`):

1. The AI only pulls quoted evidence from student answers and never scores.
2. Fixed code (`dimension_scorer.py` + `rubric_v1.yaml`) does all scoring.
3. The optimizer forms teams deterministically.
4. Personality "type" is a display label only.
5. Confidence is a per-evidence field, used only to flag weak evidence for review.

Line numbers are relative to the repo root. Unless noted, "called" means reachable from a FastAPI route registered in `backend/app/main.py:87-94`. No LLM was called during this audit, so the live Gemini paths were traced by reading the code, not by running them.

---

## Summary

- **Two profile-building paths run today, and only one follows the design.**
  - The multiple-choice path (`/assessment/analyze`) is fully deterministic and runs through the scorer.
  - The conversational interview path (`/conversation/extract`) lets Gemini assert the final label for every trait and writes those labels straight into the student record. The scorer never runs. This is the main boundary violation, and it lives in `backend/app/conversation.py`, not in `ai/gemini_analyzer.py`.
- **The `BehavioralAnalyzer` interface is dead code.** None of its three implementations (mock, Claude, Gemini) is imported by the app. Both LLM analyzers would also violate the boundary if they were wired in, and both are broken against the current prompt.
- **The Anthropic/Claude path is entirely off.** `anthropic` is commented out in `requirements.txt` and not installed, and nothing imports the Claude classes. The README TODO says to "point the demo at the Claude analyzer path", but no such path exists to point at.
- **The extraction prompt, the code and the rubric disagree** on `conflict_style` and `accountability` values, and on whether `cooperativeness` exists.
- **Team formation is deterministic for a fixed input order, but not across input orders.** Production loads students without `ORDER BY`.
- **The team's two written decisions on confidence contradict each other.** `docs/conversation_pipeline_decisions.md` §4 drops it; `backend/ai/README.md` keeps it per-evidence.

---

## 1. Findings

Tags: **[done]** works as designed · **[partial]** exists, incomplete · **[stub]** defined but not wired or not functional · **[broken]** wired or wireable but fails · **[contradicts-design]** violates a design rule.

### 1.1 What actually runs end to end

| # | Tag | Finding | Where |
|---|-----|---------|-------|
| F1 | [done] | **Multiple-choice assessment is deterministic.** `/assessment/analyze` maps option choices to `SignalUnit`s via `bank_v1.yaml`, then calls `score()`. No LLM is involved, and provenance (`rubric_version`, `bank_version`, `scorer_version`) is stored. | `backend/app/behavioral_analysis.py:145-159`, `:243-250`; `backend/scoring/mcq_adapter.py:41-62` |
| F2 | [stub] | **Free-text answers in the MCQ path are saved but never read.** `response_text` is persisted, but `responses_to_signals` only uses `scenario_id` and `option_id`. No analyzer is called. | `backend/app/behavioral_analysis.py:223-240`; `backend/scoring/mcq_adapter.py:52-54` |
| F3 | [contradicts-design] | **Conversation extraction lets the LLM score.** `/conversation/extract` sends the transcript to Gemini, parses the JSON, and writes `sig["planning_style"]`, `sig["accountability"]` and the other labels directly into `StudentDB`. `score()` is never called. `BehavioralSignatureDB` is written with `rubric_version`, `bank_version` and `scorer_version` all `None`. **This is the live boundary violation**, and it is the path the frontend's main nav uses (`frontend/components/Navigation.tsx:15`, `frontend/app/interview/page.tsx:92`). | `backend/app/conversation.py:212-221`, `:270-302`, `:323-331` |
| F4 | [contradicts-design] | **Conversation profiles feed the optimizer unchecked.** `generate_teams` loads every `StudentDB` row regardless of how it was produced, so LLM-asserted labels from F3 drive team formation and floor constraints directly. | `backend/app/matching.py:4493` |
| F5 | [broken] | **The conversation path never measures `cooperativeness`, a floor dimension.** It is missing from the extracted dimension list and the prompt, and isn't written to `collaboration_signature`. The model default `developing` applies, and `under_determined_dims` is never set (column default `[]`). The optimizer therefore treats every interviewed student as *known* `developing` on a floor dimension. | `backend/app/conversation.py:224-228`, `:288-295`; `backend/app/models.py` `CollaborationSignature.cooperativeness`; `backend/app/database.py:32` |
| F6 | [broken] | **Conversation-path values are outside the rubric.** The prompt asks for `accountability: high\|medium\|low` and `conflict_style: …\|confrontational\|…`. The code's fallback default is `"medium"`. Nothing validates the result against `allowed_values`. In matching, `"medium"` silently becomes the midpoint (`level_map.get(..., max_level // 2)`), and `"confrontational"` counts as a distinct, non-collaborative style in `_conflict_risk`. | `prompts/behavioral_assessment.txt:29,35`; `backend/app/conversation.py:291`; `backend/app/matching.py:136`, `:263-275` |
| F7 | [partial] | **MCQ assessment doesn't persist `safety_contribution` or `stress_response`.** The scorer returns them as unknown defaults, but `/analyze` copies only 8 dims into the signature. The bank has no signals for those two, so they are always in `_unknown_dims`, and they pull `confidence_score` down. That makes the maximum MCQ coverage 8/10. | `backend/app/behavioral_analysis.py:169-182`; `backend/questions/bank_v1.yaml` (no `safety_contribution` or `stress_response` traits); `backend/scoring/dimension_scorer.py:83-87`, `:105-108` |
| F8 | [done] | **Explanations and norms run after formation.** `/teams/{id}/explanation` uses `GeminiExplainer`, which falls back to `MockExplainer` without `GEMINI_API_KEY`. It writes prose and norms about teams that already exist and never feeds back into scores or team choice. This is an allowed AI use under the design. | `backend/app/explainability.py:7,11`, `:72-100`; `backend/ai/gemini_explainer.py:17-31` |
| F9 | [partial] | **Interview state lives in memory.** `_active_sessions` is a process dict, lost on restart or across workers. The fallback rebuild path covers it, but the two paths diverge (native chat vs `generate_content` with the full history). | `backend/app/conversation.py:21`, `:116-166` |

### 1.2 Interfaces and implementations

| # | Tag | Component | Exists | Called? | Works? |
|---|-----|-----------|--------|---------|--------|
| F10 | [stub] | `BehavioralAnalyzer` interface | `backend/ai/interfaces.py:19-30` | **No.** No app module imports any analyzer. | The contract (return final labels + `confidence_score`) itself contradicts design rules 1 and 5. |
| F11 | [stub] | `MockBehavioralAnalyzer` | `backend/ai/mock_analyzer.py:106-128` | No (its own docstring says so, `:4-8`). Only reachable as a fallback inside the two dead LLM analyzers. | It runs: keyword matching that emits final labels + `confidence_score`. Ties fall to the first key (`:99`). Same boundary issue as F10. |
| F12 | [broken] [contradicts-design] | `GeminiBehavioralAnalyzer` | `backend/ai/gemini_analyzer.py:14-58` | No | **The old boundary bug is still there:** it returns `json.loads(response.text)`, the model's final labels, with no scorer (`:57-58`). **It is also broken:** it fills `{responses}` (`:45`), but the shared prompt has no `{responses}` placeholder (0 occurrences). It uses `{transcript}` and `{student_name}`. The student's answers never reach the model, which receives a literal `{transcript}` and invents a profile. Other issues: no schema validation, no retry, sets an unused `self.model` (`:29`). |
| F13 | [broken] [contradicts-design] | `ClaudeBehavioralAnalyzer` | `backend/ai/claude_analyzer.py:20-70` | No | It splits the prompt on `{responses}` (`:33`), which isn't there, so the whole prompt, including the literal `{transcript}`, becomes the system message, and the dynamic suffix is empty. Same final-label return (`:70`). `anthropic` isn't installed, so it always falls back to the mock. Model id `claude-sonnet-4-6` is hardcoded (`:17`). |
| F14 | [stub] | `ExplanationGenerator` / `NormGenerator` interfaces | `backend/ai/interfaces.py:33-44` | Yes, via `GeminiExplainer` (F8) | Fine. These sit after formation, so they're allowed under the design. |
| F15 | [done] | `MockExplainer` | `backend/ai/mock_explainer.py:106` | Yes, as the fallback when `GEMINI_API_KEY` is unset | Rule-based and deterministic. |
| F16 | [done] | `GeminiExplainer` | `backend/ai/gemini_explainer.py:15-135` | Yes (`backend/app/explainability.py:11`) | It is live when the key is set. There's a covering test, `backend/tests/test_research_alignment_fixes.py:311-326`. The norms prompt is inline (`:50-65`), not in `prompts/`. |
| F17 | [stub] | `ClaudeExplainer` | `backend/ai/claude_explainer.py:18-96` | No | `suggest_norms` calls `generate()`, which reads `team.team_norms`. Teams are persisted with `team_norms=[]`, so live norms would come back empty (`:36-39`, `:66`). |

### 1.3 Where scoring happens, and boundary violations

| # | Tag | Finding | Where |
|---|-----|---------|-------|
| F18 | [done] | **`score()` is the only deterministic scorer, and the MCQ path is its only caller.** It uses plurality for categorical dims and a weighted ordinal mean for ordinal ones. | `backend/scoring/dimension_scorer.py:57-110`; called only at `backend/app/behavioral_analysis.py:159` |
| F19 | [contradicts-design] | **The LLM emits final scores in three places:** live in `conversation.py` (F3), and dead in `gemini_analyzer.py` (F12) and `claude_analyzer.py` (F13). | see F3, F12, F13 |
| F20 | [partial] | **Tie-breaking in `_plurality` is "first-seen"** (dict insertion order). It's deterministic for a fixed signal order but biased toward whichever option appears first. The decision doc §1 has already decided to change this to `unknown` plus a flag. | `backend/scoring/dimension_scorer.py:117-122`; `docs/conversation_pipeline_decisions.md:17-23` |
| F21 | [stub] | **The evidence-unit schema exists but nothing writes to it.** `EvidenceUnitDB` (trait, polarity, `rejected`) and `ConversationTurnDB` are defined but never written. `DimensionScoreDB.evidence_span` stores the LLM's quote next to an LLM-asserted value, which is the reverse of the design. | `backend/app/database.py:199`, `:232-252`; `backend/app/conversation.py:257-268` |

### 1.4 Confidence

| # | Tag | Finding | Where |
|---|-----|---------|-------|
| F22 | [contradicts-design] | **Confidence is per dimension, self-reported by the model.** The prompt asks for `confidence_scores` per dimension plus `overall_confidence`, and these are stored on `DimensionScoreDB.confidence` and `confidence_layer`. It is not per evidence, and it isn't tied to quotes. | `prompts/behavioral_assessment.txt:58-61`, `:92-103`; `backend/app/conversation.py:230`, `:266`, `:297-301` |
| F23 | [contradicts-design] | **`confidence_score` means two different things depending on the path.** On the MCQ path it is scorer coverage (rubric definition). On the conversation path it is Gemini's self-reported `overall_confidence`. Same field, different meaning. | `backend/app/behavioral_analysis.py:183`; `backend/app/conversation.py:298`; `backend/traits/rubric_v1.yaml:362-379` |
| F24 | [partial] | **Confidence is displayed to the student, not used for review.** The profile page shows "N% overall confidence" and per-dimension confidence bars. Nothing flags weak evidence for a human. | `frontend/app/profile/[id]/page.tsx:90-107`, `:185-193` |
| F25 | [done] | **Confidence never reaches matching.** `matching.py` doesn't read `confidence_layer` for scoring or team choice (it's only copied in `_db_student_to_model`, `:4185`). Rule 5's "never weight scores" holds today. | `backend/app/matching.py` |
| F26 | [contradicts-design] | **The two confidence decision records conflict.** `docs/conversation_pipeline_decisions.md` §4 (status "decided") drops per-dimension confidence from extraction entirely and shows scorer coverage instead. `backend/ai/README.md` "Decision record" (2026-11-06) keeps confidence as a per-evidence review flag. | `docs/conversation_pipeline_decisions.md:66-78`; `backend/ai/README.md:27-38` |

### 1.5 Trait vocabulary

The rubric has 10 scoreable dimensions plus the derived `confidence_score` (`backend/traits/rubric_v1.yaml:93-379`).

| Dimension | Rubric values | `models.py` default | Extraction prompt | `conversation.py` writes | MCQ bank signals | Interview covers |
|---|---|---|---|---|---|---|
| planning_style | planner / spontaneous / adaptive | adaptive | same | ✓ | 5 | #4 |
| communication_style | async / sync / mixed | mixed | same | ✓ | 7 | #3 |
| execution_style | methodical / iterative / exploratory | iterative | same | ✓ | 2 | **none** |
| conflict_style | avoidant / **assertive** / collaborative | collaborative | **confrontational** ✗ | ✓ | 12 | #1 |
| leadership_style | directive / facilitative / emergent | facilitative | same | ✓ | 8 | #5 |
| accountability | low / **developing** / high | developing | **high / medium / low** ✗ | ✓ (default **"medium"** ✗) | 2 | #2 |
| help_seeking | proactive / reactive / independent | proactive | same | ✓ | 3 | **none** |
| cooperativeness | low / developing / high | developing | **missing** ✗ | **not written** ✗ | 6 | maybe #8 |
| safety_contribution | low / medium / high | medium | same | ✓ | **0** | #7, #8 |
| stress_response | composed / reactive / withdraws | composed | same | ✓ | **0** | #6 |

Sources:
- Prompt: `prompts/behavioral_assessment.txt:18-47`
- Interview topics: `prompts/interview_system.txt:29-45`
- Models: `backend/app/models.py` (`WorkRhythmSignature`, `CollaborationSignature`)
- Conversation dims: `backend/app/conversation.py:224-228`

Other vocabulary findings:

| # | Tag | Finding | Where |
|---|-----|---------|-------|
| F27 | [broken] | **The rename from `confrontational` to `assertive` is incomplete.** Still using the old value: the extraction prompt, both frontend radar maps, and the AI README contract. | `prompts/behavioral_assessment.txt:29-30`; `frontend/app/profile/[id]/page.tsx:44`; `frontend/app/assessment/page.tsx:11`; `backend/ai/README.md:110` |
| F28 | [broken] | **The two frontend pages use different `accountability` scales.** The profile page maps `medium` (conversation vocabulary), so MCQ `developing` falls back to 0.5. The backend radar maps `developing`, so conversation `medium` falls back to 0.5. The two radars disagree on the same data. | `frontend/app/profile/[id]/page.tsx:46`; `backend/app/explainability.py:20` |
| F29 | [partial] | **The interview and extraction cover different dimensions.** The interview prompt drives "all 8" topics, while extraction asks for 9 dims. `execution_style` and `help_seeking` have no interview topic, so the model has to guess them. | `prompts/interview_system.txt:29-45`, `:96` vs `prompts/behavioral_assessment.txt:16-47` |
| F30 | [partial] | **Code hand-copies the trait list instead of reading the rubric.** `conversation.py:224-228`, `behavioral_analysis.py:169-201`, `interfaces.py:25-28`, `ai/README.md:103-112`, the prompt, and the frontend maps are all hand-maintained. The decision doc §2 plans to generate prompt enums from the rubric. | as listed |

### 1.6 Team formation entry point and determinism

| # | Tag | Finding | Where |
|---|-----|---------|-------|
| F31 | [partial] | **There's no pure team-formation function.** The entry point is the FastAPI route `generate_teams(request: TeamGenerationRequest) -> dict` at `POST /teams/generate`. It reads the DB, runs the formation sequence inline, and deletes and rewrites `TeamDB`. The sequence is `resolve_profile` → `_normalize_weights` → `_check_feasibility` → `_greedy_build` → `_two_opt_improve` → `_maximin_improve` → `_run_optional_passes` → `_build_teams`. Offline wrappers copy that sequence: `backend/evaluation/run_report.py:160-174` and `backend/eval/determinism_harness.py`. | `backend/app/matching.py:4476-4590` |
| F32 | [done] | **Formation is deterministic for a fixed input order.** Tested by `backend/eval/determinism_harness.py` on 24- and 23-student synthetic cohorts: 10 in-process runs and 3 fresh processes with different `PYTHONHASHSEED` values each gave 1 distinct partition. A random-shuffle baseline gave 10 distinct partitions in 10 runs. | `backend/eval/` |
| F33 | [partial] | **The result depends on input order.** The same 24 students in 10 shuffled orders gave 10 different partitions. `generate_teams` loads `db.query(StudentDB).all()` with no `ORDER BY`. SQLite usually returns rows in insertion order, but doesn't guarantee it. | `backend/app/matching.py:4493` |
| F34 | [done] | **Random mode is nondeterministic by design.** `random_assignment` uses the unseeded global `random`, which is expected for `formation_mode="random"`. | `backend/app/matching.py:4376-4385` |

### 1.7 Personality type

| # | Tag | Finding | Where |
|---|-----|---------|-------|
| F35 | [stub] | **No personality "type" label exists** anywhere in the backend, prompts or frontend (searched for archetype, personality, type label and MBTI). Rule 4 holds trivially. There's nothing to enforce yet. | — |

### 1.8 Dead code, duplicate paths, legacy

| # | Tag | Finding | Where |
|---|-----|---------|-------|
| F36 | [stub] | **The analyzer stack is unused.** `BehavioralAnalyzer` and its three implementations aren't imported by any app module. `ClaudeExplainer` isn't imported either. | `backend/ai/` |
| F37 | [stub] | **`_call_with_retry` is never called.** Gemini calls in `conversation.py` have no retry. | `backend/app/conversation.py:344-353` |
| F38 | [partial] | **There are three copies of `_db_student_to_model`.** The `explainability.py` copy and `main.py` route constructors drop `under_determined_dims`, so profile and explanation views can't tell "unknown" from "default". | `backend/app/matching.py:4177`; `backend/app/explainability.py:40-49`; `backend/app/main.py:111-119`, `:152-160` |
| F39 | [partial] | **The Gemini model id `gemini-3.1-flash-lite` is hardcoded in 10 places** (7 in `conversation.py`, 3 in `ai/gemini_*.py`), and `prompt_version="1.0"` is hardcoded. | `backend/app/conversation.py:67,90,122,142,213,241,327` (prompt_version `:242`); `backend/ai/gemini_*.py` |
| F40 | [partial] | **Several pieces are stale:** the `ai/README.md` "Interface Contract" and "How to swap" sections, the `interfaces.py` docstring ("instantiate in behavioral_analysis.py"), and the mock analyzer's "retained for future L3 use" note all describe the old label-emitting design. | `backend/ai/README.md:42-125`; `backend/ai/interfaces.py:1-10` |
| F41 | [partial] | **Conversation extract creates a fresh student id every call** (`s<ms>`). Re-extracting a session creates duplicate students, and `ProfileVersionDB` doesn't link back to an existing student. | `backend/app/conversation.py:233` |
| F42 | [broken] | **The rubric's `leadership_style` TODO contradicts the code.** The rubric TODO says soft-compatibility gives a "slight reward for SIMILAR styles". `leadership_style` has no `fit` key, so `_soft_compat_score` defaults to `complementary`, which rewards *diversity*. The `matching.py:161` TODO describes the code correctly. One of the two TODOs is stale. | `backend/traits/rubric_v1.yaml:203-207`; `backend/app/matching.py:161-165`, `:175-182` |

### 1.9 Config and flags that switch behavior

| Flag | Effect | Where |
|---|---|---|
| `GEMINI_API_KEY` unset | `GeminiExplainer` silently falls back to `MockExplainer`. All `/conversation/*` routes return HTTP 500 ("GEMINI_API_KEY not set"). | `backend/ai/gemini_explainer.py:18-31`; `backend/app/conversation.py:44-46` |
| `ANTHROPIC_API_KEY` + `anthropic` installed | It would switch the Claude classes from mock to live, but `anthropic` is commented out and not installed, and no app code imports the Claude classes. | `backend/requirements.txt` (last line); `backend/ai/claude_*.py:6-10`, `:23-24` |
| `google-genai` hard dependency | `conversation.py` and `gemini_*.py` import it at module level or call time. It is required even when running mock-only. | `backend/requirements.txt`; `backend/ai/gemini_explainer.py:6-7` |
| `DATABASE_URL` | Defaults to `sqlite:///./behavioral_team_formation.db`, relative to the working directory. | `backend/app/database.py:8` |
| `load_dotenv()` | Reads `.env` from the working directory at app import. There's no `.env` in `backend/` or the repo root on this machine. | `backend/app/main.py:2-3` |
| `TeamGenerationRequest.profile` / `floor_policy` / `formation_mode` | Select optimizer behavior. Default is `legacy`, `one_low_per_team`, `behavioral`. | `backend/app/models.py:784-847` |
| Turn cap `len(history) >= 22` + completion phrase match | Ends the interview. Matching on phrases is fragile. | `backend/app/conversation.py:109`, `:156-161`, `:178-183` |

### 1.10 TODOs already in the code

- `backend/app/matching.py:161`: leadership_style fit (see F42)
- `backend/app/matching.py:211`: conflict_style zeroed
- `backend/app/matching.py:4235`: greedy seeding strategy
- `backend/traits/rubric_v1.yaml:171`: conflict_style zeroed
- `backend/traits/rubric_v1.yaml:203`: leadership_style rule (stale, see F42)
- `backend/ai/README.md:38`: confirm Option B; fix or retire `gemini_analyzer.py`
- `backend/eval/*`: 8 TODOs from the offline-eval work. That work is uncommitted.
- `CHANGES.md:39`: TODO index (lists 4 items; doesn't include the README or eval TODOs)
- `docs/conversation_pipeline_decisions.md`: the decided-but-unbuilt sequence (§Build sequence, `:123-134`)

---

## 2. Recommended refactor order

The order puts docs and tests first, so later runtime changes are protected, then the boundary fix, then cleanup. The order largely follows the existing build sequence in `docs/conversation_pipeline_decisions.md:123-134`.

| # | Change | Runtime? | Risk |
|---|--------|----------|------|
| R1 | Resolve the confidence conflict (F26) and update whichever doc loses. | No (docs) | None. It unblocks R4's extraction contract. |
| R2 | Add drift and contract tests: rubric ↔ prompt ↔ `models.py` ↔ frontend value maps (F27–F30), plus a mocked-LLM test pinning `/conversation/extract`'s current output. | No (tests only) | Low. Tests will fail on today's drift, which is the point. Mark them xfail until R3. |
| R3 | Fix vocabulary drift: `confrontational`→`assertive` and `medium`→`developing` in the prompt, the `conversation.py` default, and the frontend maps; add `cooperativeness` to the extraction prompt. | **Yes** | Low–medium. It changes stored values for new interview profiles; existing DB rows keep the old values (see Q5). |
| R4 | Change extraction to evidence only: the LLM emits `{trait, indicator, polarity, strength, quote}`; an adapter checks quotes verbatim, writes `EvidenceUnitDB`, builds `SignalUnit`s, calls `score()`, and sets `under_determined_dims`. This removes F3, F4, F5, F19 and F22. | **Yes** | High. It's the core design fix and changes every interview-derived profile and therefore team composition. It needs R2's tests first. |
| R5 | Make the confidence representation match R1's decision: per-evidence field or scorer coverage. Stop showing LLM self-confidence as a per-dimension bar (F24); add a review flag if Option B wins. | **Yes** (UI + stored fields) | Medium. It changes what students see. |
| R6 | Sort students by id before formation (or add `ORDER BY id`) so input order can't change the teams (F33). | **Yes** | Low–medium. One-time change to which teams existing rosters produce, with deterministic results afterward. Needs Q7 answered first. |
| R7 | Change plurality tie-breaking to "tie → unknown + flag" (F20; already decided in decision doc §1). | **Yes** (MCQ path too) | Medium. Shared L4 code. No existing test depends on tie behavior. |
| R8 | Retire or rewrite the dead analyzer stack (F10–F13, F36) and `ClaudeExplainer` (F17), per Q3. If kept, reshape `BehavioralAnalyzer` to return evidence units, not labels. | No (never imported) | Low. Removing it can't change runtime. Deleting `gemini_analyzer.py` also removes the demo hazard the README TODO warns about. |
| R9 | Persist or explicitly mark `safety_contribution`/`stress_response` on the MCQ path, and decide whether they count in MCQ `confidence_score` (F7). | **Yes** (stored fields, displayed confidence) | Low. Weight 0, so it can't change teams. |
| R10 | Pull the inline formation sequence out of `generate_teams` into a pure `form_teams(...)` used by the route and by both eval wrappers (F31). | No, if done as a pure move | Medium. `matching.py` is actively edited by two people, and the decision doc says not to split it. Extract one function only. |
| R11 | Small cleanup: one `_db_student_to_model` that keeps `under_determined_dims` (F38); a single model-id/prompt-version config with a prompt hash (F39); wire or delete `_call_with_retry` (F37); stable student id per session (F41). | Partly (F38 and F41 change API output) | Low. |
| R12 | Update stale docs: `ai/README.md` contract and examples, the `interfaces.py` docstring, the stale rubric TODO (F42), and the `CHANGES.md` TODO index (F40). | No | None. |

---

## 3. Open questions

These are for the team to decide. This audit doesn't resolve them.

- **Q1 — TODO(team-decision):** Confidence: drop it from extraction and show scorer coverage (`docs/conversation_pipeline_decisions.md` §4), or keep it as a per-evidence review flag (`backend/ai/README.md` Decision record)? The two records conflict (F26).
- **Q2 — TODO(team-decision):** Which provider handles extraction? The only live LLM path is Gemini, inline in `conversation.py`. The README TODO says to demo through "the Claude analyzer path", which isn't wired anywhere and whose SDK isn't installed. Should the demo use Gemini through a fixed adapter, wire up Claude, or skip interviews in the demo and use MCQ only?
- **Q3 — TODO(team-decision):** Fix or retire `gemini_analyzer.py`, `claude_analyzer.py`, `mock_analyzer.py`, the `BehavioralAnalyzer` interface, and `ClaudeExplainer` (F10–F13, F17, F36)? (Carries over the existing `backend/ai/README.md:38` TODO.)
- **Q4 — TODO(team-decision):** Should the MCQ path's free-text `response_text` become evidence (through the same extraction adapter), or is MCQ deliberately option-only (F2)?
- **Q5 — TODO(team-decision):** What happens to existing interview profiles in the DB, whose labels came from the LLM (F3, F6)? Re-score from stored transcripts after R4, flag them as legacy, or exclude them from any study data?
- **Q6 — TODO(team-decision):** Which interview topic measures `cooperativeness`? Is topic #8 "making space for others" cooperativeness or safety_contribution? Should `execution_style` and `help_seeking` get interview topics, or be dropped from extraction (F5, F29)? Ties to the deferred fold-or-keep question in decision doc §3.
- **Q7 — TODO(team-decision):** Is input-order independence part of the determinism claim in the paper (F33)? If yes, R6 is required before any study run.
- **Q8 — TODO(team-decision):** The personality "type" label doesn't exist (F35). Is one planned? If so, where is it derived, and what test enforces that it never feeds scoring or formation?
- **Q9 — TODO(team-decision):** Should students see confidence at all? Under Option B confidence is a reviewer signal, but the profile page shows it to the student (F24).
- **Q10 — TODO(team-decision):** `leadership_style` fit: the rubric TODO and the code disagree about what the current rule even does (F42). Which description is the intended one?
