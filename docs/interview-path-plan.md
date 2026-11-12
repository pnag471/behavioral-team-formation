# Plan: route the interview path through the deterministic scorer

Date: 2026-11-10 · Branch: `feat/gemini-integration` · Status: **PLAN ONLY.** No source file has been changed.

**Goal.** `POST /conversation/extract` should work like the MCQ path:

1. Gemini extracts evidence items from the student's own words.
2. A new adapter turns those items into `SignalUnit`s.
3. `score()` produces the trait labels.
4. The labels are written to the same `Student` fields the MCQ path writes.
5. The raw evidence from every extraction run is saved, so scoring and team formation can be reproduced without calling the LLM again.

**Team decisions this plan applies.** These are settled and not reopened here:
- Gemini is the provider for this path.
- Per-evidence confidence is a **review flag only**. It never enters `score()` or any weight.
- Strength **does** weight the score, through the rubric's `evidence_strength`. It is provisional until exemplars are written.
- Trait values follow the rubric: `assertive` (not `confrontational`), `developing` (not `medium`).
- `cooperativeness` gets an interview topic.
- Reproducibility comes from persisting the evidence.

Related docs: `backend/ai/README.md` (Decision record, 2026-11-10), `docs/conversation_pipeline_decisions.md` (§1–§8 and its build sequence), `docs/ai-audit.md` (F3, F5, F6, F22).

---

## Part A. What exists today

All excerpts below are quoted from the files as they stand.

### A1. The MCQ path, end to end (the template)

**1. Route.** `backend/app/behavioral_analysis.py:145-161`:

```python
@router.post("/analyze")
def analyze_assessment(request: AssessmentRequest):
    """
    Score MCQ responses via the deterministic L4 scorer (no LLM in this path).
    ...
    rubric = get_rubric()
    bank = get_bank()

    # L3 → L4: convert option choices to SignalUnits, then score deterministically
    signals = responses_to_signals(request.responses)
    scored = score(signals, rubric)

    unknown_dims: List[str] = scored.pop("_unknown_dims", [])
```

**2. `SignalUnit` shape.** `backend/scoring/dimension_scorer.py:30-35`:

```python
@dataclass
class SignalUnit:
    trait: str
    value: str
    weight: float = 1.0
    source_ref: str = ""
```

There is **no confidence field** on `SignalUnit`. That keeps confidence out of `score()` by construction.

**3. Adapter.** `backend/scoring/mcq_adapter.py:51-61`. Each chosen option expands into the signals `bank_v1.yaml` lists for it, all at weight `1.0`:

```python
    for resp in responses:
        per_scenario = option_map.get(resp.scenario_id, {})
        sig_defs = per_scenario.get(resp.option_id, [])
        for sig_def in sig_defs:
            signals.append(SignalUnit(
                trait=sig_def["trait"],
                value=sig_def["value"],
                weight=1.0,
                source_ref=f"{resp.scenario_id}:{resp.option_id}",
            ))
```

**4. Writing to the Student.** `backend/app/behavioral_analysis.py:169-184` (existing student; the new-student branch at `:186-206` mirrors it):

```python
            existing.work_rhythm_signature = {
                **existing.work_rhythm_signature,
                "planning_style": scored["planning_style"],
                "communication_style": scored["communication_style"],
                "execution_style": scored["execution_style"],
            }
            existing.collaboration_signature = {
                **existing.collaboration_signature,
                "conflict_style": scored["conflict_style"],
                "leadership_style": scored["leadership_style"],
                "accountability": scored["accountability"],
                "help_seeking": scored["help_seeking"],
                "cooperativeness": scored["cooperativeness"],
            }
            existing.confidence_layer = {"confidence_score": scored["confidence_score"]}
            existing.under_determined_dims = unknown_dims
```

The fields it writes:
- **`work_rhythm_signature`:** `planning_style`, `communication_style`, `execution_style`
- **`collaboration_signature`:** `conflict_style`, `leadership_style`, `accountability`, `help_seeking`, `cooperativeness`
- **`confidence_layer.confidence_score`:** scorer coverage. Going forward this is called `measurement_coverage`.
- **`under_determined_dims`:** the list of dimensions with no signal

It does **not** write `safety_contribution` or `stress_response`. They are rubric dimensions (`rubric_v1.yaml:316-360`) but the bank has no signals for them, so they always come back unknown.

**5. Provenance.** `behavioral_analysis.py:243-250` writes `BehavioralSignatureDB(signature=scored, rubric_version=rubric["schema_version"], bank_version=bank["bank_version"], scorer_version=SCORER_VERSION)`.

### A2. The current interview path (`backend/app/conversation.py:192-341`)

**What the LLM is asked to return.** The prompt is `prompts/behavioral_assessment.txt`, loaded at `conversation.py:29-37`. It asks for **final labels** and **self-reported confidence** (`:16`, `:58-61`, `:67-104`):

```
Extract the following dimensions. Use ONLY the allowed values.
...
Return ONLY valid JSON with these exact keys:
{
  "planning_style": "...",
  "planning_style_evidence": "direct quote from transcript supporting this inference",
  ...
  "confidence_scores": { "planning_style": 0.0, ... "stress_response": 0.0 },
  "overall_confidence": 0.0,
  "consistency_flags": [...]
}
```

The value lists in the prompt that disagree with the rubric:
- `:29` `conflict_style: "avoidant" | "confrontational" | "collaborative"`
- `:35` `accountability: "high" | "medium" | "low"`
- `cooperativeness` is not requested anywhere.

**The call.** `conversation.py:212-221`. `temperature=0.1`, JSON mime type, no `response_schema`, and no validation:

```python
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
        )
    )

    sig = json.loads(response.text)
```

**What gets written, and where:**

| Table / field | What is written | Lines |
|---|---|---|
| `extraction_runs` | `model_name`, `prompt_version="1.0"` (hardcoded), `params_json={"temperature": 0.1}` | `:238-245` |
| `profile_versions.signature_json` | the raw LLM JSON (`sig`) | `:249-255` |
| `dimension_scores` | one row per dimension: `value=sig.get(dimension)` (**LLM label**), `confidence=confidence_scores.get(dimension)`, `evidence_span=sig.get(f"{dimension}_evidence")` | `:258-268` |
| `students.competence_signature` | `skills=sig["primary_skills"]`, `experience_level=sig["skill_depth"]`, `skill_depth`, `skill_no_gos` | `:273-279` |
| `students.work_rhythm_signature` | `planning_style`, `communication_style`, `execution_style` = **LLM labels**; `unavailable_times`, `timezone_notes` | `:280-287` |
| `students.collaboration_signature` | `conflict_style`, `leadership_style`, `accountability` (default **`"medium"`**), `help_seeking`, `safety_contribution`, `stress_response` = **LLM labels**. **No `cooperativeness`.** | `:288-295` |
| `students.confidence_layer` | `confidence_score=sig["overall_confidence"]` (LLM self-report), `confidence_scores`, `consistency_flags` | `:297-301` |
| `students.under_determined_dims` | **not set**, so the column default `[]` applies (`database.py:32`) | — |
| `responses` | the whole transcript as one row, `scenario_id="conversational_interview"` | `:313-320` |
| `behavioral_signatures` | `signature=sig`, `rubric_version=None`, `bank_version=None`, `scorer_version=None` | `:323-331` |

`score()` is never called, and `evidence_units` is never written. `EvidenceUnitDB` is defined at `database.py:242-253` and the table exists in the local DB with 0 rows. The dimensions loop hardcodes 9 names and leaves out `cooperativeness` (`:224-228`).

**One more leak.** `experience_level=sig.get("skill_depth")` (`:276`) is an LLM-assigned label. When `ability_score` is unset, `_student_ability_score` falls back to it (`matching.py:1260`). That fallback is only used by the optional ability pass, which is off in the default `legacy` profile, and by reporting.

### A3. The scorer contract (`backend/scoring/dimension_scorer.py`)

- **Input:** `score(signals: list[SignalUnit], rubric: dict) -> dict[str, Any]` (`:57`).
- **Output** (`:61-63`, `:104-108`): a flat dict with `{dim_name: label}` for every non-derived dimension, plus `"_unknown_dims": [...]` and `"confidence_score": float`. The confidence value is coverage: `min(1.0, len(trait_signals) / 3)` per dimension, averaged (`:93`). This is the `measurement_coverage` value.
- **Aggregation:**
  - Categorical dimensions use `_plurality`: highest total `weight`, and **ties go to the first value seen** (`:117-122`).
  - Ordinal dimensions use `_ordinal_mean`: the weight-averaged position, rounded to the nearest allowed value (`:125-142`).
  - An unknown `value` on an ordinal dimension silently counts as the midpoint (`:136`, `level_map.get(s.value, max_level // 2)`).
- **`evidence_strength` is not applied anywhere today.** The block exists at `rubric_v1.yaml:67-86` (strong 1.5, medium 1.0, weak 0.5, "Provisional numbers — not tuned against real data yet"). A repo-wide search finds no code that reads it. `score()` only reads `SignalUnit.weight`, so applying strength is the **adapter's** job: set `weight` from the rubric. `score()` does not need to change.
- **Dimensions with no signal** (`:85-91`):
  ```python
          if not trait_signals:
              unknown_dims.append(dim_name)
              coverage_fractions.append(0.0)
              # default_value is for DISPLAY only — do not assign here;
              # callers must check _unknown_dims before using the value.
              result[dim_name] = dim_def.get("default_value", "")
  ```
  The caller stores `_unknown_dims` in `Student.under_determined_dims`. The matcher then excludes unknown members from floor checks and compatibility (`matching.py:114-115` `_is_unknown`; `:352-355` `_is_floor_level` returns False for unknown).
- **Mapping labels onto the Student:** the rubric `group` decides the target. `work_rhythm` goes to `work_rhythm_signature` and `collaboration` goes to `collaboration_signature` (the matcher reads it the same way in `_get_student_dim`, `matching.py:103-111`).

### A4. Where the frontend (and docs) use the old trait values

| File:line | Old value | Fix |
|---|---|---|
| `frontend/app/profile/[id]/page.tsx:44` | `confrontational: 0.60` | `assertive: 0.60` |
| `frontend/app/profile/[id]/page.tsx:46` | `medium: 0.60` | `developing: 0.60` |
| `frontend/app/assessment/page.tsx:11` | `confrontational: 0.6` | `assertive: 0.6` |
| `frontend/app/assessment/page.tsx:13` | `medium: 0.6` | `developing: 0.6` |
| `frontend/app/profile/[id]/page.tsx:27-34` | `collaboration_signature` type has no `cooperativeness` | add it |
| `frontend/app/profile/[id]/page.tsx:172-180` | trait list shown omits `cooperativeness` | add a row |
| `frontend/app/profile/[id]/page.tsx:35-38`, `:90-92`, `:185-193` | reads and shows `confidence_scores` (LLM per-dimension confidence) to the **student** | replace (see B5) |
| `prompts/behavioral_assessment.txt:29-30`, `:35-36` | `confrontational`, `medium` | prompt is replaced (B1) |
| `backend/app/conversation.py:291` | default `"medium"` | removed by B3 |
| `backend/ai/README.md:110` | contract lists `confrontational` | docs fix |

The backend radar is already correct: `backend/app/explainability.py:18` uses `assertive` and `:20` uses `developing`. `frontend/lib/types.ts:19-21`, `:74-76` type these fields as plain `string`, so nothing changes there.

### A5. The interview question set (`prompts/interview_system.txt`)

There are 8 behavioral topics (`:29-45`), and the "8" count is repeated at `:29`, `:74` and `:96`. How the topics line up with rubric dimensions (my reading; there is no mapping in code):

| # | Topic | Likely rubric dimension |
|---|---|---|
| 1 | Tension and disagreement | conflict_style |
| 2 | Responsibility and follow-through | accountability |
| 3 | Communication and coordination | communication_style |
| 4 | Planning and deadlines | planning_style |
| 5 | Stepping up | leadership_style |
| 6 | Stress and pressure response | stress_response |
| 7 | Credit and blame | safety_contribution |
| 8 | Making space for others | safety_contribution **or** cooperativeness (ambiguous) |
| — | *(none)* | **cooperativeness, execution_style, help_seeking** |

There are also banned "leading" words at `:10`: collaborative, proactive, accountable, leadership, conflict resolution, communication skills, team player, self-starter, go-getter, initiative.

---

## Part B. The new path

```
transcript ─▶ Gemini (prompt built from rubric, response_schema)
          ─▶ evidence items {quote, dimension, indicator, polarity, strength, confidence}
          ─▶ persist raw output + every item (accepted or rejected)       [reproducibility]
          ─▶ interview_adapter.evidence_to_signals()                      [deterministic]
          ─▶ score(signals, rubric)                                       [deterministic, unchanged]
          ─▶ write labels to Student (same fields as MCQ) + under_determined_dims
```

### B1. New LLM extraction output

The LLM returns **only** evidence items. It returns no labels per dimension, no `overall_confidence` and no `consistency_flags`.

```json
{
  "evidence": [
    {
      "quote": "I messaged the group chat the night before and said I'd take the slides",
      "dimension": "accountability",
      "indicator": "high",
      "polarity": "supporting",
      "strength": "strong",
      "confidence": 0.8
    }
  ],
  "skills":       { "primary_skills": [], "skill_no_gos": [] },
  "availability": { "unavailable_times": [], "timezone_notes": "" }
}
```

What each field means:
- **`quote`:** a verbatim span copied from a **student** turn. Interviewer text doesn't count.
- **`dimension`:** a rubric dimension name. The schema enum is built from `rubric["dimensions"]`, excluding `type: derived`, so it includes `cooperativeness` automatically.
- **`indicator`:** which of that dimension's `allowed_values` the quote points toward. This is the "direction" field, and it maps to `EvidenceUnitDB.rubric_indicator`. It isn't constrained per dimension in the schema; the adapter checks it (B2), as decided in `conversation_pipeline_decisions.md` §2.
- **`polarity`:** `supporting` or `contradicting`. It maps to `EvidenceUnitDB.polarity` (`database.py:249`) and follows decision doc §5.
- **`strength`:** one of the keys of `rubric["evidence_strength"]` (`strong` / `medium` / `weak`). The prompt includes each level's rubric `definition` text, so strength is classified against the rubric's wording rather than freehand.
- **`confidence`:** a float in [0, 1]. **Stored as a review flag only** (B4).

The prompt is built at request time from `rubric_v1.yaml`: the dimension list, each dimension's `description` and `allowed_values`, and the `evidence_strength` definitions (decision doc §2 and §7). The hand-written instructions (read the whole transcript, prefer past examples, quote verbatim) stay in a new prompt file. A `response_schema` with enums on `dimension`, `polarity` and `strength` is passed to `generate_content`. `prompt_hash` and `rubric_version` are stored on the run.

**Skills and availability stay LLM-extracted facts** (lists of names and times), as they are today. They aren't rubric traits, and the scorer has nothing to say about them. `skill_depth` is the exception, because it is a label (see TODO-9).

### B2. The interview-to-signal adapter (new: `backend/scoring/interview_adapter.py`)

This mirrors `mcq_adapter.responses_to_signals`. It is a pure function with no DB access and no LLM, so it is fully unit-testable.

```python
def evidence_to_signals(items: list[EvidenceItem], transcript_student_text: str, rubric: dict
                        ) -> tuple[list[SignalUnit], list[EvidenceDecision]]
```

It handles each item in a fixed order:

1. **Validate.** Reject the item and record why if any of these hold:
   - `dimension` is not a non-derived rubric dimension → `rejected:unknown_dimension`
   - `indicator not in rubric["dimensions"][dimension]["allowed_values"]` → `rejected:indicator_not_allowed`. This is what `EvidenceUnitDB.rejected` is for (`database.py:252`), and it closes the silent midpoint fallback in `_ordinal_mean` (A3).
   - `strength not in rubric["evidence_strength"]` → `rejected:unknown_strength`
   - the quote isn't found verbatim in the student's turns → `rejected:not_verbatim` (normalization rules: TODO-4)
2. **Contradicting items:** `polarity == "contradicting"` items are **not** turned into signals. They are kept as `excluded:contradicting` and produce a deterministic `consistency_flag` (decision doc §5).
3. **Dedupe and cap** (decision doc §6): at most 2 counted items per (dimension, indicator). Drop near-duplicates first. The threshold is TODO-6.
4. **Map each surviving item to a `SignalUnit`:**
   ```python
   SignalUnit(
       trait      = item.dimension,
       value      = item.indicator,
       weight     = rubric["evidence_strength"][item.strength]["weight"],   # 1.5 / 1.0 / 0.5, provisional
       source_ref = f"{run_id}:{evidence_unit_id}",
   )
   ```
   **`item.confidence` is not read here.** `SignalUnit` has no field for it, so it can't reach `score()`.
5. **Order deterministically.** Emit signals sorted by `(dimension, start offset of quote in transcript, indicator, quote)`, not in LLM output order. `_plurality` breaks ties by first-seen value (A3), so this ordering makes re-scoring from stored evidence independent of how the LLM happened to order its output. (A tie still resolves toward whichever value appears earlier in the transcript. See TODO-8 and decision doc §1.)

It returns the signals plus a decision record per item (accepted, rejected with reason, or excluded with reason). Every item is persisted, including rejected ones.

### B3. Scoring and writing to the Student

In `conversation.py` `extract_signature`, after the LLM call:

```python
items      = parse_and_schema_check(response.text)          # malformed JSON → 502, nothing written
signals, decisions = evidence_to_signals(items, student_text, rubric)
scored     = score(signals, rubric)                          # unchanged scorer
unknown    = scored.pop("_unknown_dims", [])
coverage   = scored.pop("confidence_score")                  # = measurement_coverage
```

Then write to `StudentDB`, using the **same fields the MCQ path writes** (A1):
- **`work_rhythm_signature`:** `planning_style`, `communication_style`, `execution_style` from `scored`, plus `unavailable_times` and `timezone_notes` from the extraction (facts, not traits)
- **`collaboration_signature`:** `conflict_style`, `leadership_style`, `accountability`, `help_seeking`, `cooperativeness` from `scored`. Also `safety_contribution` and `stress_response`, because the interview *can* produce evidence for them (TODO-10).
- **`confidence_layer`:** `{"measurement_coverage": coverage}`. The legacy key `confidence_score` keeps the same value so existing readers don't break (TODO-11). LLM `confidence_scores` and `overall_confidence` are **no longer written** to the student.
- **`under_determined_dims`:** `unknown` (newly set; today it is never written on this path)
- **`competence_signature`:** skills, `skill_no_gos` (facts); `experience_level` is handled per TODO-9

Choosing which Student fields to write (`group` → signature) should go through **one shared helper**, e.g. `apply_scored_to_student(student_row, scored, unknown)`, rather than a third hand-written field list. TODO-10 covers whether the MCQ path should switch to it in this change.

**Persisting evidence per run** (reproducibility):

| Table | Write |
|---|---|
| `extraction_runs` | as today, plus **new columns** `rubric_version`, `prompt_hash`, `raw_response` (the unparsed model text) (decision doc §2). `prompt_version` is no longer hardcoded. |
| `evidence_units` | **one row per LLM item, including rejected and contradicting ones**: `trait`, `rubric_indicator`, `polarity`, `strength`, `span`, `rejected`, plus **new columns** `confidence` (Float, nullable), `needs_review` (bool), `decision` (string reason), `quote_start` (int, nullable) |
| `profile_versions.signature_json` | `scored` plus `_unknown_dims` (the **scorer** output, not the raw LLM JSON). It links `run_id`, so the profile can be traced back to its evidence. |
| `dimension_scores` | `value = scored[dim]` (the column comment at `database.py:237` already says "final label from dimension_scorer.score()"); `confidence = NULL` (TODO-11); `evidence_span` = the chosen display quote (strongest accepted supporting item, earliest in transcript) |
| `behavioral_signatures` | `signature=scored`, `rubric_version=rubric["schema_version"]`, `scorer_version=SCORER_VERSION`, `bank_version=None`, `model_version` = Gemini model id |

**Re-scoring function** (the reproducibility guarantee): `rescore_run(run_id)` reads the `evidence_units` rows for a run, rebuilds the `EvidenceItem`s, calls `evidence_to_signals` and then `score()`, and must reproduce `profile_versions.signature_json` byte for byte. It needs no LLM and no network. It is the basis for a regression test.

### B4. Confidence as a review flag

- **Stored** on `evidence_units.confidence` next to its quote.
- **`needs_review`** is set deterministically from the stored item and never changes a score. Rules:
  - (a) rejected items
  - (b) contradicting items on a floor dimension (`accountability` or `cooperativeness`); decision doc §5 says these must be surfaced
  - (c) items whose `confidence` falls below the review threshold (TODO-3)
- **It never enters `score()` or any weight.** The adapter doesn't read `confidence` (B2 step 4), and `SignalUnit` has no confidence field. A test should assert that changing every item's `confidence` leaves `scored` unchanged.
- **Who sees it:** a reviewer-facing endpoint, e.g. `GET /conversation/{session_id}/evidence`, returns the evidence rows with `needs_review`. The student profile page stops showing LLM per-dimension confidence bars (TODO-12).

### B5. Strength

- The LLM classifies each item's `strength` as `strong`, `medium` or `weak`, using the rubric's `evidence_strength.*.definition` text interpolated into the prompt.
- The weight comes **only** from `rubric["evidence_strength"][strength]["weight"]` (1.5 / 1.0 / 0.5). **No new numbers are introduced.**
- **Provisional:** the rubric already marks these numbers "Provisional … not tuned against real data yet" (`rubric_v1.yaml:70-71`), and `exemplars: {}` is empty on every dimension. Each run stores `rubric_version`, so profiles can be re-scored once psych faculty (Thao, Liesel) add exemplars or retune the weights. The profile and reviewer views should label strength-weighted scores as provisional.

### B6. Trait-name fixes

- **Prompt:** the new prompt (B1) is generated from the rubric, so `confrontational` and `medium` can't appear. `prompts/behavioral_assessment.txt` is replaced. Note that the dead `ai/gemini_analyzer.py:20-24` and `ai/claude_analyzer.py:26-30` also load that file (see the audit's F12 and F13).
- **Backend:** the `conversation.py:291` `"medium"` default goes away, because labels now come from `score()`, which only emits `allowed_values` or the rubric `default_value`.
- **Frontend:** the four map entries in A4 (`profile/[id]/page.tsx:44,46` and `assessment/page.tsx:11,13`), plus adding `cooperativeness` to the profile type (`:27-34`) and trait list (`:172-180`).
- **Docs:** `backend/ai/README.md:110`.
- **Guard:** add a drift test that asserts every value literal in the frontend radar maps and the generated prompt enums is in the rubric's `allowed_values`.

### B7. The cooperativeness topic

Add a new topic **9** to `prompts/interview_system.txt` after topic 8 (`:45`), and change "8" to "9" at `:29`, `:74` and `:96`. The wording is drawn from the rubric's cooperativeness anchors (`rubric_v1.yaml:296-306`: responds to teammates' messages, checks in, engages with and amplifies others' contributions). It avoids the banned words at `:10`:

> 9. Showing up for teammates: when a teammate posts in the group chat asking for input or help, what do they actually do, and how soon? When a teammate shares an idea they're unsure about, how do they respond? Ask for a specific recent example.

Suggested probes, phrased neutrally and past-tense first:
- "Think of the last group project. When someone posted a question in the group chat, what did you usually do?"
- "Tell me about a time a teammate suggested something you weren't sure about. What did you say?"

Topic 8 ("Making space for others") stays as written. Which dimension it feeds is TODO-7.

### B8. Every file that would change

| File | Change | Runtime? |
|---|---|---|
| `backend/scoring/interview_adapter.py` (new) | `EvidenceItem`, `evidence_to_signals`, the decision record | No on its own (pure); becomes runtime when called |
| `backend/app/conversation.py` | `/extract` rewritten per B3; new `/conversation/{session_id}/evidence` reviewer endpoint; prompt built from rubric; `response_schema` | **Yes.** Changes every interview-derived profile, and therefore team formation. |
| `prompts/behavioral_assessment.txt` → new `prompts/evidence_extraction.txt` | evidence-only instructions; enums and strength definitions filled in from the rubric | **Yes** (LLM input) |
| `prompts/interview_system.txt` | add topic 9; "8" → "9" at `:29`, `:74`, `:96` | **Yes.** Longer interviews (see TODO-13). |
| `backend/app/database.py` | new columns on `EvidenceUnitDB` (`confidence`, `needs_review`, `decision`, `quote_start`) and `ExtractionRunDB` (`rubric_version`, `prompt_hash`, `raw_response`) | **Yes** (schema) |
| `backend/alembic/versions/<new>.py` | migration for those nullable columns | **Yes** (schema; additive) |
| `backend/app/models.py` | Pydantic `EvidenceItem` / response schema; `ExtractResponse`; optional `measurement_coverage` on `ConfidenceLayer` | **Yes** (API shape) |
| shared helper (e.g. `backend/scoring/apply.py`, new) | `apply_scored_to_student` | Only for paths that call it (TODO-10) |
| `backend/app/behavioral_analysis.py` | *only if* TODO-10 says to switch MCQ to the shared helper (adds `safety_contribution` and `stress_response` writes) | **Yes**, if done |
| `frontend/app/profile/[id]/page.tsx` | `:44`, `:46` value fixes; add `cooperativeness` (`:27-34`, `:172-180`); replace per-dimension confidence bars (`:90-92`, `:185-193`) with coverage and unknown markers | **Yes** (UI) |
| `frontend/app/assessment/page.tsx` | `:11`, `:13` value fixes | **Yes** (UI; radar values for `assertive`/`developing` profiles change from the 0.5 fallback to the intended numbers) |
| `frontend/app/interview/page.tsx` | probably none: it only reads `data.student_id` (`:97-98`) | No |
| `backend/tests/test_interview_adapter.py` (new) | adapter unit tests: each rejection reason, strength→weight, confidence-invariance, deterministic order, cap/dedupe | No |
| `backend/tests/test_conversation_extract.py` (new) | `/extract` with Gemini mocked: labels come from `score()`, `under_determined_dims` set, evidence persisted, `rescore_run` reproduces the stored profile | No |
| `backend/tests/test_vocabulary_drift.py` (new) | rubric ↔ generated prompt ↔ frontend maps | No |
| `backend/ai/README.md`, `docs/conversation_pipeline_decisions.md` | docs follow-up | No |

**Not touched:** `dimension_scorer.py` (`score()` is used as is), `rubric_v1.yaml` (no new numbers; `evidence_strength` is used as written), `matching.py`, and `mcq_adapter.py`.

**Suggested order:**
1. Adapter and its tests (pure; nothing else changes)
2. Migration
3. Prompt and `/extract` rewrite, behind the mocked-LLM tests
4. Frontend value fixes and `cooperativeness` row
5. Interview topic 9
6. Reviewer endpoint

---

## Part C. TODO(team-decision)

Each item lists the options and a **safe default** the plan uses until the team decides.

- **TODO-1 (team-decision): one quote supporting more than one dimension.**
  - Options: (a) allow it, with the LLM emitting one item per dimension for the same quote, each scored independently; (b) one dimension per quote, so the LLM must choose; (c) allow it but split the weight across dimensions.
  - **Default: (a).** Scores are per-dimension, so one quote can't double-count *within* a dimension, and nothing evidential is thrown away. Each item gets its own `evidence_units` row. The §6 cap still applies per (dimension, indicator).
- **TODO-2 (team-decision): a dimension the conversation gives no evidence for.**
  - Options: (a) mark it unknown: `score()` lists it in `_unknown_dims`, it is stored in `under_determined_dims`, the matcher excludes it from floor and compatibility, and `default_value` is display-only (this is exactly how MCQ behaves); (b) have the interviewer probe the gap before closing (a coverage stopping rule, which the decision doc lists as waiting on the adapter); (c) refuse to finalize the profile.
  - **Default: (a).** It matches MCQ and the existing matcher contract. (b) is a later improvement.
  - Note: per `matching.py:352-355`, an unknown floor dimension does **not** count as a floor violation. That is existing TODO #3 in `CHANGES.md`, and it matters more here because `cooperativeness` will often be unknown until topic 9 is in place.
- **TODO-3 (team-decision): the per-evidence confidence threshold for "needs review".**
  - Options: (a) no confidence-based flag yet: store confidence, flag only structurally (rejected, or contradicting on a floor dimension), and have the reviewer sort by confidence; (b) a fixed threshold in the rubric (e.g. a new `review.confidence_threshold`); (c) a per-dimension threshold; (d) a threshold calibrated against human annotations (`human_annotations` table) once a pilot exists.
  - **Default: (a)**, with the threshold read from config and defaulting to "off", so no number is made up. LLM self-reported confidence is uncalibrated, which is the reason in the README decision record for keeping it out of scoring.
- **TODO-4 (team-decision): how strict the verbatim-quote check is.**
  - Options: exact substring; normalize whitespace, case and curly quotes; fuzzy match above a threshold.
  - **Default:** normalize whitespace, case and quote characters, then require an exact substring of **student** turns only. Reject anything else, and keep the row so a reviewer can see it.
- **TODO-5 (team-decision): meaning of "direction" for ordinal dimensions.**
  - Options: (a) `indicator` = a specific allowed level plus `polarity` (supporting or contradicting), matching the existing `EvidenceUnitDB` columns; (b) `up`/`down` relative to the middle level.
  - **Default: (a).** It fits `_ordinal_mean`, which averages level positions, and the existing schema.
- **TODO-6 (team-decision): near-duplicate threshold for dedupe** (decision doc §6 says "tune empirically").
  - **Default:** treat a quote that is a substring of another for the same dimension and indicator as a duplicate, with no Jaccard rule yet, and cap at 2 per indicator as already decided.
- **TODO-7 (team-decision): which dimension topic 8 ("Making space for others") feeds.**
  - Options: `safety_contribution`, `cooperativeness`, or both.
  - **Default:** add the dedicated topic 9 for cooperativeness and leave topic 8 unmapped in the prompt. The extraction LLM tags each quote by dimension anyway, so topics don't hard-map to dimensions. This ties to the deferred fold-or-keep decision in decision doc §3.
  - Related: `execution_style` and `help_seeking` have no interview topic either. Out of scope here, flagged.
- **TODO-8 (team-decision): plurality ties.** Today a tie goes to the first-seen value (`dimension_scorer.py:117-122`). Decision doc §1 decided "tie → unknown plus a consistency flag", but that isn't built yet and it is shared L4 code (it changes MCQ too).
  - **Default for this change:** keep `score()` unchanged and use the deterministic signal order from B2 step 5, so ties are at least reproducible. Do the §1 change as its own step.
- **TODO-9 (team-decision): `skill_depth` and `experience_level`.** Today the LLM's `skill_depth` is copied into `experience_level` (`conversation.py:276`), which feeds the ability fallback (`matching.py:1260`). That is an LLM-asserted label reaching an (optional) optimizer input.
  - Options: (a) keep it, documented as a self-report fact; (b) store `skill_depth` for display only and leave `experience_level` at its default; (c) bring skill depth into the rubric and evidence scheme.
  - **Default: (b).** It keeps LLM labels out of anything the optimizer can read.
- **TODO-10 (team-decision): should the MCQ path switch to the shared `apply_scored_to_student` helper now?** Switching makes MCQ also write `safety_contribution` and `stress_response`, which are weight 0 and always unknown on MCQ.
  - **Default:** no. Only the interview path uses the helper in this change, so MCQ behavior stays byte-identical. Switch MCQ over in a separate change.
- **TODO-11 (team-decision): `confidence_layer` naming during the transition.**
  - Options: write both `measurement_coverage` and the legacy `confidence_score` key; or rename outright and update every reader (frontend `profile/[id]/page.tsx:36,90`, `matching.py:4185` copy, `evaluation/`).
  - **Default:** write both keys with the same value until all readers move over. `dimension_scores.confidence` becomes `NULL` for new rows, or could hold per-dimension coverage if `score()` is extended to expose it (decision doc §4 asks for that).
- **TODO-12 (team-decision): what the student sees.**
  - Options: (a) labels plus measurement coverage plus an "not enough information" marker on unknown dimensions; (b) also show supporting quotes; (c) also show confidence.
  - **Default: (a).** Per-evidence confidence and `needs_review` are reviewer-only.
- **TODO-13 (team-decision): interview length.** Adding topic 9 lengthens interviews. The hard cap is `len(request.history) >= 22` (`conversation.py:109`).
  - **Default:** keep the cap and measure it on pilot transcripts. Raising it is a cost and latency decision.
- **TODO-14 (team-decision): old interview-derived profiles** (LLM-labelled, written before this change).
  - Options: re-score them from stored transcripts (needs an LLM re-run, so the evidence would differ); flag them as `legacy_llm_labels`; exclude them from study data.
  - **Default:** flag them and exclude them from study data. The local DB currently has 0 such profiles (`responses` rows with `scenario_id='conversational_interview'` = 0), but other machines may have some.
- **TODO-15 (team-decision): what to do when the LLM output is malformed or fails the schema.**
  - Options: retry (`_call_with_retry` already exists, unused, `conversation.py:344`); fail with 502 and write nothing; save the raw response with zero accepted items.
  - **Default:** retry on 429/503, otherwise 502, but **always** persist the `extraction_runs` row with `raw_response`, so failures are auditable.
