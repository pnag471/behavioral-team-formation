# Decision doc: wiring the conversation pipeline (L2/L3) into L4 scoring

Status: decided, ready to implement.

Supersedes: slide 9 of the EMNLP deck (stale framing of soft_compatibility as always "not
better/worse" — no longer true after decision #1 below).

---

## 1. Unknown detection

**Decision:** count-based. A dimension is `unknown` when it has zero evidence units after
the verbatim check (§2) filters out unverifiable ones. No confidence-threshold gating —
self-reported per-dimension confidence is dropped from the extraction contract entirely
(see §4); display shows scorer evidence coverage instead.

**Plurality ties:** a genuine tie (equal total weight across ≥2 values) does **not** fall
back to first-listed-value — that would silently bias every tie toward whichever value is
first in `allowed_values` (`directive`, `planner`, `avoidant` all happen to be first in
their lists). A tie marks the dimension `unknown` and adds a `consistency_flag`. This is a
change to `score()`/`_plurality()` in `dimension_scorer.py` — **shared L4 code**, so it
also changes MCQ-path tie behavior, not just the conversation path. No existing test
depends on tie behavior (checked `test_dimension_scorer.py`).

## 2. Prompt/rubric sync

**Decision:** generate the per-dimension `allowed_values` enum list (and the dimension
list itself) from `rubric_v1.yaml` at request time. Hand-written explanatory prose stays
in the prompt file for now (rubric has no per-value description field yet).

**Extraction contract changes shape:** the LLM no longer asserts a final label per
dimension. It emits only **evidence units** — `{trait, indicator, polarity, strength,
quote}` — and `score()` computes the final value. Use `response_schema` (confirmed
supported, google-genai 2.8.0) with enums on `trait`/`polarity`/`strength`. `indicator`
is NOT schema-enum-constrained per-trait (too fiddly cross-field) — validated instead at
the adapter level against that trait's `allowed_values`, writing `rejected=true` to
`evidence_units` on mismatch (this column already exists for exactly this).

Run the rubric/prompt drift test (§6) regardless, as a backstop.

**Provenance:** `ExtractionRunDB` gets new columns — `rubric_version`, `prompt_hash`.
`prompt_version` is currently hardcoded `"1.0"`; stop hardcoding it. New alembic
migration (nullable columns, trivial in SQLite).

## 3. Orphan dimensions (safety_contribution, stress_response)

**Decision:** add both as rubric entries, `matching_weight: 0.0`, `composition_rule:
null`. Keep native value sets (`high|medium|low`, `composed|reactive|withdraws`) — do
NOT force them into the `low|developing|high` BARS naming reserved for CATME-grounded
dims (accountability, cooperativeness). Mark `evidence_basis: weak`.

**Landmine avoided:** `_floor_dims()` filters only on `composition_rule ==
"high_mean_with_floor"` AND `type == "ordinal"` — it does **not** check weight. If either
field were given `type: ordinal` + `high_mean_with_floor` (the intuitive-looking choice),
it becomes a hard floor constraint regardless of weight 0, risking the same
infeasible-roster bug already flagged as Critical Item #1 in `recommended_changes.md`.
`composition_rule: null` is safe regardless of `type`, and safe against a future
weight-bump mistake (unlike gating on weight alone).

**Deferred (needs real conversation data, not a build-time decision):** whether to fold
safety_contribution into cooperativeness, or weight either nonzero. Their overlap is
real but incomplete — cooperativeness's anchors are about responsiveness/checking in;
safety_contribution's description also carries "admits mistakes" / "credit-taking",
closer to Edmondson (1999) psychological safety than CATME cooperativeness.

## 4. Confidence

**Decision:** drop self-reported per-dimension confidence from the extraction contract
entirely — once the LLM isn't asserting a label, there's no single inference for a
confidence number to attach to. Display the scorer's per-dimension evidence coverage
instead. Requires `dimension_scorer.score()` to expose coverage per-dimension (today it
only returns one value averaged across all dims) — small additive change, benefits the
MCQ path too.

`StudentDB.confidence_layer.confidence_score` switches from Gemini's self-reported
`overall_confidence` to the scorer's coverage-based value, for consistency with the
MCQ path's existing redefinition (CHANGES.md item 11). This changes the displayed number
for existing conversation-path profiles.

## 5. Contradicting evidence

**Decision:** a unit that explicitly argues against another unit's indicator (same
dimension) is excluded from scoring, logged to `evidence_units` (`polarity:
contradicting`), and raised as a `consistency_flag`. Two merely-different supporting
units for different values is normal plurality voting, not a contradiction — handled by
§1's tie logic when they tie.

No partial-weight cancellation — would require negative-weight support in `_plurality`/
`_ordinal_mean`, not worth it for this pass.

**UI:** contradiction flags on the two floor dimensions (accountability, cooperativeness)
get surfaced, not just logged — a hidden bad-apple signal is worse than a hidden style
mismatch. Bundle with the existing `floor_violations` surfacing gap (neither is shown in
the frontend today — recommended_changes.md Critical Item #1). This is the demo's
headline feature ("working conversation → auditable profile") — raised priority.

## 6. Verbosity bias

**Decision:** weight-by-unit-count rewards talkative students. Cap counted evidence
units at **2 per indicator**. Dedupe near-duplicate spans before counting: two units
for the same trait+indicator are the same unit if one quote is a substring of the other,
or word-overlap (Jaccard) exceeds a threshold — **threshold value TBD, tune empirically
once real transcripts exist, not guessed now.**

No formal bias-audit doc exists yet (checked — only a passing mention in
`docs/pipeline.md`). Verbosity gets tracked as a new open item, not appended to
something that already exists.

## 7. Strength definitions

**Decision:** move the weak/medium/strong definitions out of prompt-author prose and
into the rubric as a single shared (not per-dimension) policy block — e.g. "specific
past example = strong; stated general habit = medium; hypothetical = weak." Interpolated
into the generated prompt. Reusable later for the human annotation codebook.

## 8. Exemplars

**Decision:** add an empty `exemplars` field to the schema now (per allowed_value).
Content deferred — written later by psych faculty, not guessed at here.

---

## Build sequence

1. Rubric edits: weight-0 orphan entries (§3), strength definitions (§7), empty
   exemplars field (§8), fit-type decision (already committed — `1aead45`).
2. Extraction rewrite: generate prompt from rubric, evidence-only output, include
   cooperativeness.
3. Adapter + migration: verbatim check, rejected units, write to `evidence_units`,
   signals → `score()`, `under_determined_dims`, unit cap (§6), provenance columns (§2).
4. Tests: drift test, extraction contract test (mocked LLM), end-to-end test (mocked
   LLM, expected to fail at the scorer today), roster generator.
5. Frontend: evidence-to-score view + `floor_violations` surfacing (§5).
6. Small fixes: configurable `REFERENCE_SKILLS`, replace stale `docs/architecture.md`.

## Explicitly not doing

- No new optimizer passes (four optional ones already exist, off by default).
- No splitting `matching.py` (4,693 lines, actively edited by two people — a split
  guarantees the next merge isn't as clean as the last one).
- No embedding-based classifier, unless exemplars turn out insufficient.

## Waiting on research / data, not a build blocker now

conflict_style bank redesign (needs psych input on task vs. relationship conflict) ·
safety_contribution/stress_response fold-or-weight (§3) · ability-balancing default-on,
strong/weak pairing (needs faculty sign-off — still a hypothesis) · non-legacy default
profile choice · trait-coverage stopping rule (needs adapter landed first, then a
latency/cost call) · familiarity/preference intake flows (needs study design) ·
fluency/length bias audit, codebook, match_confidence calibration (needs annotation
+ pilot) · speed-dating test design (needs the pilot).
