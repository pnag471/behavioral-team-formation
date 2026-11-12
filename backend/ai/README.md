# AI Layer

This directory contains the abstract interfaces and mock implementations for all AI-powered components.

## Design rules (do not break these)

The AI and the fixed code have separate jobs, and they never overlap.

**The AI only reads language.** It takes a student's answers and pulls out
evidence: which of the student's own words point to which trait, and in which
direction. It quotes those words. It never outputs a score and never decides
teams.

**The fixed code does all the deciding.** The scorer (dimension_scorer.py +
rubric_v1.yaml) turns the AI's evidence into trait scores using fixed rules.
The optimizer builds teams from those scores. Both are deterministic: the same
inputs always give the same scores and the same teams.

Why: every score traces back to the student's own words, and the same data
always produces the same result. That is what makes the system auditable and
reproducible, which is the core claim of the paper.

**Personality type is cosmetic.** Any "type" label on a profile is just a
friendly name derived from the trait scores. It is never an input to scoring
or to team formation.

## Decision record

Decision (updated 2026-11-10, with Bernice): Confidence is a per-evidence REVIEW
FLAG only. It flags weak evidence for human review and never weights scores.
(Supersedes conversation_pipeline_decisions.md §4, which dropped confidence; the
team has now landed on keep-as-review-flag.)

Determinism: the scorer and optimizer are deterministic given the same STORED
evidence. LLM extraction is NOT deterministic, so we persist the evidence from each
run and reproduce scoring and team formation from it. Reproducibility comes from
saving the evidence, not from the LLM.

Strength is a SEPARATE field and DOES weight the score. It must be a grounded
classification (specific example vs general habit vs hypothetical) with anchored
example responses per level. Exemplars are not filled in yet (rubric: "filled later
by psych faculty"), so strength is currently UNVALIDATED. Dependency: Thao / Liesel.

Known boundary violation to fix: POST /conversation/extract (conversation.py) writes
LLM labels straight to the student DB with no scorer between. Live interview path.
Tracked for the refactor.

Naming (to stop the two "confidence" fields getting mixed up):

- `measurement_coverage` = rubric's old confidence_score (dimension signal coverage)
- per-evidence review flag = the Option B confidence above

Option B is settled (agreed with Bernice and Sandipan). TODO(team-decision): get the okay to fix or retire gemini_analyzer.py. Until then, point the demo at the Claude analyzer path only, so you’re never running through the one with the old boundary bug.

## Architecture

```
ai/
  interfaces.py      ← Abstract base classes (the contract)
  mock_analyzer.py   ← Keyword-based BehavioralAnalyzer (no LLM)
  mock_explainer.py  ← Rule-based ExplanationGenerator + NormGenerator (no LLM)
```

## How to swap in a real LLM

### 1. Behavioral Analysis

`mock_analyzer.py` uses keyword matching to infer behavioral signatures from free-text responses. To replace with Claude or another LLM:

```python
# ai/claude_analyzer.py
import anthropic
from ai.interfaces import BehavioralAnalyzer

class ClaudeBehavioralAnalyzer(BehavioralAnalyzer):
    def __init__(self):
        self.client = anthropic.Anthropic()

    def analyze(self, request):
        prompt = open("prompts/behavioral_assessment.txt").read()
        responses_json = json.dumps([r.model_dump() for r in request.responses])
        filled = prompt.replace("{responses}", responses_json)

        message = self.client.messages.create(
            model="claude-opus-4-8",
            max_tokens=512,
            messages=[{"role": "user", "content": filled}]
        )
        return json.loads(message.content[0].text)
```

Then in `app/behavioral_analysis.py`, replace:
```python
analyzer = MockBehavioralAnalyzer()
```
with:
```python
analyzer = ClaudeBehavioralAnalyzer()
```

### 2. Explanation & Norms Generation

`mock_explainer.py` uses rule-based logic. To replace with Claude:

```python
# ai/claude_explainer.py
class ClaudeExplainer(ExplanationGenerator, NormGenerator):
    def generate(self, team, students):
        prompt = open("prompts/explanation_generation.txt").read()
        # Fill in team data, call client.messages.create(...)
        # Parse response into TeamExplanation
        ...
```

Then in `app/explainability.py`, replace `MockExplainer()` with `ClaudeExplainer()`.

## Interface Contract

### `BehavioralAnalyzer.analyze(request) -> dict`

Returns a dict with exactly these keys:
- `planning_style`: `"planner" | "spontaneous" | "adaptive"`
- `communication_style`: `"async" | "sync" | "mixed"`
- `execution_style`: `"methodical" | "iterative" | "exploratory"`
- `conflict_style`: `"avoidant" | "assertive" | "collaborative"`
- `leadership_style`: `"directive" | "facilitative" | "emergent"`
- `accountability`: `"high" | "developing" | "low"`
- `help_seeking`: `"proactive" | "reactive" | "independent"`
- `confidence_score`: `float` in `[0.0, 1.0]`

### `ExplanationGenerator.generate(team, students) -> TeamExplanation`

Returns a `TeamExplanation` with prose explanation, strengths, risks, and team norms.

### `NormGenerator.suggest_norms(team, students) -> List[TeamNorm]`

Returns a list of `TeamNorm` objects (one per behavioral dimension).
