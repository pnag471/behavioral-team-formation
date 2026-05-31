# AI Layer

This directory contains the abstract interfaces and mock implementations for all AI-powered components.

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
- `conflict_style`: `"avoidant" | "confrontational" | "collaborative"`
- `leadership_style`: `"directive" | "facilitative" | "emergent"`
- `accountability`: `"high" | "medium" | "low"`
- `help_seeking`: `"proactive" | "reactive" | "independent"`
- `confidence_score`: `float` in `[0.0, 1.0]`

### `ExplanationGenerator.generate(team, students) -> TeamExplanation`

Returns a `TeamExplanation` with prose explanation, strengths, risks, and team norms.

### `NormGenerator.suggest_norms(team, students) -> List[TeamNorm]`

Returns a list of `TeamNorm` objects (one per behavioral dimension).
