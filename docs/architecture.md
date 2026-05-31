# Architecture: Behavioral Team Formation

## System Overview

Behavioral Team Formation is a research prototype that demonstrates AI-assisted student team assignment. The system moves beyond simple skill-matching by constructing *behavioral signatures* from scenario-based assessments, then optimizing team composition across four weighted dimensions.

The prototype is intentionally modular: every AI component is behind an abstract interface, making it trivial to replace mock logic with real LLM calls in a future iteration.

---

## High-Level Data Flow

```mermaid
flowchart TD
    Student["Student\n(browser)"] -->|POST /assessment/analyze| BA["Behavioral Analysis\napp/behavioral_analysis.py"]
    BA -->|keyword matching via| MA["MockBehavioralAnalyzer\nai/mock_analyzer.py"]
    MA -->|behavioral signature| SS[("students_store\nin-memory dict")]

    Instructor["Instructor\n(browser)"] -->|POST /teams/generate| ME["Matching Engine\napp/matching.py"]
    SS -->|student pool| ME
    ME -->|greedy optimization| TS[("teams_store\nin-memory dict")]

    Instructor -->|GET /teams/{id}/explanation| EE["Explainability Engine\napp/explainability.py"]
    EE -->|rule-based via| MX["MockExplainer\nai/mock_explainer.py"]
    TS --> EE
    SS --> EE
    MX -->|strengths · risks · norms · explanation| Instructor

    Instructor -->|GET /teams/{id}/radar| EE
    EE -->|RadarData| Instructor
```

---

## API Design

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/` | Health check |
| `GET`  | `/students` | List all students in the pool |
| `POST` | `/students` | Add a student |
| `GET`  | `/assessment/scenarios` | Return the 6 built-in scenarios |
| `POST` | `/assessment/analyze` | Infer behavioral signature; upsert student profile |
| `POST` | `/teams/generate` | Run greedy optimization; store teams |
| `GET`  | `/teams/teams` | List all generated teams |
| `GET`  | `/teams/{id}/explanation` | Get explanation, strengths, risks, norms for a team |
| `GET`  | `/teams/{id}/radar` | Get radar chart data (team average + per-member) |

All POST endpoints require `Content-Type: application/json`.

---

## Student Schema

Each student carries five nested signature objects:

```
Student
├── CompetenceSignature
│   ├── skills: List[str]          # domain skills
│   ├── roles: List[str]           # functional roles
│   └── experience_level: str      # beginner | intermediate | advanced
├── WorkRhythmSignature
│   ├── planning_style: str        # planner | spontaneous | adaptive
│   ├── communication_style: str   # async | sync | mixed
│   ├── execution_style: str       # methodical | iterative | exploratory
│   └── availability: List[str]    # e.g. Mon-morning, Wed-evening
├── CollaborationSignature
│   ├── conflict_style: str        # avoidant | confrontational | collaborative
│   ├── leadership_style: str      # directive | facilitative | emergent
│   ├── accountability: str        # high | medium | low
│   └── help_seeking: str          # proactive | reactive | independent
├── MotivationLayer
│   ├── interests: List[str]
│   └── learning_goals: List[str]
└── ConfidenceLayer
    └── confidence_score: float    # [0.0, 1.0]
```

---

## Matching Pipeline

### Step 1: Seed student selection
The student with the most distinct reference skills (the "connector") seeds each team.

### Step 2: Greedy member addition
For each open slot, add the unassigned student who maximizes the team's marginal weighted score.

### Step 3: 2-opt improvement
One pass of pairwise swaps between teams — if swapping member A (team i) and member B (team j) increases the combined score of both teams, perform the swap.

### Weighted Scoring Formula

```
total = w₁ × skill_coverage
      + w₂ × behavioral_compat
      + w₃ × availability_overlap
      + w₄ × shared_interests
```

Default weights: `w₁=0.40, w₂=0.30, w₃=0.20, w₄=0.10`

**Skill Coverage** (0–1): `|team_skills ∩ REFERENCE_SKILLS| / |REFERENCE_SKILLS|`
where `REFERENCE_SKILLS` = {frontend, backend, ml, data analysis, project management, testing, design, devops, algorithms, communication, documentation, research}

**Behavioral Compatibility** (0–1): Average pairwise compatibility across all C(n,2) pairs, scored on conflict style (35%), leadership style (30%), planning style (20%), and accountability (15%).

**Availability Overlap** (0–1): Average pairwise Jaccard similarity on availability slot lists.

**Shared Interests** (0–1): Jaccard similarity across the union of all members' interests + learning goals.

### Match Confidence Score

```
relative_position = (team_score - min_score) / (max_score - min_score)
confidence = 0.40 + relative_position × 0.52   # range [0.40, 0.92]
```
Scores cluster near each other → spread < 0.05 → confidence capped at 0.45 (near-tie pool).

### Conflict Risk (diagnostic, not in total)

```
conflict_risk = mismatch_rate × 0.7 + directive_penalty
```
Where `mismatch_rate` = fraction of pairs with mismatched conflict styles (excluding collaborative buffer), and `directive_penalty` = `max(0, (directive_count - 1) × 0.15)`.

---

## AI Interface Contract

The `ai/` directory decouples algorithmic logic from the rest of the application.

```
ai/
  interfaces.py      ← ABC definitions
  mock_analyzer.py   ← keyword-based BehavioralAnalyzer
  mock_explainer.py  ← rule-based ExplanationGenerator + NormGenerator
```

To upgrade to a real LLM: create a new class implementing the ABC, pass the prompt templates from `prompts/`, and swap the instantiation in `behavioral_analysis.py` / `explainability.py`. No other changes required.

---

## Behavioral Radar Chart

Five axes, each mapped from a categorical value to [0, 1]:

| Axis | Dimension | Low (0) → High (1) |
|------|-----------|---------------------|
| Communication | communication_style | async (0.35) → sync (0.90) |
| Conflict Handling | conflict_style | avoidant (0.20) → collaborative (0.85) |
| Leadership | leadership_style | emergent (0.30) → directive (0.92) |
| Accountability | accountability | low (0.20) → high (0.90) |
| Planning | planning_style | spontaneous (0.20) → planner (0.90) |

Rendered as pure SVG (no external chart library) with team average polygon overlay and per-member dashed traces.

---

## Future AI Integration

### Phase 2: LLM Behavioral Analysis

Replace `MockBehavioralAnalyzer.analyze()` with a call to the Claude API using `prompts/behavioral_assessment.txt`. Expected latency: 1–3 seconds per assessment. Enable prompt caching on the scenario list (static prefix) to reduce cost.

### Phase 3: LLM Explanation Generation

Replace `MockExplainer.generate()` with a Claude call using `prompts/explanation_generation.txt`. The structured JSON output format ensures reliable downstream parsing.

### Phase 4: Longitudinal Feedback Loop

After project completion, collect instructor-rated team performance scores and use them to fine-tune the behavioral compatibility weights via a simple regression — closing the feedback loop between behavioral prediction and actual team outcomes.

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.12 · FastAPI 0.136.3 · Pydantic v2 · Uvicorn |
| Frontend | Next.js 16.2.6 · React 19 · TypeScript · Tailwind CSS v4 |
| AI (mock) | Keyword matching · Rule-based templates |
| AI (future) | Anthropic Claude API |
| Storage | In-memory Python dicts (prototype) |
