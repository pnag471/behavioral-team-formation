# Behavioral Team Formation

> AI-assisted team formation using behavioral signatures, explainable matching, and instructor-guided team generation.
> **Research Prototype** · CS Department · 2026

---

## Overview

Most team formation tools sort students by skill lists or random assignment. This prototype takes a different approach: it builds a *behavioral signature* for each student by analyzing how they respond to realistic team scenarios, then optimizes team composition across four dimensions — skill coverage, behavioral compatibility, availability, and shared interests.

Every team assignment comes with a full explanation: compatibility score, match confidence rating, per-team behavioral radar chart, strengths, risk factors, and auto-generated working norms derived from the team's collective behavioral profile.

---

## Research Motivation

Team dynamics research consistently shows that behavioral misalignment — not technical skill gaps — is the primary cause of project failure in academic settings. This prototype operationalizes key constructs from that literature:

- **Conflict style compatibility** (Jehn & Mannix, 2001)
- **Accountability and accountability asymmetry** (Lencioni, 2002)
- **Communication channel matching** (Daft & Lengel, 1986)
- **Leadership style complementarity** (Carson et al., 2007)

The matching algorithm is transparent and auditable: every score is broken down by component, and the instructor can adjust component weights before generation.

---

## Architecture

```mermaid
flowchart TD
    Student["Student Browser"] -->|POST /assessment/analyze| BA["Behavioral Analysis Engine"]
    BA -->|keyword matching| MA["MockBehavioralAnalyzer\nai/mock_analyzer.py"]
    MA --> SS[("students_store")]

    Instructor["Instructor Browser"] -->|POST /teams/generate| ME["Matching Engine\nGreedy + 2-opt"]
    SS --> ME
    ME --> TS[("teams_store")]

    Instructor -->|GET /teams/{id}/explanation| EE["Explainability Engine"]
    EE -->|rule-based| MX["MockExplainer\nai/mock_explainer.py"]
    TS --> EE
    SS --> EE
    MX --> Instructor
```

See [docs/architecture.md](docs/architecture.md) for full documentation including the matching pipeline, scoring formula, radar chart mapping, and future LLM integration path.

---

## Screenshots

<!-- screenshot: landing page hero -->
<!-- screenshot: assessment scenarios -->
<!-- screenshot: assessment result with radar chart -->
<!-- screenshot: instructor dashboard with weight sliders -->
<!-- screenshot: generated teams list -->
<!-- screenshot: team detail with radar overlay and norms -->

---

## Local Setup

### Backend

```bash
cd backend
venv/bin/pip install -r requirements.txt   # first time only
venv/bin/uvicorn app.main:app --reload --port 8000
```

API available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### Frontend

```bash
cd frontend
npm install    # first time only
npm run dev
```

Frontend available at `http://localhost:3000`.

---

## Demo Workflow

1. **`/`** — Read the landing page overview
2. **`/assessment`** — Fill in your name, answer all 6 scenarios (select an option + optional free text), submit → see your behavioral signature and radar chart
3. **`/dashboard`** — View the student pool (20 seeded students + any from assessments), adjust matching weights, click Generate Teams
4. **`/teams`** — Browse the generated teams, see per-team metrics panel and match confidence
5. **`/team/team-1`** — Explore full team detail: members, radar chart overlay, 5-metric panel, team norms, strengths, risks, and prose explanation

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/students` | List all students |
| `GET`  | `/assessment/scenarios` | List 6 built-in scenarios |
| `POST` | `/assessment/analyze` | Infer signature + upsert student |
| `POST` | `/teams/generate` | Generate optimized teams |
| `GET`  | `/teams/teams` | List all generated teams |
| `GET`  | `/teams/{id}/explanation` | Team explanation + norms |
| `GET`  | `/teams/{id}/radar` | Radar chart data |

---

## Project Structure

```
behavioral-team-formation/
├── backend/
│   ├── app/
│   │   ├── main.py                  FastAPI app, shared state, seed loading
│   │   ├── models.py                All Pydantic v2 models
│   │   ├── behavioral_analysis.py   Assessment scenarios + analyze endpoint
│   │   ├── matching.py              Greedy optimizer + team generation
│   │   └── explainability.py        Explanation + radar endpoints
│   ├── ai/
│   │   ├── interfaces.py            Abstract base classes
│   │   ├── mock_analyzer.py         Keyword-based analyzer
│   │   ├── mock_explainer.py        Rule-based explainer + norm generator
│   │   └── README.md                LLM swap-in guide
│   ├── sample_students.json         20 diverse seed students
│   └── requirements.txt
├── frontend/
│   ├── app/
│   │   ├── page.tsx                 Landing page
│   │   ├── assessment/page.tsx      Assessment form
│   │   ├── dashboard/page.tsx       Instructor dashboard
│   │   ├── teams/page.tsx           Teams list
│   │   └── team/[id]/page.tsx       Team detail
│   ├── components/
│   │   ├── Navigation.tsx           Site header
│   │   ├── RadarChart.tsx           Pure SVG radar chart
│   │   ├── ScoreBar.tsx             Progress bar component
│   │   └── TeamMetricsPanel.tsx     5-metric panel
│   └── lib/
│       ├── types.ts                 TypeScript interfaces
│       └── api.ts                   Typed API client
├── docs/
│   └── architecture.md
└── prompts/
    ├── behavioral_assessment.txt    LLM prompt template (future)
    └── explanation_generation.txt   LLM prompt template (future)
```

---

## License

MIT © 2026 Prisha Nag
