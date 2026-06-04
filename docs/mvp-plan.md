# Behavioral Team Formation: MVP Build Plan

Build order: **1 → 2 → 3 + 4 (parallel) → 5 → 6 → 7**

Tasks 1 and 2 are hard blockers — nothing can be tested end-to-end until data persists and users exist.

---

## 1. Persistent Storage

Replace in-memory Python dicts with a real database.

- SQLite for dev, Postgres for production
- Tables: `students`, `competence_signatures`, `work_rhythm_signatures`, `collaboration_signatures`, `motivation_layers`, `assessment_sessions`, `scenario_responses`, `teams`, `team_members`
- Use Alembic for schema migrations so changes are tracked and reversible
- Seed script to load `sample_students.json` into the DB on first run (replaces the current startup-time dict population)
- Add `NEXT_PUBLIC_API_URL` env var to the frontend and replace the hardcoded `http://localhost:8000` in `api.ts` — nothing else can be shared across environments until this is done

---

## 2. Authentication

Students and instructors need accounts with distinct capabilities.

- Email + password registration and login
- JWT access tokens; two roles: `student` and `instructor`
- Protect all write endpoints; instructors-only endpoints (team generation, roster upload) require the `instructor` role
- Middleware that injects the authenticated user into every request so student submissions are automatically scoped to their account

---

## 3. Student Onboarding Flow

The assessment page exists but only captures behavioral scenarios — skills, availability, and roles are never collected from new students via the UI.

- Extend the existing `/assessment` flow into a multi-step funnel: Registration → Skills + availability form → Behavioral scenarios
- Skills form should collect: domain skills (checkboxes against the 12 reference skills), roles, experience level, availability windows (day + time-of-day slots), interests, and learning goals — all fields needed to populate the full student schema
- Progress indicator showing which step the student is on
- Each step saves to the DB before advancing — no partial data loss if the student drops off mid-flow
- Students who have already completed onboarding skip directly to their profile view

---

## 4. Assessment Pipeline

Wire the existing LLM scaffolding to persistent storage and replace the keyword-based mock analyzer with a real Claude API call.

- Swap `MockBehavioralAnalyzer` for a Claude-backed implementation using `prompts/behavioral_assessment.txt` — the abstract interface in `ai/interfaces.py` means no other code changes are required
- Enable prompt caching on the static scenario list prefix to reduce per-assessment cost
- Store every scenario response with `student_id`, `session_id`, and `timestamp`
- Save the generated behavioral signature with the model version used, so signatures can be re-scored if the model changes
- Mark sessions `complete` or `incomplete` — never discard partial data; allow resume
- Associate completed sessions with the authenticated student so the matching engine reads real profiles

---

## 5. Team Formation

Connect the matching engine to real database profiles instead of the in-memory mock data.

- Pull live student profiles from the DB at generation time, not from the startup seed
- Support at minimum two formation modes selectable from the dashboard: behavioral matching (existing algorithm) and random assignment (baseline for comparison)
- Store team assignments, per-team scores, and confidence scores in the DB so results survive a server restart
- Return a warning if any students have incomplete assessments — instructors should know which profiles have partial data before generating teams

---

## 6. Instructor Dashboard

The dashboard frontend exists with weight sliders and team generation, but reads no real data and has no roster management.

- CSV roster upload: parse name + email, create pending student accounts, trigger onboarding invite emails
- Per-student completion tracker: show each student's onboarding status (not started / in progress / complete) and assessment completion
- Team generation pulls from the DB and respects the weight parameters set in the UI
- Results view shows generated teams with scores; allow the instructor to re-run formation with different weights and compare results
- Lock formation once the instructor confirms — prevent accidental overwrites after teams have been communicated to students

---

## 7. Explainability Output

Replace template-based norms and rule-based strengths/risks with LLM-generated explanations.

- Swap `MockExplainer` for a Claude-backed implementation using `prompts/explanation_generation.txt` — same interface swap pattern as Task 4
- Each team gets: a one-paragraph plain-English compatibility summary, a dimension breakdown (skill coverage, behavioral fit, availability overlap, shared interests), and 2–4 specific risk factors with the member names involved
- Auto-generated team norms grounded in the actual behavioral profiles of that team, not generic templates
- Confidence score displayed prominently; flag teams below 0.55 with a callout prompting the instructor to review manually
- Cache explanations in the DB — don't regenerate on every page load
