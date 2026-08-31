# How This System Works

*A guided walkthrough for anyone new to this repo — no assumed background.*

This document explains three things: the problem this project is trying to solve, the system built to
address it, and how data actually flows through the code, layer by layer. It reflects the code as it
exists right now on branch `feat/gemini-integration`, including a piece that's mid-migration (see
[A pipeline in transition](#a-pipeline-in-transition) below) — this is not a polished pitch, it's a map.

---

## 1. The problem

When instructors put students into project teams, they're usually optimizing for the wrong thing.
Most tools — and most instructor intuition — group students by **skills** ("we need a frontend person
and a data person") or **schedules** ("these three are all free Tuesday evenings"). What that approach
misses is *how people actually work together*: whether someone follows through on commitments, how
they behave when a teammate drops the ball, whether they take charge or wait to be told what to do.

Research on team dynamics backs this up. Disagreement over *how* work should get done (process
conflict) tends to hurt teams more than disagreement over *what* to do. And the single most common
failure mode in student project teams is accountability asymmetry — one or two people carrying the
group while others coast. None of that shows up in a skills list or a shared free-time slot.

So the project's working hypothesis is: **team formation is a behavioral compatibility problem, not
just a scheduling and skills problem.** If you can build a reliable picture of how a student tends to
behave in a team — before the team exists — you can use that to form teams more deliberately, and to
tell instructors *why* a given grouping should work (or where it might not).

The harder problem underneath that is **measurement**. If you just ask a student "are you a team
player?", they'll say yes — everyone rates themselves well on self-report surveys (social desirability
bias). Multiple-choice options with labels like "avoidant" / "collaborative" have the same issue in a
sneakier form: if the option text and its scoring label are basically the same thing, the system isn't
really inferring anything, it's just relabeling a checkbox. This project's central research bet is that
a **conversation** — asking about specific past situations and probing vague answers — produces a more
honest signal than either of those. That bet is what's currently being built out (see below).

---

## 2. The solution, at a glance

```
 Student                          Instructor
    │                                  │
    ▼                                  │
Assessment (MCQ today,                 │
conversational interview               │
in progress)                           │
    │                                  │
    ▼                                  │
Behavioral Signature                   │
(a structured profile:                 │
 work rhythm, collaboration            │
 style, skills, confidence)            │
    │                                  │
    ▼                                  ▼
     Student Pool  ──────────►  Team Formation Engine
                                (greedy + local-search
                                 optimization over
                                 skills + behavior +
                                 schedule + interests)
                                        │
                                        ▼
                              Generated Teams + Explanations
                              (score breakdown, radar chart,
                               strengths, risks, working norms)
                                        │
                                        ▼
                              Instructor reviews & finalizes
```

Every student who completes an assessment gets a **behavioral signature** — not a single score, but a
profile across several independent dimensions (how they plan, how they communicate, how they handle
conflict, how reliably they follow through, and more). The **matching engine** then treats team
formation as an optimization problem: build teams that are strong on skill coverage, behaviorally
compatible, schedule-compatible, and share some interests — using instructor-adjustable weights. Every
generated team comes back with a plain-language explanation, not just a score, because the system is
built to be **instructor-in-the-loop**: it recommends, the instructor decides.

---

## 3. The layered architecture

The backend is deliberately split into layers, each with one job, so that "what counts as good
behavior" (a research question) is never hard-coded inside "how do we compute a match" (an
engineering question). This is documented in-repo as L0–L5:

| Layer | What it is | Where it lives |
|---|---|---|
| **L0 — Rubric** | The single source of truth for every behavioral dimension: its allowed values, why it's measured, what evidence supports it, and how it should count toward team composition. | [`backend/traits/rubric_v1.yaml`](../backend/traits/rubric_v1.yaml) |
| **L1 — Question bank** | The scenarios shown to students, and — server-side only — which answer implies which rubric value. | [`backend/questions/bank_v1.yaml`](../backend/questions/bank_v1.yaml) |
| **L2 — Elicitation** | The actual interaction that collects evidence from a student: the MCQ form, or the conversational interview. | [`backend/app/behavioral_analysis.py`](../backend/app/behavioral_analysis.py), [`backend/app/conversation.py`](../backend/app/conversation.py) |
| **L3 — Signal extraction** | Turning a raw response (an option pick, or a chunk of interview transcript) into discrete, typed "signal units" — *this trait, this value, this much confidence*. | [`backend/scoring/mcq_adapter.py`](../backend/scoring/mcq_adapter.py) |
| **L4 — Scoring** | Aggregating signal units into a final behavioral signature per student. | [`backend/scoring/dimension_scorer.py`](../backend/scoring/dimension_scorer.py) |
| **L5 — Downstream use** | Everything that *consumes* a finished profile: team matching, explanations, norms, radar charts. | [`backend/app/matching.py`](../backend/app/matching.py), [`backend/app/explainability.py`](../backend/app/explainability.py) |

The rule that makes this worth doing: **no dimension definitions live anywhere except the L0 rubric.**
Every prompt, scoring function, and matching rule reads from `rubric_v1.yaml` at runtime instead of
hard-coding "accountability has three levels." That means a researcher can change what's measured, or
how much it should count, by editing one YAML file — not by hunting through five source files.

---

## 4. A pipeline in transition

This repo currently contains **two working assessment pipelines**, because the project is mid-pivot
from one measurement approach to another. Understanding both — and which one is "live" — matters for
anyone picking this up.

### Pipeline A — MCQ assessment (shipped, linked from the homepage)

This is the original design, and it's the one a student hits today via `/assessment`.

1. `GET /assessment/scenarios` returns 6 scenario prompts (a missed deadline, a workload imbalance, a
   leadership vacuum, etc.), each with 4 answer options. The option IDs sent to the browser are
   **anonymous** (`opt_a`…`opt_d`) — the client never sees which option maps to which trait value, so
   there's no way to game the form by reading the DOM.
2. The student picks one option per scenario. `POST /assessment/analyze` looks up, server-side, what
   each chosen option *signals* (e.g. "chose `opt_c` on `missed_deadline`" → `conflict_style:
   collaborative`, `cooperativeness: high`) via [`mcq_adapter.py`](../backend/scoring/mcq_adapter.py).
3. Those signals are aggregated into a final profile by
   [`dimension_scorer.py`](../backend/scoring/dimension_scorer.py) — categorical traits are resolved by
   plurality vote across signals, ordinal traits (like accountability) by a weighted mean. **No LLM is
   involved in this scoring step at all** — it's pure, deterministic code, which means the same answers
   always produce the same profile, and the logic is fully auditable.
4. The resulting signature — plus which dimensions had *zero* signal (`under_determined_dims`) and a
   `confidence_score` (how much of the profile is actually backed by evidence vs. left at a default) —
   is written to the student's record, tagged with the exact rubric/bank/scorer version that produced
   it.

The known limitation, and the reason for Pipeline B: a multiple-choice option is a canned answer. Even
with anonymized IDs, the system is choosing among a small number of pre-written behaviors rather than
observing an open-ended one — it can't ask "wait, what did you actually say to them?"

### Pipeline B — Conversational interview (in progress, at `/interview`, not yet linked from the homepage)

This is the pivot: replace canned multiple-choice answers with a real back-and-forth conversation,
scored by an LLM (Gemini) instead of a fixed rubric lookup.

1. `POST /conversation/start` opens a chat session with Gemini, seeded by a detailed interviewer system
   prompt ([`prompts/interview_system.txt`](../prompts/interview_system.txt)). That prompt is worth
   reading directly — it's the actual research instrument. It instructs the model to: stay neutral (a
   banned-word list keeps it from saying "collaborative" or "proactive" and tipping off what a good
   answer sounds like), always ask for *specific past behavior* over hypotheticals, probe vague answers
   ("what did you actually say?"), and cover 8 behavioral dimensions plus skills and availability before
   closing — without ever telling the student which dimension is being probed.
2. `POST /conversation/turn` continues the exchange, one student message at a time (capped at ~22 turns
   so it can't run away).
3. `POST /conversation/extract` sends the full transcript through a second prompt
   ([`prompts/behavioral_assessment.txt`](../prompts/behavioral_assessment.txt)) that asks Gemini to
   read the conversation and output a structured profile as JSON — this time *with* per-dimension
   confidence scores and supporting evidence quotes, stored in dedicated tables
   (`dimension_scores`, `evidence_units`) for later research analysis.

Because this path is LLM-scored rather than rubric-scored, it's intentionally logged more heavily than
Pipeline A: every extraction run, prompt version, and profile version is recorded
(`extraction_runs`, `profile_versions` in [`database.py`](../backend/app/database.py)), so a later
study can compare conversational profiles against MCQ profiles for the same students.

**Bottom line:** if you're exploring the repo, the MCQ path is what actually runs end-to-end today for
a typical demo; the interview path is the direction the research is moving, functionally complete, but
not yet the front door.

---

## 5. Team formation: turning profiles into teams

Once students have profiles (from either pipeline), an instructor calls `POST /teams/generate` with a
target team size and a set of dimension weights. [`matching.py`](../backend/app/matching.py) does the
rest:

- **Skill coverage (default 40% weight)** — how much of a fixed reference skill list (frontend,
  backend, ML, testing, design, etc.) the team collectively covers.
- **Behavioral compatibility (default 30%)** — computed dimension-by-dimension, using whatever
  *composition rule* the L0 rubric assigns to that dimension, not a single hard-coded formula:
  - `high_mean_with_floor` (used for `accountability`, `cooperativeness`): maximize the team's
    average level, and *forbid* putting more than one weak-on-this-dimension student on the same team.
    This directly encodes the "bad apple" research finding — one low-accountability member can drag a
    whole team down, so the system spreads such students round-robin across teams rather than letting
    optimization accidentally cluster them.
  - `soft_compatibility` (used for `leadership_style`, `conflict_style`, etc.): a mild *reward* for
    having a mix of styles on a team rather than all-identical — not a penalty for mismatch.
- **Availability overlap (default 20%)** and **shared interests (default 10%)** — straightforward
  overlap between members' free-time slots and stated interests/goals.

The optimizer itself runs in three passes: greedy construction (seed each team, then repeatedly add
whichever remaining student most improves that team's score), a 2-opt pass (try swapping pairs of
students between teams if it improves both), and a maximin pass that specifically evens out the
floor-protected dimensions across teams so no single team is left worse off than the others. Before any
of this runs, the system checks feasibility — if there are more low-accountability students than
teams, for instance, it refuses to generate teams rather than silently producing an unfair split.

Every generated team also gets an **explanation** on demand
([`GET /teams/{id}/explanation`](../backend/app/explainability.py)) — a Gemini call that turns the raw
score breakdown and member profiles into plain-language strengths, risks, and five suggested team
working norms (communication cadence, conflict protocol, etc.), plus a five-axis radar chart comparing
team members visually. This is generated lazily, not baked into team formation, specifically so an LLM
call never sits inside the optimization loop.

---

## 6. Where things live, quickly

```
backend/
  traits/rubric_v1.yaml         L0 — what's measured and why
  questions/bank_v1.yaml        L1 — MCQ scenarios + hidden answer→trait mapping
  scoring/
    mcq_adapter.py               L3 — MCQ answers → signal units
    dimension_scorer.py          L4 — signal units → final profile (deterministic)
  app/
    behavioral_analysis.py       MCQ assessment endpoints (Pipeline A)
    conversation.py              Interview endpoints (Pipeline B, Gemini-backed)
    matching.py                  Team formation optimizer
    explainability.py            Team explanations, norms, radar chart (Gemini-backed)
    models.py                    All data shapes (Student, Team, Signature, ...)
    database.py                  SQLite schema, incl. research/provenance tables
  ai/
    gemini_analyzer.py / gemini_explainer.py   Real Gemini-backed implementations
    mock_analyzer.py / mock_explainer.py       Deterministic fallbacks (no API key needed)
    interfaces.py                              The abstract contract both must satisfy

frontend/app/
  assessment/     MCQ flow (Pipeline A) — linked from the homepage today
  interview/      Conversational flow (Pipeline B) — built, not yet linked from the homepage
  dashboard/      Instructor: adjust weights, generate teams
  teams/, team/[id]/   Generated team list and per-team detail + explanation
  profile/[id]/   Individual student profile view

prompts/
  interview_system.txt           The live interview instrument (Pipeline B)
  behavioral_assessment.txt      Transcript → structured profile extraction (Pipeline B)
  explanation_generation.txt     Team score breakdown → strengths/risks/explanation
```

---

## 7. A note for the non-technical reader

If the layers above are too much detail, the short version is:

1. A student answers questions about how they work in teams — either by picking from multiple choice
   options, or (in the newer, in-progress version) by having an actual typed conversation with an AI
   interviewer.
2. Their answers get turned into a structured "behavioral profile" — not a single grade, but a set of
   independent traits like *how reliably do they follow through* and *how do they handle disagreement*.
3. An instructor picks how much each trait should matter, and the system proposes teams that balance
   skills, behavior, schedules, and interests — while specifically avoiding known bad patterns, like
   putting two unreliable students on the same team.
4. Every proposed team comes with a plain-English explanation, not just a score, so the instructor can
   sanity-check and override it before anything is final. The AI recommends; the human decides.
