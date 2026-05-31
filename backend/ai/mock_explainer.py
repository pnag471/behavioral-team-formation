"""
Rule-based mock implementations of ExplanationGenerator and NormGenerator.
Replace with LLM-backed classes to upgrade the prototype.
See interfaces.py for the contract.
"""
from __future__ import annotations
from typing import List, TYPE_CHECKING
from collections import Counter

from ai.interfaces import ExplanationGenerator, NormGenerator

if TYPE_CHECKING:
    from app.models import Student, Team, TeamExplanation, TeamNorm

# ---------------------------------------------------------------------------
# Team norm templates keyed by behavioral majority
# ---------------------------------------------------------------------------
_NORMS: dict[str, dict[str, tuple[str, str]]] = {
    "communication": {
        "async":  ("Communication",
                   "Default to async — use written channels for non-urgent items; reserve live meetings for decisions only."),
        "sync":   ("Communication",
                   "Hold a 30-minute standup every other day; document decisions in a shared notes file."),
        "mixed":  ("Communication",
                   "Use async for status updates; schedule two synchronous check-ins per week."),
    },
    "conflict": {
        "collaborative": ("Conflict Resolution",
                          "Surface disagreements early in retrospectives; use structured debate — each person states their concern before proposing a solution."),
        "confrontational": ("Conflict Resolution",
                            "Establish a 24-hour cooling-off rule before escalating technical disagreements to the full team."),
        "avoidant": ("Conflict Resolution",
                     "Assign a rotating 'concern-raiser' role each sprint — that person is explicitly tasked with naming tensions before they accumulate."),
    },
    "leadership": {
        "directive":    ("Decision Making",
                         "Designate a rotating decision-owner per milestone; major decisions require async sign-off from all members within 24 hours."),
        "facilitative": ("Decision Making",
                         "Use consent-based decisions: proceed unless someone raises a concrete objection within 24 hours."),
        "emergent":     ("Decision Making",
                         "Open each sprint with a 15-minute planning session to distribute ownership explicitly — don't assume someone will step up."),
    },
    "planning": {
        "planner":      ("Planning",
                         "Maintain a shared milestone tracker; surface deadline changes at least 48 hours in advance."),
        "spontaneous":  ("Planning",
                         "Set only one-week lookahead goals; re-plan at each sync meeting rather than enforcing a fixed schedule."),
        "adaptive":     ("Planning",
                         "Keep a rolling two-week plan; flag blockers at least three days before hard deadlines."),
    },
    "accountability": {
        "high":     ("Accountability",
                     "Self-reported progress updates every two days in the shared channel; no surprises at deadline."),
        "medium":   ("Accountability",
                     "Weekly structured check-in with explicit status: Done / In Progress / Blocked."),
        "low":      ("Accountability",
                     "Pair each task with a named owner and a due date; review ownership assignments at every sync meeting."),
    },
}


def _majority(values: List[str]) -> str:
    """Return the most common value in a list."""
    return Counter(values).most_common(1)[0][0]


def _build_norms(students: "List[Student]") -> "List[TeamNorm]":
    from app.models import TeamNorm

    comm    = _majority([s.work_rhythm_signature.communication_style for s in students])
    conflict = _majority([s.collaboration_signature.conflict_style for s in students])
    leader  = _majority([s.collaboration_signature.leadership_style for s in students])
    plan    = _majority([s.work_rhythm_signature.planning_style for s in students])
    account = _majority([s.collaboration_signature.accountability for s in students])

    # Map directive majority → "directive" key even if only 1 person is directive
    directive_count = sum(1 for s in students if s.collaboration_signature.leadership_style == "directive")
    if directive_count >= 1 and leader != "directive":
        leader = "directive"

    keys = [
        ("communication", comm),
        ("conflict", conflict),
        ("leadership", leader),
        ("planning", plan),
        ("accountability", account),
    ]

    norms = []
    for dimension, key in keys:
        category, norm_text = _norms_lookup(dimension, key)
        norms.append(TeamNorm(category=category, norm=norm_text))
    return norms


def _norms_lookup(dimension: str, key: str) -> tuple[str, str]:
    bucket = _NORMS.get(dimension, {})
    if key in bucket:
        return bucket[key]
    # fallback to first entry
    return list(bucket.values())[0]


class MockExplainer(ExplanationGenerator, NormGenerator):

    def suggest_norms(self, team: "Team", students: "List[Student]") -> "List[TeamNorm]":
        return _build_norms(students)

    def generate(self, team: "Team", students: "List[Student]") -> "TeamExplanation":
        from app.models import TeamExplanation

        sb = team.score_breakdown
        member_names = [m.name for m in team.members]
        name_list = ", ".join(member_names[:-1]) + f" and {member_names[-1]}" if len(member_names) > 1 else member_names[0]

        # ---- Strengths ----
        strengths: list[str] = []
        if sb.skill_coverage >= 0.7:
            skill_union = set()
            for s in students:
                skill_union.update(s.competence_signature.skills)
            strengths.append(
                f"Strong skill breadth — the team covers {len(skill_union)} distinct domains including "
                + ", ".join(list(skill_union)[:3]) + "."
            )
        if sb.behavioral_compat >= 0.7:
            strengths.append("High behavioral compatibility: working styles complement each other across planning, communication, and execution dimensions.")
        facilitative = [s for s in students if s.collaboration_signature.leadership_style == "facilitative"]
        if facilitative:
            strengths.append(f"{facilitative[0].name} brings a facilitative leadership style, well-suited for keeping the team aligned without micromanaging.")
        if sb.availability_overlap >= 0.6:
            strengths.append("Good scheduling alignment — the team shares enough availability windows for regular synchronous collaboration.")
        collab = [s for s in students if s.collaboration_signature.conflict_style == "collaborative"]
        if len(collab) >= len(students) // 2:
            strengths.append("Majority collaborative conflict style — the team is likely to surface and resolve disagreements constructively.")
        high_account = [s for s in students if s.collaboration_signature.accountability == "high"]
        if len(high_account) >= len(students) - 1:
            strengths.append("Near-universal high accountability — members are likely to follow through on commitments without external tracking.")
        if not strengths:
            strengths.append("Team members bring a mix of perspectives that can spark creative problem-solving.")

        # ---- Risks ----
        risks: list[str] = []
        directive = [s for s in students if s.collaboration_signature.leadership_style == "directive"]
        if len(directive) >= 2:
            risks.append(
                f"Multiple directive leaders ({', '.join(s.name for s in directive)}) may create friction around ownership — establish clear decision authority early."
            )
        avoidant = [s for s in students if s.collaboration_signature.conflict_style == "avoidant"]
        if len(avoidant) >= 2:
            risks.append("Conflict-avoidant members may let tensions accumulate — recommend a structured retrospective protocol.")
        if sb.skill_coverage < 0.5:
            risks.append("Limited skill breadth — consider pairing this team with a complementary group for cross-team collaboration.")
        if sb.availability_overlap < 0.35:
            risks.append("Low scheduling overlap — the team may struggle to find synchronous meeting times; establish async decision norms early.")
        low_account = [s for s in students if s.collaboration_signature.accountability == "low"]
        if low_account:
            risks.append(
                f"{low_account[0].name} has lower self-reported accountability — explicit task ownership and check-in cadences are recommended."
            )
        if sb.conflict_risk >= 0.6:
            risks.append("Elevated conflict risk from mixed conflict-resolution styles — invest in team norms during the first week.")
        if sb.match_confidence < 0.55:
            risks.append("Moderate match confidence — instructor review recommended before finalizing this grouping.")
        if not risks:
            risks.append("No major structural risks identified. Monitor team dynamics during the first milestone.")

        # ---- Prose explanation ----
        score_label = (
            "excellent" if sb.total >= 0.8
            else "strong" if sb.total >= 0.65
            else "moderate" if sb.total >= 0.5
            else "developing"
        )
        conf_label = (
            "high confidence" if sb.match_confidence >= 0.7
            else "moderate confidence" if sb.match_confidence >= 0.5
            else "lower confidence"
        )

        explanation = (
            f"This team of {len(students)} — {name_list} — demonstrates {score_label} overall compatibility "
            f"(score: {sb.total:.0%}) with {conf_label} in the matching result ({sb.match_confidence:.0%}). "
            f"The formation was optimized across skill coverage ({sb.skill_coverage:.0%}), "
            f"behavioral compatibility ({sb.behavioral_compat:.0%}), "
            f"availability overlap ({sb.availability_overlap:.0%}), "
            f"and conflict risk ({sb.conflict_risk:.0%} — lower is better). "
            f"Members bring complementary backgrounds in "
            + ", ".join(set(role for s in students for role in s.competence_signature.roles[:1]))
            + ". The suggested team norms below are derived algorithmically from the behavioral signatures "
            f"and are intended as a starting point for the team's working agreement."
        )

        norms = _build_norms(students)

        return TeamExplanation(
            team_id=team.id,
            compatibility_score=sb.total,
            match_confidence=sb.match_confidence,
            strengths=strengths,
            risks=risks,
            explanation=explanation,
            team_norms=norms,
        )
