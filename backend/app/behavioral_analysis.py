from __future__ import annotations
import time
from typing import List

from fastapi import APIRouter, HTTPException

from app.models import (
    AssessmentRequest,
    CollaborationSignature,
    CompetenceSignature,
    ConfidenceLayer,
    MotivationLayer,
    Scenario,
    ScenarioOption,
    Student,
    WorkRhythmSignature,
)
from ai.mock_analyzer import MockBehavioralAnalyzer

router = APIRouter(prefix="/assessment", tags=["assessment"])

analyzer = MockBehavioralAnalyzer()

# ---------------------------------------------------------------------------
# Built-in scenarios
# ---------------------------------------------------------------------------
SCENARIOS: List[Scenario] = [
    Scenario(
        id="missed_deadline",
        title="Teammate Missed a Deadline",
        description=(
            "A teammate was responsible for a critical deliverable due yesterday. "
            "They haven't sent any update and the team is now blocked. "
            "How do you respond?"
        ),
        options=[
            ScenarioOption(id="avoidant",        label="Wait and hope they catch up on their own"),
            ScenarioOption(id="confrontational",  label="Directly call out the issue in the group chat"),
            ScenarioOption(id="collaborative",    label="Message them privately to understand what happened"),
            ScenarioOption(id="directive",        label="Reassign the task immediately and notify the team"),
        ],
    ),
    Scenario(
        id="workload_imbalance",
        title="Workload Imbalance",
        description=(
            "You realize you are doing significantly more work than your teammates. "
            "Two members seem to be coasting while you are pulling extra weight. "
            "How do you handle this?"
        ),
        options=[
            ScenarioOption(id="avoidant",        label="Keep working hard and hope others notice"),
            ScenarioOption(id="confrontational",  label="Raise it bluntly in the next meeting"),
            ScenarioOption(id="collaborative",    label="Bring it up with the team as a process problem to solve together"),
            ScenarioOption(id="directive",        label="Propose a formal task-tracking system with assigned owners"),
        ],
    ),
    Scenario(
        id="communication_breakdown",
        title="Communication Breakdown",
        description=(
            "The team has gone quiet. Messages are going unanswered, meetings are skipped, "
            "and nobody seems to know what others are working on. "
            "What is your move?"
        ),
        options=[
            ScenarioOption(id="avoidant",        label="Continue working on my part and wait for others to reconnect"),
            ScenarioOption(id="confrontational",  label="Send a firm message demanding everyone respond"),
            ScenarioOption(id="facilitative",     label="Schedule a quick sync and set up a shared status board"),
            ScenarioOption(id="collaborative",    label="Check in with each person individually to understand what's going on"),
        ],
    ),
    Scenario(
        id="leadership_vacuum",
        title="Leadership Vacuum",
        description=(
            "Your team has no designated leader and nobody is stepping up to coordinate the project. "
            "Decisions are stalling, work is duplicated, and the deadline is approaching. "
            "What do you do?"
        ),
        options=[
            ScenarioOption(id="directive",        label="Take charge immediately — assign roles and set a plan"),
            ScenarioOption(id="facilitative",     label="Organize a meeting to collectively decide on a coordination structure"),
            ScenarioOption(id="emergent",         label="Wait to see if someone naturally steps up"),
            ScenarioOption(id="collaborative",    label="Suggest a rotating leadership model so nobody is burdened"),
        ],
    ),
    Scenario(
        id="technical_disagreement",
        title="Technical Disagreement",
        description=(
            "You and a teammate strongly disagree on the technical approach for a core feature. "
            "Both solutions have merit but they are incompatible. The team needs to pick one. "
            "How do you handle it?"
        ),
        options=[
            ScenarioOption(id="confrontational",  label="Argue my case strongly until the team agrees"),
            ScenarioOption(id="avoidant",         label="Let them have it — it's not worth the conflict"),
            ScenarioOption(id="collaborative",    label="Propose a structured comparison — pros/cons doc and a team vote"),
            ScenarioOption(id="directive",        label="Escalate to the instructor or a neutral third party to decide"),
        ],
    ),
    Scenario(
        id="ambiguous_requirements",
        title="Ambiguous Requirements",
        description=(
            "The project brief is vague, the team is paralyzed by uncertainty, "
            "and everyone is waiting for someone to clarify the direction. "
            "What do you do?"
        ),
        options=[
            ScenarioOption(id="directive",        label="Make reasonable assumptions, document them, and move forward"),
            ScenarioOption(id="facilitative",     label="Organize a requirements clarification session with the whole team"),
            ScenarioOption(id="collaborative",    label="Reach out to the instructor or stakeholder directly for clarification"),
            ScenarioOption(id="avoidant",         label="Wait until more information becomes available"),
        ],
    ),
]


@router.get("/scenarios", response_model=List[Scenario])
def get_scenarios():
    return SCENARIOS


@router.post("/analyze")
def analyze_assessment(request: AssessmentRequest):
    from app.main import students_store

    sig = analyzer.analyze(request)

    # Build or update student in the store
    existing = students_store.get(request.student_id) if request.student_id else None

    if existing:
        # Update behavioral fields, preserve competence/motivation
        existing.work_rhythm_signature.planning_style = sig["planning_style"]
        existing.work_rhythm_signature.communication_style = sig["communication_style"]
        existing.work_rhythm_signature.execution_style = sig["execution_style"]
        existing.collaboration_signature.conflict_style = sig["conflict_style"]
        existing.collaboration_signature.leadership_style = sig["leadership_style"]
        existing.collaboration_signature.accountability = sig["accountability"]
        existing.collaboration_signature.help_seeking = sig["help_seeking"]
        existing.confidence_layer.confidence_score = sig["confidence_score"]
        student_id = existing.id
    else:
        student_id = request.student_id or f"s{int(time.time() * 1000)}"
        student = Student(
            id=student_id,
            name=request.student_name,
            work_rhythm_signature=WorkRhythmSignature(
                planning_style=sig["planning_style"],
                communication_style=sig["communication_style"],
                execution_style=sig["execution_style"],
                availability=[],
            ),
            collaboration_signature=CollaborationSignature(
                conflict_style=sig["conflict_style"],
                leadership_style=sig["leadership_style"],
                accountability=sig["accountability"],
                help_seeking=sig["help_seeking"],
            ),
            confidence_layer=ConfidenceLayer(
                confidence_score=sig["confidence_score"]
            ),
        )
        students_store[student_id] = student

    return {"student_id": student_id, "signature": sig}
