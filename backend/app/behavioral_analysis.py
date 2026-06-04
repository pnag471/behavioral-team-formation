from __future__ import annotations
import time
import uuid
from typing import List

from fastapi import APIRouter, HTTPException

from app.models import (
    AssessmentRequest,
    CollaborationSignature,
    CompetenceSignature,
    ConfidenceLayer,
    MotivationLayer,
    SaveProgressRequest,
    Scenario,
    ScenarioOption,
    StartSessionRequest,
    Student,
    WorkRhythmSignature,
)
from ai.claude_analyzer import ClaudeBehavioralAnalyzer, MODEL_ID

router = APIRouter(prefix="/assessment", tags=["assessment"])

analyzer = ClaudeBehavioralAnalyzer()

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
            ScenarioOption(id="avoidant",       label="Wait and hope they catch up on their own"),
            ScenarioOption(id="confrontational", label="Directly call out the issue in the group chat"),
            ScenarioOption(id="collaborative",   label="Message them privately to understand what happened"),
            ScenarioOption(id="directive",       label="Reassign the task immediately and notify the team"),
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
            ScenarioOption(id="avoidant",       label="Keep working hard and hope others notice"),
            ScenarioOption(id="confrontational", label="Raise it bluntly in the next meeting"),
            ScenarioOption(id="collaborative",   label="Bring it up with the team as a process problem to solve together"),
            ScenarioOption(id="directive",       label="Propose a formal task-tracking system with assigned owners"),
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
            ScenarioOption(id="avoidant",       label="Continue working on my part and wait for others to reconnect"),
            ScenarioOption(id="confrontational", label="Send a firm message demanding everyone respond"),
            ScenarioOption(id="facilitative",   label="Schedule a quick sync and set up a shared status board"),
            ScenarioOption(id="collaborative",   label="Check in with each person individually to understand what's going on"),
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
            ScenarioOption(id="directive",     label="Take charge immediately — assign roles and set a plan"),
            ScenarioOption(id="facilitative",  label="Organize a meeting to collectively decide on a coordination structure"),
            ScenarioOption(id="emergent",      label="Wait to see if someone naturally steps up"),
            ScenarioOption(id="collaborative", label="Suggest a rotating leadership model so nobody is burdened"),
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
            ScenarioOption(id="confrontational", label="Argue my case strongly until the team agrees"),
            ScenarioOption(id="avoidant",        label="Let them have it — it's not worth the conflict"),
            ScenarioOption(id="collaborative",   label="Propose a structured comparison — pros/cons doc and a team vote"),
            ScenarioOption(id="directive",       label="Escalate to the instructor or a neutral third party to decide"),
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
            ScenarioOption(id="directive",     label="Make reasonable assumptions, document them, and move forward"),
            ScenarioOption(id="facilitative",  label="Organize a requirements clarification session with the whole team"),
            ScenarioOption(id="collaborative", label="Reach out to the instructor or stakeholder directly for clarification"),
            ScenarioOption(id="avoidant",      label="Wait until more information becomes available"),
        ],
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_version() -> str:
    return MODEL_ID if analyzer._fallback is None else "mock"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/scenarios", response_model=List[Scenario])
def get_scenarios():
    return SCENARIOS


@router.post("/start")
def start_session(request: StartSessionRequest):
    """Create a student record and an incomplete assessment session."""
    from app.database import SessionLocal, StudentDB, AssessmentDB

    student_id = request.student_id or f"s{int(time.time() * 1000)}"
    session_id = str(uuid.uuid4())

    db = SessionLocal()
    try:
        existing = db.query(StudentDB).filter(StudentDB.id == student_id).first()
        if not existing:
            db.add(StudentDB(
                id=student_id,
                name=request.student_name,
                competence_signature={"skills": [], "roles": [], "experience_level": "intermediate"},
                work_rhythm_signature={"planning_style": "adaptive", "communication_style": "mixed",
                                       "execution_style": "iterative", "availability": []},
                collaboration_signature={"conflict_style": "collaborative", "leadership_style": "facilitative",
                                         "accountability": "medium", "help_seeking": "proactive"},
                motivation_layer={"interests": [], "learning_goals": []},
                confidence_layer={"confidence_score": 0.5},
            ))

        db.add(AssessmentDB(
            id=session_id,
            student_id=student_id,
            status="incomplete",
            model_version=_model_version(),
        ))
        db.commit()
    finally:
        db.close()

    return {"session_id": session_id, "student_id": student_id}


@router.post("/save-progress")
def save_progress(request: SaveProgressRequest):
    """Upsert partial responses without finalising the session."""
    from app.database import SessionLocal, ResponseDB

    db = SessionLocal()
    try:
        for response in request.responses:
            db.merge(ResponseDB(
                id=f"{request.session_id}:{response.scenario_id}",
                assessment_id=request.session_id,
                student_id=request.student_id,
                scenario_id=response.scenario_id,
                option_id=response.option_id,
                response_text=response.response_text,
            ))
        db.commit()
    finally:
        db.close()

    return {"saved": len(request.responses)}


@router.get("/session/{session_id}")
def get_session(session_id: str):
    """Return session status and all saved responses."""
    from app.database import SessionLocal, AssessmentDB, ResponseDB

    db = SessionLocal()
    try:
        session = db.query(AssessmentDB).filter(AssessmentDB.id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
        responses = db.query(ResponseDB).filter(ResponseDB.assessment_id == session_id).all()
        return {
            "session_id": session_id,
            "student_id": session.student_id,
            "status": session.status,
            "responses": [
                {
                    "scenario_id": r.scenario_id,
                    "option_id": r.option_id,
                    "response_text": r.response_text,
                }
                for r in responses
            ],
        }
    finally:
        db.close()


@router.post("/analyze")
def analyze_assessment(request: AssessmentRequest):
    from app.database import SessionLocal, StudentDB, AssessmentDB, ResponseDB, BehavioralSignatureDB

    sig = analyzer.analyze(request)

    db = SessionLocal()
    try:
        student_id = request.student_id or f"s{int(time.time() * 1000)}"

        existing = db.query(StudentDB).filter(StudentDB.id == student_id).first()
        if existing:
            existing.work_rhythm_signature = {
                **existing.work_rhythm_signature,
                "planning_style": sig["planning_style"],
                "communication_style": sig["communication_style"],
                "execution_style": sig["execution_style"],
            }
            existing.collaboration_signature = {
                **existing.collaboration_signature,
                "conflict_style": sig["conflict_style"],
                "leadership_style": sig["leadership_style"],
                "accountability": sig["accountability"],
                "help_seeking": sig["help_seeking"],
            }
            existing.confidence_layer = {"confidence_score": sig["confidence_score"]}
        else:
            db.add(StudentDB(
                id=student_id,
                name=request.student_name,
                work_rhythm_signature={
                    "planning_style": sig["planning_style"],
                    "communication_style": sig["communication_style"],
                    "execution_style": sig["execution_style"],
                    "availability": [],
                },
                collaboration_signature={
                    "conflict_style": sig["conflict_style"],
                    "leadership_style": sig["leadership_style"],
                    "accountability": sig["accountability"],
                    "help_seeking": sig["help_seeking"],
                },
                confidence_layer={"confidence_score": sig["confidence_score"]},
                competence_signature={"skills": [], "roles": [], "experience_level": "intermediate"},
                motivation_layer={"interests": [], "learning_goals": []},
            ))

        # Resolve or create the session record
        if request.session_id:
            session = db.query(AssessmentDB).filter(AssessmentDB.id == request.session_id).first()
            if not session:
                raise HTTPException(status_code=404, detail=f"Session '{request.session_id}' not found.")
            session.status = "complete"
            session.model_version = _model_version()
            session_id = request.session_id
        else:
            session_id = str(uuid.uuid4())
            db.add(AssessmentDB(
                id=session_id,
                student_id=student_id,
                status="complete",
                model_version=_model_version(),
            ))

        for response in request.responses:
            db.merge(ResponseDB(
                id=f"{session_id}:{response.scenario_id}",
                assessment_id=session_id,
                student_id=student_id,
                scenario_id=response.scenario_id,
                option_id=response.option_id,
                response_text=response.response_text,
            ))

        db.merge(BehavioralSignatureDB(
            student_id=student_id,
            signature=sig,
            model_version=_model_version(),
        ))

        db.commit()
    finally:
        db.close()

    return {"student_id": student_id, "session_id": session_id, "signature": sig}
