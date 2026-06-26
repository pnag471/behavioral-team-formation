from __future__ import annotations
import uuid
import time
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
from scoring.dimension_scorer import score, get_rubric, SCORER_VERSION
from scoring.mcq_adapter import responses_to_signals, get_scenarios_for_client, get_bank

router = APIRouter(prefix="/assessment", tags=["assessment"])


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/scenarios", response_model=List[Scenario])
def get_scenarios():
    """Return scenarios with anonymous option IDs (no trait-value leakage)."""
    return [
        Scenario(
            id=s["id"],
            title=s["title"],
            description=s["description"],
            options=[ScenarioOption(id=o["id"], label=o["label"]) for o in s["options"]],
        )
        for s in get_scenarios_for_client()
    ]


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
                collaboration_signature={
                    "conflict_style": "collaborative", "leadership_style": "facilitative",
                    "accountability": "developing", "help_seeking": "proactive",
                    "cooperativeness": "developing",
                },
                motivation_layer={"interests": [], "learning_goals": []},
                confidence_layer={"confidence_score": 0.5},
                under_determined_dims=[],
            ))

        db.add(AssessmentDB(
            id=session_id,
            student_id=student_id,
            status="incomplete",
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
            existing = (
                db.query(ResponseDB)
                .filter_by(assessment_id=request.session_id, scenario_id=response.scenario_id)
                .first()
            )
            if existing:
                existing.option_id = response.option_id
                existing.response_text = response.response_text
            else:
                db.add(ResponseDB(
                    id=str(uuid.uuid4()),
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
    """
    Score MCQ responses via the deterministic L4 scorer (no LLM in this path).

    Provenance stored: rubric_version, bank_version, scorer_version.
    """
    from app.database import SessionLocal, StudentDB, AssessmentDB, ResponseDB, BehavioralSignatureDB

    rubric = get_rubric()
    bank = get_bank()

    # L3 → L4: convert option choices to SignalUnits, then score deterministically
    signals = responses_to_signals(request.responses)
    scored = score(signals, rubric)

    unknown_dims: List[str] = scored.pop("_unknown_dims", [])

    db = SessionLocal()
    try:
        student_id = request.student_id or f"s{int(time.time() * 1000)}"

        existing = db.query(StudentDB).filter(StudentDB.id == student_id).first()
        if existing:
            existing.work_rhythm_signature = {
                **existing.work_rhythm_signature,
                "planning_style": scored["planning_style"],
                "communication_style": scored["communication_style"],
                "execution_style": scored["execution_style"],
            }
            existing.collaboration_signature = {
                **existing.collaboration_signature,
                "conflict_style": scored["conflict_style"],
                "leadership_style": scored["leadership_style"],
                "accountability": scored["accountability"],
                "help_seeking": scored["help_seeking"],
                "cooperativeness": scored["cooperativeness"],
            }
            existing.confidence_layer = {"confidence_score": scored["confidence_score"]}
            existing.under_determined_dims = unknown_dims
        else:
            db.add(StudentDB(
                id=student_id,
                name=request.student_name,
                work_rhythm_signature={
                    "planning_style": scored["planning_style"],
                    "communication_style": scored["communication_style"],
                    "execution_style": scored["execution_style"],
                    "availability": [],
                },
                collaboration_signature={
                    "conflict_style": scored["conflict_style"],
                    "leadership_style": scored["leadership_style"],
                    "accountability": scored["accountability"],
                    "help_seeking": scored["help_seeking"],
                    "cooperativeness": scored["cooperativeness"],
                },
                confidence_layer={"confidence_score": scored["confidence_score"]},
                competence_signature={"skills": [], "roles": [], "experience_level": "intermediate"},
                motivation_layer={"interests": [], "learning_goals": []},
                under_determined_dims=unknown_dims,
            ))

        # Resolve or create the session record
        if request.session_id:
            session = db.query(AssessmentDB).filter(AssessmentDB.id == request.session_id).first()
            if not session:
                raise HTTPException(status_code=404, detail=f"Session '{request.session_id}' not found.")
            session.status = "complete"
            session_id = request.session_id
        else:
            session_id = str(uuid.uuid4())
            db.add(AssessmentDB(
                id=session_id,
                student_id=student_id,
                status="complete",
            ))

        for response in request.responses:
            existing_resp = (
                db.query(ResponseDB)
                .filter_by(assessment_id=session_id, scenario_id=response.scenario_id)
                .first()
            )
            if existing_resp:
                existing_resp.option_id = response.option_id
                existing_resp.response_text = response.response_text
            else:
                db.add(ResponseDB(
                    id=str(uuid.uuid4()),
                    assessment_id=session_id,
                    student_id=student_id,
                    scenario_id=response.scenario_id,
                    option_id=response.option_id,
                    response_text=response.response_text,
                ))

        # Archive the scored signature with full provenance
        db.add(BehavioralSignatureDB(
            id=str(uuid.uuid4()),
            student_id=student_id,
            signature=scored,
            rubric_version=rubric["schema_version"],
            bank_version=bank["bank_version"],
            scorer_version=SCORER_VERSION,
        ))

        db.commit()
    finally:
        db.close()

    return {"student_id": student_id, "session_id": session_id, "signature": scored}
