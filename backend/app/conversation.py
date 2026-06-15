from __future__ import annotations
import json
import os
from typing import List
from fastapi import APIRouter, HTTPException
from app.models import (
    ConversationStartRequest,
    ConversationTurnRequest,
    ConversationResponse,
    ConversationTurn,
)

router = APIRouter(prefix="/conversation", tags=["conversation"])

# Load prompts
_INTERVIEW_SYSTEM_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "prompts", "interview_system.txt"
)
_EXTRACTION_PROMPT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "prompts", "behavioral_assessment.txt"
)

with open(_INTERVIEW_SYSTEM_PATH) as f:
    INTERVIEW_SYSTEM = f.read()

with open(_EXTRACTION_PROMPT_PATH) as f:
    EXTRACTION_PROMPT = f.read()


def _get_gemini_client():
    from google import genai
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set")
    return genai.Client(api_key=api_key)


@router.post("/start", response_model=ConversationResponse)
def start_conversation(request: ConversationStartRequest):
    """Start a new interview session — returns the first interviewer message."""
    from app.database import SessionLocal, AssessmentDB
    import uuid
    from google.genai import types

    client = _get_gemini_client()
    session_id = str(uuid.uuid4())

    # Inject student name into system prompt
    first_name = request.student_name.strip().split()[0]
    system = INTERVIEW_SYSTEM.replace("[name]", first_name)

    # Get first message from interviewer
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[{
            "role": "user",
            "parts": [{"text": f"Start the interview. The student's name is {request.student_name}."}]
        }],
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.7,
        )
    )

    first_message = response.text

    # Save session to DB
    db = SessionLocal()
    try:
        db.add(AssessmentDB(
            id=session_id,
            student_id=f"pending_{session_id}",
            status="incomplete",
            model_version="gemini-2.5-flash",
        ))
        db.commit()
    finally:
        db.close()

    return ConversationResponse(
        message=first_message,
        is_complete=False,
        session_id=session_id,
    )


@router.post("/turn", response_model=ConversationResponse)
def conversation_turn(request: ConversationTurnRequest):
    """Process one student message and return the next interviewer message."""
    from google.genai import types

    # Hard turn cap — force close before building or calling anything
    if len(request.history) >= 16:
        return ConversationResponse(
            message="That's everything I needed — thanks for being so open about all of this. I'll put together your profile now.",
            is_complete=True,
            session_id=request.session_id,
        )

    client = _get_gemini_client()
    first_name = request.student_name.strip().split()[0]
    system = INTERVIEW_SYSTEM.replace("[name]", first_name)

    contents = []
    for turn in request.history:
        role = "model" if turn.role == "interviewer" else "user"
        contents.append({
            "role": role,
            "parts": [{"text": turn.content}]
        })

    contents.append({
        "role": "user",
        "parts": [{"text": request.message}]
    })

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.7,
            max_output_tokens=150,
        )
    )

    usage = response.usage_metadata
    print(f"Turn {len(request.history) + 1} | "
        f"in: {usage.prompt_token_count} | "
        f"out: {usage.candidates_token_count} | "
        f"total: {usage.total_token_count}")

    interviewer_message = response.text

    is_complete = any(phrase in interviewer_message.lower() for phrase in [
        "that's everything i needed",
        "thanks for being open",
        "i'll put together your profile",
        "based on our conversation",
    ])

    return ConversationResponse(
        message=interviewer_message,
        is_complete=is_complete,
        session_id=request.session_id,
    )


@router.post("/extract")
def extract_signature(session_id: str, student_name: str, history: List[ConversationTurn]):
    """Extract behavioral signature from completed conversation."""
    from app.database import SessionLocal, StudentDB, AssessmentDB, BehavioralSignatureDB, ResponseDB
    from google.genai import types
    import uuid
    import time

    client = _get_gemini_client()

    # Format transcript
    transcript = ""
    for turn in history:
        label = "Interviewer" if turn.role == "interviewer" else student_name
        transcript += f"{label}: {turn.content}\n\n"

    # Fill extraction prompt
    prompt = EXTRACTION_PROMPT \
        .replace("{transcript}", transcript) \
        .replace("{student_name}", student_name)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
        )
    )

    sig = json.loads(response.text)

    # Build student record
    student_id = f"s{int(time.time() * 1000)}"

    db = SessionLocal()
    try:
        db.add(StudentDB(
            id=student_id,
            name=student_name,
            competence_signature={
                "skills": sig.get("primary_skills", []),
                "roles": [],
                "experience_level": sig.get("skill_depth", "intermediate"),
                "skill_depth": sig.get("skill_depth", "intermediate"),
                "skill_no_gos": sig.get("skill_no_gos", []),
            },
            work_rhythm_signature={
                "planning_style": sig.get("planning_style", "adaptive"),
                "communication_style": sig.get("communication_style", "mixed"),
                "execution_style": sig.get("execution_style", "iterative"),
                "availability": [],
                "unavailable_times": sig.get("unavailable_times", []),
                "timezone_notes": sig.get("timezone_notes", ""),
            },
            collaboration_signature={
                "conflict_style": sig.get("conflict_style", "collaborative"),
                "leadership_style": sig.get("leadership_style", "facilitative"),
                "accountability": sig.get("accountability", "medium"),
                "help_seeking": sig.get("help_seeking", "proactive"),
                "safety_contribution": sig.get("safety_contribution", "medium"),
                "stress_response": sig.get("stress_response", "composed"),
            },
            motivation_layer={"interests": [], "learning_goals": []},
            confidence_layer={
                "confidence_score": sig.get("overall_confidence", 0.5),
                "confidence_scores": sig.get("confidence_scores", {}),
                "consistency_flags": sig.get("consistency_flags", []),
            },
        ))

        # Update assessment session
        session = db.query(AssessmentDB).filter(
            AssessmentDB.id == session_id
        ).first()
        if session:
            session.status = "complete"
            session.student_id = student_id

        # Save full transcript as a single response
        db.add(ResponseDB(
            id=str(uuid.uuid4()),
            assessment_id=session_id,
            student_id=student_id,
            scenario_id="conversational_interview",
            option_id="n/a",
            response_text=transcript,
        ))

        # Save signature
        db.merge(BehavioralSignatureDB(
            student_id=student_id,
            signature=sig,
            model_version="gemini-2.5-flash",
        ))

        db.commit()
    finally:
        db.close()

    return {
        "student_id": student_id,
        "session_id": session_id,
        "signature": sig,
    }