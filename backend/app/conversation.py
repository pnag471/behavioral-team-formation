from __future__ import annotations
import hashlib
import json
import os
import re
import time
from collections import Counter
from typing import List, Optional
import uuid as uuid_lib
from fastapi import APIRouter, HTTPException
from pydantic import ValidationError
from app.models import (
    ConversationStartRequest,
    ConversationTurnRequest,
    ConversationResponse,
    ConversationTurn,
    EvidenceExtractionOutput,
    ExtractResponse,
    ReviewEvidenceItem,
    ReviewEvidenceResponse,
    ReviewEvidenceSummary,
)

from app import database as db_module
from app.database import (
    SessionLocal, AssessmentDB, StudentDB, ResponseDB,
    BehavioralSignatureDB, ExtractionRunDB, ProfileVersionDB,
    DimensionScoreDB, EvidenceUnitDB,
)
from scoring.apply import apply_scored_to_student
from scoring.dimension_scorer import SCORER_VERSION, get_rubric, score
from scoring.interview_adapter import (
    EvidenceDecision,
    EvidenceItem,
    evidence_to_signals,
    floor_dimensions,
)

_active_sessions: dict[str, object] = {}
_gemini_client = None
router = APIRouter(prefix="/conversation", tags=["conversation"])

# Load prompts
_INTERVIEW_SYSTEM_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "prompts", "interview_system.txt"
)
# Evidence-only extraction prompt. The old label-asserting prompt
# (prompts/behavioral_assessment.txt) is no longer used by this route; it is
# left in place because ai/gemini_analyzer.py and ai/claude_analyzer.py still
# load it (audit F12/F13).
_EXTRACTION_PROMPT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "prompts", "evidence_extraction.txt"
)

with open(_INTERVIEW_SYSTEM_PATH) as f:
    INTERVIEW_SYSTEM = f.read()

with open(_EXTRACTION_PROMPT_PATH) as f:
    EXTRACTION_PROMPT_TEMPLATE = f.read()

EXTRACTION_MODEL = "gemini-3.1-flash-lite"
EXTRACTION_TEMPERATURE = 0.1
EXTRACTION_RETRIES = 3
RETRY_DELAY_SECONDS = 10
# Student turns are joined with this before the adapter's verbatim check.
STUDENT_TURN_SEPARATOR = "\n\n"
# Safety backstop, not the normal close: the interview normally ends when the model reports coverage is complete (see prompts/interview_system.txt).
MAX_INTERVIEW_HISTORY = 60


def _get_gemini_client():
    global _gemini_client
    from google import genai
    if _gemini_client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set")
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


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

    # Create a native chat session
    chat = client.chats.create(
        model="gemini-3.1-flash-lite",
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.7,
            max_output_tokens=200,
        )
    )

    response = chat.send_message(
        f"Start the interview. The student's name is {request.student_name}."
    )
    first_message = response.text

    # store session in memory
    _active_sessions[session_id] = chat

    # Save session to DB
    db = SessionLocal()
    try:
        db.add(AssessmentDB(
            id=session_id,
            student_id=f"pending_{session_id}",
            status="incomplete",
            model_version="gemini-3.1-flash-lite",
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
    if len(request.history) >= MAX_INTERVIEW_HISTORY:
        return ConversationResponse(
            message="That's everything I needed — thanks for being so open about all of this. I'll put together your profile now.",
            is_complete=True,
            session_id=request.session_id,
        )

    chat = _active_sessions.get(request.session_id)
    if not chat:
        client = _get_gemini_client()
        first_name = request.student_name.strip().split()[0]
        system = INTERVIEW_SYSTEM.replace("[name]", first_name)
        chat = client.chats.create(
            model="gemini-3.1-flash-lite",
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.7,
                max_output_tokens=200,
            )
        )
        # Fallback: rebuild using generate_content with full history
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
            model="gemini-3.1-flash-lite",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.7,
                max_output_tokens=200,
            )
        )
        interviewer_message = response.text
        usage = response.usage_metadata
        print(f"Turn {len(request.history) + 1} [rebuilt] | "
              f"in: {usage.prompt_token_count} | "
              f"out: {usage.candidates_token_count} | "
              f"total: {usage.total_token_count}")
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

    # send just the new message
    response = chat.send_message(request.message)
    interviewer_message = response.text

    usage = response.usage_metadata
    print(f"Turn {len(request.history) + 1} | "
          f"in: {usage.prompt_token_count} | "
          f"out: {usage.candidates_token_count} | "
          f"total: {usage.total_token_count}")

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




# ===========================================================================
# POST /conversation/extract  (plan B1, B3, B4)
#
# The LLM extracts EVIDENCE only. Labels come from the deterministic scorer:
#   transcript -> Gemini (evidence JSON) -> interview_adapter.evidence_to_signals
#   -> dimension_scorer.score() -> Student (via scoring.apply)
# Every evidence item is persisted, and rescore_run() reproduces the profile
# from the stored evidence with no LLM.
# ===========================================================================

def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _scorable_dims(rubric: dict) -> list[str]:
    return [n for n, d in rubric["dimensions"].items() if d.get("type") != "derived"]


def _dimensions_block(rubric: dict) -> str:
    lines = []
    for name in _scorable_dims(rubric):
        d = rubric["dimensions"][name]
        values = ", ".join(f'"{v}"' for v in d["allowed_values"])
        order = " (ordered lowest to highest)" if d.get("type") == "ordinal" else ""
        lines.append(f"- {name}: indicator values {values}{order}")
        if d.get("description"):
            lines.append(f"  {_collapse(d['description'])}")
    return "\n".join(lines)


def _strength_block(rubric: dict) -> str:
    return "\n".join(
        f'    "{k}" = {_collapse(v.get("definition", ""))}'
        for k, v in rubric["evidence_strength"].items()
    )


def build_extraction_prompt(rubric: dict, transcript: str, student_name: str) -> tuple[str, str]:
    """
    Return (prompt, prompt_version). Dimension names, indicator values and the
    strength definitions are interpolated from the rubric at request time, so
    the prompt cannot drift from the rubric. prompt_version hashes the prompt
    BEFORE the transcript goes in, so it identifies the prompt+rubric version,
    not the individual run.
    """
    base = (EXTRACTION_PROMPT_TEMPLATE
            .replace("{dimensions_block}", _dimensions_block(rubric))
            .replace("{strength_block}", _strength_block(rubric)))
    prompt_version = "evidence_extraction:" + hashlib.sha256(base.encode()).hexdigest()[:12]
    prompt = base.replace("{student_name}", student_name).replace("{transcript}", transcript)
    return prompt, prompt_version


def build_response_schema(rubric: dict):
    """Gemini response_schema: enums on dimension, polarity and strength.
    indicator is NOT enum-constrained here (it depends on the dimension); the
    adapter checks it and records rejected:indicator_not_allowed."""
    from google.genai import types
    S, T = types.Schema, types.Type
    string_list = S(type=T.ARRAY, items=S(type=T.STRING))
    item = S(
        type=T.OBJECT,
        properties={
            "quote": S(type=T.STRING),
            "dimension": S(type=T.STRING, enum=_scorable_dims(rubric)),
            "indicator": S(type=T.STRING),
            "polarity": S(type=T.STRING, enum=["supporting", "contradicting"]),
            "strength": S(type=T.STRING, enum=list(rubric["evidence_strength"])),
            "confidence": S(type=T.NUMBER),
        },
        required=["quote", "dimension", "indicator", "polarity", "strength", "confidence"],
    )
    return S(
        type=T.OBJECT,
        properties={
            "evidence": S(type=T.ARRAY, items=item),
            "primary_skills": string_list,
            "skill_depth": S(type=T.STRING, enum=["beginner", "intermediate", "advanced"]),
            "skill_no_gos": string_list,
            "unavailable_times": string_list,
            "timezone_notes": S(type=T.STRING),
        },
        required=["evidence"],
    )


def format_transcript(history: List[ConversationTurn], student_name: str) -> str:
    return "".join(
        f"{'Interviewer' if t.role == 'interviewer' else student_name}: {t.content}\n\n"
        for t in history
    )


def student_text_from_history(history: List[ConversationTurn]) -> str:
    """Only the student's turns: the adapter rejects quotes of the interviewer."""
    return STUDENT_TURN_SEPARATOR.join(t.content for t in history if t.role != "interviewer")


def parse_extraction(raw_text: Optional[str]) -> EvidenceExtractionOutput:
    """JSON parse + shape check. Raises ValueError on anything malformed."""
    if not raw_text:
        raise ValueError("empty model response")
    try:
        return EvidenceExtractionOutput.model_validate(json.loads(raw_text))
    except (json.JSONDecodeError, ValidationError, TypeError) as e:
        raise ValueError(f"{type(e).__name__}: {e}") from e


def _is_retryable(e: Exception) -> bool:
    return getattr(e, "code", None) in (429, 503) or "429" in str(e) or "503" in str(e)


def _call_with_retry(fn, retries: int = EXTRACTION_RETRIES):
    """Retry on 429/503 (plan TODO-15); anything else is raised immediately."""
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            if attempt < retries - 1 and _is_retryable(e):
                print(f"Gemini unavailable, retrying in {RETRY_DELAY_SECONDS}s... (attempt {attempt + 1})")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                raise


def consistency_flags(decisions: List[EvidenceDecision], rubric: dict) -> list[dict]:
    """
    Decision doc §5: each contradicting item becomes a consistency flag. The
    adapter leaves this to the caller. {dimension, note} is the shape the
    profile page already renders; the other keys are for reviewers.
    """
    floors = set(floor_dimensions(rubric))
    return [
        {
            "dimension": d.item.dimension,
            "note": f'Contradicting evidence against "{d.item.indicator}": "{d.item.quote}"',
            "indicator": d.item.indicator,
            "quote": d.item.quote,
            "floor_dimension": d.item.dimension in floors,
            "needs_review": d.needs_review,
        }
        for d in decisions
        if d.decision == "excluded:contradicting"
    ]


def display_quote(dimension: str, label: str, decisions: List[EvidenceDecision], rubric: dict) -> Optional[str]:
    """
    The quote shown next to a score: the strongest accepted item, earliest in
    the transcript on ties. Items pointing at the FINAL label are preferred,
    so the displayed quote supports what is displayed; if none does (possible
    for an ordinal mean that lands between indicators), any accepted item for
    the dimension is used.
    """
    accepted = [d for d in decisions if d.status == "accepted" and d.item.dimension == dimension]
    pool = [d for d in accepted if d.item.indicator == label] or accepted
    if not pool:
        return None
    weights = rubric["evidence_strength"]
    best = min(pool, key=lambda d: (-weights[d.item.strength]["weight"], d.quote_start))
    return best.item.quote


def _evidence_rows(run_id: str, session_id: str, decisions: List[EvidenceDecision]) -> list[EvidenceUnitDB]:
    return [
        EvidenceUnitDB(
            id=str(uuid_lib.uuid4()),
            run_id=run_id,
            session_id=session_id,
            trait=d.item.dimension,
            rubric_indicator=d.item.indicator,
            polarity=d.item.polarity,
            strength=d.item.strength,
            span=d.item.quote,
            rejected="true" if d.status == "rejected" else "false",
            confidence=d.item.confidence,
            needs_review="true" if d.needs_review else "false",
            decision=d.decision,
            quote_start=d.quote_start,
        )
        for d in decisions
    ]


@router.post("/extract", response_model=ExtractResponse)
def extract_signature(session_id: str, student_name: str, history: List[ConversationTurn]):
    """Extract evidence from a completed interview, then score it deterministically."""
    from google.genai import types

    rubric = get_rubric()
    client = _get_gemini_client()

    transcript = format_transcript(history, student_name)
    student_text = student_text_from_history(history)
    prompt, prompt_version = build_extraction_prompt(rubric, transcript, student_name)
    run_id = str(uuid_lib.uuid4())

    # TODO(team-decision): where the student text for re-scoring lives.
    # rescore_run() must re-run the adapter's verbatim check, so it needs the
    # exact student text this run saw. No column holds it (the migration is
    # fixed for this step), so it goes in params_json. A dedicated column, or
    # per-run rows in the unused conversation_turns table, would be cleaner.
    params = {"temperature": EXTRACTION_TEMPERATURE, "student_text": student_text}

    def _run_row(raw_response: Optional[str], extra: Optional[dict] = None) -> ExtractionRunDB:
        return ExtractionRunDB(
            id=run_id,
            session_id=session_id,
            model_name=EXTRACTION_MODEL,
            prompt_version=prompt_version,
            params_json={**params, **(extra or {})},
            run_purpose="production",
            rubric_version=rubric["schema_version"],
            prompt_hash=hashlib.sha256(prompt.encode()).hexdigest(),
            raw_response=raw_response,
        )

    # ---- LLM call + parse. On ANY failure: persist the run row (with the raw
    # ---- text when there is one), write no profile, return 502 (TODO-15).
    raw_text: Optional[str] = None
    try:
        config = types.GenerateContentConfig(
            temperature=EXTRACTION_TEMPERATURE,
            response_mime_type="application/json",
            response_schema=build_response_schema(rubric),
        )
        response = _call_with_retry(lambda: client.models.generate_content(
            model=EXTRACTION_MODEL, contents=prompt, config=config,
        ))
        raw_text = response.text
        extraction = parse_extraction(raw_text)
    except Exception as e:
        db = db_module.SessionLocal()
        try:
            db.add(_run_row(raw_text, {"error": f"{type(e).__name__}: {str(e)[:500]}"}))
            db.commit()
        finally:
            db.close()
        raise HTTPException(status_code=502, detail=f"Evidence extraction failed (run {run_id}); nothing was scored.")

    # ---- Evidence -> signals -> deterministic scores
    items = [EvidenceItem(**i.model_dump()) for i in extraction.evidence]
    signals, decisions = evidence_to_signals(items, student_text, rubric)
    scored = score(signals, rubric)
    signature = dict(scored)          # full scorer output, incl. _unknown_dims; stored as the profile
    unknown = list(scored["_unknown_dims"])
    coverage = scored["confidence_score"]

    student_id = f"s{int(time.time() * 1000)}"

    db = db_module.SessionLocal()
    try:
        db.add(_run_row(raw_text))
        db.add_all(_evidence_rows(run_id, session_id, decisions))

        profile_version_id = str(uuid_lib.uuid4())
        db.add(ProfileVersionDB(
            id=profile_version_id,
            session_id=session_id,
            student_id=student_id,
            run_id=run_id,
            signature_json=signature,
        ))

        for dim in _scorable_dims(rubric):
            # TODO(team-decision): no dimension_scores row for unknown
            # dimensions. Their value would only be the rubric's display
            # default, which would read as a measurement. Plan B3 says
            # "value = scored[dim]" without addressing unknowns.
            if dim in unknown:
                continue
            db.add(DimensionScoreDB(
                id=str(uuid_lib.uuid4()),
                profile_version_id=profile_version_id,
                dimension=dim,
                value=scored[dim],
                confidence=None,   # LLM confidence is a per-evidence review flag only (TODO-11)
                evidence_span=display_quote(dim, scored[dim], decisions, rubric),
            ))

        student = StudentDB(
            id=student_id,
            name=student_name,
            competence_signature={
                "skills": extraction.primary_skills,
                "roles": [],
                # TODO-9 default: the LLM's skill_depth is display-only. It is
                # NOT copied into experience_level, which the matcher's ability
                # fallback reads; experience_level stays at its default.
                "experience_level": "intermediate",
                "skill_depth": extraction.skill_depth,
                "skill_no_gos": extraction.skill_no_gos,
            },
            work_rhythm_signature={
                "availability": [],
                "unavailable_times": extraction.unavailable_times,
                "timezone_notes": extraction.timezone_notes,
            },
            collaboration_signature={},
            motivation_layer={"interests": [], "learning_goals": []},
        )
        apply_scored_to_student(student, scored, unknown, rubric)
        student.confidence_layer = {
            **student.confidence_layer,
            "consistency_flags": consistency_flags(decisions, rubric),
        }
        db.add(student)

        session = db.query(AssessmentDB).filter(AssessmentDB.id == session_id).first()
        if session:
            session.status = "complete"
            session.student_id = student_id

        db.add(ResponseDB(
            id=str(uuid_lib.uuid4()),
            assessment_id=session_id,
            student_id=student_id,
            scenario_id="conversational_interview",
            option_id="n/a",
            response_text=transcript,
        ))

        db.add(BehavioralSignatureDB(
            id=str(uuid_lib.uuid4()),
            student_id=student_id,
            signature=signature,
            model_version=EXTRACTION_MODEL,
            rubric_version=rubric["schema_version"],
            bank_version=None,
            scorer_version=SCORER_VERSION,
        ))

        db.commit()
    finally:
        db.close()

    return ExtractResponse(
        student_id=student_id,
        session_id=session_id,
        run_id=run_id,
        signature=signature,
        under_determined_dims=unknown,
        measurement_coverage=coverage,
        evidence_counts=dict(sorted(Counter(d.decision for d in decisions).items())),
    )


def rescore_run(run_id: str, db=None) -> dict:
    """
    Re-score a stored extraction run from its persisted evidence. No LLM, no
    network. Returns the score() output, which must equal the run's
    profile_versions.signature_json byte for byte (same JSON serialization).

    Raises LookupError if the run doesn't exist, ValueError if it can't be
    reproduced (rubric version changed, or the run predates stored student text).
    """
    own_session = db is None
    db = db or db_module.SessionLocal()
    try:
        run = db.get(ExtractionRunDB, run_id)
        if run is None:
            raise LookupError(f"extraction run {run_id!r} not found")
        rubric = get_rubric()
        if run.rubric_version != rubric["schema_version"]:
            raise ValueError(
                f"run {run_id} was scored with rubric {run.rubric_version!r}; "
                f"the loaded rubric is {rubric['schema_version']!r}"
            )
        student_text = (run.params_json or {}).get("student_text")
        if student_text is None:
            raise ValueError(f"run {run_id} has no stored student text; it cannot be re-scored")

        rows = db.query(EvidenceUnitDB).filter(EvidenceUnitDB.run_id == run_id).all()
        items = [
            EvidenceItem(
                quote=r.span or "",
                dimension=r.trait,
                indicator=r.rubric_indicator,
                polarity=r.polarity,
                strength=r.strength,
                confidence=r.confidence,
            )
            for r in rows
        ]
        signals, _ = evidence_to_signals(items, student_text, rubric)
        return score(signals, rubric)
    finally:
        if own_session:
            db.close()


# ===========================================================================
# GET /conversation/{session_id}/evidence  (plan B4: reviewer view)
#
# Read-only. Returns the stored evidence of a session's extraction run so a
# human can check the items flagged needs_review. No auth, like the rest of
# the app.
# ===========================================================================

def _as_bool(flag: Optional[str]) -> Optional[bool]:
    """evidence_units stores flags as "true"/"false" strings; NULL = not assessed."""
    return None if flag is None else flag == "true"


@router.get("/{session_id}/evidence", response_model=ReviewEvidenceResponse)
def get_session_evidence(session_id: str):
    db = db_module.SessionLocal()
    try:
        runs = (
            db.query(ExtractionRunDB)
            .filter(ExtractionRunDB.session_id == session_id)
            .order_by(ExtractionRunDB.created_at.desc(), ExtractionRunDB.id.desc())
            .all()
        )
        # TODO(team-decision): which run to show when a session has several
        # (reruns). Default: the LATEST run by created_at, skipping failed runs
        # (a 502 run has an "error" in params_json and no evidence, so showing
        # it would hide the real evidence behind an empty list). The
        # alternative is returning every run. runs_for_session tells the
        # reviewer when there were others.
        run = next((r for r in runs if "error" not in (r.params_json or {})), None)
        if run is None:
            detail = (f"No successful extraction run for session '{session_id}' "
                      f"({len(runs)} failed run(s))." if runs
                      else f"No extraction run for session '{session_id}'.")
            raise HTTPException(status_code=404, detail=detail)

        rows = db.query(EvidenceUnitDB).filter(EvidenceUnitDB.run_id == run.id).all()

        # TODO(team-decision): which items to return. Default: ALL items, so the
        # reviewer sees flagged items in context, each with its needs_review
        # flag. A ?needs_review_only=true filter is the alternative.
        # Order: flagged first, then by dimension and transcript position.
        rows.sort(key=lambda r: (
            _as_bool(r.needs_review) is not True,
            r.trait,
            r.quote_start is None, r.quote_start or 0,
            r.rubric_indicator,
            r.span or "",
            r.decision or "",
        ))
        items = [
            ReviewEvidenceItem(
                dimension=r.trait,
                indicator=r.rubric_indicator,
                polarity=r.polarity,
                strength=r.strength,
                quote=r.span,
                confidence=r.confidence,
                decision=r.decision,
                needs_review=_as_bool(r.needs_review),
            )
            for r in rows
        ]
        summary = ReviewEvidenceSummary(
            total=len(items),
            needs_review=sum(1 for i in items if i.needs_review),
            by_decision=dict(sorted(Counter(i.decision or "unknown" for i in items).items())),
        )
        return ReviewEvidenceResponse(
            session_id=session_id,
            run_id=run.id,
            run_created_at=run.created_at.isoformat() if run.created_at else None,
            rubric_version=run.rubric_version,
            model_name=run.model_name,
            runs_for_session=len(runs),
            summary=summary,
            items=items,
        )
    finally:
        db.close()
