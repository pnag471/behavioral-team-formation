from __future__ import annotations
from typing import List

from fastapi import APIRouter, HTTPException

from app.models import MemberRadarData, RadarData, Student, Team, TeamExplanation
from ai.claude_explainer import ClaudeExplainer

router = APIRouter(prefix="/teams", tags=["explainability"])

explainer = ClaudeExplainer()

# ---------------------------------------------------------------------------
# Categorical → numeric radar mappings (5 axes)
# labels: Communication, Conflict Handling, Leadership, Accountability, Planning
# ---------------------------------------------------------------------------
_COMM_MAP    = {"async": 0.35, "mixed": 0.65, "sync": 0.90}
_CONFLICT_MAP = {"avoidant": 0.20, "assertive": 0.60, "collaborative": 0.85}
_LEADER_MAP  = {"emergent": 0.30, "facilitative": 0.70, "directive": 0.92}
_ACCOUNT_MAP = {"low": 0.20, "developing": 0.60, "high": 0.90}
_PLAN_MAP    = {"spontaneous": 0.20, "adaptive": 0.60, "planner": 0.90}


def _student_radar_values(student: Student) -> List[float]:
    ws = student.work_rhythm_signature
    cs = student.collaboration_signature
    return [
        _COMM_MAP.get(ws.communication_style, 0.5),
        _CONFLICT_MAP.get(cs.conflict_style, 0.5),
        _LEADER_MAP.get(cs.leadership_style, 0.5),
        _ACCOUNT_MAP.get(cs.accountability, 0.5),
        _PLAN_MAP.get(ws.planning_style, 0.5),
    ]


def _avg_radar(value_lists: List[List[float]]) -> List[float]:
    n = len(value_lists)
    return [round(sum(col) / n, 3) for col in zip(*value_lists)]

def _db_student_to_model(s) -> Student:
    return Student(
        id=s.id,
        name=s.name,
        competence_signature=s.competence_signature,
        work_rhythm_signature=s.work_rhythm_signature,
        collaboration_signature=s.collaboration_signature,
        motivation_layer=s.motivation_layer,
        confidence_layer=s.confidence_layer,
    )

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/{team_id}", response_model=Team)
def get_team(team_id: str):
    from app.database import SessionLocal, TeamDB
    db = SessionLocal()
    try:
        team = db.query(TeamDB).filter(TeamDB.id == team_id).first()
        if not team:
            raise HTTPException(status_code=404, detail=f"Team '{team_id}' not found.")
        return Team(
            id=team.id,
            members=team.members,
            score_breakdown=team.score_breakdown,
            team_norms=team.team_norms,
        )
    finally:
        db.close()

@router.get("/{team_id}/explanation", response_model=TeamExplanation)
def get_explanation(team_id: str):
    from app.database import SessionLocal, TeamDB, StudentDB
    db = SessionLocal()
    try:
        team_db = db.query(TeamDB).filter(TeamDB.id == team_id).first()
        if not team_db:
            raise HTTPException(status_code=404, detail=f"Team '{team_id}' not found.")
        
        team = Team(
            id=team_db.id,
            members=team_db.members,
            score_breakdown=team_db.score_breakdown,
            team_norms=team_db.team_norms,
        )
    
        student_ids = [m["student_id"] for m in team_db.members]
        db_students = db.query(StudentDB).filter(StudentDB.id.in_(student_ids)).all()
        students = [_db_student_to_model(s) for s in db_students]

        return explainer.generate(team, students)
    finally:
        db.close()

@router.get("/{team_id}/radar", response_model=RadarData)
def get_radar(team_id: str):
    from app.database import SessionLocal, TeamDB, StudentDB
    db = SessionLocal()
    try:
        team_db = db.query(TeamDB).filter(TeamDB.id == team_id).first()
        if not team_db:
            raise HTTPException(status_code=404, detail=f"Team '{team_id}' not found.")
        
        student_ids = [m["student_id"] for m in team_db.members]
        db_students = db.query(StudentDB).filter(StudentDB.id.in_(student_ids)).all()
        students = [_db_student_to_model(s) for s in db_students]    
        
        per_member = [_student_radar_values(s) for s in students]
        avg = _avg_radar(per_member) if per_member else [0.5] * 5

        members_radar = [
            MemberRadarData(student_id=s.id, name=s.name, values=vals)
            for s, vals in zip(students, per_member)
        ]

        return RadarData(values=avg, members_radar=members_radar)
    finally:
        db.close()