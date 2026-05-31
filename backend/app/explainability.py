from __future__ import annotations
from typing import List

from fastapi import APIRouter, HTTPException

from app.models import MemberRadarData, RadarData, Team, TeamExplanation

router = APIRouter(prefix="/teams", tags=["explainability"])

# ---------------------------------------------------------------------------
# Categorical → numeric radar mappings (5 axes)
# labels: Communication, Conflict Handling, Leadership, Accountability, Planning
# ---------------------------------------------------------------------------
_COMM_MAP    = {"async": 0.35, "mixed": 0.65, "sync": 0.90}
_CONFLICT_MAP = {"avoidant": 0.20, "confrontational": 0.60, "collaborative": 0.85}
_LEADER_MAP  = {"emergent": 0.30, "facilitative": 0.70, "directive": 0.92}
_ACCOUNT_MAP = {"low": 0.20, "medium": 0.60, "high": 0.90}
_PLAN_MAP    = {"spontaneous": 0.20, "adaptive": 0.60, "planner": 0.90}


def _student_radar_values(student) -> List[float]:
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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/{team_id}", response_model=Team)
def get_team(team_id: str):
    from app.main import teams_store
    team = teams_store.get(team_id)
    if not team:
        raise HTTPException(status_code=404, detail=f"Team '{team_id}' not found.")
    return team


@router.get("/{team_id}/explanation", response_model=TeamExplanation)
def get_explanation(team_id: str):
    from app.main import teams_store, students_store
    from ai.mock_explainer import MockExplainer

    team = teams_store.get(team_id)
    if not team:
        raise HTTPException(status_code=404, detail=f"Team '{team_id}' not found. Generate teams first.")

    students = [students_store[m.student_id] for m in team.members if m.student_id in students_store]
    return MockExplainer().generate(team, students)


@router.get("/{team_id}/radar", response_model=RadarData)
def get_radar(team_id: str):
    from app.main import teams_store, students_store

    team = teams_store.get(team_id)
    if not team:
        raise HTTPException(status_code=404, detail=f"Team '{team_id}' not found.")

    students = [students_store[m.student_id] for m in team.members if m.student_id in students_store]
    per_member = [_student_radar_values(s) for s in students]
    avg = _avg_radar(per_member) if per_member else [0.5] * 5

    members_radar = [
        MemberRadarData(student_id=s.id, name=s.name, values=vals)
        for s, vals in zip(students, per_member)
    ]

    return RadarData(values=avg, members_radar=members_radar)
