from __future__ import annotations
from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class CompetenceSignature(BaseModel):
    skills: List[str] = []
    roles: List[str] = []
    experience_level: str = "intermediate"  # beginner | intermediate | advanced
    skill_depth: str = "intermediate"
    skill_no_gos: List[str] = []


class WorkRhythmSignature(BaseModel):
    planning_style: str = "adaptive"        # planner | spontaneous | adaptive
    communication_style: str = "mixed"      # async | sync | mixed
    execution_style: str = "iterative"      # methodical | iterative | exploratory
    availability: List[str] = []            # e.g. ["Mon-morning", "Wed-evening"]
    unavailable_times: List[str] = []
    timezone_notes: str = ""


class CollaborationSignature(BaseModel):
    conflict_style: str = "collaborative"   # avoidant | confrontational | collaborative
    leadership_style: str = "facilitative"  # directive | facilitative | emergent
    accountability: str = "medium"          # high | medium | low
    help_seeking: str = "proactive"         # proactive | reactive | independent
    safety_contribution: str = "medium"
    stress_response: str = "composed"


class MotivationLayer(BaseModel):
    interests: List[str] = []
    learning_goals: List[str] = []


class ConfidenceLayer(BaseModel):
    confidence_score: float = Field(default=0.6, ge=0.0, le=1.0)


class Student(BaseModel):
    id: str
    name: str
    competence_signature: CompetenceSignature = Field(default_factory=CompetenceSignature)
    work_rhythm_signature: WorkRhythmSignature = Field(default_factory=WorkRhythmSignature)
    collaboration_signature: CollaborationSignature = Field(default_factory=CollaborationSignature)
    motivation_layer: MotivationLayer = Field(default_factory=MotivationLayer)
    confidence_layer: ConfidenceLayer = Field(default_factory=ConfidenceLayer)


# Assessment models

class ScenarioOption(BaseModel):
    id: str
    label: str


class Scenario(BaseModel):
    id: str
    title: str
    description: str
    options: List[ScenarioOption]


class AssessmentResponse(BaseModel):
    scenario_id: str
    option_id: str
    response_text: str = ""


class AssessmentRequest(BaseModel):
    student_name: str
    student_id: Optional[str] = None   # if provided, updates existing profile
    session_id: Optional[str] = None   # if provided, completes that existing session
    responses: List[AssessmentResponse]


class StartSessionRequest(BaseModel):
    student_name: str
    student_id: Optional[str] = None


class SaveProgressRequest(BaseModel):
    session_id: str
    student_id: str
    responses: List[AssessmentResponse]


# Team models

class TeamMember(BaseModel):
    student_id: str
    name: str
    roles: List[str] = []
    experience_level: str = "intermediate"


class TeamScoreBreakdown(BaseModel):
    skill_coverage: float = Field(ge=0.0, le=1.0)
    behavioral_compat: float = Field(ge=0.0, le=1.0)
    availability_overlap: float = Field(ge=0.0, le=1.0)
    conflict_risk: float = Field(ge=0.0, le=1.0)
    match_confidence: float = Field(ge=0.0, le=1.0)
    total: float = Field(ge=0.0, le=1.0)


class TeamNorm(BaseModel):
    category: str
    norm: str


class Team(BaseModel):
    id: str
    members: List[TeamMember]
    score_breakdown: TeamScoreBreakdown
    team_norms: List[TeamNorm] = []


class TeamGenerationRequest(BaseModel):
    team_size: int = Field(default=4, ge=2, le=8)
    formation_mode: str = "behavioral"  # "behavioral" | "skill" | "random"
    weights: Dict[str, float] = Field(default_factory=lambda: {
        "skill_coverage": 0.4,
        "behavioral_compat": 0.3,
        "availability_overlap": 0.2,
        "shared_interests": 0.1,
    })


class TeamExplanation(BaseModel):
    team_id: str
    compatibility_score: float
    match_confidence: float
    strengths: List[str]
    risks: List[str]
    explanation: str
    team_norms: List[TeamNorm]


# Radar chart model

class MemberRadarData(BaseModel):
    student_id: str
    name: str
    values: List[float]  # 5 values in [0,1]


class RadarData(BaseModel):
    labels: List[str] = Field(default_factory=lambda: [
        "Communication",
        "Conflict Handling",
        "Leadership",
        "Accountability",
        "Planning",
    ])
    values: List[float]               # team-average, 5 values in [0,1]
    members_radar: List[MemberRadarData] = []

class ConversationTurn(BaseModel):
    role: str   # "interviewer" or "student"
    content: str

class ConversationStartRequest(BaseModel):
    student_name: str

class ConversationTurnRequest(BaseModel):
    session_id: str
    student_name: str
    message: str
    history: List[ConversationTurn]

class ConversationResponse(BaseModel):
    message: str
    is_complete: bool
    session_id: str
    