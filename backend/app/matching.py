from __future__ import annotations
import itertools
import random
from typing import List

from fastapi import APIRouter, HTTPException

from app.models import (
    Student,
    Team,
    TeamGenerationRequest,
    TeamMember,
    TeamNorm,
    TeamScoreBreakdown,
)

router = APIRouter(prefix="/teams", tags=["teams"])

# ---------------------------------------------------------------------------
# Reference skill set for coverage scoring
# ---------------------------------------------------------------------------
REFERENCE_SKILLS = {
    "frontend", "backend", "ml", "data analysis", "project management",
    "testing", "design", "devops", "algorithms", "communication",
    "documentation", "research",
}


# ---------------------------------------------------------------------------
# Compatibility tables — pairwise scoring per behavioral dimension
# ---------------------------------------------------------------------------
_CONFLICT_COMPAT = {
    ("collaborative",   "collaborative"):   1.0,
    ("collaborative",   "confrontational"): 0.6,
    ("collaborative",   "avoidant"):        0.7,
    ("confrontational", "confrontational"): 0.3,
    ("confrontational", "avoidant"):        0.4,
    ("avoidant",        "avoidant"):        0.5,
}

_LEADERSHIP_COMPAT = {
    ("facilitative", "facilitative"): 0.9,
    ("facilitative", "directive"):    0.85,
    ("facilitative", "emergent"):     0.8,
    ("directive",    "directive"):    0.3,
    ("directive",    "emergent"):     0.6,
    ("emergent",     "emergent"):     0.65,
}

_PLANNING_COMPAT = {
    ("planner",      "planner"):     0.8,
    ("planner",      "adaptive"):    0.9,
    ("planner",      "spontaneous"): 0.65,
    ("adaptive",     "adaptive"):    0.85,
    ("adaptive",     "spontaneous"): 0.75,
    ("spontaneous",  "spontaneous"): 0.6,
}

_ACCOUNTABILITY_COMPAT = {
    ("high",   "high"):   1.0,
    ("high",   "medium"): 0.75,
    ("high",   "low"):    0.4,
    ("medium", "medium"): 0.7,
    ("medium", "low"):    0.5,
    ("low",    "low"):    0.35,
}


def _lookup(table: dict, a: str, b: str) -> float:
    key = (a, b) if (a, b) in table else (b, a)
    return table.get(key, 0.5)


# ---------------------------------------------------------------------------
# Individual score components
# ---------------------------------------------------------------------------

def _skill_coverage(students: List[Student]) -> float:
    team_skills = set()
    for s in students:
        team_skills.update(s.competence_signature.skills)
    overlap = team_skills & REFERENCE_SKILLS
    return len(overlap) / len(REFERENCE_SKILLS)


def _behavioral_compat(students: List[Student]) -> float:
    if len(students) < 2:
        return 1.0
    scores = []
    for a, b in itertools.combinations(students, 2):
        cs = a.collaboration_signature
        bs = b.collaboration_signature
        pair_score = (
            _lookup(_CONFLICT_COMPAT, cs.conflict_style, bs.conflict_style) * 0.35
            + _lookup(_LEADERSHIP_COMPAT, cs.leadership_style, bs.leadership_style) * 0.30
            + _lookup(_PLANNING_COMPAT, a.work_rhythm_signature.planning_style,
                      b.work_rhythm_signature.planning_style) * 0.20
            + _lookup(_ACCOUNTABILITY_COMPAT, cs.accountability, bs.accountability) * 0.15
        )
        scores.append(pair_score)
    return sum(scores) / len(scores)


def _availability_overlap(students: List[Student]) -> float:
    if len(students) < 2:
        return 1.0
    scores = []
    for a, b in itertools.combinations(students, 2):
        set_a = set(a.work_rhythm_signature.availability)
        set_b = set(b.work_rhythm_signature.availability)
        union = set_a | set_b
        if not union:
            scores.append(0.5)
        else:
            scores.append(len(set_a & set_b) / len(union))
    return sum(scores) / len(scores)


def _shared_interests(students: List[Student]) -> float:
    all_interests: list[set] = [
        set(s.motivation_layer.interests + s.motivation_layer.learning_goals)
        for s in students
    ]
    union = set().union(*all_interests)
    if not union:
        return 0.0
    intersection = all_interests[0]
    for s in all_interests[1:]:
        intersection = intersection & s
    return len(intersection) / len(union)


def _conflict_risk(students: List[Student]) -> float:
    if len(students) < 2:
        return 0.0
    mismatch_count = 0
    total_pairs = 0
    for a, b in itertools.combinations(students, 2):
        total_pairs += 1
        ca, cb = a.collaboration_signature.conflict_style, b.collaboration_signature.conflict_style
        if ca != cb and not ("collaborative" in (ca, cb)):
            mismatch_count += 1
    mismatch_rate = mismatch_count / total_pairs
    directive_count = sum(1 for s in students if s.collaboration_signature.leadership_style == "directive")
    directive_penalty = max(0.0, (directive_count - 1) * 0.15)
    return min(1.0, mismatch_rate * 0.7 + directive_penalty)


def _total_score(students: List[Student], weights: dict) -> float:
    w = weights
    return (
        w.get("skill_coverage", 0.4) * _skill_coverage(students)
        + w.get("behavioral_compat", 0.3) * _behavioral_compat(students)
        + w.get("availability_overlap", 0.2) * _availability_overlap(students)
        + w.get("shared_interests", 0.1) * _shared_interests(students)
    )


def _match_confidence(team_score: float, all_scores: List[float]) -> float:
    if len(all_scores) <= 1:
        return 0.6
    lo, hi = min(all_scores), max(all_scores)
    spread = hi - lo
    if spread < 0.05:
        return 0.45
    relative = (team_score - lo) / spread
    return round(min(0.92, 0.40 + relative * 0.52), 2)


def _normalize_weights(weights: dict) -> dict:
    total = sum(weights.values())
    if total <= 0:
        return weights
    return {k: v / total for k, v in weights.items()}


# ---------------------------------------------------------------------------
# SQLite helper — mirrors the same pattern in explainability.py
# ---------------------------------------------------------------------------

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
# Greedy team builder (unchanged — used by both behavioral and skill modes)
# ---------------------------------------------------------------------------

def _greedy_build(students: List[Student], team_size: int, weights: dict) -> List[List[Student]]:
    remaining = list(students)
    teams: List[List[Student]] = []

    while len(remaining) >= team_size:
        seed = max(remaining, key=lambda s: len(set(s.competence_signature.skills) & REFERENCE_SKILLS))
        team = [seed]
        remaining.remove(seed)

        while len(team) < team_size and remaining:
            best_student = max(
                remaining,
                key=lambda candidate: _total_score(team + [candidate], weights),
            )
            team.append(best_student)
            remaining.remove(best_student)

        teams.append(team)

    for leftover in remaining:
        best_team_idx = max(
            range(len(teams)),
            key=lambda i: _total_score(teams[i] + [leftover], weights),
        )
        teams[best_team_idx].append(leftover)

    return teams


def _two_opt_improve(teams: List[List[Student]], weights: dict) -> List[List[Student]]:
    improved = True
    while improved:
        improved = False
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                for a_idx in range(len(teams[i])):
                    for b_idx in range(len(teams[j])):
                        original = _total_score(teams[i], weights) + _total_score(teams[j], weights)
                        teams[i][a_idx], teams[j][b_idx] = teams[j][b_idx], teams[i][a_idx]
                        after = _total_score(teams[i], weights) + _total_score(teams[j], weights)
                        if after > original + 1e-6:
                            improved = True
                        else:
                            teams[i][a_idx], teams[j][b_idx] = teams[j][b_idx], teams[i][a_idx]
    return teams


# ---------------------------------------------------------------------------
# Formation mode implementations
# ---------------------------------------------------------------------------

def random_assignment(students: List[Student], team_size: int) -> List[List[Student]]:
    """Shuffle students randomly and slice into teams. No optimization."""
    pool = list(students)
    random.shuffle(pool)
    teams: List[List[Student]] = []
    while len(pool) >= team_size:
        teams.append(pool[:team_size])
        pool = pool[team_size:]
    # Distribute leftover students across teams
    for i, leftover in enumerate(pool):
        teams[i % len(teams)].append(leftover)
    return teams


def skill_based_assignment(students: List[Student], team_size: int) -> List[List[Student]]:
    """Greedy + 2-opt optimizing skill coverage only — ignores behavioral dimensions."""
    skill_weights = {
        "skill_coverage": 1.0,
        "behavioral_compat": 0.0,
        "availability_overlap": 0.0,
        "shared_interests": 0.0,
    }
    raw_teams = _greedy_build(students, team_size, skill_weights)
    return _two_opt_improve(raw_teams, skill_weights)


# ---------------------------------------------------------------------------
# Shared team-object builder
# ---------------------------------------------------------------------------

def _build_teams(
    raw_groups: List[List[Student]],
    mode: str,
    behavioral_weights: dict,
) -> List[Team]:
    from ai.gemini_explainer import GeminiExplainer

    explainer = GeminiExplainer()
    skill_weights = {"skill_coverage": 1.0, "behavioral_compat": 0.0,
                     "availability_overlap": 0.0, "shared_interests": 0.0}

    # For confidence calculation we need the "primary" score of each group
    if mode == "random":
        primary_scores = [0.0] * len(raw_groups)
    elif mode == "skill":
        primary_scores = [_skill_coverage(g) for g in raw_groups]
    else:
        primary_scores = [_total_score(g, behavioral_weights) for g in raw_groups]

    result: List[Team] = []

    for idx, group in enumerate(raw_groups):
        team_id = f"team-{idx + 1}"

        sc = _skill_coverage(group)
        cr = _conflict_risk(group)

        if mode == "random":
            bc = ao = 0.0
            mc = 0.0
            total = 0.0
        elif mode == "skill":
            bc = ao = 0.0
            mc = _match_confidence(sc, primary_scores)
            total = round(sc, 3)
        else:
            bc = _behavioral_compat(group)
            ao = _availability_overlap(group)
            si = _shared_interests(group)
            raw_total = (
                behavioral_weights.get("skill_coverage", 0.4) * sc
                + behavioral_weights.get("behavioral_compat", 0.3) * bc
                + behavioral_weights.get("availability_overlap", 0.2) * ao
                + behavioral_weights.get("shared_interests", 0.1) * si
            )
            mc = _match_confidence(raw_total, primary_scores)
            total = round(min(raw_total, 1.0), 3)

        breakdown = TeamScoreBreakdown(
            skill_coverage=round(sc, 3),
            behavioral_compat=round(bc, 3),
            availability_overlap=round(ao, 3),
            conflict_risk=round(cr, 3),
            match_confidence=mc,
            total=total,
        )

        members = [
            TeamMember(
                student_id=s.id,
                name=s.name,
                roles=s.competence_signature.roles,
                experience_level=s.competence_signature.experience_level,
            )
            for s in group
        ]

        norms = explainer.suggest_norms(
            Team(id=team_id, members=members, score_breakdown=breakdown),
            group,
        )

        result.append(Team(id=team_id, members=members, score_breakdown=breakdown, team_norms=norms))

    return result


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/generate")
def generate_teams(request: TeamGenerationRequest):
    from app.database import SessionLocal, TeamDB, StudentDB

    db = SessionLocal()
    try:
        db_students = db.query(StudentDB).all()
        students = [_db_student_to_model(s) for s in db_students]
    finally:
        db.close()

    if len(students) < request.team_size:
        raise HTTPException(status_code=400, detail="Not enough students to form a team.")

    mode = request.formation_mode
    behavioral_weights = _normalize_weights(request.weights)

    if mode == "random":
        raw_groups = random_assignment(students, request.team_size)
    elif mode == "skill":
        raw_groups = skill_based_assignment(students, request.team_size)
    else:
        raw_groups = _greedy_build(students, request.team_size, behavioral_weights)
        raw_groups = _two_opt_improve(raw_groups, behavioral_weights)

    result = _build_teams(raw_groups, mode, behavioral_weights)

    # Persist to SQLite
    db = SessionLocal()
    try:
        db.query(TeamDB).delete()
        for team in result:
            db.add(TeamDB(
                id=team.id,
                members=[m.model_dump() for m in team.members],
                score_breakdown=team.score_breakdown.model_dump(),
                team_norms=[n.model_dump() for n in team.team_norms],
                formation_mode=mode,
            ))
        db.commit()
    finally:
        db.close()

    return {"teams": result}


@router.get("/teams")
def list_teams():
    from app.database import SessionLocal, TeamDB

    db = SessionLocal()
    try:
        rows = db.query(TeamDB).all()
        teams = [
            Team(
                id=r.id,
                members=r.members,
                score_breakdown=r.score_breakdown,
                team_norms=r.team_norms,
            )
            for r in rows
        ]
        return {"teams": teams}
    finally:
        db.close()
