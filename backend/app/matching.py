from __future__ import annotations
import itertools
import random
from typing import List, Tuple

from fastapi import APIRouter, HTTPException

from app.models import (
    Student,
    Team,
    TeamGenerationRequest,
    TeamMember,
    TeamNorm,
    TeamScoreBreakdown,
)
from scoring.dimension_scorer import get_rubric

router = APIRouter(prefix="/teams", tags=["teams"])

# ---------------------------------------------------------------------------
# Reference skill set for complementary_coverage scoring
# ---------------------------------------------------------------------------
REFERENCE_SKILLS = {
    "frontend", "backend", "ml", "data analysis", "project management",
    "testing", "design", "devops", "algorithms", "communication",
    "documentation", "research",
}


# ---------------------------------------------------------------------------
# Rubric accessors
# ---------------------------------------------------------------------------

def _get_student_dim(student: Student, dim_name: str, dim_def: dict):
    """Read a dimension value from the appropriate Student sub-object."""
    group = dim_def.get("group", "collaboration")
    default = dim_def.get("default_value")
    if group == "work_rhythm":
        return getattr(student.work_rhythm_signature, dim_name, default)
    if group == "collaboration":
        return getattr(student.collaboration_signature, dim_name, default)
    return default


def _is_unknown(student: Student, dim_name: str) -> bool:
    return dim_name in (student.under_determined_dims or [])


# ---------------------------------------------------------------------------
# Per-team scoring for each composition rule
# ---------------------------------------------------------------------------

def _ordinal_team_mean(students: List[Student], dim_name: str, dim_def: dict) -> float:
    """
    Mean ordinal level for known members.  Unknown members are excluded.
    Returns a value in [0, 1].  Returns 0.5 when all members are unknown.

    composition_rule: high_mean_with_floor
    Floor is enforced separately in _check_floor_violations(); this function
    only scores the mean so the matcher can rank candidate teams.
    """
    allowed: List[str] = dim_def["allowed_values"]
    level_map = {v: i for i, v in enumerate(allowed)}
    max_level = len(allowed) - 1

    known_levels = [
        level_map.get(_get_student_dim(s, dim_name, dim_def), max_level // 2)
        for s in students
        if not _is_unknown(s, dim_name)
    ]

    if not known_levels:
        return 0.5
    return sum(known_levels) / (len(known_levels) * max_level)


def _soft_compat_score(students: List[Student], dim_name: str, dim_def: dict) -> float:
    """
    Slight REWARD for value diversity in a team: more distinct values among
    known members → higher score.  Range [0.7, 1.0].

    composition_rule: soft_compatibility

    Implementation: reward-for-diversity (not penalise-mismatch).
    A fully homogeneous team scores 0.7; a fully diverse team scores 1.0.
    """
    # TODO(team-decision): leadership_style is soft_compatibility (rewards SIMILAR
    # styles when there is only one member with each style and n_distinct/n is high).
    # This might contradict complementary-leadership intent — consider switching to
    # complementary_coverage, which explicitly rewards covering all styles.
    # File: backend/app/matching.py  — confirm which rule is correct.
    values = [
        _get_student_dim(s, dim_name, dim_def)
        for s in students
        if not _is_unknown(s, dim_name)
    ]
    if not values:
        return 0.85  # all unknown → neutral
    n_distinct = len(set(values))
    return 0.7 + 0.3 * (n_distinct / len(values))


def _behavioral_compat(students: List[Student]) -> float:
    """
    Team-level behavioral compatibility score, driven by rubric composition_rule.

    Normalised by sum of non-zero weights so missing a dimension does not
    deflate the score.
    """
    if len(students) < 2:
        return 1.0

    rubric = get_rubric()
    weighted_score = 0.0
    total_weight = 0.0

    for dim_name, dim_def in rubric["dimensions"].items():
        if dim_def.get("type") == "derived":
            continue
        rule = dim_def.get("composition_rule")
        weight = dim_def.get("matching_weight", 0.0)
        if weight == 0.0 or not rule:
            continue

        if rule == "high_mean_with_floor":
            dim_score = _ordinal_team_mean(students, dim_name, dim_def)
        elif rule == "soft_compatibility":
            # TODO(team-decision): conflict_style is zeroed (weight=0.0 in rubric)
            # until the question bank distinguishes task vs relationship conflict.
            # File: backend/app/matching.py  — confirm this is correct.
            dim_score = _soft_compat_score(students, dim_name, dim_def)
        else:
            continue

        weighted_score += weight * dim_score
        total_weight += weight

    return weighted_score / total_weight if total_weight > 0 else 0.5


# ---------------------------------------------------------------------------
# Individual score components (deterministic)
# ---------------------------------------------------------------------------

def _skill_coverage(students: List[Student]) -> float:
    """complementary_coverage: fraction of REFERENCE_SKILLS covered by the team."""
    team_skills: set[str] = set()
    for s in students:
        team_skills.update(s.competence_signature.skills)
    overlap = team_skills & REFERENCE_SKILLS
    return len(overlap) / len(REFERENCE_SKILLS)


def _availability_overlap(students: List[Student]) -> float:
    if len(students) < 2:
        return 1.0
    scores = []
    for a, b in itertools.combinations(students, 2):
        set_a = set(a.work_rhythm_signature.availability)
        set_b = set(b.work_rhythm_signature.availability)
        union = set_a | set_b
        scores.append(len(set_a & set_b) / len(union) if union else 0.5)
    return sum(scores) / len(scores)


def _shared_interests(students: List[Student]) -> float:
    all_interests = [
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
        ca = a.collaboration_signature.conflict_style
        cb = b.collaboration_signature.conflict_style
        if ca != cb and "collaborative" not in (ca, cb):
            mismatch_count += 1
    mismatch_rate = mismatch_count / total_pairs
    directive_count = sum(
        1 for s in students
        if s.collaboration_signature.leadership_style == "directive"
    )
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
# Floor and balance (high_mean_with_floor objective)
# ---------------------------------------------------------------------------

def _floor_dims() -> dict:
    rubric = get_rubric()
    return {
        name: dim
        for name, dim in rubric["dimensions"].items()
        if dim.get("composition_rule") == "high_mean_with_floor"
        and dim.get("type") == "ordinal"
    }


def _is_floor_level(student: Student, dim_name: str, dim_def: dict) -> bool:
    if _is_unknown(student, dim_name):
        return False  # unknown does not violate the floor
    value = _get_student_dim(student, dim_name, dim_def)
    return value == dim_def["allowed_values"][0]  # lowest level


def _check_feasibility(
    students: List[Student], n_teams: int
) -> Tuple[bool, dict]:
    """
    Returns (is_feasible, {dim_name: count_floor_level_students}).

    Infeasible when any floor dim has more students at the lowest level
    than there are teams — cannot spread them one-per-team.
    """
    dims = _floor_dims()
    infeasible: dict[str, int] = {}
    for dim_name, dim_def in dims.items():
        count = sum(1 for s in students if _is_floor_level(s, dim_name, dim_def))
        if count > n_teams:
            infeasible[dim_name] = count
    return len(infeasible) == 0, infeasible


def _check_floor_violations(
    teams: List[List[Student]],
) -> Tuple[bool, List[str], List[str]]:
    """
    Returns (feasible, floor_violation_messages, under_determined_team_ids).

    floor_violation_messages: list of strings describing each violation.
    under_determined_team_ids: teams where ≥1 member has unknown floor dims.
    """
    dims = _floor_dims()
    violations: List[str] = []
    under_determined: List[str] = []

    for team_idx, team in enumerate(teams):
        team_id = f"team-{team_idx + 1}"
        has_unknown = False
        for dim_name, dim_def in dims.items():
            for student in team:
                if _is_unknown(student, dim_name):
                    has_unknown = True
                elif _is_floor_level(student, dim_name, dim_def):
                    violations.append(
                        f"{team_id}: '{student.name}' has {dim_name}="
                        f"{dim_def['allowed_values'][0]} (floor level)"
                    )
        if has_unknown:
            under_determined.append(team_id)

    return len(violations) == 0, violations, under_determined


def _partition_floor_min(teams: List[List[Student]]) -> float:
    """
    Minimum per-team floor-dim mean across all teams and floor dims.
    Used by maximin improve to equalize team quality.
    """
    dims = _floor_dims()
    if not dims:
        return 1.0
    scores = [
        _ordinal_team_mean(team, dim_name, dim_def)
        for team in teams
        for dim_name, dim_def in dims.items()
    ]
    return min(scores) if scores else 0.5


def _maximin_improve(teams: List[List[Student]]) -> List[List[Student]]:
    """
    Pairwise swap pass that maximises min(team_floor_mean across all teams).

    Accepts a swap only if it increases the partition floor minimum.
    This equalises team means on validated ordinal dims (maximin objective)
    without creating a within-team variance penalty (which would concentrate
    weak members — the Bell 2007 bad-apple problem).
    """
    improved = True
    while improved:
        improved = False
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                for a_idx in range(len(teams[i])):
                    for b_idx in range(len(teams[j])):
                        original = _partition_floor_min(teams)
                        teams[i][a_idx], teams[j][b_idx] = teams[j][b_idx], teams[i][a_idx]
                        new_val = _partition_floor_min(teams)
                        if new_val > original + 1e-6:
                            improved = True
                        else:
                            teams[i][a_idx], teams[j][b_idx] = teams[j][b_idx], teams[i][a_idx]
    return teams


# ---------------------------------------------------------------------------
# DB helper
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
        under_determined_dims=s.under_determined_dims or [],
    )


# ---------------------------------------------------------------------------
# Greedy team builder with floor-first seeding
# ---------------------------------------------------------------------------

def _greedy_build(
    students: List[Student], team_size: int, weights: dict
) -> List[List[Student]]:
    """
    Greedy team builder.

    Floor-level students (known lowest level on any floor dim) are assigned
    round-robin first so at most one lands per team.  Remaining students are
    added greedily by total score.

    TODO(team-decision): seeding is currently skills-coverage-first for non-floor
    students.  Consider floor-dim-score-first seeding to balance high-quality
    members across teams before the maximin pass.
    File: backend/app/matching.py  — confirm seeding strategy.
    """
    dims = _floor_dims()

    def is_floor_student(s: Student) -> bool:
        return any(_is_floor_level(s, dn, dd) for dn, dd in dims.items())

    floor_students = [s for s in students if is_floor_student(s)]
    other_students = [s for s in students if not is_floor_student(s)]

    n_teams = len(students) // team_size
    teams: List[List[Student]] = [[] for _ in range(n_teams)]

    # Assign floor-level students round-robin (one per team max)
    for i, s in enumerate(floor_students):
        teams[i % n_teams].append(s)

    # Fill remaining spots greedily by total score
    remaining = list(other_students)

    # Seed each team that lacks a starter with the highest-skill remaining student
    for team in teams:
        if not team and remaining:
            seed = max(
                remaining,
                key=lambda s: len(set(s.competence_signature.skills) & REFERENCE_SKILLS),
            )
            team.append(seed)
            remaining.remove(seed)

    while remaining:
        # Find best (team, candidate) pair
        best_score = -float("inf")
        best_team_idx = 0
        best_student = remaining[0]

        for team_idx, team in enumerate(teams):
            if len(team) >= team_size:
                continue
            for candidate in remaining:
                s = _total_score(team + [candidate], weights)
                if s > best_score:
                    best_score = s
                    best_team_idx = team_idx
                    best_student = candidate

        teams[best_team_idx].append(best_student)
        remaining.remove(best_student)

    # Distribute any true leftovers (when len(students) % team_size != 0)
    leftover = [s for s in students if not any(s in t for t in teams)]
    for s in leftover:
        best_team_idx = max(
            range(len(teams)),
            key=lambda i: _total_score(teams[i] + [s], weights),
        )
        teams[best_team_idx].append(s)

    return teams


def _two_opt_improve(
    teams: List[List[Student]], weights: dict
) -> List[List[Student]]:
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
    pool = list(students)
    random.shuffle(pool)
    teams: List[List[Student]] = []
    while len(pool) >= team_size:
        teams.append(pool[:team_size])
        pool = pool[team_size:]
    for i, leftover in enumerate(pool):
        teams[i % len(teams)].append(leftover)
    return teams


def skill_based_assignment(students: List[Student], team_size: int) -> List[List[Student]]:
    skill_weights = {"skill_coverage": 1.0, "behavioral_compat": 0.0,
                     "availability_overlap": 0.0, "shared_interests": 0.0}
    raw = _greedy_build(students, team_size, skill_weights)
    return _two_opt_improve(raw, skill_weights)


# ---------------------------------------------------------------------------
# Shared team-object builder (norms generated separately, on demand)
# ---------------------------------------------------------------------------

def _build_teams(
    raw_groups: List[List[Student]],
    mode: str,
    behavioral_weights: dict,
) -> List[Team]:
    """
    Convert raw student groups to Team objects.  Norms are NOT generated here
    (LLM calls don't belong inside formation); they are generated on-demand
    in GET /teams/{id}/explanation.
    """
    skill_weights = {"skill_coverage": 1.0, "behavioral_compat": 0.0,
                     "availability_overlap": 0.0, "shared_interests": 0.0}

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

        result.append(Team(id=team_id, members=members, score_breakdown=breakdown, team_norms=[]))

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
    n_teams = len(students) // request.team_size

    # Feasibility check for floor dimensions before running optimization
    if mode == "behavioral":
        feasible_pre, infeasible_dims = _check_feasibility(students, n_teams)
        if not feasible_pre:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "infeasible_partition",
                    "message": (
                        "Too many students at the lowest level on floor dimensions "
                        "to spread them one-per-team. Cannot form valid teams."
                    ),
                    "infeasible_dims": {
                        dim: f"{count} students at floor level but only {n_teams} teams"
                        for dim, count in infeasible_dims.items()
                    },
                },
            )

    if mode == "random":
        raw_groups = random_assignment(students, request.team_size)
    elif mode == "skill":
        raw_groups = skill_based_assignment(students, request.team_size)
    else:
        raw_groups = _greedy_build(students, request.team_size, behavioral_weights)
        raw_groups = _two_opt_improve(raw_groups, behavioral_weights)
        raw_groups = _maximin_improve(raw_groups)

    result = _build_teams(raw_groups, mode, behavioral_weights)

    # Post-formation floor check (informational — optimization may not eliminate all violations)
    floor_ok, floor_violations, under_determined_teams = _check_floor_violations(raw_groups)

    # Persist teams
    db = SessionLocal()
    try:
        db.query(TeamDB).delete()
        for team in result:
            db.add(TeamDB(
                id=team.id,
                members=[m.model_dump() for m in team.members],
                score_breakdown=team.score_breakdown.model_dump(),
                team_norms=[],
                formation_mode=mode,
            ))
        db.commit()
    finally:
        db.close()

    return {
        "teams": result,
        "feasible": floor_ok,
        "floor_violations": floor_violations,
        "under_determined_teams": under_determined_teams,
    }


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
