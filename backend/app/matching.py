from __future__ import annotations
import itertools
import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException

from app.models import (
    AbilityConfig,
    AbilitySummary,
    AbilityTeamStats,
    FairnessConfig,
    FairnessSummary,
    FamiliarityClusterStats,
    FamiliarityConfig,
    FamiliarityEdge,
    FamiliaritySummary,
    FamiliarityTeamStats,
    PreferenceConfig,
    PreferenceEntry,
    PreferenceStudentStats,
    PreferenceSummary,
    ProfileResolution,
    Student,
    StudentUtility,
    Team,
    TeamGenerationRequest,
    TeamMember,
    TeamNorm,
    TeamScoreBreakdown,
    TeamSizeConfig,
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
# Floor policy for accountability/cooperativeness (high_mean_with_floor dims)
#
# rubric_v1.yaml states a zero-tolerance floor ("no team may include a
# member at the 'low' level"), but the matcher has always shipped a
# one-per-team tolerance (_check_feasibility/_greedy_build spread floor-level
# students round-robin, at most one per team). These are not equivalent
# (research_alignment_audit.md Critical #1). Rather than silently picking
# one, both are implemented and selectable per-request; the default below
# preserves the behavior this system has always actually run in production.
# ---------------------------------------------------------------------------
FLOOR_POLICY_ONE_PER_TEAM = "one_low_per_team"
FLOOR_POLICY_ZERO_TOLERANCE = "zero_tolerance"
DEFAULT_FLOOR_POLICY = FLOOR_POLICY_ONE_PER_TEAM

# ---------------------------------------------------------------------------
# Optimizer safety constants (Part 1.5 — constraint-safe 2-opt/maximin)
#
# SWAP_IMPROVEMENT_TOLERANCE: how much a candidate swap must strictly improve
#   its own stage's objective by to be accepted (both _two_opt_improve's total
#   score and _maximin_improve's partition floor-minimum use this same bar).
#   Also the numeric noise band for deterministic tie-breaking: a swap that
#   "improves" by less than this is treated as a tie and rejected, so
#   floating-point jitter can never flip an accept/reject decision.
# _FLOAT_EPS: much tighter tolerance used only to distinguish genuine hard-
#   constraint regressions from floating-point noise (never used to decide
#   whether a swap is "an improvement").
# MAX_OPTIMIZATION_SWEEPS: hard iteration cap for both local-search loops.
#   Both loops already terminate by construction (each accepted swap must
#   strictly improve a bounded, monotonically-tracked quantity — see
#   _two_opt_improve/_maximin_improve docstrings) — this is a documented
#   backstop, not the expected exit path.
# HARD_SKILL_COVERAGE_FLOOR: optional ABSOLUTE minimum per-team skill
#   coverage. 0.0 (default) means "not configured" — the always-on
#   protection is the *relative* one (never regress below what greedy
#   already achieved, see validate_assignment). Raise this to add a stricter,
#   instructor-configured absolute guarantee on top of that.
# MAXIMIN_OBJECTIVE_TOLERANCE: how much _maximin_improve is allowed to lower
#   the primary total-score objective _two_opt_improve already optimized.
#   0.0 (default) = no regression allowed at all; _two_opt_improve's optimum
#   is a floor for _maximin_improve, not a starting point it may trade away
#   (see recommended_changes.md #9 — this is the fix for that finding).
# ---------------------------------------------------------------------------
SWAP_IMPROVEMENT_TOLERANCE = 1e-6
_FLOAT_EPS = 1e-9
MAX_OPTIMIZATION_SWEEPS = 500
HARD_SKILL_COVERAGE_FLOOR = 0.0
MAXIMIN_OBJECTIVE_TOLERANCE = 0.0


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


# Default objective weights, used by _maximin_improve when a caller doesn't
# supply the weights the preceding _two_opt_improve pass actually used (e.g.
# existing call sites written before _maximin_improve needed to know the
# primary objective at all). generate_teams() always passes the request's
# real weights explicitly; this is a safe fallback, not silently wrong data.
_DEFAULT_WEIGHTS = _normalize_weights({
    "skill_coverage": 0.4, "behavioral_compat": 0.3,
    "availability_overlap": 0.2, "shared_interests": 0.1,
})


# ---------------------------------------------------------------------------
# Floor and balance (high_mean_with_floor objective)
# ---------------------------------------------------------------------------

_floor_dims_cache: Optional[dict] = None


def _floor_dims() -> dict:
    """
    Memoized (Part 6.5): a pure filter over get_rubric()'s own already-
    memoized, process-lifetime-immutable dict, so caching this too changes
    nothing observable — it was previously re-filtered on every call
    (_check_floor_violations, _greedy_build, and every optimizer pass call
    this at least once per candidate swap via _team_floor_violation_count),
    which cProfile showed as measurable overhead at n=80 (docs/
    performance.md). test_floor_dims_is_memoized_and_matches_fresh_
    computation proves the cached and from-scratch results are identical.
    """
    global _floor_dims_cache
    if _floor_dims_cache is None:
        rubric = get_rubric()
        _floor_dims_cache = {
            name: dim
            for name, dim in rubric["dimensions"].items()
            if dim.get("composition_rule") == "high_mean_with_floor"
            and dim.get("type") == "ordinal"
        }
    return _floor_dims_cache


def _is_floor_level(student: Student, dim_name: str, dim_def: dict) -> bool:
    if _is_unknown(student, dim_name):
        return False  # unknown does not violate the floor
    value = _get_student_dim(student, dim_name, dim_def)
    return value == dim_def["allowed_values"][0]  # lowest level


def _check_feasibility(
    students: List[Student], n_teams: int, policy: str = DEFAULT_FLOOR_POLICY
) -> Tuple[bool, dict]:
    """
    Returns (is_feasible, {dim_name: count_floor_level_students}).

    one_low_per_team: infeasible when any floor dim has more students at the
      lowest level than there are teams — cannot spread them one-per-team.
    zero_tolerance: infeasible when any floor dim has ANY student at the
      lowest level at all — the rubric's stated "no team may include a
      member at the low level" guarantee cannot be honored for any roster
      that contains such a student.
    """
    dims = _floor_dims()
    infeasible: dict[str, int] = {}
    limit = 0 if policy == FLOOR_POLICY_ZERO_TOLERANCE else n_teams
    for dim_name, dim_def in dims.items():
        count = sum(1 for s in students if _is_floor_level(s, dim_name, dim_def))
        if count > limit:
            infeasible[dim_name] = count
    return len(infeasible) == 0, infeasible


def _check_floor_violations(
    teams: List[List[Student]], policy: str = DEFAULT_FLOOR_POLICY
) -> Tuple[bool, List[str], List[str]]:
    """
    Returns (feasible, floor_violation_messages, under_determined_team_ids).

    floor_violation_messages: list of strings describing each violation.
    under_determined_team_ids: teams where ≥1 member has unknown floor dims.

    zero_tolerance: flags every floor-level member on every team (matches
      rubric_v1.yaml's literal wording).
    one_low_per_team: flags a team only when it has TWO OR MORE floor-level
      members on the SAME dimension — i.e. only when the "one per team"
      spread that _check_feasibility/_greedy_build target has actually
      failed, not merely when a single floor-level student (which the
      feasibility check explicitly allowed) is present.
    """
    dims = _floor_dims()
    violations: List[str] = []
    under_determined: List[str] = []

    for team_idx, team in enumerate(teams):
        team_id = f"team-{team_idx + 1}"
        has_unknown = False
        for dim_name, dim_def in dims.items():
            floor_members = []
            for student in team:
                if _is_unknown(student, dim_name):
                    has_unknown = True
                elif _is_floor_level(student, dim_name, dim_def):
                    floor_members.append(student)

            if policy == FLOOR_POLICY_ZERO_TOLERANCE:
                flagged = floor_members
            else:
                # one_low_per_team: only ≥2 floor-level members on the same
                # dim within one team violates the "spread one-per-team"
                # guarantee; a single one is the accepted tolerance.
                flagged = floor_members if len(floor_members) >= 2 else []

            for student in flagged:
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


def _partition_skill_coverage_min(teams: List[List[Student]]) -> float:
    """Minimum per-team required-skill coverage across the whole partition."""
    if not teams:
        return 1.0
    return min(_skill_coverage(t) for t in teams)


def _stage_objective(teams: List[List[Student]], weights: dict) -> float:
    """Sum of per-team _total_score — the SAME primary objective _two_opt_
    improve maximizes. Shared by _maximin_improve's acceptance rule and the
    stage diagnostics below so both reason about "the objective" identically."""
    return sum(_total_score(t, weights) for t in teams)


# ---------------------------------------------------------------------------
# Assignment invariants — one centralized hard-constraint validator, used by
# every optimization stage (2-opt, maximin) instead of each stage growing its
# own slightly-different constraint logic.
#
# Hard constraints (a candidate assignment may NEVER violate these — a
# proposed move that would is rejected outright, regardless of how much it
# improves any objective):
#   - every student present in the baseline appears exactly once; no unknown
#     student appears
#   - every team's size matches its target size; no team is unexpectedly
#     empty
#   - the partition's minimum team skill-coverage never drops below the
#     value captured right after greedy construction (research_alignment_
#     audit.md recommended_changes.md #7), and never drops below the
#     optional absolute HARD_SKILL_COVERAGE_FLOOR if one is configured
#   - the configured floor_policy's violation count never exceeds the count
#     captured right after greedy construction (normally 0)
#
# Soft objectives (may trade off against each other; NOT enforced here —
# each stage's own acceptance rule decides how to weigh them):
#   _total_score's weighted components (skill_coverage weight, behavioral_
#   compat, availability_overlap, shared_interests), _partition_floor_min
#   (maximin's own target).
#
# Evaluation-only metrics (informational; never gate acceptance anywhere in
# app/matching.py — surfaced instead via evaluation/metrics.py):
#   match_confidence, conflict_risk, blocking_moves_score_based, ability/
#   familiarity/preference/individual-utility metrics.
# ---------------------------------------------------------------------------


@dataclass
class AssignmentBaseline:
    """
    Hard-constraint reference values captured ONCE, immediately after greedy
    construction, before any optimization pass runs. Every later candidate
    swap (2-opt or maximin) is validated against this SAME baseline, so no
    later stage can silently erode a guarantee an earlier stage established.

    min_sizes/max_sizes (Part 5.5, both default None): OPT-IN per-team size
    RANGE, for validate_assignment() to accept a candidate whose team sizes
    deviate from target_sizes within [min_sizes[i], max_sizes[i]]. Every
    existing caller of .capture() (2-opt, maximin, ability/fairness/
    familiarity/preference-repair, every test written before Part 5.5)
    leaves both None, which validate_assignment() treats IDENTICALLY to
    before this field existed — exact-size enforcement against
    target_sizes, byte-for-byte the same reasons/messages. See
    app.models.TeamSizeConfig and compute_unilateral_vacancy_moves for the
    one caller that actually sets these.
    """
    student_ids: frozenset
    target_sizes: Tuple[int, ...]
    min_skill_coverage: float
    floor_policy: str
    max_floor_violations: int
    min_sizes: Optional[Tuple[int, ...]] = None
    max_sizes: Optional[Tuple[int, ...]] = None

    @staticmethod
    def capture(
        teams: List[List[Student]],
        floor_policy: str = DEFAULT_FLOOR_POLICY,
        *,
        min_sizes: Optional[Tuple[int, ...]] = None,
        max_sizes: Optional[Tuple[int, ...]] = None,
    ) -> "AssignmentBaseline":
        ids = frozenset(s.id for team in teams for s in team)
        _, violations, _ = _check_floor_violations(teams, floor_policy)
        return AssignmentBaseline(
            student_ids=ids,
            target_sizes=tuple(len(t) for t in teams),
            min_skill_coverage=_partition_skill_coverage_min(teams),
            floor_policy=floor_policy,
            max_floor_violations=len(violations),
            min_sizes=min_sizes,
            max_sizes=max_sizes,
        )


@dataclass
class ValidationResult:
    ok: bool
    reasons: List[str] = field(default_factory=list)


def _team_floor_violation_count(team: List[Student], floor_policy: str) -> int:
    """
    Number of floor-policy violations WITHIN a single team. Delegates to
    _check_floor_violations (as a one-team partition) so the violation RULE
    is defined in exactly one place; this just exposes a per-team view of it
    for the optimizer hot loops' incremental cache below.
    """
    return len(_check_floor_violations([team], floor_policy)[1])


def validate_assignment(
    teams: List[List[Student]],
    baseline: AssignmentBaseline,
    hard_skill_coverage_floor: float = HARD_SKILL_COVERAGE_FLOOR,
    *,
    precomputed_min_skill_coverage: Optional[float] = None,
    precomputed_floor_violation_count: Optional[int] = None,
) -> ValidationResult:
    """
    THE single hard-constraint check. Every optimization stage calls this
    (rather than re-implementing overlapping checks) before accepting any
    candidate assignment. See module docstring above for the exact hard vs.
    soft vs. evaluation-only split this enforces.

    `precomputed_min_skill_coverage` / `precomputed_floor_violation_count`
    let a caller that already knows these values more cheaply (e.g. an
    incremental per-team cache that only re-scores the 1-2 teams an in-
    progress local search actually touched — see _two_opt_improve /
    _maximin_improve) skip the full-partition rescan below. The comparison
    RULES are identical either way; only how the input numbers were
    obtained differs. Omit both for a plain, obviously-correct full scan
    (what a direct/manual caller — tests, API validation — gets by default).
    """
    reasons: List[str] = []

    seen_ids = [s.id for team in teams for s in team]
    if len(seen_ids) != len(set(seen_ids)):
        dupes = sorted({sid for sid in seen_ids if seen_ids.count(sid) > 1})
        reasons.append(f"student(s) appear on more than one team: {dupes}")
    seen_set = set(seen_ids)
    missing = baseline.student_ids - seen_set
    unknown = seen_set - baseline.student_ids
    if missing:
        reasons.append(f"missing student(s): {sorted(missing)}")
    if unknown:
        reasons.append(f"unknown student(s) not in the original roster: {sorted(unknown)}")

    if len(teams) != len(baseline.target_sizes):
        reasons.append(f"expected {len(baseline.target_sizes)} teams, got {len(teams)}")
    elif baseline.min_sizes is None and baseline.max_sizes is None:
        # DEFAULT exact-size mode — byte-identical to every pre-Part-5.5
        # check (same conditions, same messages, same order), so a caller
        # that never sets min_sizes/max_sizes sees NO behavior change.
        for idx, (team, target) in enumerate(zip(teams, baseline.target_sizes)):
            if len(team) != target:
                reasons.append(f"team-{idx + 1} size {len(team)} != target size {target}")
            elif target > 0 and len(team) == 0:
                reasons.append(f"team-{idx + 1} is unexpectedly empty")
    else:
        # RANGE mode — only reachable when a caller explicitly set
        # min_sizes/max_sizes on the baseline (see AssignmentBaseline.capture
        # / app.models.TeamSizeConfig / compute_unilateral_vacancy_moves).
        # A side left None falls back to the EXACT target size for that
        # side, so "only min_sizes set" means "may shrink, may never grow".
        min_sizes = baseline.min_sizes if baseline.min_sizes is not None else baseline.target_sizes
        max_sizes = baseline.max_sizes if baseline.max_sizes is not None else baseline.target_sizes
        for idx, (team, target, lo, hi) in enumerate(zip(teams, baseline.target_sizes, min_sizes, max_sizes)):
            size = len(team)
            if size < lo:
                reasons.append(f"team-{idx + 1} size {size} is below the configured minimum size {lo} (target {target})")
            elif size > hi:
                reasons.append(f"team-{idx + 1} size {size} exceeds the configured maximum size {hi} (target {target})")
            elif size == 0 and hi > 0:
                reasons.append(f"team-{idx + 1} is unexpectedly empty")

    # Coverage/floor checks only mean anything once membership & sizes hold —
    # a malformed partition shouldn't also report a spurious coverage number.
    if not reasons:
        cov_min = (
            precomputed_min_skill_coverage
            if precomputed_min_skill_coverage is not None
            else _partition_skill_coverage_min(teams)
        )
        if cov_min < baseline.min_skill_coverage - _FLOAT_EPS:
            reasons.append(
                f"partition minimum skill coverage regressed: {cov_min:.4f} < "
                f"baseline {baseline.min_skill_coverage:.4f}"
            )
        if hard_skill_coverage_floor > 0.0 and cov_min < hard_skill_coverage_floor - _FLOAT_EPS:
            reasons.append(
                f"partition minimum skill coverage {cov_min:.4f} is below the "
                f"configured hard floor {hard_skill_coverage_floor:.4f}"
            )

        violation_count = (
            precomputed_floor_violation_count
            if precomputed_floor_violation_count is not None
            else len(_check_floor_violations(teams, baseline.floor_policy)[1])
        )
        if violation_count > baseline.max_floor_violations:
            reasons.append(
                f"floor_policy ('{baseline.floor_policy}') violations increased: "
                f"{violation_count} > baseline {baseline.max_floor_violations}"
            )

    return ValidationResult(ok=not reasons, reasons=reasons)


def _stage_diagnostics(
    teams: List[List[Student]],
    weights: dict,
    floor_policy: str = DEFAULT_FLOOR_POLICY,
    target_sizes: Optional[List[int]] = None,
) -> dict:
    """
    Internal, testable snapshot of one pipeline stage's assignment quality —
    used to compare greedy vs. post-2-opt vs. post-maximin vs. final without
    duplicating the metric computations that live in evaluation/metrics.py.
    NOT part of the public API response — POST /teams/generate has no debug
    mode today, so this is not wired into it; a caller that wants this
    should import and call it directly (as the tests do).
    """
    per_team_scores = [_total_score(t, weights) for t in teams]
    per_team_coverage = [_skill_coverage(t) for t in teams]
    sizes = [len(t) for t in teams]
    floor_ok, floor_violations, under_determined = _check_floor_violations(teams, floor_policy)
    return {
        "objective_score": _stage_objective(teams, weights),
        "worst_team_score": min(per_team_scores) if per_team_scores else None,
        "min_skill_coverage": min(per_team_coverage) if per_team_coverage else None,
        "mean_skill_coverage": (sum(per_team_coverage) / len(per_team_coverage)) if per_team_coverage else None,
        "floor_ok": floor_ok,
        "floor_violations": floor_violations,
        "under_determined_teams": under_determined,
        "team_sizes": sizes,
        "team_sizes_valid": (sizes == list(target_sizes)) if target_sizes is not None else None,
        "partition_floor_min": _partition_floor_min(teams),
    }


def _maximin_improve(
    teams: List[List[Student]],
    weights: Optional[dict] = None,
    floor_policy: str = DEFAULT_FLOOR_POLICY,
    hard_skill_coverage_floor: float = HARD_SKILL_COVERAGE_FLOOR,
    objective_tolerance: float = MAXIMIN_OBJECTIVE_TOLERANCE,
    swap_tolerance: float = SWAP_IMPROVEMENT_TOLERANCE,
    max_sweeps: int = MAX_OPTIMIZATION_SWEEPS,
) -> List[List[Student]]:
    """
    Pairwise swap pass that maximises min(team_floor_mean across all teams)
    — the worst team's accountability/cooperativeness mean — WITHOUT undoing
    the primary total-score optimum _two_opt_improve already reached.

    _maximin_improve optimizes a DIFFERENT objective than _two_opt_improve
    (partition floor-minimum vs. summed total score) and runs AFTER it in
    the production pipeline (generate_teams()). Naively accepting any
    floor-min-improving swap can silently re-open total-score regressions
    2-opt had already eliminated: on the real 20-student seed roster,
    unconstrained maximin dropped the summed total score by ~5% (3.084 ->
    2.922) while reintroducing 10 swaps 2-opt would have accepted (measured
    via evaluation.metrics.blocking_moves_score_based) —
    recommended_changes.md #9, now fixed by the rule below.

    Acceptance is LEXICOGRAPHIC:
      1. validate_assignment(...) must hold — hard constraints first.
      2. the primary total-score objective may not regress by more than
         `objective_tolerance` (default 0.0: no regression at all —
         2-opt's optimum is a floor for this pass, not a budget to spend).
      3. only then must the swap strictly improve the partition floor-
         minimum by more than `swap_tolerance`.
      4. deterministic tie-break: teams/members are scanned in a fixed
         nested order and the FIRST swap satisfying 1-3 in a sweep is taken
         — no randomness, so identical input always produces identical
         output.

    Termination: every accepted swap strictly increases the partition floor
    minimum (bounded in [0, 1]) by more than `swap_tolerance` while never
    lowering the objective beyond `objective_tolerance` — a swap cannot
    "undo" a prior accepted swap's floor-min gain without decreasing it,
    which is disallowed, so the sequence cannot cycle. `max_sweeps` is a
    documented hard backstop in case that invariant is ever violated by a
    future edit.
    """
    weights = weights or _DEFAULT_WEIGHTS
    baseline = AssignmentBaseline.capture(teams, floor_policy)
    dims = _floor_dims()

    # Incremental per-team caches. A swap only ever changes two teams, so
    # rescoring every OTHER team's total_score / skill-coverage / floor-
    # violation-count / floor-dim mean on every candidate (which naive
    # full-partition recomputation would do) is wasted work that dominates
    # runtime at scale (recommended_changes.md #16) — this feeds
    # validate_assignment cheaper precomputed inputs, it does not change
    # what counts as valid or as an improvement.
    score_cache = [_total_score(t, weights) for t in teams]
    coverage_cache = [_skill_coverage(t) for t in teams]
    violation_cache = [_team_floor_violation_count(t, floor_policy) for t in teams]
    floor_mean_cache = [{dn: _ordinal_team_mean(t, dn, dd) for dn, dd in dims.items()} for t in teams]

    def _floor_min_of(cache):
        vals = [v for means in cache for v in means.values()]
        return min(vals) if vals else 1.0

    sweeps = 0
    improved = True
    while improved and sweeps < max_sweeps:
        improved = False
        sweeps += 1
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                others = [k for k in range(len(teams)) if k != i and k != j]
                others_score = sum(score_cache[k] for k in others)
                others_min_cov = min((coverage_cache[k] for k in others), default=1.0)
                others_violations = sum(violation_cache[k] for k in others)
                others_floor_vals = [v for k in others for v in floor_mean_cache[k].values()]

                for a_idx in range(len(teams[i])):
                    for b_idx in range(len(teams[j])):
                        original_floor_min = _floor_min_of(floor_mean_cache)
                        original_objective = score_cache[i] + score_cache[j] + others_score

                        teams[i][a_idx], teams[j][b_idx] = teams[j][b_idx], teams[i][a_idx]

                        new_score_i = _total_score(teams[i], weights)
                        new_score_j = _total_score(teams[j], weights)
                        new_cov_i = _skill_coverage(teams[i])
                        new_cov_j = _skill_coverage(teams[j])
                        new_viol_i = _team_floor_violation_count(teams[i], floor_policy)
                        new_viol_j = _team_floor_violation_count(teams[j], floor_policy)
                        new_means_i = {dn: _ordinal_team_mean(teams[i], dn, dd) for dn, dd in dims.items()}
                        new_means_j = {dn: _ordinal_team_mean(teams[j], dn, dd) for dn, dd in dims.items()}

                        new_objective = new_score_i + new_score_j + others_score
                        new_floor_min = min(
                            [*new_means_i.values(), *new_means_j.values(), *others_floor_vals],
                            default=1.0,
                        )
                        cov_min_candidate = min(new_cov_i, new_cov_j, others_min_cov)
                        violations_candidate = new_viol_i + new_viol_j + others_violations

                        validation = validate_assignment(
                            teams, baseline, hard_skill_coverage_floor,
                            precomputed_min_skill_coverage=cov_min_candidate,
                            precomputed_floor_violation_count=violations_candidate,
                        )

                        accept = (
                            validation.ok
                            and new_objective >= original_objective - objective_tolerance
                            and new_floor_min > original_floor_min + swap_tolerance
                        )
                        if accept:
                            improved = True
                            score_cache[i], score_cache[j] = new_score_i, new_score_j
                            coverage_cache[i], coverage_cache[j] = new_cov_i, new_cov_j
                            violation_cache[i], violation_cache[j] = new_viol_i, new_viol_j
                            floor_mean_cache[i], floor_mean_cache[j] = new_means_i, new_means_j
                        else:
                            teams[i][a_idx], teams[j][b_idx] = teams[j][b_idx], teams[i][a_idx]
    return teams


# ---------------------------------------------------------------------------
# Individual-utility / fairness framework (Part 2)
#
# Everything up to here scores a TEAM (_total_score, _behavioral_compat,
# _skill_coverage, ...) — every member of a team receives the identical
# number. This section adds a PER-STUDENT utility: how beneficial the
# proposed assignment is for one specific student, decomposed into a
# breakdown of contributing factors, built entirely from data the schema
# already has (skills, behavioral signature, availability, interests/goals).
# No preference or familiarity data is invented — those genuinely don't
# exist in the schema (see research_alignment_audit.md §4/§7) and stay
# unsupported here exactly as evaluation/metrics.py already documents.
#
# Design: each component answers "how much does THIS student, specifically,
# gain from being on THIS team" (not "how much does the student contribute
# to the team" — that's a different, contribution-oriented question):
#   skill_fit         — of the reference skills this student does NOT
#                        already have, what fraction do TEAMMATES cover?
#                        (complementary skill access — None if the student's
#                        own skills already span every reference skill,
#                        since the ratio is then undefined, not zero.)
#   behavioral_fit     — _behavioral_compat() of the student's TEAMMATES
#                        ONLY (excluding the student) — "how strong/
#                        compatible are the people I have to work with,"
#                        the Bell (2007) bad-apple cost from the RECEIVING
#                        end. Reuses the existing rubric-driven scorer
#                        as-is, so it inherits its existing unknown-data
#                        handling with no new logic.
#   availability_fit   — this student's own mean pairwise availability
#                        overlap with teammates only (None if there are no
#                        teammates to compare against).
#   interest_fit       — overlap between this student's own interests/goals
#                        and teammates' combined interests/goals (None if
#                        neither side has any interest data at all).
#
# All four components are already bounded to [0, 1] by construction (same
# normalization convention _total_score's components use), so combining
# them is a straight weighted average — no additional rescaling needed.
# Weights are configurable (FairnessConfig.weights) and RENORMALIZED over
# only the non-missing components for a given student, mirroring
# _behavioral_compat's own established renormalization idiom — a missing
# component is excluded from the weighted average, never silently treated
# as a zero.
# ---------------------------------------------------------------------------

DEFAULT_UTILITY_WEIGHTS: Dict[str, float] = {
    "skill_fit": 0.4,
    "behavioral_fit": 0.3,
    "availability_fit": 0.2,
    "interest_fit": 0.1,
}
DEFAULT_ACCEPTABLE_UTILITY_FLOOR = 0.5
MAX_EXACT_REGRET_ROSTER_SIZE = 20
# _fairness_improve's acceptance tolerances reuse the same documented
# defaults _maximin_improve established: zero regression tolerated on
# either the primary objective or the floor-minimum it protects.
FAIRNESS_OBJECTIVE_TOLERANCE = MAXIMIN_OBJECTIVE_TOLERANCE
FAIRNESS_FLOOR_MIN_TOLERANCE = MAXIMIN_OBJECTIVE_TOLERANCE


def _skill_fit(student: Student, team: List[Student]) -> Optional[float]:
    own = set(student.competence_signature.skills) & REFERENCE_SKILLS
    lacking = REFERENCE_SKILLS - own
    if not lacking:
        return None  # student's own skills already span every reference skill — ratio undefined
    teammate_skills: set = set()
    for s in team:
        if s.id != student.id:
            teammate_skills.update(s.competence_signature.skills)
    covered = lacking & teammate_skills
    return len(covered) / len(lacking)


def _behavioral_fit(student: Student, team: List[Student]) -> Optional[float]:
    teammates = [s for s in team if s.id != student.id]
    # _behavioral_compat already returns 1.0 for <2 students (no one to
    # clash with) — an existing, deliberate convention this reuses as-is
    # rather than re-deciding what "no teammates" should mean here.
    return _behavioral_compat(teammates)


def _availability_fit(student: Student, team: List[Student]) -> Optional[float]:
    teammates = [s for s in team if s.id != student.id]
    if not teammates:
        return None
    own_avail = set(student.work_rhythm_signature.availability)
    scores = []
    for mate in teammates:
        mate_avail = set(mate.work_rhythm_signature.availability)
        union = own_avail | mate_avail
        scores.append(len(own_avail & mate_avail) / len(union) if union else 0.5)
    return sum(scores) / len(scores)


def _interest_fit(student: Student, team: List[Student]) -> Optional[float]:
    teammates = [s for s in team if s.id != student.id]
    if not teammates:
        return None
    own = set(student.motivation_layer.interests + student.motivation_layer.learning_goals)
    others: set = set()
    for mate in teammates:
        others.update(mate.motivation_layer.interests + mate.motivation_layer.learning_goals)
    union = own | others
    if not union:
        return None  # neither this student nor any teammate has any interest data at all
    return len(own & others) / len(union)


def _individual_utility(
    student: Student,
    team: List[Student],
    utility_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """
    One utility value in [0, 1] for `student` on `team`, plus the component
    breakdown that produced it. Deterministic: a pure function of `student`
    and `team`'s data, no randomness, no cross-call state.
    """
    utility_weights = utility_weights or DEFAULT_UTILITY_WEIGHTS
    components: Dict[str, Optional[float]] = {
        "skill_fit": _skill_fit(student, team),
        "behavioral_fit": _behavioral_fit(student, team),
        "availability_fit": _availability_fit(student, team),
        "interest_fit": _interest_fit(student, team),
    }
    known = {k: v for k, v in components.items() if v is not None}
    missing = [k for k in components if k not in known]
    total_w = sum(utility_weights.get(k, 0.0) for k in known)
    utility = (
        sum(utility_weights.get(k, 0.0) * v for k, v in known.items()) / total_w
        if total_w > 0
        else 0.5  # every component missing — neutral fallback, mirrors _behavioral_compat's own convention
    )
    return {
        "student_id": student.id,
        "utility": utility,
        "components": components,
        "missing_components": missing,
    }


def _best_alternative_utility(
    student: Student,
    roster: List[Student],
    team_size: int,
    utility_weights: Dict[str, float],
) -> float:
    """
    Exact brute-force search: the highest utility `student` could achieve
    on ANY feasible team_size-sized team drawn from `roster` (including
    their currently assigned teammates as one of the candidate combos).
    Combinatorial — callers must bound roster size (see
    MAX_EXACT_REGRET_ROSTER_SIZE / FairnessConfig.max_exact_regret_roster_size).
    """
    others = [s for s in roster if s.id != student.id]
    best = 0.0
    for combo in itertools.combinations(others, max(0, team_size - 1)):
        hypothetical_team = [student, *combo]
        u = _individual_utility(student, hypothetical_team, utility_weights)["utility"]
        if u > best:
            best = u
    return best


def compute_assignment_utilities(
    teams: List[List[Student]],
    utility_weights: Optional[Dict[str, float]] = None,
    acceptable_utility_floor: float = DEFAULT_ACCEPTABLE_UTILITY_FLOOR,
    max_exact_regret_roster_size: int = MAX_EXACT_REGRET_ROSTER_SIZE,
) -> Dict[str, object]:
    """
    Per-student utility + assignment-level fairness metrics for a proposed
    partition. Pure evaluation — never mutates `teams` and never influences
    which assignment gets returned; that only happens if the caller
    separately runs _fairness_improve() (gated by FairnessConfig.enabled).

    Returns {"per_student": {student_id: {...}}, "summary": {...}}. See
    module docstring above for the welfare-concept mapping:
      total welfare        = summary["total_utility"]      (sum over students)
      average welfare      = summary["mean_utility"]
      worst-off welfare    = summary["min_utility"]
      equality of welfare  = summary["utility_variance"] / ["utility_stdev"]
                              (lower = more equal); summary["utility_range"]
                              is the same idea in the metric's own units.
    """
    utility_weights = utility_weights or DEFAULT_UTILITY_WEIGHTS
    roster = [s for team in teams for s in team]
    per_student: Dict[str, Dict[str, object]] = {}
    for team in teams:
        for student in team:
            per_student[student.id] = _individual_utility(student, team, utility_weights)

    n_students = len(roster)
    team_size = max((len(t) for t in teams), default=0)
    exact = 0 < n_students <= max_exact_regret_roster_size and team_size > 0

    if exact:
        for student in roster:
            assigned = per_student[student.id]["utility"]
            best_alt = _best_alternative_utility(student, roster, team_size, utility_weights)
            per_student[student.id]["regret"] = round(max(0.0, best_alt - assigned), 6)
    else:
        for student in roster:
            per_student[student.id]["regret"] = None

    utilities = [v["utility"] for v in per_student.values()]
    regrets = [v["regret"] for v in per_student.values() if v["regret"] is not None]
    below_floor = [u for u in utilities if u < acceptable_utility_floor]

    summary = {
        "mean_utility": statistics.fmean(utilities) if utilities else None,
        "min_utility": min(utilities) if utilities else None,
        "max_utility": max(utilities) if utilities else None,
        "utility_range": (max(utilities) - min(utilities)) if utilities else None,
        "utility_variance": statistics.pvariance(utilities) if len(utilities) > 1 else (0.0 if utilities else None),
        "utility_stdev": statistics.pstdev(utilities) if len(utilities) > 1 else (0.0 if utilities else None),
        "total_utility": sum(utilities) if utilities else None,
        "worst_case_regret": max(regrets) if regrets else (0.0 if exact else None),
        "mean_regret": statistics.fmean(regrets) if regrets else (0.0 if exact else None),
        "exact_regret_search": exact,
        "acceptable_utility_floor": acceptable_utility_floor,
        "students_below_floor_count": len(below_floor),
        "students_below_floor_fraction": (len(below_floor) / n_students) if n_students else None,
        "note": None if exact else (
            f"Roster of {n_students} students exceeds max_exact_regret_roster_size="
            f"{max_exact_regret_roster_size}; worst_case_regret/mean_regret were not "
            "computed (brute-force alternative-team search is combinatorial)."
        ),
    }
    return {"per_student": per_student, "summary": summary}


def _partition_min_utility(teams: List[List[Student]], utility_weights: Dict[str, float]) -> float:
    vals = [
        _individual_utility(student, team, utility_weights)["utility"]
        for team in teams
        for student in team
    ]
    return min(vals) if vals else 1.0


def _fairness_improve(
    teams: List[List[Student]],
    weights: Optional[dict] = None,
    utility_weights: Optional[Dict[str, float]] = None,
    floor_policy: str = DEFAULT_FLOOR_POLICY,
    hard_skill_coverage_floor: float = HARD_SKILL_COVERAGE_FLOOR,
    objective_tolerance: float = FAIRNESS_OBJECTIVE_TOLERANCE,
    floor_min_tolerance: float = FAIRNESS_FLOOR_MIN_TOLERANCE,
    swap_tolerance: float = SWAP_IMPROVEMENT_TOLERANCE,
    max_sweeps: int = MAX_OPTIMIZATION_SWEEPS,
) -> List[List[Student]]:
    """
    OPTIONAL post-maximin pass — only ever invoked when the caller sets
    FairnessConfig.enabled=True (generate_teams() does not call this
    otherwise, so default matching results are unaffected by this
    function's mere existence). Raises the worst-off student's individual
    utility across the WHOLE partition, without undoing what the earlier
    stages already established.

    Acceptance is LEXICOGRAPHIC, extending the same pattern
    _maximin_improve uses (see its docstring):
      1. validate_assignment(...) hard constraints must hold.
      2. the primary total-score objective may not regress by more than
         `objective_tolerance` (default 0.0 — no regression at all).
      3. the partition floor-minimum _maximin_improve raised (accountability/
         cooperativeness) may not regress by more than `floor_min_tolerance`
         (default 0.0) — this pass must not undo THAT pass's work either.
      4. only then must the swap strictly raise the partition's MINIMUM
         individual utility by more than `swap_tolerance`.
      5. deterministic tie-break: same fixed nested scan order as every
         other pass — no randomness, identical input -> identical output.

    Termination: every accepted swap strictly increases a quantity bounded
    in [0, 1] without lowering either protected quantity beyond its
    tolerance, so the sequence cannot cycle; `max_sweeps` is a documented
    hard backstop, matching the other two passes.
    """
    weights = weights or _DEFAULT_WEIGHTS
    utility_weights = utility_weights or DEFAULT_UTILITY_WEIGHTS
    baseline = AssignmentBaseline.capture(teams, floor_policy)
    dims = _floor_dims()

    # Incremental per-team caches (Part 6.5) — identical idiom to
    # _maximin_improve/_two_opt_improve/_ability_balance_improve (see their
    # docstrings). _partition_min_utility(teams, ...) is the min, over ALL
    # students, of _individual_utility() on their OWN team — algebraically
    # equal to min(per-team min-utility), so it decomposes into a per-team
    # cache exactly like objective/floor-min do. cProfile at n=40 measured
    # this pass's full-partition _stage_objective/_partition_floor_min/
    # _partition_min_utility recompute (called twice per CANDIDATE swap, an
    # O(T) operation each) as >85% of this pass's own runtime (docs/
    # performance.md) — the dominant contributor to research_balanced's
    # overall cost, since this pass runs on every roster it's enabled for.
    def _team_min_utility(team: List[Student]) -> float:
        vals = [_individual_utility(s, team, utility_weights)["utility"] for s in team]
        return min(vals) if vals else 1.0

    score_cache = [_total_score(t, weights) for t in teams]
    coverage_cache = [_skill_coverage(t) for t in teams]
    violation_cache = [_team_floor_violation_count(t, floor_policy) for t in teams]
    floor_mean_cache = [{dn: _ordinal_team_mean(t, dn, dd) for dn, dd in dims.items()} for t in teams]
    min_utility_cache = [_team_min_utility(t) for t in teams]

    def _floor_min_of(cache):
        vals = [v for means in cache for v in means.values()]
        return min(vals) if vals else 1.0

    sweeps = 0
    improved = True
    while improved and sweeps < max_sweeps:
        improved = False
        sweeps += 1
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                others = [k for k in range(len(teams)) if k != i and k != j]
                others_score = sum(score_cache[k] for k in others)
                others_min_cov = min((coverage_cache[k] for k in others), default=1.0)
                others_violations = sum(violation_cache[k] for k in others)
                others_floor_vals = [v for k in others for v in floor_mean_cache[k].values()]
                others_min_util = min((min_utility_cache[k] for k in others), default=1.0)

                for a_idx in range(len(teams[i])):
                    for b_idx in range(len(teams[j])):
                        original_objective = score_cache[i] + score_cache[j] + others_score
                        original_floor_min = _floor_min_of(floor_mean_cache)
                        original_min_utility = min(min_utility_cache[i], min_utility_cache[j], others_min_util)

                        teams[i][a_idx], teams[j][b_idx] = teams[j][b_idx], teams[i][a_idx]

                        new_score_i = _total_score(teams[i], weights)
                        new_score_j = _total_score(teams[j], weights)
                        new_cov_i = _skill_coverage(teams[i])
                        new_cov_j = _skill_coverage(teams[j])
                        new_viol_i = _team_floor_violation_count(teams[i], floor_policy)
                        new_viol_j = _team_floor_violation_count(teams[j], floor_policy)
                        new_means_i = {dn: _ordinal_team_mean(teams[i], dn, dd) for dn, dd in dims.items()}
                        new_means_j = {dn: _ordinal_team_mean(teams[j], dn, dd) for dn, dd in dims.items()}
                        new_min_util_i = _team_min_utility(teams[i])
                        new_min_util_j = _team_min_utility(teams[j])

                        new_objective = new_score_i + new_score_j + others_score
                        new_floor_min = min(
                            [*new_means_i.values(), *new_means_j.values(), *others_floor_vals],
                            default=1.0,
                        )
                        new_min_utility = min(new_min_util_i, new_min_util_j, others_min_util)
                        cov_min_candidate = min(new_cov_i, new_cov_j, others_min_cov)
                        violations_candidate = new_viol_i + new_viol_j + others_violations

                        validation = validate_assignment(
                            teams, baseline, hard_skill_coverage_floor,
                            precomputed_min_skill_coverage=cov_min_candidate,
                            precomputed_floor_violation_count=violations_candidate,
                        )

                        accept = (
                            validation.ok
                            and new_objective >= original_objective - objective_tolerance
                            and new_floor_min >= original_floor_min - floor_min_tolerance
                            and new_min_utility > original_min_utility + swap_tolerance
                        )
                        if accept:
                            improved = True
                            score_cache[i], score_cache[j] = new_score_i, new_score_j
                            coverage_cache[i], coverage_cache[j] = new_cov_i, new_cov_j
                            violation_cache[i], violation_cache[j] = new_viol_i, new_viol_j
                            floor_mean_cache[i], floor_mean_cache[j] = new_means_i, new_means_j
                            min_utility_cache[i], min_utility_cache[j] = new_min_util_i, new_min_util_j
                        else:
                            teams[i][a_idx], teams[j][b_idx] = teams[j][b_idx], teams[i][a_idx]
    return teams


def _student_utility_models(
    utility_report: Dict[str, object],
) -> Dict[str, StudentUtility]:
    """Convert compute_assignment_utilities()'s per_student dict into
    StudentUtility pydantic models, keyed by student_id."""
    return {
        sid: StudentUtility(
            student_id=sid,
            utility=data["utility"],
            components=data["components"],
            missing_components=data["missing_components"],
            regret=data["regret"],
        )
        for sid, data in utility_report["per_student"].items()
    }


def _fairness_summary_model(utility_report: Dict[str, object]) -> FairnessSummary:
    return FairnessSummary(**utility_report["summary"])


# ---------------------------------------------------------------------------
# Ability-balance framework (Part 3)
#
# Distinguishes SKILL COVERAGE (does the team collectively possess each
# required capability — _skill_coverage(), unchanged, purely about the SET
# of skills present) from ABILITY (how strong is each individual member —
# a new, separate signal). A team can have perfect skill coverage while
# every member is a beginner, or zero skill coverage while every member is
# highly able at skills the team just doesn't happen to need — these are
# orthogonal axes, and conflating them (as `experience_level`-as-proxy did
# implicitly before this) hides exactly the failure mode
# research_alignment_audit.md recommended_changes.md #2 flagged.
#
# ability_score lives on CompetenceSignature (app/models.py) as an
# Optional[float] in [0.0, 1.0], source-tagged via ability_source. Most
# students in this repo's seed data (and any pre-Part-3 record) have no
# ability_score at all — _student_ability_score() below falls back to the
# existing experience_level ordinal (beginner/intermediate/advanced ->
# 0.0/0.5/1.0) so every metric and the optional soft objective both still
# work on data that predates this schema extension, exactly as required.
# ---------------------------------------------------------------------------

_EXPERIENCE_LEVEL_TO_ABILITY_SCORE: Dict[str, float] = {
    "beginner": 0.0,
    "intermediate": 0.5,
    "advanced": 1.0,
}
DEFAULT_ABILITY_LEVEL_THRESHOLDS: Dict[str, float] = {
    "developing_max": 0.4,
    "medium_max": 0.7,
}
ABILITY_SOURCE_DIRECT = "ability_score"
ABILITY_SOURCE_FALLBACK = "experience_level_fallback"
ABILITY_SOURCE_UNKNOWN = "unknown"


def _student_ability_score(student: Student) -> Tuple[Optional[float], str]:
    """
    Resolves ONE ability value for `student` on the documented [0.0, 1.0]
    scale, plus which source produced it:
      ("ability_score", <value>)             — CompetenceSignature.ability_score was set directly
      ("experience_level_fallback", <value>) — no ability_score; derived from experience_level
      ("unknown", None)                      — neither is resolvable (experience_level not one of
                                                the 3 known values — should not happen given its
                                                default, but handled rather than assumed away)

    Never invents a value from behavioral data — mirrors the audit's own
    warning against reintroducing "checkbox relabeling"
    (research_alignment_audit.md recommended_changes.md #2).
    """
    direct = student.competence_signature.ability_score
    if direct is not None:
        return direct, ABILITY_SOURCE_DIRECT
    fallback = _EXPERIENCE_LEVEL_TO_ABILITY_SCORE.get(student.competence_signature.experience_level)
    if fallback is not None:
        return fallback, ABILITY_SOURCE_FALLBACK
    return None, ABILITY_SOURCE_UNKNOWN


def _ability_level(score: float, thresholds: Optional[Dict[str, float]] = None) -> str:
    """Buckets a continuous ability score into a 3-tier label. Thresholds
    are configurable (AbilityConfig.ability_level_thresholds) — there is no
    validated cutoff for this population; the defaults are a documented
    placeholder, same status as EvalConfig.min_acceptable_level."""
    thresholds = thresholds or DEFAULT_ABILITY_LEVEL_THRESHOLDS
    if score < thresholds["developing_max"]:
        return "developing"
    if score <= thresholds["medium_max"]:
        return "medium"
    return "high"


def _team_ability_scores(team: List[Student]) -> List[float]:
    """Known (non-unknown-source) ability scores for a team's members."""
    return [
        score
        for score, source in (_student_ability_score(s) for s in team)
        if score is not None
    ]


def _team_mean_ability(team: List[Student]) -> Optional[float]:
    scores = _team_ability_scores(team)
    return statistics.fmean(scores) if scores else None


def _partition_ability_gap(teams: List[List[Student]]) -> float:
    """Strongest-team-mean minus weakest-team-mean. 0.0 when fewer than 2
    teams have a resolvable mean (nothing to compare)."""
    means = [m for m in (_team_mean_ability(t) for t in teams) if m is not None]
    return (max(means) - min(means)) if len(means) >= 2 else 0.0


def compute_ability_metrics(
    teams: List[List[Student]],
    ability_level_thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """
    Assignment-level ability metrics, computed for whatever partition is
    passed in. Pure evaluation — never mutates `teams` and never
    influences which assignment gets returned (that only happens via
    _ability_balance_improve(), gated by AbilityConfig.enabled).

    Distinguishes, per the task definitions:
      within-team ability diversity -> per_team[i]["ability_range"/"ability_variance"]
      between-team balance          -> between_team_mean_variance / strongest_weakest_gap
    """
    thresholds = ability_level_thresholds or DEFAULT_ABILITY_LEVEL_THRESHOLDS

    per_team: List[Dict[str, object]] = []
    all_scores: List[float] = []
    overall_distribution = {"developing": 0, "medium": 0, "high": 0}

    for i, team in enumerate(teams):
        scores = _team_ability_scores(team)
        unknown_count = len(team) - len(scores)
        distribution = {"developing": 0, "medium": 0, "high": 0}
        for s in scores:
            distribution[_ability_level(s, thresholds)] += 1
            overall_distribution[_ability_level(s, thresholds)] += 1
        all_scores.extend(scores)
        per_team.append({
            "team_index": i,
            "mean_ability": statistics.fmean(scores) if scores else None,
            "ability_range": (max(scores) - min(scores)) if scores else None,
            "ability_variance": statistics.pvariance(scores) if len(scores) > 1 else (0.0 if scores else None),
            "ability_stdev": statistics.pstdev(scores) if len(scores) > 1 else (0.0 if scores else None),
            "distribution": distribution,
            "unknown_count": unknown_count,
        })

    team_means = [t["mean_ability"] for t in per_team if t["mean_ability"] is not None]
    coverage_rates = [_skill_coverage(t) for t in teams]

    correlation = None
    correlation_note = None
    if len(team_means) >= 3 and len(team_means) == len(coverage_rates) and len(set(team_means)) > 1 and len(set(coverage_rates)) > 1:
        # Pearson correlation between each team's mean ability and its
        # skill coverage rate. Needs >=3 teams with actual variance in
        # both series to be a remotely meaningful number — below that it's
        # reported as None with a note rather than a noisy/undefined value.
        try:
            correlation = statistics.correlation(team_means, coverage_rates)
        except statistics.StatisticsError:
            correlation = None
            correlation_note = "correlation undefined for this partition (zero variance in one series)."
    else:
        correlation_note = (
            f"Not computed: needs >=3 teams with non-constant mean ability AND non-constant "
            f"skill coverage to be meaningful (this partition has {len(team_means)} team(s) "
            f"with a resolvable mean ability)."
        )

    return {
        "per_team": per_team,
        "between_team_mean_variance": statistics.pvariance(team_means) if len(team_means) > 1 else (0.0 if team_means else None),
        "between_team_mean_stdev": statistics.pstdev(team_means) if len(team_means) > 1 else (0.0 if team_means else None),
        "strongest_weakest_gap": _partition_ability_gap(teams),
        "distribution": overall_distribution,
        "skill_coverage_ability_correlation": correlation,
        "correlation_note": correlation_note,
        "ability_level_thresholds": dict(thresholds),
        "note": (
            "mean_ability/ability_range/ability_variance use ability_score when a student has "
            "one, falling back to experience_level (beginner/intermediate/advanced -> "
            "0.0/0.5/1.0) otherwise — see app.matching._student_ability_score. 'Balanced ability' "
            "is a research hypothesis, not a validated target — see AbilityConfig's docstring."
        ),
    }


def _ability_summary_model(report: Dict[str, object]) -> AbilitySummary:
    return AbilitySummary(
        per_team=[AbilityTeamStats(**t) for t in report["per_team"]],
        between_team_mean_variance=report["between_team_mean_variance"],
        between_team_mean_stdev=report["between_team_mean_stdev"],
        strongest_weakest_gap=report["strongest_weakest_gap"],
        distribution=report["distribution"],
        skill_coverage_ability_correlation=report["skill_coverage_ability_correlation"],
        correlation_note=report["correlation_note"],
        ability_level_thresholds=report["ability_level_thresholds"],
        note=report["note"],
    )


def _ability_balance_improve(
    teams: List[List[Student]],
    weights: Optional[dict] = None,
    floor_policy: str = DEFAULT_FLOOR_POLICY,
    hard_skill_coverage_floor: float = HARD_SKILL_COVERAGE_FLOOR,
    objective_tolerance: float = MAXIMIN_OBJECTIVE_TOLERANCE,
    floor_min_tolerance: float = MAXIMIN_OBJECTIVE_TOLERANCE,
    swap_tolerance: float = SWAP_IMPROVEMENT_TOLERANCE,
    max_sweeps: int = MAX_OPTIMIZATION_SWEEPS,
) -> List[List[Student]]:
    """
    OPTIONAL post-maximin pass — only ever invoked when the caller sets
    AbilityConfig.enabled=True (generate_teams() does not call this
    otherwise, so default matching results are unaffected by this
    function's mere existence). Narrows the gap between the strongest and
    weakest team's MEAN ability across the whole partition, without undoing
    what earlier stages already established.

    Unlike _maximin_improve (raise the worst team's floor-dim mean) or
    _fairness_improve (raise the worst-off STUDENT's utility), this pass
    targets a BETWEEN-team comparability goal: it accepts a swap only if it
    STRICTLY REDUCES (strongest_team_mean_ability - weakest_team_mean_
    ability) — i.e. it is trying to make teams more comparable in overall
    strength, not to maximize anyone's ability or push every team's mean as
    high as possible. Whether comparable-strength teams are actually
    desirable is a research question this repo does not have outcome data
    to answer (see AbilityConfig's docstring) — this pass makes the
    HYPOTHESIS operational and testable, it does not assert it is correct.

    Acceptance is LEXICOGRAPHIC, same pattern as _maximin_improve/
    _fairness_improve:
      1. validate_assignment(...) hard constraints must hold.
      2. the primary total-score objective may not regress by more than
         `objective_tolerance` (default 0.0).
      3. the accountability/cooperativeness floor-minimum _maximin_improve
         established may not regress by more than `floor_min_tolerance`
         (default 0.0).
      4. only then must the swap strictly reduce the strongest-weakest
         ability gap by more than `swap_tolerance`.
      5. deterministic tie-break: same fixed nested scan order as every
         other pass.

    Termination: every accepted swap strictly decreases a quantity bounded
    below by 0, without lowering either protected quantity beyond its
    tolerance; `max_sweeps` is the same documented hard backstop the other
    passes use.

    KNOWN LIMITATION: if BOTH ability.enabled and fairness.enabled are set,
    generate_teams() runs this pass BEFORE _fairness_improve, and
    _fairness_improve does not itself protect the ability gap this pass
    narrows — running both is not guaranteed to preserve both properties
    simultaneously. Documented rather than silently assumed away.
    """
    weights = weights or _DEFAULT_WEIGHTS
    baseline = AssignmentBaseline.capture(teams, floor_policy)
    dims = _floor_dims()

    # Incremental per-team caches (Part 6.5) — identical idiom to
    # _maximin_improve/_two_opt_improve (see their docstrings): a swap only
    # ever changes two teams, so every OTHER team's score/coverage/floor-
    # violation-count/floor-mean/ability-mean is read from cache instead of
    # rescanned on every candidate. cProfile at n=40/n=80 (docs/
    # performance.md) showed this full-partition rescan was the dominant
    # runtime cost of every optional pass that lacked it; this does not
    # change what counts as valid or as an improvement, only how the
    # comparison values are obtained (proven in
    # test_partition_score_cache_matches_from_scratch_computation and the
    # Part 6.5 before/after equivalence harness).
    score_cache = [_total_score(t, weights) for t in teams]
    coverage_cache = [_skill_coverage(t) for t in teams]
    violation_cache = [_team_floor_violation_count(t, floor_policy) for t in teams]
    floor_mean_cache = [{dn: _ordinal_team_mean(t, dn, dd) for dn, dd in dims.items()} for t in teams]
    ability_cache: List[Optional[float]] = [_team_mean_ability(t) for t in teams]

    def _floor_min_of(cache):
        vals = [v for means in cache for v in means.values()]
        return min(vals) if vals else 1.0

    def _gap_of(values):
        known = [v for v in values if v is not None]
        return (max(known) - min(known)) if len(known) >= 2 else 0.0

    sweeps = 0
    improved = True
    while improved and sweeps < max_sweeps:
        improved = False
        sweeps += 1
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                others = [k for k in range(len(teams)) if k != i and k != j]
                others_score = sum(score_cache[k] for k in others)
                others_min_cov = min((coverage_cache[k] for k in others), default=1.0)
                others_violations = sum(violation_cache[k] for k in others)
                others_floor_vals = [v for k in others for v in floor_mean_cache[k].values()]
                others_ability = [ability_cache[k] for k in others]

                for a_idx in range(len(teams[i])):
                    for b_idx in range(len(teams[j])):
                        original_objective = score_cache[i] + score_cache[j] + others_score
                        original_floor_min = _floor_min_of(floor_mean_cache)
                        original_gap = _gap_of([ability_cache[i], ability_cache[j], *others_ability])

                        teams[i][a_idx], teams[j][b_idx] = teams[j][b_idx], teams[i][a_idx]

                        new_score_i = _total_score(teams[i], weights)
                        new_score_j = _total_score(teams[j], weights)
                        new_cov_i = _skill_coverage(teams[i])
                        new_cov_j = _skill_coverage(teams[j])
                        new_viol_i = _team_floor_violation_count(teams[i], floor_policy)
                        new_viol_j = _team_floor_violation_count(teams[j], floor_policy)
                        new_means_i = {dn: _ordinal_team_mean(teams[i], dn, dd) for dn, dd in dims.items()}
                        new_means_j = {dn: _ordinal_team_mean(teams[j], dn, dd) for dn, dd in dims.items()}
                        new_ability_i = _team_mean_ability(teams[i])
                        new_ability_j = _team_mean_ability(teams[j])

                        new_objective = new_score_i + new_score_j + others_score
                        new_floor_min = min(
                            [*new_means_i.values(), *new_means_j.values(), *others_floor_vals],
                            default=1.0,
                        )
                        new_gap = _gap_of([new_ability_i, new_ability_j, *others_ability])
                        cov_min_candidate = min(new_cov_i, new_cov_j, others_min_cov)
                        violations_candidate = new_viol_i + new_viol_j + others_violations

                        validation = validate_assignment(
                            teams, baseline, hard_skill_coverage_floor,
                            precomputed_min_skill_coverage=cov_min_candidate,
                            precomputed_floor_violation_count=violations_candidate,
                        )

                        accept = (
                            validation.ok
                            and new_objective >= original_objective - objective_tolerance
                            and new_floor_min >= original_floor_min - floor_min_tolerance
                            and new_gap < original_gap - swap_tolerance
                        )
                        if accept:
                            improved = True
                            score_cache[i], score_cache[j] = new_score_i, new_score_j
                            coverage_cache[i], coverage_cache[j] = new_cov_i, new_cov_j
                            violation_cache[i], violation_cache[j] = new_viol_i, new_viol_j
                            floor_mean_cache[i], floor_mean_cache[j] = new_means_i, new_means_j
                            ability_cache[i], ability_cache[j] = new_ability_i, new_ability_j
                        else:
                            teams[i][a_idx], teams[j][b_idx] = teams[j][b_idx], teams[i][a_idx]
    return teams


# ---------------------------------------------------------------------------
# Familiarity / anti-clique framework (Part 4, revised Part 4.5)
#
# Familiarity is a RELATIONSHIP between two students (app.models.
# FamiliarityEdge / the `familiarity` DB table, app/familiarity.py), not a
# property of one — this section builds a graph from those edges and
# computes team-level/assignment-level metrics over it, plus an optional
# soft objective that nudges each team's density toward a configured
# target band.
#
# GRAPH DEFINITION (plain Python, no networkx — the roster sizes this
# system targets, per recommended_changes.md #16, don't warrant the
# dependency):
#   nodes = every student across all teams passed in
#   edges = FamiliarityEdge rows; strength IS the edge weight. A pair with
#           NO recorded row is simply absent from the graph — this is what
#           keeps "unknown" distinguishable from "confirmed strength 0.0"
#           everywhere below (a strength=0.0 edge DOES appear in the
#           graph, contributing a known-but-zero data point; a missing
#           pair does not appear at all and is excluded from density
#           averages rather than silently pulling them toward zero).
#
# connected_familiarity_cluster (Part 4.5 — renamed from "pre-existing
# subgroup", and the underlying computation was corrected, not just
# relabeled): for ONE team, build the TEAM-INDUCED subgraph — nodes are
# that team's own members ONLY, edges are familiarity edges BETWEEN THOSE
# MEMBERS ONLY (any edge to a student outside the team is excluded), kept
# only when >= clique_edge_threshold — then take its connected components.
# The Part 4 implementation instead computed connected components on the
# GLOBAL graph (every student across every team) and intersected each
# component with a team afterward. That could count two teammates as
# "clustered" when they were connected only through a PATH THROUGH A
# STUDENT ON A DIFFERENT TEAM — e.g. A-C and C-B both strong, A and B on
# the same team, C on another team: A and B share no direct or
# team-internal edge at all, yet the old code placed them in the same
# reported subgroup. The team-induced subgraph cannot make that mistake —
# a path through an outside student is structurally absent from it.
#
# This is also why the concept is called a CLUSTER, not a "clique": a
# connected component only requires a PATH between any two members through
# OTHER MEMBERS OF THE SAME CLUSTER — it does NOT require every pair to be
# directly connected (that would be a clique, a strictly stronger claim).
# A chain (A-B-C-D, each link strong, but A-C/A-D/B-D unknown) and a star
# (one hub connected to 3 spokes who aren't connected to each other) both
# form ONE connected cluster of size 4 despite looking nothing like a
# clique — FamiliarityClusterStats' cohesion fields (known_internal_edge_
# coverage, mean_known_internal_strength, thresholded_internal_edge_
# density, is_quasi_clique) exist specifically so a caller can tell a loose
# chain/star apart from a tightly-connected group without this module ever
# asserting a cluster IS a clique.
# ---------------------------------------------------------------------------

DEFAULT_FAMILIAR_PAIR_THRESHOLD = 0.3
DEFAULT_CLIQUE_EDGE_THRESHOLD = 0.5
DEFAULT_FAMILIARITY_LOWER_BOUND = 0.15
DEFAULT_FAMILIARITY_UPPER_BOUND = 0.6
DEFAULT_MAX_CLIQUE_FRACTION = 0.5
DEFAULT_MINIMUM_KNOWN_PAIR_COVERAGE = 0.5
DEFAULT_QUASI_CLIQUE_DENSITY_THRESHOLD = 0.75
FAMILIARITY_OBJECTIVE_TOLERANCE = MAXIMIN_OBJECTIVE_TOLERANCE
FAMILIARITY_FLOOR_MIN_TOLERANCE = MAXIMIN_OBJECTIVE_TOLERANCE


def build_familiarity_graph(
    students: List[Student],
    edges: List[FamiliarityEdge],
) -> Dict[str, Dict[str, float]]:
    """
    Adjacency-dict graph: student_id -> {other_student_id: strength}, built
    ONLY from `edges` whose both endpoints are present in `students`.
    Symmetric by construction (see FamiliarityEdge's directionality
    docstring): every edge is inserted in both directions. A pair absent
    from the resulting graph is UNKNOWN, not zero.

    Iterates `students` in its OWN given order (not a set) when seeding the
    graph's keys, so the graph's key order — and therefore every
    downstream deterministic tie-break that scans it — depends only on the
    deterministic order `students` was built in, never on Python's
    per-process string-hash randomization.
    """
    roster_ids = [s.id for s in students]
    graph: Dict[str, Dict[str, float]] = {}
    for sid in roster_ids:
        graph.setdefault(sid, {})
    for edge in edges:
        a, b = edge.student_a_id, edge.student_b_id
        if a in graph and b in graph and a != b:
            graph[a][b] = edge.strength
            graph[b][a] = edge.strength
    return graph


def _known_familiarity(graph: Dict[str, Dict[str, float]], a: str, b: str) -> Optional[float]:
    """None means UNKNOWN (no edge recorded) — never coerced to 0.0."""
    return graph.get(a, {}).get(b)


def _is_familiar_pair(graph: Dict[str, Dict[str, float]], a: str, b: str, threshold: float) -> bool:
    """True only for a KNOWN edge >= threshold. An unknown pair is never
    treated as familiar regardless of threshold (including threshold=0.0 —
    unlike a naive `strength or sentinel` check, this never conflates a
    confirmed strength of exactly 0.0 with 'no data')."""
    s = _known_familiarity(graph, a, b)
    return s is not None and s >= threshold


def _connected_components(
    graph: Dict[str, Dict[str, float]],
    edge_threshold: float,
) -> List[frozenset]:
    """
    Connected components of the subgraph restricted to edges with
    strength >= edge_threshold. Every node in `graph` appears in exactly
    one component (a node with no qualifying edge forms its own singleton
    component). Plain BFS over the adjacency dict — no external graph
    library needed at this scale. Component ORDER in the returned list
    follows `graph`'s own key order, which callers rely on for
    deterministic tie-breaking (see e.g. _largest_team_cluster).
    """
    visited: set = set()
    components: List[frozenset] = []
    for start in graph:
        if start in visited:
            continue
        component = set()
        queue = [start]
        visited.add(start)
        while queue:
            current = queue.pop()
            component.add(current)
            for neighbor, strength in graph.get(current, {}).items():
                if strength >= edge_threshold and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        components.append(frozenset(component))
    return components


def _induced_team_graph(
    team: List[Student],
    graph: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """
    Subgraph of `graph` restricted to `team`'s members and the edges
    BETWEEN THEM ONLY — any edge to a student outside this team is
    excluded. This is what makes connected-component analysis over the
    result reflect TEAM-INTERNAL structure exclusively: a path through an
    outside student cannot exist here (see this section's module
    docstring for the Part 4 bug this fixes).

    Iterates `team` in its OWN given order (not a set) so the resulting
    dict's key order — and therefore _connected_components' deterministic
    tie-breaking on it — depends only on `team`'s own deterministic order.
    """
    team_ids = [s.id for s in team]
    member_set = set(team_ids)
    induced: Dict[str, Dict[str, float]] = {sid: {} for sid in team_ids}
    for sid in team_ids:
        for other_id, strength in graph.get(sid, {}).items():
            if other_id in member_set:
                induced[sid][other_id] = strength
    return induced


def _team_pair_stats(team: List[Student], graph: Dict[str, Dict[str, float]]) -> Dict[str, object]:
    """
    Within-team pairwise familiarity stats, independent of any threshold:
    known/unknown/confirmed-zero pair counts, density (mean strength over
    KNOWN pairs), and known_pair_coverage (known / total_possible). Total
    possible pairs is 0 for a team of size 0 or 1 — handled explicitly
    (coverage/density are None, never a division error).
    """
    n = len(team)
    total = n * (n - 1) // 2
    known = 0
    confirmed_zero = 0
    strengths: List[float] = []
    for a, b in itertools.combinations(team, 2):
        s = _known_familiarity(graph, a.id, b.id)
        if s is None:
            continue
        known += 1
        strengths.append(s)
        if s == 0.0:
            confirmed_zero += 1
    unknown = total - known
    density = statistics.fmean(strengths) if strengths else None
    coverage = (known / total) if total else None
    return {
        "total_possible_pairs": total,
        "known_pair_count": known,
        "unknown_pair_count": unknown,
        "confirmed_zero_pair_count": confirmed_zero,
        "density": density,
        "known_pair_coverage": coverage,
    }


def _team_familiarity_density(
    team: List[Student],
    graph: Dict[str, Dict[str, float]],
) -> Tuple[Optional[float], int, int]:
    """DEPRECATED (Part 4.5) — use _team_pair_stats for the fuller known/
    unknown/confirmed-zero/coverage breakdown this delegates to. Kept only
    for backward compatibility with existing call sites; returns exactly
    (density, known_pair_count, unknown_pair_count) as before."""
    stats = _team_pair_stats(team, graph)
    return stats["density"], stats["known_pair_count"], stats["unknown_pair_count"]


def _team_familiar_pair_count(team: List[Student], graph: Dict[str, Dict[str, float]], familiar_pair_threshold: float) -> int:
    return sum(
        1 for a, b in itertools.combinations(team, 2)
        if _is_familiar_pair(graph, a.id, b.id, familiar_pair_threshold)
    )


def _cluster_stats(
    members: frozenset,
    graph: Dict[str, Dict[str, float]],
    clique_edge_threshold: float,
    quasi_clique_density_threshold: Optional[float],
) -> Dict[str, object]:
    """
    Cohesion measures for ONE connected_familiarity_cluster, computed over
    pairs WITHIN `members` only (never against anyone outside the
    cluster). `thresholded_internal_edge_density` — the fraction of ALL
    possible internal pairs with a KNOWN edge >= clique_edge_threshold —
    is what "how complete/tight is this cluster's own subgraph" means
    here; it is 1.0 only when the cluster is a genuine clique (every pair
    directly, strongly connected), and meaningfully LESS than 1.0 for a
    chain or star, which is exactly the distinction this field exists to
    surface. `known_internal_edge_coverage` is a DIFFERENT thing — how
    much DATA exists among internal pairs, regardless of strength.
    """
    member_list = sorted(members)
    n = len(member_list)
    total_pairs = n * (n - 1) // 2
    known = 0
    strengths: List[float] = []
    thresholded = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = _known_familiarity(graph, member_list[i], member_list[j])
            if s is None:
                continue
            known += 1
            strengths.append(s)
            if s >= clique_edge_threshold:
                thresholded += 1
    unknown = total_pairs - known
    coverage = (known / total_pairs) if total_pairs else None
    mean_strength = statistics.fmean(strengths) if strengths else None
    density = (thresholded / total_pairs) if total_pairs else None
    is_quasi_clique = None
    if quasi_clique_density_threshold is not None and total_pairs > 0:
        is_quasi_clique = density is not None and density >= quasi_clique_density_threshold
    return {
        "size": n,
        "known_internal_pairs": known,
        "unknown_internal_pairs": unknown,
        "total_internal_pairs": total_pairs,
        "known_internal_edge_coverage": coverage,
        "mean_known_internal_strength": mean_strength,
        "thresholded_internal_edge_density": density,
        "is_quasi_clique": is_quasi_clique,
    }


def _largest_team_cluster(
    team: List[Student],
    graph: Dict[str, Dict[str, float]],
    clique_edge_threshold: float,
    quasi_clique_density_threshold: Optional[float] = None,
) -> Dict[str, object]:
    """
    The largest connected_familiarity_cluster on `team`'s OWN team-induced
    subgraph (see _induced_team_graph) — it can NEVER include a
    "connection" that only exists through a student on a different team.

    Deterministic tie-breaking: components are discovered by scanning
    _induced_team_graph's keys in `team`'s own given order (that
    function's docstring), and Python's max() returns the FIRST
    maximal-length element it encounters — so the same team composition
    always yields the same "largest" cluster, matching every other pass's
    tie-breaking convention in this module. When no pair meets
    clique_edge_threshold at all, this reports a trivial size-1 cluster
    (the team's own first member) rather than a size-0/None result — the
    same "isolated node is its own singleton component" convention
    _connected_components already uses.
    """
    if not team:
        stats = _cluster_stats(frozenset(), graph, clique_edge_threshold, quasi_clique_density_threshold)
        stats["fraction_of_team"] = 0.0
        return stats
    induced = _induced_team_graph(team, graph)
    clusters = _connected_components(induced, clique_edge_threshold)
    non_trivial = [c for c in clusters if len(c) > 1]
    largest = max(non_trivial, key=len) if non_trivial else frozenset([team[0].id])
    stats = _cluster_stats(largest, graph, clique_edge_threshold, quasi_clique_density_threshold)
    stats["fraction_of_team"] = stats["size"] / len(team)
    return stats


def _band_status(
    density: Optional[float],
    lower: float,
    upper: float,
    *,
    known_pair_coverage: Optional[float] = None,
    minimum_known_pair_coverage: float = 0.0,
    total_possible_pairs: Optional[int] = None,
) -> Optional[str]:
    """
    "not_applicable"      total_possible_pairs == 0 (team of size 0 or 1 —
                           there is structurally nothing to classify)
    "insufficient_data"   zero known pairs (density is None while there ARE
                           possible pairs), OR known_pair_coverage is below
                           minimum_known_pair_coverage — not enough
                           recorded data among these specific teammates to
                           trust a density judgment
    "below" / "within" / "above"   normal band classification

    Backward compatible: a caller that omits the keyword-only coverage
    arguments entirely (as pre-Part-4.5 callers do) gets exactly the old
    behavior — None only when density is None, otherwise a plain
    below/within/above classification — since total_possible_pairs stays
    None (never triggers "not_applicable") and known_pair_coverage stays
    None (never triggers the coverage-based "insufficient_data" branch).
    """
    if total_possible_pairs == 0:
        return "not_applicable"
    if density is None:
        return "insufficient_data" if total_possible_pairs is not None else None
    if known_pair_coverage is not None and known_pair_coverage < minimum_known_pair_coverage:
        return "insufficient_data"
    if density < lower:
        return "below"
    if density > upper:
        return "above"
    return "within"


def _band_violation(
    density: Optional[float],
    lower: float,
    upper: float,
    *,
    known_pair_coverage: Optional[float] = None,
    minimum_known_pair_coverage: float = 0.0,
    penalize_insufficient_data: bool = False,
) -> float:
    """
    0.0 when density is unknown, within the band, or exactly at a bound;
    otherwise the distance outside the band. Used only by the optional
    soft objective below.

    Missing-data policy (Part 4.5): when `known_pair_coverage` is supplied
    and below `minimum_known_pair_coverage`, this returns 0.0 regardless of
    density UNLESS `penalize_insufficient_data=True` — the optimizer must
    not "confidently" chase a density-band target it doesn't have enough
    evidence to judge. Default False matches FamiliarityConfig's
    conservative default. Backward compatible: a caller that omits the
    coverage arguments (pre-Part-4.5 call sites) never triggers this gate
    at all (known_pair_coverage stays None), so behavior is unchanged.
    """
    if not penalize_insufficient_data and known_pair_coverage is not None and known_pair_coverage < minimum_known_pair_coverage:
        return 0.0
    if density is None:
        return 0.0
    if density < lower:
        return lower - density
    if density > upper:
        return density - upper
    return 0.0


def _team_clique_violation(
    team: List[Student],
    graph: Dict[str, Dict[str, float]],
    clique_edge_threshold: float,
    max_clique_fraction: float,
) -> float:
    """
    How far this team's largest connected_familiarity_cluster (computed
    from the TEAM-INDUCED subgraph, never a global-component intersection
    — see this section's module docstring) exceeds `max_clique_fraction`
    of the team, as a fraction (0.0 = within bounds).

    This is DELIBERATELY separate from the density band violation: density
    is a mean over KNOWN pairs only, so a uniformly-strong cluster's
    density does not change as long as >=2 of its members remain together
    on a team (removing one member still leaves the same strength among
    the rest) — density alone is therefore insensitive to shrinking a
    cluster one member at a time. Cluster SIZE, in contrast, responds
    immediately: swapping one cluster member out for an outsider directly
    reduces the team-induced largest-cluster size. See the Part 4 report's
    limitations section for the full explanation.
    """
    if not team:
        return 0.0
    cluster = _largest_team_cluster(team, graph, clique_edge_threshold, quasi_clique_density_threshold=None)
    fraction = cluster["size"] / len(team) if cluster["size"] > 1 else 0.0
    return max(0.0, fraction - max_clique_fraction)


def _team_familiarity_violation(
    team: List[Student],
    graph: Dict[str, Dict[str, float]],
    lower: float,
    upper: float,
    clique_edge_threshold: float,
    max_clique_fraction: float,
    minimum_known_pair_coverage: float,
    penalize_insufficient_data: bool,
) -> float:
    """Combined density-band + clique-concentration violation for ONE
    team — see _band_violation and _team_clique_violation."""
    stats = _team_pair_stats(team, graph)
    density_violation = _band_violation(
        stats["density"], lower, upper,
        known_pair_coverage=stats["known_pair_coverage"],
        minimum_known_pair_coverage=minimum_known_pair_coverage,
        penalize_insufficient_data=penalize_insufficient_data,
    )
    cluster_violation = _team_clique_violation(team, graph, clique_edge_threshold, max_clique_fraction)
    return density_violation + cluster_violation


def _partition_familiarity_violation(
    teams: List[List[Student]],
    graph: Dict[str, Dict[str, float]],
    lower: float,
    upper: float,
    clique_edge_threshold: float = DEFAULT_CLIQUE_EDGE_THRESHOLD,
    max_clique_fraction: float = DEFAULT_MAX_CLIQUE_FRACTION,
    minimum_known_pair_coverage: float = DEFAULT_MINIMUM_KNOWN_PAIR_COVERAGE,
    penalize_insufficient_data: bool = False,
) -> float:
    """Sum of _team_familiarity_violation over every team. Convenience
    one-off entry point (tests, diagnostics); _familiarity_balance_improve
    caches per-team violation values internally rather than calling this
    on the whole partition per candidate swap."""
    return sum(
        _team_familiarity_violation(
            team, graph, lower, upper, clique_edge_threshold, max_clique_fraction,
            minimum_known_pair_coverage, penalize_insufficient_data,
        )
        for team in teams
    )


def compute_familiarity_metrics(
    teams: List[List[Student]],
    edges: List[FamiliarityEdge],
    familiar_pair_threshold: float = DEFAULT_FAMILIAR_PAIR_THRESHOLD,
    clique_edge_threshold: float = DEFAULT_CLIQUE_EDGE_THRESHOLD,
    lower_bound: float = DEFAULT_FAMILIARITY_LOWER_BOUND,
    upper_bound: float = DEFAULT_FAMILIARITY_UPPER_BOUND,
    minimum_known_pair_coverage: float = DEFAULT_MINIMUM_KNOWN_PAIR_COVERAGE,
    quasi_clique_density_threshold: Optional[float] = DEFAULT_QUASI_CLIQUE_DENSITY_THRESHOLD,
) -> Dict[str, object]:
    """
    Assignment-level familiarity metrics for a proposed partition, given
    the FULL set of known familiarity edges (typically every row in the
    `familiarity` table). Pure evaluation — never mutates `teams` and never
    influences which assignment gets returned (that only happens via
    _familiarity_balance_improve(), gated by FamiliarityConfig.enabled).
    """
    all_students = [s for team in teams for s in team]
    graph = build_familiarity_graph(all_students, edges)

    per_team: List[Dict[str, object]] = []
    total_known = 0
    total_possible = 0
    total_confirmed_zero = 0
    total_familiar = 0

    for i, team in enumerate(teams):
        stats = _team_pair_stats(team, graph)
        familiar_count = _team_familiar_pair_count(team, graph, familiar_pair_threshold)

        total_possible += stats["total_possible_pairs"]
        total_known += stats["known_pair_count"]
        total_confirmed_zero += stats["confirmed_zero_pair_count"]
        total_familiar += familiar_count

        # Structural warnings (who has no familiar teammate) are preserved
        # even under insufficient_data — they're supported by whatever
        # known edges DO exist; only the density-BAND classification (which
        # needs enough coverage to trust a single aggregate number) is
        # suppressed. known_pair_coverage on this same record is the
        # "evidence coverage" label a caller uses to weigh isolated_members
        # accordingly.
        isolated: List[str] = []
        if len(team) >= 2:
            for s in team:
                has_familiar_teammate = any(
                    _is_familiar_pair(graph, s.id, o.id, familiar_pair_threshold)
                    for o in team if o.id != s.id
                )
                if not has_familiar_teammate:
                    isolated.append(s.name)

        cluster = _largest_team_cluster(team, graph, clique_edge_threshold, quasi_clique_density_threshold)

        band = _band_status(
            stats["density"], lower_bound, upper_bound,
            known_pair_coverage=stats["known_pair_coverage"],
            minimum_known_pair_coverage=minimum_known_pair_coverage,
            total_possible_pairs=stats["total_possible_pairs"],
        )

        per_team.append({
            "team_index": i,
            "total_possible_pairs": stats["total_possible_pairs"],
            "known_pair_count": stats["known_pair_count"],
            "unknown_pair_count": stats["unknown_pair_count"],
            "confirmed_zero_pair_count": stats["confirmed_zero_pair_count"],
            "known_pair_coverage": stats["known_pair_coverage"],
            "familiar_pair_count": familiar_count,
            "familiar_pair_fraction": (familiar_count / stats["known_pair_count"]) if stats["known_pair_count"] else None,
            "density": stats["density"],
            "isolated_members": isolated,
            "largest_connected_cluster": cluster,
            "band_status": band,
        })

    densities = [t["density"] for t in per_team if t["density"] is not None]
    band_counts = {"below": 0, "within": 0, "above": 0, "insufficient_data": 0, "not_applicable": 0}
    for t in per_team:
        if t["band_status"] is not None:
            band_counts[t["band_status"]] += 1

    return {
        "per_team": per_team,
        "mean_density": statistics.fmean(densities) if densities else None,
        "cross_team_density_variance": statistics.pvariance(densities) if len(densities) > 1 else (0.0 if densities else None),
        "cross_team_density_stdev": statistics.pstdev(densities) if len(densities) > 1 else (0.0 if densities else None),
        "total_known_pairs": total_known,
        "total_possible_pairs": total_possible,
        "total_confirmed_zero_pairs": total_confirmed_zero,
        "total_familiar_pairs": total_familiar,
        "teams_below_band": band_counts["below"],
        "teams_within_band": band_counts["within"],
        "teams_above_band": band_counts["above"],
        "teams_insufficient_data": band_counts["insufficient_data"],
        "teams_not_applicable": band_counts["not_applicable"],
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "minimum_known_pair_coverage": minimum_known_pair_coverage,
        "note": (
            "density is the mean familiarity strength over KNOWN pairs only; a pair with no "
            "recorded FamiliarityDB row is excluded from the average entirely, not treated as "
            "0.0. 'largest_connected_cluster' is computed from the TEAM-INDUCED subgraph (edges "
            f"between this team's own members only, at/above clique_edge_threshold={clique_edge_threshold}) "
            "— a connected COMPONENT, not necessarily a clique (a chain or star can form one "
            "cluster without every pair being directly connected; see its cohesion fields to tell "
            "them apart). band_status is 'not_applicable' for teams with 0-1 members, "
            "'insufficient_data' when known_pair_coverage is below minimum_known_pair_coverage="
            f"{minimum_known_pair_coverage} (not enough recorded data among these specific "
            "teammates to trust a density judgment — isolated_members and the cluster above "
            "remain populated from whatever known edges DO exist, only the band classification "
            "is suppressed), otherwise below/within/above the configured band. Neither the band "
            "bounds nor the pair/cluster/coverage thresholds are scientifically established for "
            "this population — see FamiliarityConfig's docstring."
        ),
    }


def _familiarity_summary_model(report: Dict[str, object]) -> FamiliaritySummary:
    return FamiliaritySummary(
        per_team=[
            FamiliarityTeamStats(
                **{k: v for k, v in t.items() if k != "largest_connected_cluster"},
                largest_connected_cluster=FamiliarityClusterStats(**t["largest_connected_cluster"]),
            )
            for t in report["per_team"]
        ],
        mean_density=report["mean_density"],
        cross_team_density_variance=report["cross_team_density_variance"],
        cross_team_density_stdev=report["cross_team_density_stdev"],
        total_known_pairs=report["total_known_pairs"],
        total_possible_pairs=report["total_possible_pairs"],
        total_confirmed_zero_pairs=report["total_confirmed_zero_pairs"],
        total_familiar_pairs=report["total_familiar_pairs"],
        teams_below_band=report["teams_below_band"],
        teams_within_band=report["teams_within_band"],
        teams_above_band=report["teams_above_band"],
        teams_insufficient_data=report["teams_insufficient_data"],
        teams_not_applicable=report["teams_not_applicable"],
        lower_bound=report["lower_bound"],
        upper_bound=report["upper_bound"],
        minimum_known_pair_coverage=report["minimum_known_pair_coverage"],
        note=report["note"],
    )


def _familiarity_balance_improve(
    teams: List[List[Student]],
    edges: List[FamiliarityEdge],
    weights: Optional[dict] = None,
    floor_policy: str = DEFAULT_FLOOR_POLICY,
    hard_skill_coverage_floor: float = HARD_SKILL_COVERAGE_FLOOR,
    lower_bound: float = DEFAULT_FAMILIARITY_LOWER_BOUND,
    upper_bound: float = DEFAULT_FAMILIARITY_UPPER_BOUND,
    clique_edge_threshold: float = DEFAULT_CLIQUE_EDGE_THRESHOLD,
    max_clique_fraction: float = DEFAULT_MAX_CLIQUE_FRACTION,
    minimum_known_pair_coverage: float = DEFAULT_MINIMUM_KNOWN_PAIR_COVERAGE,
    penalize_insufficient_data: bool = False,
    objective_tolerance: float = FAMILIARITY_OBJECTIVE_TOLERANCE,
    floor_min_tolerance: float = FAMILIARITY_FLOOR_MIN_TOLERANCE,
    swap_tolerance: float = SWAP_IMPROVEMENT_TOLERANCE,
    max_sweeps: int = MAX_OPTIMIZATION_SWEEPS,
) -> List[List[Student]]:
    """
    OPTIONAL post-fairness pass — only ever invoked when the caller sets
    FamiliarityConfig.enabled=True (generate_teams() does not call this
    otherwise, so default matching results are unaffected by this
    function's mere existence, and a roster with zero recorded familiarity
    edges is a no-op even when enabled — every team's density is None,
    every violation term is 0.0, there is nothing to improve).

    Targets a TWO-PART violation (_team_familiarity_violation), not a
    single min/max like _maximin_improve or a gap like
    _ability_balance_improve — a swap is accepted only if it STRICTLY
    REDUCES the sum of:
      (a) density-band violation — a team's density below lower_bound (may
          lack coordination support) or above upper_bound (possible clique
          concentration); SUPPRESSED (0.0) for a team whose
          known_pair_coverage is below minimum_known_pair_coverage unless
          `penalize_insufficient_data=True` — the pass never "confidently"
          chases a target it doesn't have enough evidence to judge.
      (b) clique-concentration violation — a team's largest
          connected_familiarity_cluster (from the TEAM-INDUCED subgraph)
          exceeding max_clique_fraction of the team.
    Term (b) exists because (a) alone is insensitive to shrinking a
    uniformly-strong cluster one member at a time (density is a mean over
    KNOWN pairs only, so removing one cluster member doesn't change the
    remaining members' known pairwise strength) — see
    _team_clique_violation's docstring and the Part 4 report's limitations
    section. Together they operationalize the stated goal: encourage
    enough familiarity for coordination WITHOUT placing an entire
    pre-existing cluster on one team.

    Acceptance is LEXICOGRAPHIC, same pattern as the other optional passes:
      1. validate_assignment(...) hard constraints must hold.
      2. the primary total-score objective may not regress by more than
         `objective_tolerance` (default 0.0).
      3. the accountability/cooperativeness floor-minimum _maximin_improve
         established may not regress by more than `floor_min_tolerance`
         (default 0.0).
      4. only then must the swap strictly reduce total violation by more
         than `swap_tolerance`.
      5. deterministic tie-break: same fixed nested scan order as every
         other pass, and _largest_team_cluster's own tie-break (see its
         docstring) never depends on hash-seed-sensitive set ordering.

    KNOWN LIMITATION (same shape as _ability_balance_improve's): if
    ability.enabled and/or fairness.enabled are ALSO set, this pass (which
    generate_teams() runs last) does not itself protect the ability gap or
    individual-utility minimum those passes achieved — running all three
    together is not guaranteed to preserve every property simultaneously.
    Documented rather than silently assumed away.
    """
    weights = weights or _DEFAULT_WEIGHTS
    baseline = AssignmentBaseline.capture(teams, floor_policy)
    all_students = [s for team in teams for s in team]
    graph = build_familiarity_graph(all_students, edges)
    dims = _floor_dims()

    def _team_viol(team):
        return _team_familiarity_violation(
            team, graph, lower_bound, upper_bound, clique_edge_threshold, max_clique_fraction,
            minimum_known_pair_coverage, penalize_insufficient_data,
        )

    # Per-team violation cache: a swap only ever changes two teams' OWN
    # induced subgraphs (unlike Part 4's global components, which never
    # changed with team assignment, a team-induced cluster genuinely does
    # — so this must be recomputed per candidate for the 2 touched teams,
    # not precomputed once). Every OTHER team's violation is unaffected by
    # a swap between i and j and is read from cache — same "others" idiom
    # _ability_balance_improve/_maximin_improve use.
    violation_cache = [_team_viol(t) for t in teams]

    # Incremental per-team caches for the UNIVERSAL protected metrics
    # (objective/floor-min/skill-coverage/floor-violation-count) — Part
    # 6.5, same idiom as _maximin_improve/_two_opt_improve/
    # _ability_balance_improve/_fairness_improve. Previously this pass
    # recomputed _stage_objective/_partition_floor_min from scratch (full
    # partition) on every candidate and never passed validate_assignment
    # any precomputed skill-coverage/floor-violation values either —
    # cProfile showed this dominating the pass's runtime at scale (docs/
    # performance.md).
    score_cache = [_total_score(t, weights) for t in teams]
    coverage_cache = [_skill_coverage(t) for t in teams]
    floor_violation_cache = [_team_floor_violation_count(t, floor_policy) for t in teams]
    floor_mean_cache = [{dn: _ordinal_team_mean(t, dn, dd) for dn, dd in dims.items()} for t in teams]

    def _floor_min_of(cache):
        vals = [v for means in cache for v in means.values()]
        return min(vals) if vals else 1.0

    sweeps = 0
    improved = True
    while improved and sweeps < max_sweeps:
        improved = False
        sweeps += 1
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                others = [k for k in range(len(teams)) if k != i and k != j]
                others_violation = sum(violation_cache[k] for k in others)
                others_score = sum(score_cache[k] for k in others)
                others_min_cov = min((coverage_cache[k] for k in others), default=1.0)
                others_floor_violations = sum(floor_violation_cache[k] for k in others)
                others_floor_vals = [v for k in others for v in floor_mean_cache[k].values()]

                for a_idx in range(len(teams[i])):
                    for b_idx in range(len(teams[j])):
                        original_objective = score_cache[i] + score_cache[j] + others_score
                        original_floor_min = _floor_min_of(floor_mean_cache)
                        original_violation = violation_cache[i] + violation_cache[j] + others_violation

                        teams[i][a_idx], teams[j][b_idx] = teams[j][b_idx], teams[i][a_idx]

                        new_score_i = _total_score(teams[i], weights)
                        new_score_j = _total_score(teams[j], weights)
                        new_cov_i = _skill_coverage(teams[i])
                        new_cov_j = _skill_coverage(teams[j])
                        new_floor_viol_i = _team_floor_violation_count(teams[i], floor_policy)
                        new_floor_viol_j = _team_floor_violation_count(teams[j], floor_policy)
                        new_means_i = {dn: _ordinal_team_mean(teams[i], dn, dd) for dn, dd in dims.items()}
                        new_means_j = {dn: _ordinal_team_mean(teams[j], dn, dd) for dn, dd in dims.items()}

                        validation = validate_assignment(
                            teams, baseline, hard_skill_coverage_floor,
                            precomputed_min_skill_coverage=min(new_cov_i, new_cov_j, others_min_cov),
                            precomputed_floor_violation_count=(
                                new_floor_viol_i + new_floor_viol_j + others_floor_violations
                            ),
                        )
                        new_objective = new_score_i + new_score_j + others_score
                        new_floor_min = min(
                            [*new_means_i.values(), *new_means_j.values(), *others_floor_vals],
                            default=1.0,
                        )
                        new_viol_i = _team_viol(teams[i])
                        new_viol_j = _team_viol(teams[j])
                        new_violation = new_viol_i + new_viol_j + others_violation

                        accept = (
                            validation.ok
                            and new_objective >= original_objective - objective_tolerance
                            and new_floor_min >= original_floor_min - floor_min_tolerance
                            and new_violation < original_violation - swap_tolerance
                        )
                        if accept:
                            improved = True
                            violation_cache[i], violation_cache[j] = new_viol_i, new_viol_j
                            score_cache[i], score_cache[j] = new_score_i, new_score_j
                            coverage_cache[i], coverage_cache[j] = new_cov_i, new_cov_j
                            floor_violation_cache[i], floor_violation_cache[j] = new_floor_viol_i, new_floor_viol_j
                            floor_mean_cache[i], floor_mean_cache[j] = new_means_i, new_means_j
                        else:
                            teams[i][a_idx], teams[j][b_idx] = teams[j][b_idx], teams[i][a_idx]
    return teams


# ---------------------------------------------------------------------------
# Preference / stability framework (Part 5)
#
# Preference data is a DIRECTED statement FROM one student ABOUT a target
# (another student, or a team characteristic) — see app.models.
# PreferenceEntry's docstring for the full schema (ranked/tied/avoid/
# missing/characteristic). This section computes preference-satisfaction
# metrics (always) and formal stability diagnostics (always), plus an
# OPTIONAL repair pass (_preference_repair_improve, gated by
# PreferenceConfig.enabled) that may change the returned assignment.
#
# TIER SCORING (how a ranked/tied/unranked preference list becomes a
# [0,1] satisfaction score):
#   For student s, collect every "prefer" entry s submitted. Entries that
#   share the same `rank` value form ONE TIER (this is how ties/
#   indifference are expressed — see PreferenceEntry). Distinct rank
#   values, sorted ascending, become tiers 0, 1, 2, ... (tier 0 = most
#   preferred). Entries left unranked (rank=None) form ONE ADDITIONAL tier
#   placed AFTER every ranked tier (a ranked preference signals stronger
#   intent than an unranked one, by convention). n_tiers(s) = number of
#   distinct ranks + (1 if any unranked entry exists, else 0).
#     - A student with ZERO "prefer" entries has satisfaction_score = None
#       (MISSING — no opinion — excluded from every mean/min/max/rate
#       below, never scored as 0.0; see PreferenceEntry's docstring).
#     - A student WITH entries whose target(s) never appear on their team
#       scores exactly 0.0 (a real, meaningful outcome: distinct from
#       "had no opinion").
#     - Otherwise, score = (n_tiers - matched_tier) / n_tiers, where
#       matched_tier is the BEST (lowest-numbered) tier containing an
#       entry whose target IS on the student's team. Matching a tier-0
#       (top) preference scores 1.0; matching only the worst tier scores
#       1/n_tiers (small but > 0 — "got something, just their least-
#       preferred option").
#   A "prefer" entry with a `target_characteristic` participates in the
#   SAME tiering as named-student entries; it is "matched" if
#   _characteristic_satisfied() can resolve and confirm it (currently only
#   the "skill" key is resolved — see that function).
#
# FORMAL STABILITY DEFINITIONS (many-to-one team assignment, NOT the
# classical one-to-one stable-roommates/stable-marriage problem — this
# section adapts the same family of ideas Gale & Shapley (1962) and the
# Hospitals/Residents literature (Roth 1984) formalize for many-to-one
# matching, to a setting where every team has a FIXED size, so the natural
# "move" operation this codebase's own optimizer already uses throughout
# (2-opt, maximin, ability/fairness/familiarity balance) is a 1-for-1
# EXCHANGE, not an unconstrained capacity-based transfer):
#
#   Let π be an assignment (a partition of students into fixed-size
#   teams). For student s and a candidate set of teammates C (C excludes
#   s), let sat(s, C) be s's tier-scored satisfaction (above) if s's
#   OTHER teammates were exactly C. sat(s, π) is s's ACTUAL satisfaction
#   under π. "s strictly prefers C to π" means sat(s, C) > sat(s, π) with
#   both defined (a student with no expressed preference — sat is None —
#   can never be said to strictly prefer anything; see each function's
#   `if score is None: skip` guard).
#
#   BLOCKING PAIR — a diagnostic ENVY signal, the classical stable-
#   matching notion, adapted with named-target tier comparison rather
#   than full hypothetical-team scoring (cheap, O(1) per pair beyond
#   lookups): an unordered pair {i, j} on DIFFERENT teams such that i
#   ranks j STRICTLY AHEAD of every one of i's own current teammates
#   (i.e. j's tier is lower than the best tier at which i named any
#   current teammate, treating an unnamed teammate as tied for i's own
#   worst tier), AND j SYMMETRICALLY ranks i ahead of every one of j's
#   current teammates. This says "i and j would each rather be near the
#   other than near anyone they currently have" — it does NOT by itself
#   certify that swapping them would leave either better off once the
#   REST of their new teammates are taken into account (see compute_
#   blocking_swaps). See compute_blocking_pairs.
#
#   BLOCKING SWAP — the EXECUTABLE, mutually-improving 1-for-1 exchange:
#   an ordered pair (i ∈ team A, j ∈ team B), A ≠ B, such that i strictly
#   prefers B's POST-swap composition ((B \ {j}) ∪ {i}) to A's actual
#   current composition, AND j strictly prefers A's post-swap composition
#   ((A \ {i}) ∪ {j}) to B's actual current composition — using the FULL
#   tier-scored sat() function against the real post-swap teammate set,
#   not the cheaper named-teammate shortcut blocking pairs use. A blocking
#   pair is NEITHER necessary NOR sufficient for a blocking swap to exist
#   between the same two students (i might rank j above i's own
#   teammates yet still end up worse off once j's OTHER current teammates
#   become i's new teammates instead). Team sizes are preserved by
#   construction, so this diagnostic is always feasibility-neutral
#   w.r.t. capacity — it says nothing about skill-coverage/behavioral
#   hard constraints (see beneficial feasible move below). See
#   compute_blocking_swaps.
#
#   STABLE ASSIGNMENT (this system's DEFAULT, documented headline
#   definition — see PreferenceConfig.stability_definition): π is stable
#   iff no blocking swap exists. This is the definition
#   _preference_repair_improve's own swap-based mechanism can actually
#   act on (it repairs EXACTLY by hunting for and applying improving
#   swaps, same mechanism as every other optional pass in this file), so
#   it is the criterion PreferenceSummary.is_stable reports by default.
#   blocking_pair and beneficial_feasible_move counts are ALWAYS
#   additionally computed and available (compute_stability_metrics) as
#   supplementary diagnostics regardless of which definition is
#   "authoritative" — PreferenceConfig.stability_definition lets a caller
#   pick a different one of the three as authoritative instead, per this
#   part's own instruction to keep one clearly-documented default and
#   make alternatives configurable. Choosing "beneficial_feasible_move" as
#   the authoritative definition does NOT yield a classical stable-
#   matching guarantee — see that definition's own caveat below, repeated
#   in compute_stability_metrics()'s returned `note` whenever selected.
#
#   BENEFICIAL FEASIBLE MOVE (Part 5.5 — renamed from "blocking move";
#   NOT a classical blocking-coalition concept, hence the rename: see
#   PreferenceSummary's TERMINOLOGY note) — a WEAKER-on-preference,
#   STRICTER-on-feasibility, intentionally ONE-SIDED sibling of blocking
#   swap, directly answering this part's own framing ("state whose
#   utility must improve and whether the receiving team's constraints
#   must remain satisfied"): student i (team A) has a beneficial feasible
#   move to team B (A ≠ B) if there EXISTS some j ∈ B such that swapping i
#   and j would leave i STRICTLY better off (sat(i, (B\{j})∪{i}) >
#   sat(i, π)) — j's OWN preference is NEVER checked, only i's utility
#   must improve — AND the resulting FULL partition (both A and B, not
#   just one side) still satisfies EVERY hard constraint validate_
#   assignment() enforces (team sizes, skill-coverage floor, behavioral
#   floor) — the receiving team's STRUCTURAL constraints, not its
#   members' preferences or consent, must remain satisfied. Because this
#   diagnostic never requires j (or B's other members) to be better off —
#   or even indifferent — it is explicitly documented as a PROJECT-
#   SPECIFIC ONE-SIDED DEVIATION MEASURE, not a classical blocking
#   coalition and not a stable-matching guarantee in the Gale-Shapley/
#   Hospitals-Residents sense; do not cite `is_stable` under this
#   definition as proof of two-sided stability. See
#   compute_beneficial_feasible_moves — which, being swap-mediated (this
#   system's teams have exact target sizes with no free vacancy), is
#   size-preserving by construction and whole-partition-validated via a
#   single validate_assignment() call per candidate, transitively
#   confirming BOTH the sending and receiving team's sizes, skill
#   coverage, and floor-policy compliance in that one check.
#
#   UNILATERAL VACANCY MOVE (Part 5.5, compute_unilateral_vacancy_moves) —
#   a DIFFERENT, genuinely non-swap-mediated operation: student i relocates
#   from A to B with NO displaced counterpart (A shrinks by one, B grows by
#   one). This is a distinct diagnostic from beneficial_feasible_move
#   above (which is always swap-mediated and therefore always size-
#   preserving) and is NEVER part of compute_stability_metrics()'s
#   headline counts — it only ever runs, and only ever finds anything,
#   when a caller EXPLICITLY configures a team-size RANGE via
#   app.models.TeamSizeConfig (min_team_size/max_team_size/
#   allow_unilateral_moves). Under this system's default EXACT team sizes,
#   a genuine unilateral move is ALWAYS infeasible — there is no vacancy
#   for it to fill — and this function returns `[]` immediately rather
#   than silently reinterpreting exact target sizes as flexible. Every
#   candidate is still validated against the WHOLE partition in one
#   validate_assignment() call (against a size-RANGE baseline), covering
#   the shrinking sending team, the growing receiving team, skill
#   coverage (including any configured absolute HARD_SKILL_COVERAGE_FLOOR),
#   floor-policy compliance, and complete membership preservation.
#
# WHAT THE OPTIONAL REPAIR PASS TARGETS: _preference_repair_improve does
# not directly search for "a blocking swap" as a boolean; it minimizes a
# continuous per-team violation (unmet satisfaction + a weighted penalty
# per unmet HIGH-priority preference + a weighted penalty per avoidance
# violation), using the exact same lexicographic acceptance rule
# (hard constraints -> objective non-regression -> floor-min non-
# regression -> strict violation reduction -> deterministic tie-break) as
# _familiarity_balance_improve. Every swap it accepts is, by construction,
# one both swapping students are individually and jointly better off
# from in aggregate — driving toward, but not guaranteeing, the absence
# of blocking swaps (same caveat every other optional pass in this file
# documents: local swap search is not a global stability guarantee).
#
# COMPUTATIONAL COMPLEXITY (N = total students, T = number of teams,
# S = team size, P = a student's own preference-entry count — all small
# in this system's target deployment scale, same regime as the existing
# O(T^2 S^2) 2-opt/familiarity/ability/fairness passes):
#   compute_preference_metrics:          O(N * (S + P))
#   compute_blocking_pairs:              O(N^2 * S * P) (pairwise envy check)
#   compute_blocking_swaps:              O(T^2 * S^3)   (every cross-team
#                                         student pair, each requiring an
#                                         O(S) satisfaction recomputation
#                                         on both sides)
#   compute_beneficial_feasible_moves:   O(T^2 * S^2 * N) (validate_assignment
#                                         is O(N) per candidate)
#   compute_unilateral_vacancy_moves:    O(T^2 * S * N), same order, no
#                                         inner j-loop (no displaced
#                                         counterpart to search over)
#   _preference_repair_improve:          O(sweeps * T^2 * S^2 * (S + P)),
#                                         matching the existing swap-search
#                                         passes' order.
# ---------------------------------------------------------------------------

DEFAULT_HIGH_PRIORITY_WEIGHT = 3.0
DEFAULT_AVOIDANCE_VIOLATION_WEIGHT = 5.0
PREFERENCE_OBJECTIVE_TOLERANCE = MAXIMIN_OBJECTIVE_TOLERANCE
PREFERENCE_FLOOR_MIN_TOLERANCE = MAXIMIN_OBJECTIVE_TOLERANCE


def _build_preference_index(entries: List[PreferenceEntry]) -> Dict[str, dict]:
    """
    Organizes raw PreferenceEntry rows by submitting student and computes
    each "prefer" entry's TIER once (see module docstring's tier-scoring
    section) so every downstream function reuses identical numbers rather
    than each recomputing tiers independently.

    index[sid] = {
        "prefer": [{"target_student_id", "target_characteristic", "rank", "tier", "priority"}, ...],  # sorted by tier
        "avoid":  [{"target_student_id", "priority"}, ...],
        "n_tiers": int,  # 0 when sid has no "prefer" entries at all
    }
    A student with no PreferenceEntry rows at all is simply absent from
    this dict — every reader below treats a missing key as "no opinion",
    never as an empty-but-present (and therefore falsely 0-scored) record.
    """
    by_student: Dict[str, List[PreferenceEntry]] = {}
    for e in entries:
        by_student.setdefault(e.student_id, []).append(e)

    index: Dict[str, dict] = {}
    for sid, es in by_student.items():
        prefers = [e for e in es if e.direction == "prefer"]
        avoids = [e for e in es if e.direction == "avoid"]

        ranked_values = sorted({e.rank for e in prefers if e.rank is not None})
        tier_of_rank = {r: i for i, r in enumerate(ranked_values)}
        has_unranked = any(e.rank is None for e in prefers)
        unranked_tier = len(ranked_values)
        n_tiers = len(ranked_values) + (1 if has_unranked else 0)

        prefer_records = [
            {
                "target_student_id": e.target_student_id,
                "target_characteristic": e.target_characteristic,
                "rank": e.rank,
                "tier": tier_of_rank[e.rank] if e.rank is not None else unranked_tier,
                "priority": e.priority,
            }
            for e in prefers
        ]
        prefer_records.sort(key=lambda r: r["tier"])

        avoid_records = [
            {"target_student_id": e.target_student_id, "priority": e.priority}
            for e in avoids
        ]

        index[sid] = {"prefer": prefer_records, "avoid": avoid_records, "n_tiers": n_tiers}
    return index


def _characteristic_satisfied(other_members: List[Student], characteristic: Dict[str, str]) -> Optional[bool]:
    """
    Whether `other_members` (a prospective/actual team's members OTHER
    than the student expressing the preference) satisfies a
    target_characteristic preference. Returns None — "cannot be resolved;
    excluded from scoring, never silently counted as unmet" — for any key
    other than "skill", the only one this system currently knows how to
    check (against CompetenceSignature.skills). Extending this to other
    characteristics (leadership_style, availability, ...) is future work —
    see the Part 5 report's limitations.
    """
    if set(characteristic.keys()) != {"skill"}:
        return None
    wanted_skill = characteristic["skill"]
    return any(wanted_skill in m.competence_signature.skills for m in other_members)


def _student_satisfaction(
    sid: str,
    other_member_ids: List[str],
    other_members: List[Student],
    pref_index: Dict[str, dict],
) -> Dict[str, object]:
    """
    sid's tier-scored preference satisfaction (module docstring) against a
    REAL OR HYPOTHETICAL team whose members OTHER than sid are exactly
    `other_members` — used both for the actual assignment (evaluation) and
    for post-swap "what if" comparisons (stability diagnostics, repair
    pass). `other_member_ids`/`other_members` must correspond 1:1 and
    exclude sid itself.
    """
    rec = pref_index.get(sid)
    if rec is None or not rec["prefer"]:
        return {"score": None, "got_a_preferred_teammate": None, "matched_tier": None,
                "n_tiers": None, "n_unmet_high_priority": 0}

    other_id_set = set(other_member_ids)
    n_tiers = rec["n_tiers"]
    matched_tier: Optional[int] = None
    n_unmet_high_priority = 0
    for p in rec["prefer"]:
        if p["target_student_id"] is not None:
            matched = p["target_student_id"] in other_id_set
        else:
            matched = bool(_characteristic_satisfied(other_members, p["target_characteristic"]))
        if matched:
            if matched_tier is None or p["tier"] < matched_tier:
                matched_tier = p["tier"]
        elif p["priority"] == "high":
            n_unmet_high_priority += 1

    if matched_tier is None:
        return {"score": 0.0, "got_a_preferred_teammate": False, "matched_tier": None,
                "n_tiers": n_tiers, "n_unmet_high_priority": n_unmet_high_priority}
    return {
        "score": (n_tiers - matched_tier) / n_tiers, "got_a_preferred_teammate": True,
        "matched_tier": matched_tier, "n_tiers": n_tiers,
        "n_unmet_high_priority": n_unmet_high_priority,
    }


def _avoidance_violations_for_team(team_ids: List[str], pref_index: Dict[str, dict]) -> List[Tuple[str, str, bool]]:
    """
    Every (a, b) pair of DISTINCT ids in `team_ids` such that EITHER a has
    an "avoid" entry targeting b OR b has one targeting a — checked BOTH
    directions since avoidance is honored UNILATERALLY (module docstring;
    PreferenceEntry's docstring). Returns (a, b, is_high_priority) tuples;
    is_high_priority is True if the triggering entry (either direction)
    was priority="high".
    """
    violations: List[Tuple[str, str, bool]] = []
    for a, b in itertools.combinations(team_ids, 2):
        a_avoids_b = next((r for r in pref_index.get(a, {}).get("avoid", []) if r["target_student_id"] == b), None)
        b_avoids_a = next((r for r in pref_index.get(b, {}).get("avoid", []) if r["target_student_id"] == a), None)
        if a_avoids_b or b_avoids_a:
            is_high = bool(
                (a_avoids_b and a_avoids_b["priority"] == "high")
                or (b_avoids_a and b_avoids_a["priority"] == "high")
            )
            violations.append((a, b, is_high))
    return violations


def compute_preference_metrics(
    teams: List[List[Student]],
    entries: List[PreferenceEntry],
) -> Dict[str, object]:
    """
    Assignment-level preference-satisfaction metrics. Pure evaluation —
    never mutates `teams`, never influences which assignment is returned
    (only _preference_repair_improve does that, gated by
    PreferenceConfig.enabled).

    PRIVACY: this raw dict includes per-student identifying fields (student
    ids) intended for INTERNAL callers (generate_teams' repair pass,
    tests, this module's own stability functions) — a caller building a
    public-facing response must go through _preference_summary_model(),
    which strips this down to the aggregate-only PreferenceStudentStats
    shape. See app/preferences.py's module docstring for the full privacy
    design this split supports.
    """
    pref_index = _build_preference_index(entries)
    members_by_id: Dict[str, Student] = {s.id: s for team in teams for s in team}

    per_student: List[Dict[str, object]] = []
    scores: List[float] = []
    total_unmet_high = 0
    total_avoid_violations = 0
    total_avoid_high = 0

    for i, team in enumerate(teams):
        team_ids = [s.id for s in team]
        avoid_viol = _avoidance_violations_for_team(team_ids, pref_index)
        avoid_count_by_student: Dict[str, int] = {}
        high_flag_by_student: Dict[str, bool] = {}
        for a, b, is_high in avoid_viol:
            avoid_count_by_student[a] = avoid_count_by_student.get(a, 0) + 1
            avoid_count_by_student[b] = avoid_count_by_student.get(b, 0) + 1
            if is_high:
                high_flag_by_student[a] = True
                high_flag_by_student[b] = True
        total_avoid_violations += len(avoid_viol)
        total_avoid_high += sum(1 for _, _, h in avoid_viol if h)

        for s in team:
            other_ids = [sid for sid in team_ids if sid != s.id]
            other_members = [members_by_id[sid] for sid in other_ids]
            sat = _student_satisfaction(s.id, other_ids, other_members, pref_index)
            rec = pref_index.get(s.id, {"prefer": [], "avoid": []})
            n_avoid_viol = avoid_count_by_student.get(s.id, 0)
            per_student.append({
                "student_id": s.id,
                "team_index": i,
                "has_preferences": bool(rec["prefer"]),
                "n_prefer_entries": len(rec["prefer"]),
                "n_avoid_entries": len(rec["avoid"]),
                "satisfaction_score": sat["score"],
                "got_a_preferred_teammate": sat["got_a_preferred_teammate"],
                "matched_tier": sat["matched_tier"],
                "n_tiers": sat["n_tiers"],
                "n_unmet_high_priority_preferences": sat["n_unmet_high_priority"],
                "avoidance_violated": n_avoid_viol > 0,
                "n_avoidance_violations": n_avoid_viol,
                "violated_high_priority_avoidance": high_flag_by_student.get(s.id, False),
            })
            if sat["score"] is not None:
                scores.append(sat["score"])
            total_unmet_high += sat["n_unmet_high_priority"]

    n_with_prefs = sum(1 for p in per_student if p["has_preferences"])
    got_rates = [1.0 if p["got_a_preferred_teammate"] else 0.0 for p in per_student if p["has_preferences"]]

    return {
        "per_student": per_student,
        "n_students_with_preferences": n_with_prefs,
        "mean_satisfaction": statistics.fmean(scores) if scores else None,
        "min_satisfaction": min(scores) if scores else None,
        "max_satisfaction": max(scores) if scores else None,
        "satisfaction_stdev": statistics.pstdev(scores) if len(scores) > 1 else (0.0 if scores else None),
        "fraction_with_at_least_one_preferred_teammate": (sum(got_rates) / len(got_rates)) if got_rates else None,
        "worst_case_preference_regret": (1 - min(scores)) if scores else None,
        "total_unmet_high_priority_preferences": total_unmet_high,
        "total_avoidance_violations": total_avoid_violations,
        "total_high_priority_avoidance_violations": total_avoid_high,
        "note": (
            "satisfaction_score is None (MISSING, not 0.0) for a student with zero 'prefer' "
            "entries -- excluded from mean/min/max/stdev/fraction_with_at_least_one_preferred_"
            "teammate. A student WITH entries none of which matched their team scores exactly "
            "0.0 -- a real outcome, distinct from having no opinion. Ties/indifference are "
            "expressed by multiple entries sharing the same rank (or all left unranked); see "
            "this module's tier-scoring docstring for the exact formula. Avoidance is honored "
            "UNILATERALLY -- a violation is counted whenever EITHER teammate in a pair named the "
            "other as an 'avoid' target, regardless of reciprocity."
        ),
    }


def _named_prefer_tier(sid: str, target_id: str, pref_index: Dict[str, dict]) -> Optional[int]:
    """The tier at which sid's preferences name target_id specifically (a
    target_student_id entry — a target_characteristic entry can NEVER
    match here, since it has no target_student_id at all: characteristic
    preferences are structurally invisible to the blocking-pair shortcut,
    see _prefers_target_to_all_current_teammates); None if sid never named
    target_id in any 'prefer' entry (including when sid submitted NO
    'prefer' entries at all, or no entries whatsoever). AVOID entries are
    never consulted here or anywhere in the blocking-pair path — a
    simultaneous (contradictory) 'prefer' entry for the same target still
    counts fully; avoidance never overrides/cancels a preference in THIS
    diagnostic (app.preferences.upsert_preference rejects that
    contradiction at write time as of Part 5.5, but existing rows written
    before that guard existed, or rows inserted directly at the DB layer,
    are not retroactively reconciled)."""
    rec = pref_index.get(sid)
    if not rec:
        return None
    for p in rec["prefer"]:
        if p["target_student_id"] == target_id:
            return p["tier"]
    return None


def _prefers_target_to_all_current_teammates(
    sid: str, target_id: str, current_teammate_ids: List[str], pref_index: Dict[str, dict],
) -> bool:
    """
    Named-target half of the blocking-PAIR check (module docstring) — a
    DELIBERATELY CONSERVATIVE envy diagnostic, stronger than "sid would
    rather have target_id than SOME current teammate": True iff sid named
    target_id in a 'prefer' entry AND ranks it STRICTLY ahead of EVERY one
    of sid's own current teammates (not merely the worst one). Documented
    edge-case behavior (Part 5.5 — verified against the implementation
    below, not merely asserted):

      no ranked/named current teammate at all (current_teammate_ids=[])
        -> returns True as soon as sid has named target_id at any tier —
        there is nothing to be "worse" than.

      an unranked (rank=None) but NAMED current teammate
        -> substitutes that teammate's ACTUAL unranked tier (a real,
        finite tier index — see the tier-scoring section above), NOT a
        sentinel. This is stronger evidence than an unnamed teammate (see
        next) — sid took the trouble to name them.

      an UNNAMED current teammate (sid never mentioned them at all)
        -> substitutes `worst_tier = n_tiers` — this is ONE TIER WORSE
        than sid's own worst NAMED tier (valid tier indices only run
        0..n_tiers-1), NOT "tied" with it. An earlier revision of this
        docstring incorrectly described this as a tie; corrected here.
        Net effect: naming target_id at ANY real tier, even sid's least-
        preferred one, beats having only unnamed current teammates.

      tied ranks (target_id shares a tier with a current teammate)
        -> returns False for that comparison (strict `<`, not `<=`): a
        tie is indifference, not a strict preference for the target, so
        it can never by itself create envy toward that teammate.

      sid submitted no 'prefer' entries at all (or never named target_id
      specifically, even if sid has OTHER entries)
        -> _named_prefer_tier returns None -> this function returns False
        immediately. Consequence for compute_blocking_pairs: a student
        with NO preference data can NEVER be one side of a blocking pair
        (both directions require a NAMED target), so missing preference
        data can never MANUFACTURE a false blocking pair — it can only
        ever suppress one that named data might otherwise have revealed.
    """
    target_tier = _named_prefer_tier(sid, target_id, pref_index)
    if target_tier is None:
        return False
    rec = pref_index.get(sid, {"n_tiers": 0})
    worst_tier = rec.get("n_tiers", 0)
    if not current_teammate_ids:
        return True
    teammate_tiers = [
        _named_prefer_tier(sid, m, pref_index)
        if _named_prefer_tier(sid, m, pref_index) is not None else worst_tier
        for m in current_teammate_ids
    ]
    return target_tier < min(teammate_tiers)


def compute_blocking_pairs(teams: List[List[Student]], entries: List[PreferenceEntry]) -> List[Tuple[str, str]]:
    """BLOCKING PAIR — see module docstring's formal definitions, and
    _prefers_target_to_all_current_teammates' docstring for the exhaustive,
    verified edge-case behavior (missing data, ties, unranked/unnamed
    teammates, characteristic preferences, avoidance non-interaction) this
    conservative envy diagnostic exhibits."""
    pref_index = _build_preference_index(entries)
    team_of: Dict[str, int] = {s.id: i for i, team in enumerate(teams) for s in team}
    ids_by_team: Dict[int, List[str]] = {i: [s.id for s in team] for i, team in enumerate(teams)}

    pairs: List[Tuple[str, str]] = []
    ids = list(team_of.keys())
    for a, b in itertools.combinations(ids, 2):
        if team_of[a] == team_of[b]:
            continue
        a_teammates = [x for x in ids_by_team[team_of[a]] if x != a]
        b_teammates = [x for x in ids_by_team[team_of[b]] if x != b]
        if (
            _prefers_target_to_all_current_teammates(a, b, a_teammates, pref_index)
            and _prefers_target_to_all_current_teammates(b, a, b_teammates, pref_index)
        ):
            pairs.append((a, b))
    return pairs


def compute_blocking_swaps(teams: List[List[Student]], entries: List[PreferenceEntry]) -> List[Tuple[str, str]]:
    """BLOCKING SWAP — see module docstring's formal definitions. This is
    the DEFAULT criterion PreferenceSummary.is_stable is checked against."""
    pref_index = _build_preference_index(entries)
    members_by_id: Dict[str, Student] = {s.id: s for team in teams for s in team}

    swaps: List[Tuple[str, str]] = []
    for a_idx in range(len(teams)):
        for b_idx in range(a_idx + 1, len(teams)):
            team_a_ids = [s.id for s in teams[a_idx]]
            team_b_ids = [s.id for s in teams[b_idx]]
            for i_id in team_a_ids:
                i_current_others = [x for x in team_a_ids if x != i_id]
                i_current = _student_satisfaction(
                    i_id, i_current_others, [members_by_id[x] for x in i_current_others], pref_index
                )["score"]
                if i_current is None:
                    continue
                for j_id in team_b_ids:
                    j_current_others = [x for x in team_b_ids if x != j_id]
                    j_current = _student_satisfaction(
                        j_id, j_current_others, [members_by_id[x] for x in j_current_others], pref_index
                    )["score"]
                    if j_current is None:
                        continue
                    i_post_others = [x for x in team_b_ids if x != j_id]
                    j_post_others = [x for x in team_a_ids if x != i_id]
                    i_post = _student_satisfaction(
                        i_id, i_post_others, [members_by_id[x] for x in i_post_others], pref_index
                    )["score"]
                    j_post = _student_satisfaction(
                        j_id, j_post_others, [members_by_id[x] for x in j_post_others], pref_index
                    )["score"]
                    if i_post is not None and j_post is not None and \
                            i_post > i_current + 1e-9 and j_post > j_current + 1e-9:
                        swaps.append((i_id, j_id))
    return swaps


def compute_beneficial_feasible_moves(
    teams: List[List[Student]],
    entries: List[PreferenceEntry],
    baseline: "AssignmentBaseline",
    hard_skill_coverage_floor: float = HARD_SKILL_COVERAGE_FLOOR,
) -> List[Tuple[str, int]]:
    """
    BENEFICIAL FEASIBLE MOVE (Part 5.5 — renamed from "blocking move"; see
    module docstring's formal definitions and PreferenceSummary's
    TERMINOLOGY note for why): a diagnostic for student i (team A) who
    could relocate to team B (A != B) such that (1) i's own utility
    strictly improves, and (2) the resulting FULL partition remains
    hard-constraint-valid. NOT a classical blocking-coalition notion —
    the displaced counterpart j's own preference is never checked, so this
    does NOT claim j (or B's other members) accept, consent to, or benefit
    from the move. "Feasible" refers strictly to hard constraints
    (team sizes, skill coverage, behavioral floor), never to j's consent.

    MECHANISM, and why this is size-preserving by construction: this
    system's teams have EXACT target sizes everywhere (see
    AssignmentBaseline.target_sizes) — there is no vacancy for i to move
    into "for free". So a candidate here is realized as a 1-for-1 EXCHANGE
    (i take's j's spot, j takes i's) purely to keep every team at its
    exact target size; only i's utility is required to improve, j is
    simply whichever current occupant of B makes the resulting FULL
    candidate hard-constraint-valid. Every candidate is checked via
    validate_assignment() against the WHOLE partition (both the shrinking-
    then-regrowing team A and the growing-then-shrinking team B, in the
    same call) — this transitively verifies: every student appears exactly
    once (validate_assignment's membership check), BOTH team A's and team
    B's sizes stay exactly at target (validate_assignment's size check,
    trivially satisfied here since a swap never changes any team's size),
    BOTH team A's and team B's skill coverage stay >= baseline (the
    partition-wide min-coverage check), and the floor-policy violation
    count does not increase FOR THE WHOLE PARTITION (not just the two
    touched teams). See compute_unilateral_vacancy_moves for the
    DIFFERENT, genuinely non-swap-mediated notion of a move into an
    explicitly configured vacancy.

    Returns (student_id, target_team_index) pairs; a student with multiple
    feasible target teams is reported once per distinct target team.
    """
    pref_index = _build_preference_index(entries)
    members_by_id: Dict[str, Student] = {s.id: s for team in teams for s in team}

    moves: List[Tuple[str, int]] = []
    for a_idx in range(len(teams)):
        team_a_ids = [s.id for s in teams[a_idx]]
        for i_id in team_a_ids:
            i_current_others = [x for x in team_a_ids if x != i_id]
            i_current = _student_satisfaction(
                i_id, i_current_others, [members_by_id[x] for x in i_current_others], pref_index
            )["score"]
            if i_current is None:
                continue
            for b_idx in range(len(teams)):
                if b_idx == a_idx:
                    continue
                team_b_ids = [s.id for s in teams[b_idx]]
                found = False
                for j_id in team_b_ids:
                    i_post_others = [x for x in team_b_ids if x != j_id]
                    i_post = _student_satisfaction(
                        i_id, i_post_others, [members_by_id[x] for x in i_post_others], pref_index
                    )["score"]
                    if i_post is None or i_post <= i_current + 1e-9:
                        continue
                    candidate = [list(t) for t in teams]
                    ai = candidate[a_idx].index(members_by_id[i_id])
                    bj = candidate[b_idx].index(members_by_id[j_id])
                    candidate[a_idx][ai], candidate[b_idx][bj] = candidate[b_idx][bj], candidate[a_idx][ai]
                    if validate_assignment(candidate, baseline, hard_skill_coverage_floor).ok:
                        found = True
                        break
                if found:
                    moves.append((i_id, b_idx))
    return moves


def compute_unilateral_vacancy_moves(
    teams: List[List[Student]],
    entries: List[PreferenceEntry],
    baseline: "AssignmentBaseline",
    team_size_config: "TeamSizeConfig",
    hard_skill_coverage_floor: float = HARD_SKILL_COVERAGE_FLOOR,
) -> List[Tuple[str, int]]:
    """
    A TRUE UNILATERAL move (Part 5.5): student i relocates from team A to
    team B with NO displaced counterpart — team A shrinks by one member,
    team B grows by one member. This is a genuinely DIFFERENT operation
    from compute_beneficial_feasible_moves' swap-mediated one: it requires
    an actual size RANGE to be explicitly configured (see
    app.models.TeamSizeConfig) — under this system's default EXACT team
    sizes there is no vacancy for a unilateral move to fill, so this
    returns `[]` immediately unless `team_size_config.allow_unilateral_
    moves` is True AND at least one of `min_team_size`/`max_team_size` is
    set. This is a deliberate refusal to silently reinterpret exact target
    sizes as flexible, per this part's own instruction.

    Every candidate is checked via validate_assignment() against a size-
    RANGE baseline (min_sizes/max_sizes derived from team_size_config,
    falling back to the EXACT original target size on whichever side isn't
    configured) covering the WHOLE partition in one call — this verifies,
    for BOTH the shrinking sending team and the growing receiving team in
    the same pass: every student still appears exactly once, both team
    sizes stay within the configured range, both teams' skill coverage
    stays >= the baseline (and >= any configured absolute hard floor), and
    the floor-policy violation count does not increase anywhere in the
    partition.
    """
    if not team_size_config.allow_unilateral_moves:
        return []
    if team_size_config.min_team_size is None and team_size_config.max_team_size is None:
        return []

    n = len(teams)
    range_min = tuple(
        team_size_config.min_team_size if team_size_config.min_team_size is not None else baseline.target_sizes[i]
        for i in range(n)
    )
    range_max = tuple(
        team_size_config.max_team_size if team_size_config.max_team_size is not None else baseline.target_sizes[i]
        for i in range(n)
    )
    range_baseline = AssignmentBaseline(
        student_ids=baseline.student_ids,
        target_sizes=baseline.target_sizes,
        min_skill_coverage=baseline.min_skill_coverage,
        floor_policy=baseline.floor_policy,
        max_floor_violations=baseline.max_floor_violations,
        min_sizes=range_min,
        max_sizes=range_max,
    )

    pref_index = _build_preference_index(entries)
    members_by_id: Dict[str, Student] = {s.id: s for team in teams for s in team}

    moves: List[Tuple[str, int]] = []
    for a_idx in range(n):
        team_a_ids = [s.id for s in teams[a_idx]]
        for i_id in team_a_ids:
            others = [x for x in team_a_ids if x != i_id]
            i_current = _student_satisfaction(
                i_id, others, [members_by_id[x] for x in others], pref_index
            )["score"]
            if i_current is None:
                continue
            for b_idx in range(n):
                if b_idx == a_idx:
                    continue
                team_b_ids = [s.id for s in teams[b_idx]]
                i_post = _student_satisfaction(
                    i_id, team_b_ids, [members_by_id[x] for x in team_b_ids], pref_index
                )["score"]
                if i_post is None or i_post <= i_current + 1e-9:
                    continue
                candidate = [list(t) for t in teams]
                mover = members_by_id[i_id]
                candidate[a_idx].remove(mover)
                candidate[b_idx].append(mover)
                if validate_assignment(candidate, range_baseline, hard_skill_coverage_floor).ok:
                    moves.append((i_id, b_idx))
    return moves


def compute_stability_metrics(
    teams: List[List[Student]],
    entries: List[PreferenceEntry],
    baseline: Optional["AssignmentBaseline"] = None,
    hard_skill_coverage_floor: float = HARD_SKILL_COVERAGE_FLOOR,
    floor_policy: str = DEFAULT_FLOOR_POLICY,
    definition: str = "blocking_swap",
) -> Dict[str, object]:
    """
    Computes all three stability diagnostics (blocking_pair, blocking_swap,
    beneficial_feasible_move) and reports `is_stable` against whichever
    `definition` is authoritative (default "blocking_swap" — see module
    docstring). The other two are always also computed and returned, never
    hidden, regardless of `definition`. `beneficial_feasible_move` (Part
    5.5 — renamed from "blocking_move") is deliberately NOT a classical
    blocking-coalition notion — see compute_beneficial_feasible_moves'
    docstring — and choosing it as `definition` reports a project-specific
    one-sided deviation measure under the SAME `is_stable` flag, not a
    stable-matching guarantee; the note below says so explicitly whenever
    it is selected.
    """
    if definition not in ("blocking_pair", "blocking_swap", "beneficial_feasible_move"):
        raise ValueError(f"unknown stability definition: {definition!r}")
    if baseline is None:
        baseline = AssignmentBaseline.capture(teams, floor_policy)

    blocking_pairs = compute_blocking_pairs(teams, entries)
    blocking_swaps = compute_blocking_swaps(teams, entries)
    beneficial_feasible_moves = compute_beneficial_feasible_moves(teams, entries, baseline, hard_skill_coverage_floor)

    counts = {
        "blocking_pair": len(blocking_pairs),
        "blocking_swap": len(blocking_swaps),
        "beneficial_feasible_move": len(beneficial_feasible_moves),
    }
    note = (
        f"is_stable reflects the '{definition}' definition (see app/matching.py's "
        "preference-framework module docstring for all three formal definitions and why "
        "blocking_swap is the documented default). The other two counts are always computed as "
        "supplementary diagnostics regardless of which definition is authoritative here."
    )
    if definition == "beneficial_feasible_move":
        note += (
            " NOTE: beneficial_feasible_move is a project-specific ONE-SIDED deviation measure "
            "(only the moving student's utility must improve; the receiving side's consent or "
            "benefit is never checked) — is_stable under this definition is NOT a classical "
            "stable-matching guarantee."
        )
    return {
        "definition": definition,
        "is_stable": counts[definition] == 0,
        "blocking_pairs": blocking_pairs,
        "blocking_swaps": blocking_swaps,
        "beneficial_feasible_moves": beneficial_feasible_moves,
        "n_blocking_pairs": counts["blocking_pair"],
        "n_blocking_swaps": counts["blocking_swap"],
        "n_beneficial_feasible_moves": counts["beneficial_feasible_move"],
        "note": note,
    }


def _preference_summary_model(pref_report: Dict[str, object], stability_report: Dict[str, object]) -> PreferenceSummary:
    per_student = [
        PreferenceStudentStats(
            student_id=p["student_id"],
            has_preferences=p["has_preferences"],
            n_prefer_entries=p["n_prefer_entries"],
            n_avoid_entries=p["n_avoid_entries"],
            satisfaction_score=p["satisfaction_score"],
            got_a_preferred_teammate=p["got_a_preferred_teammate"],
            n_unmet_high_priority_preferences=p["n_unmet_high_priority_preferences"],
            avoidance_violated=p["avoidance_violated"],
            n_avoidance_violations=p["n_avoidance_violations"],
            violated_high_priority_avoidance=p["violated_high_priority_avoidance"],
        )
        for p in pref_report["per_student"]
    ]
    return PreferenceSummary(
        per_student=per_student,
        n_students_with_preferences=pref_report["n_students_with_preferences"],
        mean_satisfaction=pref_report["mean_satisfaction"],
        min_satisfaction=pref_report["min_satisfaction"],
        max_satisfaction=pref_report["max_satisfaction"],
        satisfaction_stdev=pref_report["satisfaction_stdev"],
        fraction_with_at_least_one_preferred_teammate=pref_report["fraction_with_at_least_one_preferred_teammate"],
        worst_case_preference_regret=pref_report["worst_case_preference_regret"],
        total_unmet_high_priority_preferences=pref_report["total_unmet_high_priority_preferences"],
        total_avoidance_violations=pref_report["total_avoidance_violations"],
        total_high_priority_avoidance_violations=pref_report["total_high_priority_avoidance_violations"],
        stability_definition=stability_report["definition"],
        is_stable=stability_report["is_stable"],
        n_blocking_pairs=stability_report["n_blocking_pairs"],
        n_blocking_swaps=stability_report["n_blocking_swaps"],
        n_beneficial_feasible_moves=stability_report["n_beneficial_feasible_moves"],
        note=pref_report["note"] + " " + stability_report["note"],
    )


def _hard_avoidance_repair(
    teams: List[List[Student]],
    entries: List[PreferenceEntry],
    weights: Optional[dict] = None,
    floor_policy: str = DEFAULT_FLOOR_POLICY,
    hard_skill_coverage_floor: float = HARD_SKILL_COVERAGE_FLOOR,
    objective_tolerance: float = PREFERENCE_OBJECTIVE_TOLERANCE,
    floor_min_tolerance: float = PREFERENCE_FLOOR_MIN_TOLERANCE,
    swap_tolerance: float = SWAP_IMPROVEMENT_TOLERANCE,
    max_sweeps: int = MAX_OPTIMIZATION_SWEEPS,
) -> List[List[Student]]:
    """
    Part 5.5 — ALWAYS-ON avoidance enforcement, run by generate_teams()
    whenever PreferenceConfig.hard_avoidance=True, INDEPENDENT of
    PreferenceConfig.enabled. This is what makes "hard avoidance" an
    actual hard constraint rather than merely a heavier soft weight —
    contrast the THREE avoidance tiers this system distinguishes:

      SOFT avoidance preference (direction="avoid", priority="normal") —
        purely a reported signal (compute_preference_metrics'
        avoidance_violated/n_avoidance_violations) plus, ONLY when
        PreferenceConfig.enabled=True, a weighted term in
        _preference_repair_improve's violation function
        (avoidance_violation_weight). Never structurally enforced.

      HIGH-PRIORITY avoidance (direction="avoid", priority="high") — same
        mechanics as soft avoidance, just weighted more heavily and
        surfaced separately in reporting
        (violated_high_priority_avoidance / total_high_priority_
        avoidance_violations). STILL not a hard constraint by priority
        alone — priority is a WEIGHT, not a promotion to hard-constraint
        status.

      HARD avoidance CONSTRAINT (PreferenceConfig.hard_avoidance=True) —
        promotes AVOIDANCE AS A WHOLE (every avoid entry, any priority)
        into a constraint THIS function actively hunts to eliminate,
        REGARDLESS of whether PreferenceConfig.enabled is set at all.
        Unlike _preference_repair_improve's hard_avoidance gate (which
        only ever refuses to let ITS OWN swaps increase violations while
        chasing an UNRELATED general-satisfaction objective), this
        function's SOLE objective is minimizing avoidance-violation-pair
        COUNT — it runs whether or not general preference optimization is
        wanted, exactly like skill-coverage/floor-policy are always
        enforced whether or not ability/fairness/familiarity optimization
        is enabled.

    Why this matters for "does the pipeline actually prevent a prohibited
    pairing": _greedy_build/_two_opt_improve/_maximin_improve, and the
    optional ability/fairness/familiarity passes that may run before this
    one, have NO avoidance awareness at all — any of them could construct
    or reintroduce a prohibited pairing while pursuing their own
    objective. This function runs AFTER all of them (see generate_teams())
    and repairs whatever it can: it does not make the EARLIER passes
    avoidance-aware (that would be the combined-optimizer integration this
    part explicitly defers) — it makes the FINAL returned assignment
    avoidance-clean whenever a hard-constraint-safe fix exists, which is
    what actually reaches the API response and the database.

    Same lexicographic acceptance as every other pass in this file (hard
    constraints -> objective non-regression -> floor-min non-regression ->
    STRICT reduction in avoidance-violation-pair count -> deterministic
    tie-break): this pass actively REDUCES existing violations (that is
    its entire purpose, not merely a side effect of refusing to worsen
    them) but, like every other pass, never accepts a swap that would
    violate validate_assignment()'s hard constraints (team size, skill
    coverage, behavioral floor) to do so. A violation that cannot be
    resolved without breaking one of THOSE hard constraints is left in
    place — this is the documented, expected "unavoidable violation" case
    — and remains fully visible afterward in compute_preference_metrics()'s
    total_avoidance_violations / per-student avoidance_violated (that
    report always reflects the ACTUAL final assignment, never suppressed
    or hidden regardless of what this pass could or could not fix).
    """
    weights = weights or _DEFAULT_WEIGHTS
    baseline = AssignmentBaseline.capture(teams, floor_policy)
    dims = _floor_dims()
    pref_index = _build_preference_index(entries)

    def _team_avoid_count(team: List[Student]) -> int:
        ids = [s.id for s in team]
        return len(_avoidance_violations_for_team(ids, pref_index))

    avoid_cache = [_team_avoid_count(t) for t in teams]

    # Incremental per-team caches for the UNIVERSAL protected metrics
    # (Part 6.5 — same idiom as every other optional pass; see
    # _maximin_improve's docstring and docs/performance.md).
    score_cache = [_total_score(t, weights) for t in teams]
    coverage_cache = [_skill_coverage(t) for t in teams]
    floor_violation_cache = [_team_floor_violation_count(t, floor_policy) for t in teams]
    floor_mean_cache = [{dn: _ordinal_team_mean(t, dn, dd) for dn, dd in dims.items()} for t in teams]

    def _floor_min_of(cache):
        vals = [v for means in cache for v in means.values()]
        return min(vals) if vals else 1.0

    sweeps = 0
    improved = True
    while improved and sweeps < max_sweeps:
        improved = False
        sweeps += 1
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                others = [k for k in range(len(teams)) if k != i and k != j]
                others_count = sum(avoid_cache[k] for k in others)
                others_score = sum(score_cache[k] for k in others)
                others_min_cov = min((coverage_cache[k] for k in others), default=1.0)
                others_floor_violations = sum(floor_violation_cache[k] for k in others)
                others_floor_vals = [v for k in others for v in floor_mean_cache[k].values()]

                for a_idx in range(len(teams[i])):
                    for b_idx in range(len(teams[j])):
                        original_objective = score_cache[i] + score_cache[j] + others_score
                        original_floor_min = _floor_min_of(floor_mean_cache)
                        original_count = avoid_cache[i] + avoid_cache[j] + others_count

                        teams[i][a_idx], teams[j][b_idx] = teams[j][b_idx], teams[i][a_idx]

                        new_score_i = _total_score(teams[i], weights)
                        new_score_j = _total_score(teams[j], weights)
                        new_cov_i = _skill_coverage(teams[i])
                        new_cov_j = _skill_coverage(teams[j])
                        new_floor_viol_i = _team_floor_violation_count(teams[i], floor_policy)
                        new_floor_viol_j = _team_floor_violation_count(teams[j], floor_policy)
                        new_means_i = {dn: _ordinal_team_mean(teams[i], dn, dd) for dn, dd in dims.items()}
                        new_means_j = {dn: _ordinal_team_mean(teams[j], dn, dd) for dn, dd in dims.items()}

                        validation = validate_assignment(
                            teams, baseline, hard_skill_coverage_floor,
                            precomputed_min_skill_coverage=min(new_cov_i, new_cov_j, others_min_cov),
                            precomputed_floor_violation_count=(
                                new_floor_viol_i + new_floor_viol_j + others_floor_violations
                            ),
                        )
                        new_objective = new_score_i + new_score_j + others_score
                        new_floor_min = min(
                            [*new_means_i.values(), *new_means_j.values(), *others_floor_vals],
                            default=1.0,
                        )
                        new_count_i = _team_avoid_count(teams[i])
                        new_count_j = _team_avoid_count(teams[j])
                        new_count = new_count_i + new_count_j + others_count

                        accept = (
                            validation.ok
                            and new_objective >= original_objective - objective_tolerance
                            and new_floor_min >= original_floor_min - floor_min_tolerance
                            and new_count < original_count - swap_tolerance
                        )
                        if accept:
                            improved = True
                            avoid_cache[i], avoid_cache[j] = new_count_i, new_count_j
                            score_cache[i], score_cache[j] = new_score_i, new_score_j
                            coverage_cache[i], coverage_cache[j] = new_cov_i, new_cov_j
                            floor_violation_cache[i], floor_violation_cache[j] = new_floor_viol_i, new_floor_viol_j
                            floor_mean_cache[i], floor_mean_cache[j] = new_means_i, new_means_j
                        else:
                            teams[i][a_idx], teams[j][b_idx] = teams[j][b_idx], teams[i][a_idx]
    return teams


def _team_preference_violation(
    team_ids: List[str],
    team_members_by_id: Dict[str, Student],
    pref_index: Dict[str, dict],
    high_priority_weight: float,
    avoidance_violation_weight: float,
) -> Tuple[float, int]:
    """Combined preference violation for ONE team: (1 - satisfaction) for
    every member with an expressed preference, PLUS high_priority_weight
    per unmet high-priority 'prefer' entry, PLUS avoidance_violation_weight
    per avoidance-violation pair within the team. Returns (violation,
    n_avoidance_violations) — the count is returned separately so
    _preference_repair_improve's hard_avoidance gate can check it directly
    without re-deriving it from the weighted total."""
    violation = 0.0
    for sid in team_ids:
        others = [x for x in team_ids if x != sid]
        other_members = [team_members_by_id[x] for x in others]
        sat = _student_satisfaction(sid, others, other_members, pref_index)
        if sat["score"] is not None:
            violation += (1.0 - sat["score"])
        violation += high_priority_weight * sat["n_unmet_high_priority"]
    avoid_viol = _avoidance_violations_for_team(team_ids, pref_index)
    violation += avoidance_violation_weight * len(avoid_viol)
    return violation, len(avoid_viol)


def _preference_repair_improve(
    teams: List[List[Student]],
    entries: List[PreferenceEntry],
    weights: Optional[dict] = None,
    floor_policy: str = DEFAULT_FLOOR_POLICY,
    hard_skill_coverage_floor: float = HARD_SKILL_COVERAGE_FLOOR,
    high_priority_weight: float = DEFAULT_HIGH_PRIORITY_WEIGHT,
    avoidance_violation_weight: float = DEFAULT_AVOIDANCE_VIOLATION_WEIGHT,
    hard_avoidance: bool = False,
    objective_tolerance: float = PREFERENCE_OBJECTIVE_TOLERANCE,
    floor_min_tolerance: float = PREFERENCE_FLOOR_MIN_TOLERANCE,
    swap_tolerance: float = SWAP_IMPROVEMENT_TOLERANCE,
    max_sweeps: int = MAX_OPTIMIZATION_SWEEPS,
) -> List[List[Student]]:
    """
    OPTIONAL post-familiarity pass — only ever invoked when the caller sets
    PreferenceConfig.enabled=True (generate_teams() does not call this
    otherwise, so default matching results are unaffected by this
    function's mere existence; a roster with zero recorded preference
    entries is a no-op even when enabled, exactly like
    _familiarity_balance_improve's zero-edges case).

    Targets _team_preference_violation, summed per-team, with the SAME
    lexicographic acceptance rule as every other optional pass in this
    file: (1) validate_assignment()'s hard constraints must hold; (2) the
    primary total-score objective may not regress by more than
    `objective_tolerance`; (3) the accountability/cooperativeness
    floor-minimum may not regress by more than `floor_min_tolerance`;
    (4) — the "explicitly configured institutional restriction" this
    repair phase additionally guarantees — when `hard_avoidance=True`,
    the swap is ALSO refused if it would INCREASE the avoidance-violation
    PAIR COUNT on either team it touches (pre-existing violations may
    still shrink as a side effect of ordinary violation minimization —
    this pass just never WORSENS avoidance on its own initiative); only
    then must the swap strictly reduce total preference violation by more
    than `swap_tolerance`; (5) deterministic tie-break, same fixed nested
    scan order as every other pass. Per-team violation is CACHED and only
    the two touched teams are recomputed per candidate (same incremental
    pattern as _familiarity_balance_improve, and for the same reason:
    unlike the OLD global-familiarity-components case, a team's own
    preference violation depends only on ITS OWN membership, so caching
    "others" and recomputing just the 2 touched teams is both correct and
    cheap).

    KNOWN LIMITATION (same shape as every other optional pass's): if
    ability/fairness/familiarity are ALSO enabled, this pass (which
    generate_teams() runs last) does not itself re-protect whatever those
    earlier passes achieved beyond the objective/floor-min non-regression
    already checked — running every optional pass together is not
    guaranteed to jointly optimize every soft objective simultaneously.
    """
    weights = weights or _DEFAULT_WEIGHTS
    baseline = AssignmentBaseline.capture(teams, floor_policy)
    dims = _floor_dims()
    pref_index = _build_preference_index(entries)

    def _team_viol(team: List[Student]) -> Tuple[float, int]:
        ids = [s.id for s in team]
        by_id = {s.id: s for s in team}
        return _team_preference_violation(ids, by_id, pref_index, high_priority_weight, avoidance_violation_weight)

    cache = [_team_viol(t) for t in teams]
    violation_cache = [c[0] for c in cache]
    avoid_cache = [c[1] for c in cache]

    # Incremental per-team caches for the UNIVERSAL protected metrics
    # (Part 6.5 — same idiom as every other optional pass; see
    # _maximin_improve's docstring and docs/performance.md).
    score_cache = [_total_score(t, weights) for t in teams]
    coverage_cache = [_skill_coverage(t) for t in teams]
    floor_violation_cache = [_team_floor_violation_count(t, floor_policy) for t in teams]
    floor_mean_cache = [{dn: _ordinal_team_mean(t, dn, dd) for dn, dd in dims.items()} for t in teams]

    def _floor_min_of(cache_):
        vals = [v for means in cache_ for v in means.values()]
        return min(vals) if vals else 1.0

    sweeps = 0
    improved = True
    while improved and sweeps < max_sweeps:
        improved = False
        sweeps += 1
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                others = [k for k in range(len(teams)) if k != i and k != j]
                others_violation = sum(violation_cache[k] for k in others)
                others_score = sum(score_cache[k] for k in others)
                others_min_cov = min((coverage_cache[k] for k in others), default=1.0)
                others_floor_violations = sum(floor_violation_cache[k] for k in others)
                others_floor_vals = [v for k in others for v in floor_mean_cache[k].values()]

                for a_idx in range(len(teams[i])):
                    for b_idx in range(len(teams[j])):
                        original_objective = score_cache[i] + score_cache[j] + others_score
                        original_floor_min = _floor_min_of(floor_mean_cache)
                        original_violation = violation_cache[i] + violation_cache[j] + others_violation
                        original_avoid_i, original_avoid_j = avoid_cache[i], avoid_cache[j]

                        teams[i][a_idx], teams[j][b_idx] = teams[j][b_idx], teams[i][a_idx]

                        new_score_i = _total_score(teams[i], weights)
                        new_score_j = _total_score(teams[j], weights)
                        new_cov_i = _skill_coverage(teams[i])
                        new_cov_j = _skill_coverage(teams[j])
                        new_floor_viol_i = _team_floor_violation_count(teams[i], floor_policy)
                        new_floor_viol_j = _team_floor_violation_count(teams[j], floor_policy)
                        new_means_i = {dn: _ordinal_team_mean(teams[i], dn, dd) for dn, dd in dims.items()}
                        new_means_j = {dn: _ordinal_team_mean(teams[j], dn, dd) for dn, dd in dims.items()}

                        validation = validate_assignment(
                            teams, baseline, hard_skill_coverage_floor,
                            precomputed_min_skill_coverage=min(new_cov_i, new_cov_j, others_min_cov),
                            precomputed_floor_violation_count=(
                                new_floor_viol_i + new_floor_viol_j + others_floor_violations
                            ),
                        )
                        new_objective = new_score_i + new_score_j + others_score
                        new_floor_min = min(
                            [*new_means_i.values(), *new_means_j.values(), *others_floor_vals],
                            default=1.0,
                        )
                        new_viol_i, new_avoid_i = _team_viol(teams[i])
                        new_viol_j, new_avoid_j = _team_viol(teams[j])
                        new_violation = new_viol_i + new_viol_j + others_violation

                        avoidance_ok = True
                        if hard_avoidance:
                            avoidance_ok = new_avoid_i <= original_avoid_i and new_avoid_j <= original_avoid_j

                        accept = (
                            validation.ok
                            and avoidance_ok
                            and new_objective >= original_objective - objective_tolerance
                            and new_floor_min >= original_floor_min - floor_min_tolerance
                            and new_violation < original_violation - swap_tolerance
                        )
                        if accept:
                            improved = True
                            violation_cache[i], violation_cache[j] = new_viol_i, new_viol_j
                            avoid_cache[i], avoid_cache[j] = new_avoid_i, new_avoid_j
                            score_cache[i], score_cache[j] = new_score_i, new_score_j
                            coverage_cache[i], coverage_cache[j] = new_cov_i, new_cov_j
                            floor_violation_cache[i], floor_violation_cache[j] = new_floor_viol_i, new_floor_viol_j
                            floor_mean_cache[i], floor_mean_cache[j] = new_means_i, new_means_j
                        else:
                            teams[i][a_idx], teams[j][b_idx] = teams[j][b_idx], teams[i][a_idx]
    return teams


# ---------------------------------------------------------------------------
# Part 6 — combined optimization framework: criterion registry, a
# normalized cross-criterion objective REPORT, named configuration
# profiles, and an explicit cross-pass acceptance policy.
#
# ARCHITECTURE DECISION (Section 6 of this part's spec: "Option A: unified
# weighted objective" vs. "Option B: lexicographic multi-objective
# ordering" — pick the one the CURRENT architecture best supports):
# every optional pass already shipped through Part 5.6
# (_ability_balance_improve, _fairness_improve, _familiarity_balance_
# improve, _hard_avoidance_repair, _preference_repair_improve) makes its
# OWN per-candidate accept/reject decision via an IDENTICAL lexicographic
# rule: hard constraints (validate_assignment) -> primary total-score
# objective non-regression -> floor-minimum non-regression -> strict
# improvement in THAT pass's own target -> deterministic tie-break.
# Rewriting all five into one monolithic weighted-sum optimizer (Option A)
# would mean re-deriving and re-proving correctness for logic that is
# ALREADY tested extensively per-pass — high risk for no architectural
# gain, since a single scalar objective cannot express "never regress
# floor-minimum" as cleanly as a priority order can. Part 6 therefore
# takes **Option B**, EXTENDED one level: on top of each pass's own
# internal per-swap lexicographic rule, _run_optional_passes() below adds
# an ORCHESTRATION-level lexicographic check comparing each pass's
# COMPLETE before/after effect (not swap-by-swap) against a small set of
# PROTECTED metrics — this is what actually closes the "a later stage
# must not silently undo an earlier protected objective" requirement,
# since a pass's own per-swap tolerance could in principle accumulate
# drift across many accepted swaps that no single swap's own check would
# catch (see _run_optional_passes' docstring for the exact mechanism).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CriterionClassification:
    """One row of CRITERION_REGISTRY (below) — see that registry's own
    docstring for how to read `classification` and `condition`."""
    key: str
    criterion_area: str  # which of the 7 research criteria (or "other") this maps to
    name: str
    classification: str  # "hard_constraint" | "soft_objective" | "evaluation_only" | "optional_post_processing"
    condition: str
    rationale: str


CRITERION_REGISTRY: Tuple[CriterionClassification, ...] = (
    CriterionClassification(
        "membership_complete", "other", "Every student assigned exactly once",
        "hard_constraint", "always",
        "validate_assignment() rejects any duplicate/missing/unknown student unconditionally — "
        "no profile or config can disable this check.",
    ),
    CriterionClassification(
        "team_sizes_valid", "other", "Valid team sizes",
        "hard_constraint", "always (exact target size by default; a range when TeamSizeConfig "
        "explicitly configures min/max — see compute_unilateral_vacancy_moves)",
        "validate_assignment() rejects any size outside the (possibly ranged) target — sizes are "
        "never silently reinterpreted as flexible.",
    ),
    CriterionClassification(
        "skill_coverage_floor", "1_skill_coverage", "Required minimum skill coverage",
        "hard_constraint", "always non-regressing vs. the post-greedy baseline; additionally an "
        "ABSOLUTE floor when hard_skill_coverage_floor > 0.0 (HARD_SKILL_COVERAGE_FLOOR, default 0.0)",
        "validate_assignment() enforces both the non-regression rule (every pass) and, when "
        "configured, an absolute floor — this criterion is NEVER purely a soft objective.",
    ),
    CriterionClassification(
        "behavioral_floor_policy", "3_accountability_cooperativeness", "Behavioral minimum floor "
        "(accountability/cooperativeness)",
        "hard_constraint", "per `floor_policy` ('one_low_per_team' tolerates one floor-level "
        "member per dimension per team; 'zero_tolerance' tolerates none)",
        "_check_floor_violations()/validate_assignment() enforce whichever policy is configured — "
        "the POLICY is a request-level choice, but once chosen it is a hard constraint, not a "
        "target to trade off against other objectives.",
    ),
    CriterionClassification(
        "hard_avoidance", "7_preference_stability", "Hard avoidance constraint",
        "hard_constraint", "when preference.hard_avoidance=True (an explicitly configured "
        "institutional restriction, off by default)",
        "_hard_avoidance_repair() runs independent of preference.enabled and actively reduces "
        "violations rather than merely capping them — see Part 5.5's report.",
    ),
    CriterionClassification(
        "behavioral_compatibility", "3_accountability_cooperativeness", "Overall behavioral "
        "compatibility (mean, beyond the floor)",
        "soft_objective", "always measured; only optimized for via the primary total-score "
        "objective's `behavioral_compat` weight",
        "Rubric composition rules (high_mean_with_floor / soft_compatibility) score a MEAN — the "
        "floor is the hard constraint above, the mean above the floor is a soft target.",
    ),
    CriterionClassification(
        "skill_diversity_beyond_minimum", "1_skill_coverage", "Skill diversity beyond the required "
        "minimum",
        "soft_objective", "always measured (skill_coverage's own weight in _total_score); no "
        "dedicated diversity-beyond-coverage term exists yet",
        "No research citation establishes a target beyond 'cover the required skills' — treated as "
        "an extension of the skill_coverage weight, not a separate hard requirement.",
    ),
    CriterionClassification(
        "ability_balance", "2_ability_comparable_strength", "Mixed ability / comparable team "
        "strength",
        "soft_objective", "measured always; optimized only when ability.enabled=True",
        "AbilityConfig's own docstring: 'balanced ability' is a documented HYPOTHESIS, not an "
        "established finding for this population — never silently promoted to a hard constraint.",
    ),
    CriterionClassification(
        "familiarity_target_band", "4_familiarity_clustering", "Familiarity target band (enough "
        "for coordination, not so much it signals unaddressed clustering)",
        "soft_objective", "measured always; optimized only when familiarity.enabled=True, and "
        "never confidently acted on below minimum_known_pair_coverage (band_status becomes "
        "'insufficient_data')",
        "Band bounds are documented hypotheses (FamiliarityConfig's docstring) — insufficient "
        "evidence suppresses the band judgment entirely rather than guessing.",
    ),
    CriterionClassification(
        "anti_cluster_concentration", "4_familiarity_clustering", "Anti-cluster concentration "
        "(largest connected_familiarity_cluster fraction)",
        "soft_objective", "measured always; optimized only when familiarity.enabled=True",
        "A direct, size-based term separate from the band above, since density alone is "
        "insensitive to shrinking a uniformly-strong cluster one member at a time (Part 4 report).",
    ),
    CriterionClassification(
        "comparable_overall_quality", "5_comparable_quality", "Comparable overall team quality "
        "(team_balance_score / between-team score variance)",
        "soft_objective", "always measured (evaluation.metrics.team_strength_comparison); folded "
        "into the primary total-score objective's cross-team spread implicitly via 2-opt/maximin",
        "No dedicated hard hard hard floor — 'comparable' is a spread-minimization target, not a "
        "guarantee.",
    ),
    CriterionClassification(
        "individual_utility_fairness", "6_fair_individual_benefit", "Fair distribution of "
        "individual benefit (individual utility, worst-case regret)",
        "soft_objective", "measured always; optimized only when fairness.enabled=True",
        "compute_assignment_utilities() is pure evaluation — _fairness_improve() is the only thing "
        "that can change the assignment, gated by an explicit flag (Part 2's own requirement).",
    ),
    CriterionClassification(
        "preference_satisfaction", "7_preference_stability", "Preference satisfaction (tier-scored)",
        "soft_objective", "measured always; optimized only when preference.enabled=True",
        "compute_preference_metrics() never depends on preference.enabled — only _preference_"
        "repair_improve() (the modification path) does.",
    ),
    CriterionClassification(
        "blocking_pair_swap_counts", "7_preference_stability", "Blocking-pair / blocking-swap "
        "counts (stability diagnostics)",
        "evaluation_only", "always computed (compute_stability_metrics); ALSO an optional repair "
        "target via _preference_repair_improve's violation term when preference.enabled=True",
        "This is deliberately listed as BOTH an evaluation metric (always on) AND an optional "
        "repair target (opt-in) — the two roles are not in tension since the metric never gates "
        "on the repair flag.",
    ),
    CriterionClassification(
        "explanation_quality", "other", "Explanation quality (Gemini-generated team narrative)",
        "optional_post_processing", "generated lazily, on demand, via GET /teams/{id}/explanation — "
        "never part of team FORMATION",
        "app/explainability.py runs strictly after formation and never feeds back into which "
        "assignment was chosen — a pure post-processing/presentation step.",
    ),
)


def criterion_registry_as_dicts() -> List[Dict[str, str]]:
    """JSON-serializable view of CRITERION_REGISTRY, for API/docs surfacing."""
    return [
        {
            "key": c.key, "criterion_area": c.criterion_area, "name": c.name,
            "classification": c.classification, "condition": c.condition, "rationale": c.rationale,
        }
        for c in CRITERION_REGISTRY
    ]


# ---------------------------------------------------------------------------
# Normalized cross-criterion objective REPORT (Section 4). This is an
# EVALUATION-LAYER construct for transparency/comparison (surfaced on the
# API response and used by the profile-comparison tooling) — it does NOT
# itself drive which assignment is returned; the existing per-pass
# mechanism (_run_optional_passes below) still does that. Building a
# reporting-only representation, rather than rewiring every pass to
# consume it directly, avoids re-deriving already-tested optimizer logic
# while still satisfying "one normalized representation to reason about
# all criteria together."
# ---------------------------------------------------------------------------

DEFAULT_OBJECTIVE_WEIGHTS: Dict[str, float] = {
    "skill_quality": 0.20,
    "behavioral_compatibility": 0.15,
    "team_balance": 0.15,
    "ability_balance": 0.10,
    "familiarity_band_fit": 0.10,
    "anti_cluster": 0.10,
    "individual_fairness": 0.10,
    "preference_satisfaction": 0.10,
    "stability_penalty": 0.00,  # optional component, opt-in only (see below)
}

# Documents each component's fixed [0.0, 1.0] range, directionality, and
# missing-data policy ONCE — every component is defined so that HIGHER
# ALWAYS MEANS BETTER (this is why familiarity/anti-cluster/stability
# components below are framed as "fit"/"anti-" rather than raw
# density/cluster-size/blocking-count, which would need to be inverted).
OBJECTIVE_COMPONENT_SPECS: Dict[str, Dict[str, str]] = {
    "skill_quality": {
        "range": "[0.0, 1.0]", "direction": "higher_is_better",
        "definition": "mean per-team fraction of REFERENCE_SKILLS covered",
        "missing_data_policy": "None only if there are zero teams; otherwise always computable.",
    },
    "behavioral_compatibility": {
        "range": "[0.0, 1.0] (approx. — component sub-scores are individually bounded this way)",
        "direction": "higher_is_better",
        "definition": "mean per-team rubric-weighted behavioral compatibility (_behavioral_compat)",
        "missing_data_policy": "None only if there are zero teams.",
    },
    "team_balance": {
        "range": "[0.0, 1.0]", "direction": "higher_is_better",
        "definition": "1 - stdev of per-team total scores, clamped to [0,1]",
        "missing_data_policy": "None only if there are zero teams; a single team scores 1.0 (no spread to measure).",
    },
    "ability_balance": {
        "range": "[0.0, 1.0]", "direction": "higher_is_better",
        "definition": "1 - strongest/weakest team mean-ability gap (AbilityConfig's [0,1] ability scale)",
        "missing_data_policy": "None when no team has a resolvable mean ability (no student has "
        "ability_score or experience_level) — EXCLUDED from the weighted total, not scored 0.",
    },
    "familiarity_band_fit": {
        "range": "[0.0, 1.0]", "direction": "higher_is_better",
        "definition": "fraction of teams WITHIN the configured density band, among only teams with "
        "a resolvable band_status (below/within/above) — insufficient_data/not_applicable teams "
        "are excluded from both numerator and denominator",
        "missing_data_policy": "None when NO team has a resolvable band_status (e.g. no "
        "familiarity data at all) — EXCLUDED, not scored 0. evidence_coverage reports what "
        "fraction of teams contributed.",
    },
    "anti_cluster": {
        "range": "[0.0, 1.0]", "direction": "higher_is_better",
        "definition": "1 - mean(largest_connected_cluster.fraction_of_team) over teams with >=2 "
        "members (0/1-member teams excluded, not scored as maximally clustered)",
        "missing_data_policy": "None when every team has <2 members. evidence_coverage reports "
        "what fraction of teams contributed.",
    },
    "individual_fairness": {
        "range": "[0.0, 1.0]", "direction": "higher_is_better",
        "definition": "minimum per-student utility across the whole assignment (worst-off welfare)",
        "missing_data_policy": "None only if there are zero students.",
    },
    "preference_satisfaction": {
        "range": "[0.0, 1.0]", "direction": "higher_is_better",
        "definition": "mean tier-scored satisfaction over students who submitted >=1 'prefer' entry",
        "missing_data_policy": "None when NO student submitted any preference data at all — "
        "EXCLUDED, not scored 0 (this system's Part 5 'missing != zero' rule, extended here).",
    },
    "stability_penalty": {
        "range": "[0.0, 1.0]", "direction": "higher_is_better",
        "definition": "1 - (n_blocking_swaps / max(1, n_students)), a rough bounded proxy — OPTIONAL, "
        "weight 0.0 by default; give it a nonzero weight only when blocking-swap count should "
        "itself factor into the total objective (distinct from preference_satisfaction, which "
        "measures a different thing — see this module's overlap-prevention note below).",
        "missing_data_policy": "None only if there are zero students.",
    },
}


def _objective_component(raw: Optional[float], weight: float, coverage: float) -> Dict[str, object]:
    normalized = max(0.0, min(1.0, raw)) if raw is not None else None
    return {"raw": raw, "normalized": normalized, "weight": weight, "weighted_contribution": 0.0, "evidence_coverage": coverage}


def compute_normalized_objective(
    teams: List[List[Student]],
    familiarity_edges: Optional[List[FamiliarityEdge]] = None,
    preference_entries: Optional[List[PreferenceEntry]] = None,
    objective_weights: Optional[Dict[str, float]] = None,
    ability_level_thresholds: Optional[Dict[str, float]] = None,
    familiarity_config: Optional[FamiliarityConfig] = None,
    utility_weights: Optional[Dict[str, float]] = None,
    stability_definition: str = "blocking_swap",
) -> Dict[str, object]:
    """
    See this section's module docstring for the "reporting layer, not a
    driver" design decision, and OBJECTIVE_COMPONENT_SPECS for each
    component's exact range/direction/missing-data policy.

    OVERLAP CHECK (Section 4's explicit instruction to inspect these):
      - behavioral_compatibility (team-level rubric score) vs.
        individual_fairness (per-student utility, whose OWN
        `behavioral_fit` component is computed per-student against that
        student's teammates specifically) — related but NOT the same
        number: one is a team aggregate under composition rules, the
        other is a per-student comparison against teammates' work-rhythm/
        collaboration signatures. Kept as separate components since
        zeroing either weight independently is a meaningful choice a
        profile might want to make.
      - skill_quality (team-level coverage fraction) vs. individual_
        fairness's `skill_fit` sub-component (per-student skill overlap
        with the team) — again related, not identical; team coverage can
        be complete while one student's own personal skill_fit is low
        (they contribute a DIFFERENT skill than their own listed
        interest area might suggest).
      - familiarity_band_fit / anti_cluster vs. preference_satisfaction —
        familiarity concerns PRIOR relationships (FamiliarityDB, no
        student opinion involved); preference concerns STATED,
        student-submitted desires (PreferenceDB) — a teammate can be
        "familiar" (worked together before) without ever being
        "preferred" (named in a PreferenceEntry), and vice versa. No
        double-count: these are structurally independent data sources.
      - team_balance (spread of team_total_score) vs. ability_balance
        (spread of mean ability specifically) — ability is ONE input to
        team_total_score only insofar as ability correlates with skill/
        behavioral fit (it is not itself a _total_score component), so
        these measure related but distinct spreads; both are kept since
        a profile may care about one without the other.
      - stability_penalty (blocking-SWAP count) vs. preference_
        satisfaction (mean tier-scored SATISFACTION) — a low blocking-
        swap count does not imply high satisfaction (students may simply
        lack the data to form a blocking swap) and vice versa; kept as
        genuinely separate components, with stability_penalty defaulting
        to weight 0.0 specifically so it never silently double-counts
        against preference_satisfaction unless a profile deliberately
        enables it.
    """
    objective_weights = dict(objective_weights) if objective_weights else dict(DEFAULT_OBJECTIVE_WEIGHTS)
    familiarity_config = familiarity_config or FamiliarityConfig()
    familiarity_edges = familiarity_edges or []
    preference_entries = preference_entries or []
    non_empty_teams = [t for t in teams if t]

    components: Dict[str, Dict[str, object]] = {}

    coverages = [_skill_coverage(t) for t in non_empty_teams]
    components["skill_quality"] = _objective_component(
        statistics.fmean(coverages) if coverages else None,
        objective_weights.get("skill_quality", 0.0), 1.0 if coverages else 0.0,
    )

    compats = [_behavioral_compat(t) for t in non_empty_teams]
    components["behavioral_compatibility"] = _objective_component(
        statistics.fmean(compats) if compats else None,
        objective_weights.get("behavioral_compatibility", 0.0), 1.0 if compats else 0.0,
    )

    scores = [_total_score(t, _DEFAULT_WEIGHTS) for t in non_empty_teams]
    team_balance_raw = None
    if scores:
        team_balance_raw = 1.0 if len(scores) < 2 else max(0.0, 1.0 - min(1.0, statistics.pstdev(scores)))
    components["team_balance"] = _objective_component(
        team_balance_raw, objective_weights.get("team_balance", 0.0), 1.0 if scores else 0.0,
    )

    ability_report = compute_ability_metrics(teams, ability_level_thresholds)
    gap = ability_report.get("strongest_weakest_gap")
    components["ability_balance"] = _objective_component(
        (1.0 - gap) if gap is not None else None,
        objective_weights.get("ability_balance", 0.0), 1.0 if gap is not None else 0.0,
    )

    fam_report = compute_familiarity_metrics(
        teams, familiarity_edges,
        familiar_pair_threshold=familiarity_config.familiar_pair_threshold,
        clique_edge_threshold=familiarity_config.clique_edge_threshold,
        lower_bound=familiarity_config.lower_bound,
        upper_bound=familiarity_config.upper_bound,
        minimum_known_pair_coverage=familiarity_config.minimum_known_pair_coverage,
        quasi_clique_density_threshold=familiarity_config.quasi_clique_density_threshold,
    )
    evaluable = fam_report["teams_within_band"] + fam_report["teams_below_band"] + fam_report["teams_above_band"]
    components["familiarity_band_fit"] = _objective_component(
        (fam_report["teams_within_band"] / evaluable) if evaluable else None,
        objective_weights.get("familiarity_band_fit", 0.0), (evaluable / len(teams)) if teams else 0.0,
    )

    cluster_fracs = [t["largest_connected_cluster"]["fraction_of_team"] for t in fam_report["per_team"] if t["total_possible_pairs"] > 0]
    components["anti_cluster"] = _objective_component(
        (1.0 - statistics.fmean(cluster_fracs)) if cluster_fracs else None,
        objective_weights.get("anti_cluster", 0.0), (len(cluster_fracs) / len(teams)) if teams else 0.0,
    )

    utility_report = compute_assignment_utilities(teams, utility_weights=utility_weights)
    min_utility = utility_report["summary"]["min_utility"]
    components["individual_fairness"] = _objective_component(
        min_utility, objective_weights.get("individual_fairness", 0.0), 1.0 if min_utility is not None else 0.0,
    )

    pref_report = compute_preference_metrics(teams, preference_entries)
    n_students = sum(len(t) for t in teams)
    components["preference_satisfaction"] = _objective_component(
        pref_report["mean_satisfaction"], objective_weights.get("preference_satisfaction", 0.0),
        (pref_report["n_students_with_preferences"] / n_students) if n_students else 0.0,
    )

    stability_raw = None
    if n_students:
        stability_report = compute_stability_metrics(teams, preference_entries, definition=stability_definition)
        stability_raw = max(0.0, 1.0 - (stability_report["n_blocking_swaps"] / max(1, n_students)))
    components["stability_penalty"] = _objective_component(
        stability_raw, objective_weights.get("stability_penalty", 0.0), 1.0 if stability_raw is not None else 0.0,
    )

    usable = [(k, c) for k, c in components.items() if c["raw"] is not None and c["weight"] > 0]
    total_weight = sum(c["weight"] for _, c in usable)
    total = None
    if total_weight > 0:
        total = sum(c["weight"] * c["normalized"] for _, c in usable) / total_weight
        for k, c in components.items():
            if (k, c) in usable:
                c["weighted_contribution"] = c["weight"] * c["normalized"] / total_weight
    renormalized = total_weight > 0 and abs(total_weight - sum(w for w in objective_weights.values() if w > 0)) > 1e-9

    return {
        "components": components,
        "total_weighted_objective": total,
        "weights_renormalized_over_usable_components": renormalized,
        "note": (
            "Every component's `normalized` value is on a documented [0,1] scale where higher is "
            "always better (see OBJECTIVE_COMPONENT_SPECS). A component with raw=None (missing/"
            "insufficient data) is EXCLUDED from total_weighted_objective, and the remaining "
            "components' weights are renormalized to sum to 1.0 over just the usable ones — never "
            "silently scored as 0. `evidence_coverage` is the fraction of this assignment's teams/"
            "students that contributed a data point to that component's raw value."
        ),
    }


# ---------------------------------------------------------------------------
# Named configuration profiles (Section 3). Every profile resolves, via
# resolve_profile(), into one COMPLETE, immutable TeamGenerationRequest —
# precedence is ALWAYS system defaults -> named profile's overrides ->
# explicit request-level field overrides (highest), using Pydantic's own
# model_fields_set to distinguish "the caller actually sent this field"
# from "this is just the class default" at every level of nesting
# TeamGenerationRequest has (one level: `ability`, `fairness`,
# `familiarity`, `preference`, `weights`).
# ---------------------------------------------------------------------------

OPTIMIZATION_PROFILES: Dict[str, Dict[str, object]] = {
    "legacy": {
        "is_experimental": False,
        "primary_metric": None,
        "description": (
            "Reproduces this system's shipped-by-default behavior from Parts 1 through 5.6: every "
            "optional research pass (ability/fairness/familiarity/preference) OFF, hard_avoidance "
            "OFF. An existing client that omits `profile` entirely gets EXACTLY this — no research "
            "objective is ever silently enabled."
        ),
        "overrides": {},
    },
    "research_balanced": {
        "is_experimental": True,
        "primary_metric": None,
        "description": (
            "EXPERIMENTAL. Enables every supported research-criterion optimization pass "
            "(ability, fairness, familiarity, preference) with the conservative, documented "
            "defaults each Config class already ships (no threshold here is independently "
            "re-tuned). Pursues all seven research criteria together. NOT validated against real "
            "student outcomes and NOT claimed optimal for any population — see "
            "docs/optimization_profiles.md's limitations before using this in a real deployment."
        ),
        "overrides": {
            "ability": {"enabled": True},
            "fairness": {"enabled": True},
            "familiarity": {"enabled": True},
            "preference": {"enabled": True},
        },
    },
    "fairness_priority": {
        "is_experimental": True,
        "primary_metric": "min_utility",
        "description": (
            "EXPERIMENTAL. Prioritizes the worst-off individual student's utility and worst-case "
            "regret over other soft objectives, while still preserving every hard constraint. "
            "Enables the fairness pass and raises acceptable_utility_floor for reporting; other "
            "optional passes stay off unless explicitly overridden."
        ),
        "overrides": {"fairness": {"enabled": True, "acceptable_utility_floor": 0.6}},
    },
    "skill_priority": {
        "is_experimental": True,
        "primary_metric": "skill_coverage_min",
        "description": (
            "EXPERIMENTAL. Prioritizes required skill coverage and skill-based complementarity by "
            "raising skill_coverage's weight in the primary total-score objective. This does not "
            "claim ability/fairness/familiarity/preference are unimportant — only that skill "
            "coverage is THIS profile's stated priority; every other criterion is still measured "
            "and reported regardless."
        ),
        "overrides": {"weights": {
            "skill_coverage": 0.7, "behavioral_compat": 0.15,
            "availability_overlap": 0.1, "shared_interests": 0.05,
        }},
    },
    "preference_priority": {
        "is_experimental": True,
        "primary_metric": "mean_preference_satisfaction",
        "description": (
            "EXPERIMENTAL. Prioritizes preference satisfaction and blocking-swap reduction. "
            "Enables the preference-repair pass with heavier high-priority/avoidance weighting, "
            "AND enables hard_avoidance (an explicit institutional restriction, not merely a soft "
            "preference) since a profile whose whole purpose is preference-aware matching should "
            "not silently leave avoidance requests unenforced."
        ),
        "overrides": {"preference": {
            "enabled": True, "hard_avoidance": True,
            "high_priority_weight": 4.0, "avoidance_violation_weight": 6.0,
        }},
    },
}


def _deep_update(base: dict, overrides: dict) -> dict:
    """Plain recursive dict merge — `overrides` wins at every leaf, dicts merge key-wise."""
    merged = dict(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_update(merged[k], v)
        else:
            merged[k] = v
    return merged


def _merge_explicit_overrides(base: dict, request: TeamGenerationRequest) -> dict:
    """
    Overlays ONLY the fields `request` was actually CONSTRUCTED with (per
    Pydantic's own model_fields_set at every nesting level this schema
    has) onto `base` — this is what gives explicit request-level overrides
    the HIGHEST precedence without accidentally re-applying a nested
    config's own class defaults over a profile's chosen value for a
    sibling field the caller did not mention.
    """
    merged = dict(base)
    for field_name in request.model_fields_set:
        if field_name == "profile":
            continue
        value = getattr(request, field_name)
        if hasattr(value, "model_fields_set") and isinstance(merged.get(field_name), dict):
            sub_merged = dict(merged[field_name])
            for sub_field in value.model_fields_set:
                sub_value = getattr(value, sub_field)
                sub_merged[sub_field] = sub_value.model_dump() if hasattr(sub_value, "model_dump") else sub_value
            merged[field_name] = sub_merged
        else:
            merged[field_name] = value.model_dump() if hasattr(value, "model_dump") else value
    return merged


def resolve_profile(request: TeamGenerationRequest) -> Tuple[TeamGenerationRequest, ProfileResolution]:
    """
    Resolves `request.profile` into ONE COMPLETE, immutable
    TeamGenerationRequest (a fresh Pydantic instance — never a mutated
    shared object, so there is no hidden global state and repeated calls
    with the same input are fully deterministic) plus a ProfileResolution
    record for the API response. Precedence, always: system defaults ->
    named profile's overrides -> explicit request-level field overrides.
    Raises ValueError for an unknown profile name (generate_teams()
    converts this to a 422) — a typo is never silently treated as "legacy".
    """
    profile_name = request.profile
    if profile_name not in OPTIMIZATION_PROFILES:
        raise ValueError(
            f"unknown optimization profile: {profile_name!r}; valid profiles: "
            f"{sorted(OPTIMIZATION_PROFILES)}"
        )
    profile_def = OPTIMIZATION_PROFILES[profile_name]

    resolved_data = TeamGenerationRequest().model_dump()
    resolved_data = _deep_update(resolved_data, profile_def["overrides"])
    resolved_data = _merge_explicit_overrides(resolved_data, request)
    resolved_data["profile"] = profile_name
    resolved = TeamGenerationRequest(**resolved_data)

    return resolved, ProfileResolution(
        name=profile_name,
        is_experimental=profile_def["is_experimental"],
        description=profile_def["description"],
        effective_settings=resolved.model_dump(),
    )


# ---------------------------------------------------------------------------
# Cross-pass acceptance policy (Section 6) — see this section's module
# docstring above for why this is "Option B, extended one level" rather
# than a from-scratch unified objective.
# ---------------------------------------------------------------------------

PROTECTED_METRIC_TOLERANCE = SWAP_IMPROVEMENT_TOLERANCE


def _protected_metrics(
    teams: List[List[Student]], weights: dict, floor_policy: str,
    extra_metric: Optional[str] = None,
    preference_entries: Optional[List[PreferenceEntry]] = None,
    utility_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """The universal protected triple (total_score, floor_min,
    skill_coverage_min) — checked after EVERY optional pass — plus,
    optionally, ONE additional profile-designated `extra_metric` (see
    OPTIMIZATION_PROFILES' `primary_metric`), giving the currently
    resolved profile's own stated priority the SAME non-regression
    protection the universal three already get."""
    metrics = {
        "total_score": _stage_objective(teams, weights),
        "floor_min": _partition_floor_min(teams),
        "skill_coverage_min": _partition_skill_coverage_min(teams),
    }
    if extra_metric == "min_utility":
        metrics["min_utility"] = compute_assignment_utilities(teams, utility_weights=utility_weights)["summary"]["min_utility"] or 0.0
    elif extra_metric == "mean_preference_satisfaction":
        sat = compute_preference_metrics(teams, preference_entries or [])["mean_satisfaction"]
        metrics["mean_preference_satisfaction"] = sat if sat is not None else 1.0  # no data -> nothing to protect against regressing
    return metrics


def _run_optional_passes(
    raw_groups: List[List[Student]],
    request: TeamGenerationRequest,
    weights: dict,
    floor_policy: str,
    familiarity_edges: List[FamiliarityEdge],
    preference_entries: List[PreferenceEntry],
    primary_metric: Optional[str] = None,
) -> Tuple[List[List[Student]], List[Dict[str, object]]]:
    """
    Runs every optional pass `request.*.enabled` turns on, in this
    codebase's FIXED, long-established order (ability -> fairness ->
    familiarity -> hard_avoidance -> preference — unchanged from Part
    5.6). See this Part 6 section's module docstring for the acceptance
    policy this adds ON TOP of each pass's own internal guarantees: after
    EVERY pass, the PROTECTED metrics (total_score, floor_min,
    skill_coverage_min, plus the resolved profile's own `primary_metric`
    if any) are recomputed; if ANY regressed beyond
    PROTECTED_METRIC_TOLERANCE relative to their value immediately before
    that pass ran, the pass's ENTIRE effect is discarded (reverted to the
    pre-pass assignment) — catching cross-pass or cumulative-within-pass
    drift no single pass's own per-swap tolerance check could see on its
    own. Returns (final_teams, trace); `trace` records, per pass that
    actually ran, the exact before/after protected-metric values and
    whether it was accepted or reverted — this is the "record before-and-
    after component values for accepted swaps during debug evaluation"
    requirement, surfaced on the API response's `optimization_trace`.
    """
    trace: List[Dict[str, object]] = []
    teams = raw_groups

    def _maybe_run(name: str, enabled: bool, fn):
        nonlocal teams
        if not enabled:
            return
        before = _protected_metrics(teams, weights, floor_policy, primary_metric, preference_entries, request.fairness.weights)
        candidate = fn([list(t) for t in teams])
        after = _protected_metrics(candidate, weights, floor_policy, primary_metric, preference_entries, request.fairness.weights)

        # Part 6.5 orchestration-audit fix: `after[k] < before[k] - tol` is
        # SILENTLY False whenever either side is NaN (Python's NaN
        # comparisons are always False), which would let a broken/NaN
        # metric slip through as "not regressed" and get accepted into the
        # final assignment. Any NaN or non-finite value on EITHER side is
        # therefore treated as an unconditional regression — a metric
        # computation producing NaN/inf is a bug to surface loudly, never
        # a silently-accepted pass. See test_nan_metric_forces_pass_
        # rejection_rather_than_silently_bypassing_the_regression_check.
        non_finite_keys = [
            k for k in before
            if not math.isfinite(before[k]) or not math.isfinite(after[k])
        ]
        regressed = bool(non_finite_keys) or any(
            after[k] < before[k] - PROTECTED_METRIC_TOLERANCE for k in before
        )
        accepted = not regressed
        trace.append({
            "pass": name, "before": before, "after": after, "accepted": accepted,
            "non_finite_metric_keys": non_finite_keys,
        })
        if accepted:
            teams = candidate

    _maybe_run("ability", request.ability.enabled, lambda t: _ability_balance_improve(
        t, weights=weights, floor_policy=floor_policy,
    ))
    _maybe_run("fairness", request.fairness.enabled, lambda t: _fairness_improve(
        t, weights=weights, utility_weights=request.fairness.weights, floor_policy=floor_policy,
    ))
    _maybe_run("familiarity", request.familiarity.enabled, lambda t: _familiarity_balance_improve(
        t, edges=familiarity_edges, weights=weights, floor_policy=floor_policy,
        lower_bound=request.familiarity.lower_bound, upper_bound=request.familiarity.upper_bound,
        clique_edge_threshold=request.familiarity.clique_edge_threshold,
        max_clique_fraction=request.familiarity.max_clique_fraction,
        minimum_known_pair_coverage=request.familiarity.minimum_known_pair_coverage,
        penalize_insufficient_data=request.familiarity.penalize_insufficient_data,
    ))
    _maybe_run("hard_avoidance", request.preference.hard_avoidance, lambda t: _hard_avoidance_repair(
        t, entries=preference_entries, weights=weights, floor_policy=floor_policy,
    ))
    _maybe_run("preference", request.preference.enabled, lambda t: _preference_repair_improve(
        t, entries=preference_entries, weights=weights, floor_policy=floor_policy,
        high_priority_weight=request.preference.high_priority_weight,
        avoidance_violation_weight=request.preference.avoidance_violation_weight,
        hard_avoidance=request.preference.hard_avoidance,
    ))
    return teams, trace


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
# Team-size balancing
# ---------------------------------------------------------------------------

def _compute_n_teams(n_students: int, team_size: int) -> int:
    """
    Number of teams that keeps team sizes as close to the requested
    `team_size` as possible. Rosters that don't divide evenly are rounded to
    the NEAREST team count (not floor) so leftover students form their own
    fair share of a team rather than being piled onto existing teams — e.g.
    11 students at team_size=4 forms 3 teams (round(11/4)=3), not 2.
    """
    if n_students <= 0 or team_size <= 0:
        return 0
    return max(1, round(n_students / team_size))


def _target_team_sizes(n_students: int, n_teams: int) -> List[int]:
    """
    Split n_students into n_teams groups as evenly as possible: sizes never
    differ by more than 1 (the larger teams come first in the returned list).
    """
    if n_teams <= 0:
        return []
    base, extra = divmod(n_students, n_teams)
    return [base + 1] * extra + [base] * (n_teams - extra)


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
    added greedily by total score, respecting each team's balanced target
    capacity (see _target_team_sizes) so rosters that don't divide evenly by
    team_size still produce teams whose sizes differ by at most 1 — no team
    is ever left to absorb every leftover student.

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

    n_teams = _compute_n_teams(len(students), team_size)
    target_sizes = _target_team_sizes(len(students), n_teams)
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
        # Find best (team, candidate) pair among teams under their target size
        best_score = -float("inf")
        best_team_idx = None
        best_student = None

        for team_idx, team in enumerate(teams):
            if len(team) >= target_sizes[team_idx]:
                continue
            for candidate in remaining:
                s = _total_score(team + [candidate], weights)
                if best_team_idx is None or s > best_score:
                    best_score = s
                    best_team_idx = team_idx
                    best_student = candidate

        # target_sizes always sums to len(students), so remaining capacity
        # across all teams always matches len(remaining) here.
        teams[best_team_idx].append(best_student)
        remaining.remove(best_student)

    return teams


def _two_opt_improve(
    teams: List[List[Student]],
    weights: dict,
    floor_policy: str = DEFAULT_FLOOR_POLICY,
    hard_skill_coverage_floor: float = HARD_SKILL_COVERAGE_FLOOR,
    tolerance: float = SWAP_IMPROVEMENT_TOLERANCE,
    max_sweeps: int = MAX_OPTIMIZATION_SWEEPS,
) -> List[List[Student]]:
    """
    Pairwise swap pass that maximises the summed total_score of each swapped
    pair of teams — constraint-safe: a swap that raises the aggregate score
    by trading away required-skill coverage (or any other hard constraint)
    is rejected outright, never merely "compensated for" by the score gain
    (research_alignment_audit.md recommended_changes.md #7 — confirmed:
    the greedy_local_optimum fixture had an available swap that raised the
    summed score 1.370 -> 1.403 while dropping one team's skill coverage
    from 0.583 to 0.417, below the *other* team's pre-swap 0.500 minimum;
    that swap is now rejected).

    Acceptance rule, in order:
      1. validate_assignment(...) must hold (hard constraints — including
         "the partition's minimum skill coverage may not regress below what
         greedy already achieved", which is what rejects the swap above).
      2. only then must the swap raise the pair's summed total_score by more
         than `tolerance` — an equal-or-smaller change is a no-op/tie and is
         rejected, so repeated runs on identical input are identical.

    Termination: every accepted swap strictly raises the summed pair score
    by more than `tolerance`, and total_score is bounded above by 1.0 per
    team; `max_sweeps` is a documented hard backstop.
    """
    baseline = AssignmentBaseline.capture(teams, floor_policy)

    # Incremental per-team caches — see _maximin_improve for the rationale
    # (identical pattern): a swap only ever changes two teams, so skill
    # coverage / floor-violation counts for every other team are read from
    # cache instead of rescanned on every candidate.
    coverage_cache = [_skill_coverage(t) for t in teams]
    violation_cache = [_team_floor_violation_count(t, floor_policy) for t in teams]

    sweeps = 0
    improved = True
    while improved and sweeps < max_sweeps:
        improved = False
        sweeps += 1
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                others = [k for k in range(len(teams)) if k != i and k != j]
                others_min_cov = min((coverage_cache[k] for k in others), default=1.0)
                others_violations = sum(violation_cache[k] for k in others)

                for a_idx in range(len(teams[i])):
                    for b_idx in range(len(teams[j])):
                        original = _total_score(teams[i], weights) + _total_score(teams[j], weights)
                        teams[i][a_idx], teams[j][b_idx] = teams[j][b_idx], teams[i][a_idx]

                        new_cov_i = _skill_coverage(teams[i])
                        new_cov_j = _skill_coverage(teams[j])
                        new_viol_i = _team_floor_violation_count(teams[i], floor_policy)
                        new_viol_j = _team_floor_violation_count(teams[j], floor_policy)
                        cov_min_candidate = min(new_cov_i, new_cov_j, others_min_cov)
                        violations_candidate = new_viol_i + new_viol_j + others_violations

                        validation = validate_assignment(
                            teams, baseline, hard_skill_coverage_floor,
                            precomputed_min_skill_coverage=cov_min_candidate,
                            precomputed_floor_violation_count=violations_candidate,
                        )
                        after = _total_score(teams[i], weights) + _total_score(teams[j], weights)

                        if validation.ok and after > original + tolerance:
                            improved = True
                            coverage_cache[i], coverage_cache[j] = new_cov_i, new_cov_j
                            violation_cache[i], violation_cache[j] = new_viol_i, new_viol_j
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
    from app.database import SessionLocal, TeamDB, StudentDB, FamiliarityDB, PreferenceDB

    # Part 6 — resolve the named profile FIRST (system defaults -> named
    # profile -> explicit request overrides), so every line below reads
    # from `request` (now the RESOLVED, complete, immutable config) rather
    # than threading two objects through the whole function. The caller's
    # ORIGINAL request is never mutated — resolve_profile always returns a
    # fresh instance.
    try:
        request, profile_resolution = resolve_profile(request)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    db = SessionLocal()
    try:
        db_students = db.query(StudentDB).all()
        students = [_db_student_to_model(s) for s in db_students]
        familiarity_edges = [
            FamiliarityEdge(student_a_id=r.student_a_id, student_b_id=r.student_b_id, strength=r.strength, source=r.source)
            for r in db.query(FamiliarityDB).all()
        ]
        preference_entries = [
            PreferenceEntry(
                student_id=r.student_id, direction=r.direction,
                target_student_id=r.target_student_id, target_characteristic=r.target_characteristic,
                rank=r.rank, priority=r.priority or "normal",
                reason_category=r.reason_category, source=r.source,
            )
            for r in db.query(PreferenceDB).all()
        ]
    finally:
        db.close()

    if len(students) < request.team_size:
        raise HTTPException(status_code=400, detail="Not enough students to form a team.")

    # Section 5 — roster validation for preference targets: an unknown
    # target (not present in this roster) is REPORTED (a count only — see
    # PreferenceSummary's privacy design, never a target identity) and
    # otherwise passed through unchanged; compute_preference_metrics /
    # compute_stability_metrics already treat an unmatched target as
    # "never satisfied" structurally, so a stale ID can never silently
    # inflate a score, only ever fail to match (documented policy — see
    # evaluation.metrics.legacy_preferences_to_entries' identical policy
    # for the legacy-dict boundary).
    roster_ids = {s.id for s in students}
    n_unknown_preference_targets = sum(
        1 for e in preference_entries
        if e.target_student_id is not None and e.target_student_id not in roster_ids
    )

    mode = request.formation_mode
    behavioral_weights = _normalize_weights(request.weights)
    floor_policy = request.floor_policy
    # Same n_teams calculation _greedy_build uses (round-to-nearest, not
    # floor) so the pre-formation feasibility check reasons about the same
    # partition shape the algorithm will actually produce.
    n_teams = _compute_n_teams(len(students), request.team_size)

    # Feasibility check for floor dimensions before running optimization
    if mode == "behavioral":
        feasible_pre, infeasible_dims = _check_feasibility(students, n_teams, floor_policy)
        if not feasible_pre:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "infeasible_partition",
                    "message": (
                        "Too many students at the lowest level on floor dimensions "
                        "to satisfy the requested floor_policy "
                        f"('{floor_policy}')."
                    ),
                    "infeasible_dims": {
                        dim: f"{count} students at floor level but only {n_teams} teams"
                        for dim, count in infeasible_dims.items()
                    },
                },
            )

    optimization_trace: List[Dict[str, object]] = []
    if mode == "random":
        raw_groups = random_assignment(students, request.team_size)
    elif mode == "skill":
        raw_groups = skill_based_assignment(students, request.team_size)
    else:
        raw_groups = _greedy_build(students, request.team_size, behavioral_weights)
        raw_groups = _two_opt_improve(raw_groups, behavioral_weights, floor_policy=floor_policy)
        raw_groups = _maximin_improve(raw_groups, weights=behavioral_weights, floor_policy=floor_policy)
        # Every optional pass (ability/fairness/familiarity/hard_avoidance/
        # preference) is OFF by default and a no-op with no relevant data —
        # each one only runs, and can only change the returned assignment,
        # when the RESOLVED profile (system defaults -> named profile ->
        # explicit request overrides — see resolve_profile()) turns its
        # flag on. Every metric below is still computed and reported
        # regardless (Part 2/3/4/5 requirement, unchanged). See
        # _run_optional_passes' docstring for the cross-pass acceptance
        # policy (Part 6, Section 6) applied on top of each pass's own
        # internal guarantees.
        raw_groups, optimization_trace = _run_optional_passes(
            raw_groups, request, behavioral_weights, floor_policy,
            familiarity_edges, preference_entries,
            primary_metric=OPTIMIZATION_PROFILES[request.profile]["primary_metric"],
        )

    result = _build_teams(raw_groups, mode, behavioral_weights)

    # Individual-utility / fairness reporting — evaluation-only, computed
    # for whatever assignment was actually produced above; never feeds back
    # into which assignment was chosen (that only happens via
    # _fairness_improve, gated above by request.fairness.enabled).
    utility_report = compute_assignment_utilities(
        raw_groups,
        utility_weights=request.fairness.weights,
        acceptable_utility_floor=request.fairness.acceptable_utility_floor,
        max_exact_regret_roster_size=request.fairness.max_exact_regret_roster_size,
    )
    student_utilities = _student_utility_models(utility_report)
    fairness_summary = _fairness_summary_model(utility_report)

    # Ability reporting — same evaluation-only relationship to matching as
    # utility above (see _ability_balance_improve, gated by request.ability.enabled).
    ability_report = compute_ability_metrics(
        raw_groups,
        ability_level_thresholds=request.ability.ability_level_thresholds,
    )
    ability_summary = _ability_summary_model(ability_report)
    student_ability = {
        s.id: _student_ability_score(s)
        for team in raw_groups for s in team
    }
    for team in result:
        for member in team.members:
            member.utility = student_utilities.get(member.student_id)
            score, source = student_ability.get(member.student_id, (None, ABILITY_SOURCE_UNKNOWN))
            member.ability_score = score
            member.ability_source_used = source
            member.ability_level = _ability_level(score, request.ability.ability_level_thresholds) if score is not None else None

    # Familiarity reporting — same evaluation-only relationship to matching
    # as utility/ability above (see _familiarity_balance_improve, gated by
    # request.familiarity.enabled).
    familiarity_report = compute_familiarity_metrics(
        raw_groups,
        edges=familiarity_edges,
        familiar_pair_threshold=request.familiarity.familiar_pair_threshold,
        clique_edge_threshold=request.familiarity.clique_edge_threshold,
        lower_bound=request.familiarity.lower_bound,
        upper_bound=request.familiarity.upper_bound,
        minimum_known_pair_coverage=request.familiarity.minimum_known_pair_coverage,
        quasi_clique_density_threshold=request.familiarity.quasi_clique_density_threshold,
    )
    familiarity_summary = _familiarity_summary_model(familiarity_report)

    # Preference/stability reporting — same evaluation-only relationship to
    # matching as utility/ability/familiarity above (see
    # _preference_repair_improve, gated by request.preference.enabled).
    preference_report = compute_preference_metrics(raw_groups, preference_entries)
    stability_baseline = AssignmentBaseline.capture(raw_groups, floor_policy)
    stability_report = compute_stability_metrics(
        raw_groups, preference_entries, baseline=stability_baseline,
        floor_policy=floor_policy, definition=request.preference.stability_definition,
    )
    preference_summary = _preference_summary_model(preference_report, stability_report)

    # Part 6 — normalized cross-criterion objective REPORT (Section 4):
    # evaluation-only, reported for transparency/profile-comparison; never
    # itself drives which assignment was returned (see this module's Part
    # 6 section docstring for why).
    normalized_objective = compute_normalized_objective(
        raw_groups, familiarity_edges=familiarity_edges, preference_entries=preference_entries,
        ability_level_thresholds=request.ability.ability_level_thresholds,
        familiarity_config=request.familiarity, utility_weights=request.fairness.weights,
        stability_definition=request.preference.stability_definition,
    )

    # Post-formation floor check (informational — optimization may not eliminate all violations)
    floor_ok, floor_violations, under_determined_teams = _check_floor_violations(raw_groups, floor_policy)

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
        "floor_policy": floor_policy,
        "fairness_summary": fairness_summary,
        "ability_summary": ability_summary,
        "familiarity_summary": familiarity_summary,
        "preference_summary": preference_summary,
        "profile": profile_resolution,
        "normalized_objective": normalized_objective,
        "optimization_trace": optimization_trace,
        "n_unknown_preference_targets": n_unknown_preference_targets,
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
