from __future__ import annotations
from typing import Annotated, List, Literal, Optional, Dict
from pydantic import BaseModel, Field, model_validator


# Ability scale (Part 3): both fields below live on the SAME [0.0, 1.0]
# continuous scale documented here once and reused everywhere ability is
# read (app/matching.py, evaluation/metrics.py). 0.0 = weakest observed
# proficiency, 1.0 = strongest — NOT a percentile, NOT a grade, and NOT
# assumed to be GPA: `ability_source` records where the number actually
# came from (see its docstring) so instructors can plug in whatever
# validated-or-not measure they trust, or leave it entirely unset.
#
# `Optional[float] = None` is the field default (not 0.0) specifically so
# "no ability data was ever recorded for this student" stays distinguishable
# from "this student's recorded ability is genuinely at the bottom of the
# scale" everywhere downstream — see app/matching.py:_student_ability_score.
_AbilityUnitInterval = Annotated[float, Field(ge=0.0, le=1.0)]


class CompetenceSignature(BaseModel):
    skills: List[str] = []
    roles: List[str] = []
    experience_level: str = "intermediate"  # beginner | intermediate | advanced
    skill_depth: str = "intermediate"
    skill_no_gos: List[str] = []
    # General ability proxy on the documented [0.0, 1.0] scale above.
    # None = not assessed (existing records without this field deserialize
    # with ability_score=None automatically — no migration required for the
    # JSON-backed competence_signature column, see alembic/versions for the
    # accompanying documentation-only revision). Out-of-range or
    # non-numeric values raise a pydantic ValidationError at construction.
    ability_score: Optional[_AbilityUnitInterval] = None
    # Where ability_score came from. Deliberately a free-form string, not a
    # hardcoded enum of "gpa" — the research team may prefer an instructor
    # holistic rating, a placement exam, a prior-course grade, or something
    # else entirely (research_alignment_audit.md's own "Questions that must
    # be answered by the research team" #2 flags GPA as one option among
    # several, not a default). Documented common values (informational
    # only, NOT enforced): "instructor_rating", "prior_course_grade",
    # "placement_exam", "gpa", "self_report".
    ability_source: Optional[str] = None
    # OPTIONAL per-skill proficiency on the same [0.0, 1.0] scale, keyed by
    # skill name (values from `skills` above, but not required to cover all
    # of them). A skill present in `skills` but absent from this dict means
    # "proficiency not assessed for that skill" — distinct from an entry
    # explicitly recorded at 0.0. Empty dict (the default) means no
    # skill-specific proficiency was ever recorded, which is also how every
    # existing student record deserializes today.
    skill_proficiency: Dict[str, _AbilityUnitInterval] = Field(default_factory=dict)


class WorkRhythmSignature(BaseModel):
    planning_style: str = "adaptive"
    communication_style: str = "mixed"
    execution_style: str = "iterative"
    availability: List[str] = []
    unavailable_times: List[str] = []
    timezone_notes: str = ""


class CollaborationSignature(BaseModel):
    conflict_style: str = "collaborative"
    leadership_style: str = "facilitative"
    accountability: str = "developing"      # low | developing | high
    help_seeking: str = "proactive"
    cooperativeness: str = "developing"     # low | developing | high
    safety_contribution: str = "medium"
    stress_response: str = "composed"


class MotivationLayer(BaseModel):
    interests: List[str] = []
    learning_goals: List[str] = []


class ConfidenceLayer(BaseModel):
    # Scorer measurement quality, NOT student self-assurance (see rubric_v1.yaml).
    confidence_score: float = Field(default=0.5, ge=0.0, le=1.0)


class Student(BaseModel):
    id: str
    name: str
    competence_signature: CompetenceSignature = Field(default_factory=CompetenceSignature)
    work_rhythm_signature: WorkRhythmSignature = Field(default_factory=WorkRhythmSignature)
    collaboration_signature: CollaborationSignature = Field(default_factory=CollaborationSignature)
    motivation_layer: MotivationLayer = Field(default_factory=MotivationLayer)
    confidence_layer: ConfidenceLayer = Field(default_factory=ConfidenceLayer)
    # Dimensions for which the scorer received zero MCQ signals.
    # Matcher treats these as unknown — they do NOT satisfy or violate floor.
    under_determined_dims: List[str] = Field(default_factory=list)


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
    option_id: str           # anonymous: opt_a | opt_b | opt_c | opt_d
    response_text: str = ""


class AssessmentRequest(BaseModel):
    student_name: str
    student_id: Optional[str] = None
    session_id: Optional[str] = None
    responses: List[AssessmentResponse]


class StartSessionRequest(BaseModel):
    student_name: str
    student_id: Optional[str] = None


class SaveProgressRequest(BaseModel):
    session_id: str
    student_id: str
    responses: List[AssessmentResponse]


# Team models

class StudentUtility(BaseModel):
    """
    Per-student breakdown of how beneficial a proposed assignment is for
    ONE student — distinct from TeamScoreBreakdown, which is a team-level
    aggregate identical for every member. See app/matching.py's individual-
    utility framework (Part 2) for the exact formulas.

    `components` may contain `None` for a factor that could not be
    meaningfully computed for this student on this team (e.g. a team of one
    has no teammates to compare availability against) — that is DISTINCT
    from a genuine 0.0, which means the factor WAS computed and came out at
    the low end of its scale. `missing_components` names every such factor
    explicitly so a `None` is never silently treated as a zero downstream.
    """
    student_id: str
    utility: float = Field(ge=0.0, le=1.0)
    components: Dict[str, Optional[float]] = Field(default_factory=dict)
    missing_components: List[str] = Field(default_factory=list)
    regret: Optional[float] = None  # None when exact regret search wasn't run (see FairnessSummary.exact_regret_search)


class FairnessConfig(BaseModel):
    """
    Instructor-facing configuration for the individual-utility / fairness
    framework. `enabled=False` (default) means fairness is evaluated and
    reported but NEVER changes which assignment /teams/generate returns —
    matching behavior is byte-for-byte identical to a request with no
    fairness config at all. Setting `enabled=True` turns on an additional,
    constraint-safe optimization pass (_fairness_improve in app/matching.py)
    that may change the returned assignment to raise the worst-off student's
    utility, without regressing the total-score objective or the
    accountability/cooperativeness floor-minimum earlier passes established.
    """
    enabled: bool = False
    weights: Dict[str, float] = Field(default_factory=lambda: {
        "skill_fit": 0.4,
        "behavioral_fit": 0.3,
        "availability_fit": 0.2,
        "interest_fit": 0.1,
    })
    # A student below this utility is counted in FairnessSummary's
    # students_below_floor metric. Not a hard constraint — purely reporting.
    acceptable_utility_floor: float = Field(default=0.5, ge=0.0, le=1.0)
    # Worst-case regret requires an exact brute-force alternative-team
    # search (combinatorial); above this roster size it is skipped and
    # FairnessSummary.exact_regret_search is False.
    max_exact_regret_roster_size: int = Field(default=20, ge=0)


class FairnessSummary(BaseModel):
    """
    Assignment-level fairness metrics computed over every student's
    StudentUtility.utility. See app/matching.py compute_assignment_utilities()
    for formulas; see the Part 2 report for the welfare-concept mapping
    (total / average / worst-off / equality of welfare).
    """
    mean_utility: Optional[float] = None
    min_utility: Optional[float] = None
    max_utility: Optional[float] = None
    utility_range: Optional[float] = None
    utility_variance: Optional[float] = None
    utility_stdev: Optional[float] = None
    total_utility: Optional[float] = None
    worst_case_regret: Optional[float] = None
    mean_regret: Optional[float] = None
    exact_regret_search: bool = False
    acceptable_utility_floor: float = 0.5
    students_below_floor_count: int = 0
    students_below_floor_fraction: Optional[float] = None
    note: Optional[str] = None


class AbilityConfig(BaseModel):
    """
    Instructor-facing configuration for the ability-balance framework
    (Part 3). `enabled=False` (default) means ability is evaluated and
    reported but NEVER changes which assignment /teams/generate returns —
    identical to how FairnessConfig.enabled behaves. Setting `enabled=True`
    turns on an additional, constraint-safe optimization pass
    (_ability_balance_improve in app/matching.py) that may change the
    returned assignment to narrow the gap between the strongest and weakest
    team's mean ability, without regressing the total-score objective or
    the accountability/cooperativeness floor-minimum earlier passes
    established.

    IMPORTANT: "balanced ability across every team" is treated here as a
    research HYPOTHESIS the audit's brief asked to be able to test, NOT an
    established finding — unlike the accountability/cooperativeness floor
    (which cites Bell 2007 / Peeters et al. 2006 directly), no citation in
    this repo establishes that mixed-ability teams outperform stacked ones
    for this population. See research_alignment_audit.md's Part 3 update
    for the explicit "requires outcome validation" note.
    """
    enabled: bool = False
    # Score thresholds bucketing the continuous [0.0, 1.0] ability scale
    # into 3 reporting tiers. Configurable rather than hardcoded because
    # there is no validated cutoff for this population (mirrors
    # EvalConfig.min_acceptable_level's own "chosen to be reasonable, not
    # derived from a validated source" caveat).
    ability_level_thresholds: Dict[str, float] = Field(default_factory=lambda: {
        "developing_max": 0.4,   # score < this -> "developing"
        "medium_max": 0.7,       # this <= score <= this -> "medium"; above -> "high"
    })


class AbilityTeamStats(BaseModel):
    team_index: int
    mean_ability: Optional[float] = None
    ability_range: Optional[float] = None
    ability_variance: Optional[float] = None
    ability_stdev: Optional[float] = None
    distribution: Dict[str, int] = Field(default_factory=dict)  # {"developing": n, "medium": n, "high": n}
    unknown_count: int = 0  # students with neither ability_score nor a resolvable experience_level


class AbilitySummary(BaseModel):
    """
    Assignment-level ability metrics. See app/matching.py
    compute_ability_metrics() for formulas, and the Part 3 report for the
    within-team vs. between-team distinction (task definitions):
      within-team ability diversity  -> per_team[i].{ability_range,ability_variance}
      between-team balance           -> between_team_variance / strongest_weakest_gap
    """
    per_team: List[AbilityTeamStats] = Field(default_factory=list)
    between_team_mean_variance: Optional[float] = None
    between_team_mean_stdev: Optional[float] = None
    strongest_weakest_gap: Optional[float] = None
    distribution: Dict[str, int] = Field(default_factory=dict)
    skill_coverage_ability_correlation: Optional[float] = None
    correlation_note: Optional[str] = None
    ability_level_thresholds: Dict[str, float] = Field(default_factory=dict)
    note: Optional[str] = None


# ---------------------------------------------------------------------------
# Familiarity (Part 4) — a RELATIONSHIP between two students, not a
# property of one. See app/familiarity.py for the API surface and
# app/matching.py's familiarity-balance framework for the graph algorithms
# and optional soft objective built on top of this data model.
# ---------------------------------------------------------------------------

class FamiliarityEdge(BaseModel):
    """
    Prior familiarity between exactly two students, identified by their
    stable Student.id values.

    DIRECTIONALITY: this repo's brief does not specify whether familiarity
    should be directed (A recognizes B without B necessarily recognizing A)
    or symmetric (mutual). Per this part's own instruction to default to
    symmetric when directionality isn't specified, familiarity here is
    SYMMETRIC: an edge between A and B means each is equally familiar with
    the other. Storage canonicalizes (student_a_id, student_b_id) so
    order does not matter — see app.familiarity.canonical_pair(). Directed
    familiarity is NOT modeled; see the Part 4 report's limitations.

    SCALE (documented, NOT independently validated — no published
    instrument for peer familiarity is cited anywhere in this repo, unlike
    e.g. the accountability/cooperativeness floor's Bell 2007 citation):
      0.0   confirmed NO prior contact (explicitly asked and confirmed —
            NOT the same as never having asked at all; see below)
      ~0.3  passing acquaintance (same class/section, rarely interacted)
      ~0.6  worked together before (prior group project, study group)
      1.0   close prior collaborator / friend
    A MISSING edge (no FamiliarityDB row for this pair) is UNKNOWN — no one
    has recorded anything about this pair. That is DISTINCT from an edge
    explicitly recorded at strength=0.0. Every metric and the optional soft
    objective in app/matching.py preserves this distinction rather than
    treating a missing edge as zero.
    """
    student_a_id: str
    student_b_id: str
    strength: float = Field(ge=0.0, le=1.0)
    # e.g. "roster_survey", "lms_co_enrollment", "self_report". Free-form,
    # not enforced — mirrors CompetenceSignature.ability_source's rationale
    # for not hardcoding a single collection method.
    source: Optional[str] = None

    @model_validator(mode="after")
    def _reject_self_relationship(self) -> "FamiliarityEdge":
        if self.student_a_id == self.student_b_id:
            raise ValueError("a student cannot have a familiarity relationship with themself")
        return self


class FamiliarityConfig(BaseModel):
    """
    Instructor-facing configuration for the familiarity / anti-clique
    framework. `enabled=False` (default) means familiarity is evaluated and
    reported but NEVER changes which assignment /teams/generate returns —
    same pattern as FairnessConfig/AbilityConfig. Setting `enabled=True`
    turns on an additional, constraint-safe optimization pass
    (_familiarity_balance_improve in app/matching.py) that may change the
    returned assignment to pull each team's familiarity density toward the
    [lower_bound, upper_bound] band below, without regressing the
    total-score objective or the accountability/cooperativeness
    floor-minimum earlier passes established.

    NONE of the thresholds below are scientifically established for this
    population — they are documented hypotheses to test empirically, in
    the same spirit as AbilityConfig's "balanced ability" hypothesis and
    EvalConfig's pre-existing clique_fraction_threshold/
    min_familiarity_density defaults (which this supersedes with a real
    graph-based measure — see evaluation/metrics.py:familiarity_balance()).
    """
    enabled: bool = False
    # A pair counts as "familiar" (density/isolation metrics) when a KNOWN
    # edge exists with strength >= this. Unknown pairs never count as
    # familiar, regardless of this threshold.
    familiar_pair_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    # Edges at/above this strength are used when building a TEAM-INDUCED
    # subgraph (edges between teammates only — see
    # app.matching.compute_familiarity_metrics) to find its
    # connected_familiarity_cluster(s). Usually >= familiar_pair_threshold
    # (a stronger bar for "part of the same prior group" than merely "these
    # two know each other").
    clique_edge_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    # Target band for per-team familiarity density:
    #   density < lower_bound             -> team may lack coordination support
    #   lower_bound <= density <= upper_bound -> acceptable familiarity
    #   density > upper_bound             -> possible clique concentration
    # Only applied when the team's known_pair_coverage meets
    # minimum_known_pair_coverage below — otherwise band_status is
    # "insufficient_data" instead of below/within/above.
    lower_bound: float = Field(default=0.15, ge=0.0, le=1.0)
    upper_bound: float = Field(default=0.6, ge=0.0, le=1.0)
    # Direct anti-clique term for the optional soft objective, separate
    # from the density band above: above this FRACTION of a team belonging
    # to the team's largest connected_familiarity_cluster, the pass treats
    # it as excessive concentration and looks for a swap that shrinks it —
    # this responds correctly even when a cluster's KNOWN internal edges
    # are uniformly strong (in which case density alone is insensitive to
    # losing one member; see the Part 4 report's limitations section for
    # why density alone isn't enough to drive de-clustering).
    max_clique_fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    # Part 4.5 — coverage / missing-data handling. A team's
    # known_pair_coverage (known teammate pairs / all possible teammate
    # pairs) below this makes band_status "insufficient_data" instead of a
    # below/within/above classification — there simply isn't enough
    # recorded familiarity data among these specific teammates to trust a
    # density-based judgment. Teams with zero possible pairs (size 0 or 1)
    # get "not_applicable" instead, regardless of this threshold.
    minimum_known_pair_coverage: float = Field(default=0.5, ge=0.0, le=1.0)
    # Missing-data policy switch: by default (False), a team classified
    # "insufficient_data" contributes ZERO to the optional soft objective's
    # density-band violation term — the optimizer never "confidently"
    # chases a band target it doesn't have enough evidence to judge. Set
    # True only with a documented rationale for why extrapolating from
    # sparse data is acceptable for a given deployment; this repo's default
    # is the conservative (off) choice.
    penalize_insufficient_data: bool = False
    # Optional: a connected_familiarity_cluster is additionally flagged
    # "is_quasi_clique" when its thresholded_internal_edge_density (fraction
    # of ALL possible internal pairs with a KNOWN edge >= clique_edge_
    # threshold) is at or above this. Purely a reporting label — never
    # itself a hard requirement or an optimization target. A cluster with
    # fewer than 2 members never gets a quasi-clique classification (there
    # is no internal structure to assess).
    quasi_clique_density_threshold: float = Field(default=0.75, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _bounds_are_ordered(self) -> "FamiliarityConfig":
        if self.lower_bound > self.upper_bound:
            raise ValueError(
                f"familiarity.lower_bound ({self.lower_bound}) must be <= "
                f"familiarity.upper_bound ({self.upper_bound})"
            )
        return self


class FamiliarityClusterStats(BaseModel):
    """
    A CONNECTED COMPONENT of the TEAM-INDUCED familiarity subgraph — nodes
    are this ONE team's members, edges are familiarity edges BETWEEN THOSE
    MEMBERS ONLY (edges to students outside the team are excluded) that
    meet clique_edge_threshold. Connectivity only requires a PATH between
    any two members through other members of the SAME cluster — it does
    NOT require every pair within the cluster to be directly connected, so
    this is deliberately NOT called a "clique" (see is_quasi_clique for the
    one place this model does make a stricter, still-approximate claim).

    Renamed from the Part 4 "largest_preexisting_subgroup_size" (a bare
    int) — that field was computed from GLOBAL connected components
    intersected with a team after the fact, which could incorrectly count
    two teammates as clustered when they were connected only through a
    student on a DIFFERENT team (research_alignment_audit.md Part 4.5
    update). This model is computed directly from the team-induced
    subgraph, which cannot make that mistake — a path through an outside
    student is structurally absent from this graph.
    """
    size: int = 0
    fraction_of_team: float = 0.0
    known_internal_pairs: int = 0
    unknown_internal_pairs: int = 0
    total_internal_pairs: int = 0
    # known_internal_pairs / total_internal_pairs; None if total_internal_pairs == 0
    known_internal_edge_coverage: Optional[float] = None
    # mean strength over KNOWN internal pairs only; None if zero known internal pairs
    mean_known_internal_strength: Optional[float] = None
    # (internal pairs with a KNOWN edge >= clique_edge_threshold) / total_internal_pairs;
    # None if total_internal_pairs == 0. This is what "how complete/tight is
    # this cluster's own subgraph" means here — NOT the same quantity as
    # known_internal_edge_coverage (which only measures how much DATA exists,
    # regardless of strength).
    thresholded_internal_edge_density: Optional[float] = None
    # None unless the cluster has >=2 members AND total_internal_pairs > 0;
    # otherwise True/False against quasi_clique_density_threshold.
    is_quasi_clique: Optional[bool] = None


class FamiliarityTeamStats(BaseModel):
    team_index: int
    total_possible_pairs: int = 0
    known_pair_count: int = 0
    unknown_pair_count: int = 0
    confirmed_zero_pair_count: int = 0
    # known_pair_count / total_possible_pairs; None if total_possible_pairs == 0
    known_pair_coverage: Optional[float] = None
    familiar_pair_count: int = 0
    familiar_pair_fraction: Optional[float] = None  # of KNOWN pairs only
    density: Optional[float] = None  # mean strength over KNOWN pairs; None if zero known pairs
    isolated_members: List[str] = Field(default_factory=list)
    largest_connected_cluster: FamiliarityClusterStats = Field(default_factory=FamiliarityClusterStats)
    # "below" | "within" | "above" | "insufficient_data" (known_pair_coverage
    # below minimum_known_pair_coverage) | "not_applicable" (0 or 1 members,
    # i.e. total_possible_pairs == 0)
    band_status: Optional[str] = None


class FamiliaritySummary(BaseModel):
    """
    Assignment-level familiarity metrics. See app/matching.py
    compute_familiarity_metrics() for formulas and the Part 4/4.5 report
    for the graph definitions (nodes/edges/connected components) behind
    them.
    """
    per_team: List[FamiliarityTeamStats] = Field(default_factory=list)
    mean_density: Optional[float] = None
    cross_team_density_variance: Optional[float] = None
    cross_team_density_stdev: Optional[float] = None
    total_known_pairs: int = 0
    total_possible_pairs: int = 0
    total_confirmed_zero_pairs: int = 0
    total_familiar_pairs: int = 0
    teams_below_band: int = 0
    teams_within_band: int = 0
    teams_above_band: int = 0
    teams_insufficient_data: int = 0
    teams_not_applicable: int = 0
    lower_bound: float = 0.15
    upper_bound: float = 0.6
    minimum_known_pair_coverage: float = 0.5
    note: Optional[str] = None


class PreferenceEntry(BaseModel):
    """
    ONE preference statement FROM `student_id`, DIRECTED (asymmetric — an
    entry from A about B says nothing about B's own preferences unless B
    separately submits one) — unlike FamiliarityEdge, which is symmetric
    by design. Exactly one of `target_student_id` / `target_characteristic`
    must be set (see the validator below):

      target_student_id        a named-teammate preference/avoidance.
      target_characteristic    a preference over a TEAM TRAIT rather than a
                                named student, e.g. {"skill": "backend"} —
                                "avoid" direction is NOT supported for
                                characteristics (see `direction`'s note).
                                Only the "skill" key is resolved by
                                app.matching's scoring (checked against
                                teammates' CompetenceSignature.skills); any
                                other key is accepted and stored but
                                excluded from scoring until support is
                                added — never silently scored as unmet.

    direction:
      "prefer"   student_id would like to be teamed with the target (or a
                 team exhibiting the target characteristic).
      "avoid"    student_id would like to NOT be teamed with the target.
                 Only meaningful for target_student_id — see
                 `_reject_avoid_characteristic` below. Honored UNILATERALLY:
                 an avoid entry from A about B protects A regardless of
                 whether B reciprocates or even knows it exists (see
                 app.matching's avoidance-violation check, which looks at
                 BOTH directions of every teammate pair). This is a
                 deliberate ethical/institutional design choice, not an
                 oversight — see this class's module-level privacy notes
                 in app/preferences.py.

    rank: 1 = most preferred/most avoided AMONG this student's OWN entries
      of the SAME direction. Multiple entries may share the same rank —
      that is how TIES / INDIFFERENCE are expressed (e.g. two entries both
      at rank=1 means "equally my top choice"). `rank=None` means "listed,
      but not placed in a priority order" — still a real, honored
      preference, just an unranked one; see app.matching's tier-scoring
      docstring for exactly how unranked entries are scored relative to
      ranked ones. A student with NO PreferenceEntry rows at all has
      MISSING preference data — every metric in app/matching.py treats
      that as "no opinion" (excluded from means/rates), never as
      dissatisfaction or a score of zero.

    priority: "high" marks a preference/avoidance as significantly more
      important than an ordinary one (e.g. a safety- or conflict-driven
      avoidance request, or a genuinely load-bearing teammate need) —
      weighted more heavily by the optional repair pass
      (PreferenceConfig.high_priority_weight /
      avoidance_violation_weight) and surfaced separately in
      PreferenceSummary (unmet/violated HIGH-PRIORITY counts) rather than
      being buried in an aggregate mean. Purely a self-reported signal,
      not independently verified — see this system's privacy notes for why
      `reason_category` is never surfaced back through any student-facing
      API or explanation.
    """
    student_id: str
    direction: Literal["prefer", "avoid"]
    target_student_id: Optional[str] = None
    target_characteristic: Optional[Dict[str, str]] = None
    rank: Optional[int] = Field(default=None, ge=1)
    priority: Literal["normal", "high"] = "normal"
    # Free-form, e.g. "prior_conflict" | "safety" | "worked_well_before".
    # INTERNAL ONLY — never echoed back through app/preferences.py's API
    # responses to any student other than the one who submitted it, and
    # never passed into app/explainability.py's team-explanation prompts.
    reason_category: Optional[str] = None
    source: Optional[str] = None  # e.g. "onboarding_survey", "self_report"

    @model_validator(mode="after")
    def _exactly_one_target(self) -> "PreferenceEntry":
        has_student = self.target_student_id is not None
        has_characteristic = bool(self.target_characteristic)
        if has_student == has_characteristic:  # both set, or neither
            raise ValueError(
                "PreferenceEntry requires exactly one of target_student_id "
                "or target_characteristic"
            )
        if has_student and self.target_student_id == self.student_id:
            raise ValueError("a student cannot name themself as a preference target")
        if has_characteristic and self.direction == "avoid":
            raise ValueError(
                "avoid direction is only supported for target_student_id, not "
                "target_characteristic (there is no well-defined way to 'avoid' "
                "a team trait rather than decline a specific person)"
            )
        return self


class PreferenceConfig(BaseModel):
    """
    Instructor-facing configuration for the preference / stability
    framework (Part 5). Same "evaluated always, optimized only if enabled"
    pattern as FairnessConfig/AbilityConfig/FamiliarityConfig: preference
    satisfaction and stability metrics are ALWAYS computed and returned
    (see PreferenceSummary on /teams/generate), and `enabled=False`
    (default) means they NEVER change which assignment is returned. Setting
    `enabled=True` turns on `_preference_repair_improve` in app/matching.py,
    a constraint-safe optimization pass that may change the returned
    assignment to reduce unmet high-priority preferences / avoidance
    violations / general preference dissatisfaction, without regressing the
    total-score objective or the accountability/cooperativeness
    floor-minimum earlier passes established (same lexicographic
    acceptance rule as _familiarity_balance_improve).
    """
    enabled: bool = False
    # When True AND enabled=True, the repair pass additionally refuses any
    # swap of its OWN that would INCREASE the number of avoidance-violation
    # pairs on either team it touches (it may still reduce pre-existing
    # ones as part of normal violation minimization). This is the
    # "explicitly configured institutional restriction" hook the repair
    # phase's hard-constraint guarantee extends to cover avoidance, on top
    # of the always-enforced team-size/skill-coverage/behavioral-floor
    # constraints from validate_assignment(). Default False: avoidance
    # remains purely a soft, reported signal unless an institution opts in.
    hard_avoidance: bool = False
    # Multiplier applied to a HIGH-priority unmet "prefer" entry in the
    # repair pass's violation term, relative to an ordinary (weight 1.0)
    # unmet preference. NOT scientifically validated for this population —
    # a documented hypothesis, same caveat as every other Part 3/4 threshold.
    high_priority_weight: float = Field(default=3.0, ge=1.0)
    # Multiplier applied to each avoidance-violation PAIR in the repair
    # pass's violation term (independent of hard_avoidance, which is a
    # binary must-not-worsen gate rather than a weight — this weight shapes
    # how hard the pass looks for a fix even when hard_avoidance=False).
    avoidance_violation_weight: float = Field(default=5.0, ge=0.0)
    # Which formal stability definition compute_stability_metrics()'s
    # `is_stable` / headline count is reported against. blocking_pair /
    # blocking_swap / beneficial_feasible_move are ALWAYS all three
    # computed and available from app.matching.compute_stability_metrics()
    # regardless of this setting — this only selects which one
    # PreferenceSummary treats as authoritative. See app/matching.py's
    # stability-definitions docstring for the full formal treatment of why
    # "blocking_swap" (the default) is the one this system's swap-based
    # repair pass can actually act on, and why "beneficial_feasible_move"
    # (Part 5.5 — renamed from "blocking_move") is deliberately NOT called
    # a classical blocking pair/coalition: it requires only the MOVING
    # student's utility to improve, never the receiving side's, so it is a
    # project-specific one-sided deviation measure, not a stable-matching
    # guarantee.
    stability_definition: Literal["blocking_pair", "blocking_swap", "beneficial_feasible_move"] = "blocking_swap"

    @model_validator(mode="after")
    def _weights_are_nonnegative(self) -> "PreferenceConfig":
        if self.high_priority_weight < 1.0:
            raise ValueError("preference.high_priority_weight must be >= 1.0 (it is a MULTIPLIER on the base weight of 1.0)")
        return self


class PreferenceStudentStats(BaseModel):
    """
    PRIVACY NOTE: deliberately contains no target student IDs, no ranked
    lists, and no `reason_category` text — only aggregate counts/booleans
    about `student_id`'s OWN outcome. This is what a team-generation
    response, or any explanation built from it, is allowed to expose about
    a student's preferences to ANYONE other than that student themself; see
    app/preferences.py's module docstring for the full privacy design and
    why the richer, name-bearing internal report
    (app.matching.compute_preference_metrics()'s raw dict return) is never
    serialized onto this model or any other public response.
    """
    student_id: str
    has_preferences: bool = False  # any "prefer" entries submitted at all
    n_prefer_entries: int = 0
    n_avoid_entries: int = 0
    # None === MISSING (no "prefer" entries submitted) -- never 0.0.
    satisfaction_score: Optional[float] = None
    got_a_preferred_teammate: Optional[bool] = None
    n_unmet_high_priority_preferences: int = 0
    avoidance_violated: bool = False
    n_avoidance_violations: int = 0
    violated_high_priority_avoidance: bool = False


class PreferenceSummary(BaseModel):
    """
    Assignment-level preference-satisfaction and stability metrics. See
    app/matching.py compute_preference_metrics() / compute_stability_metrics()
    for the exact formulas, tier-scoring rules, and the formal blocking-pair
    / blocking-swap / beneficial-feasible-move / stable-assignment
    definitions behind these numbers.

    TERMINOLOGY (Part 5.5): `n_beneficial_feasible_moves` was named
    `n_blocking_moves` through Part 5. Renamed because that diagnostic
    requires only the MOVING student's utility to strictly improve — it
    never requires the receiving team or any of its members to accept,
    consent to, or benefit from the move — so calling it a "blocking move"
    alongside `blocking_pair`/`blocking_swap` (both of which DO require
    mutual improvement) overclaimed a classical stable-matching guarantee
    this diagnostic does not provide. No deprecated alias is kept (pre-1.0
    prototype, no external consumers, same precedent as Part 4.5's
    `largest_preexisting_subgroup_size` → `largest_connected_cluster`
    rename) — see app/matching.py's stability-definitions docstring for
    the full formal treatment.
    """
    per_student: List[PreferenceStudentStats] = Field(default_factory=list)
    n_students_with_preferences: int = 0
    mean_satisfaction: Optional[float] = None
    min_satisfaction: Optional[float] = None
    max_satisfaction: Optional[float] = None
    satisfaction_stdev: Optional[float] = None
    # fraction (of students WITH >=1 "prefer" entry) who got >=1 preferred
    # teammate; None if no student submitted any "prefer" entries at all.
    fraction_with_at_least_one_preferred_teammate: Optional[float] = None
    # 1 - min_satisfaction, reported separately by name since "worst-case
    # regret" is the framing this part's spec asks for explicitly.
    worst_case_preference_regret: Optional[float] = None
    total_unmet_high_priority_preferences: int = 0
    total_avoidance_violations: int = 0
    total_high_priority_avoidance_violations: int = 0
    stability_definition: str = "blocking_swap"
    is_stable: Optional[bool] = None
    n_blocking_pairs: int = 0
    n_blocking_swaps: int = 0
    # one-sided diagnostic — see this class's TERMINOLOGY note above.
    n_beneficial_feasible_moves: int = 0
    note: Optional[str] = None


class TeamSizeConfig(BaseModel):
    """
    Part 5.5 — explicit, OPT-IN configuration for whether teams may deviate
    from their exact target size. Every existing optimization pass in this
    codebase (2-opt, maximin, ability/fairness/familiarity/preference-
    repair) uses EXACT team sizes via `AssignmentBaseline.target_sizes` and
    NEVER reads this config — it exists solely for the stability-diagnostic
    layer (app.matching.compute_unilateral_vacancy_moves) to reason about a
    genuinely UNILATERAL move (one student relocating with no displaced
    counterpart — the sending team shrinks by one, the receiving team
    grows by one) WITHOUT silently reinterpreting every other pass's
    exact-size guarantee as flexible.

    Default (`min_team_size=None, max_team_size=None,
    allow_unilateral_moves=False`) means team sizes remain exactly what
    they were before this config existed — a genuine unilateral move is
    then ALWAYS infeasible (there is no vacancy for it to fill), by
    design, not by omission. Setting `allow_unilateral_moves=True` without
    also setting `min_team_size`/`max_team_size` is deliberately still a
    no-op (see compute_unilateral_vacancy_moves) — a vacancy must be
    EXPLICITLY sized, never implied.
    """
    min_team_size: Optional[int] = Field(default=None, ge=1)
    max_team_size: Optional[int] = Field(default=None, ge=1)
    allow_unilateral_moves: bool = False

    @model_validator(mode="after")
    def _range_is_ordered(self) -> "TeamSizeConfig":
        if self.min_team_size is not None and self.max_team_size is not None \
                and self.min_team_size > self.max_team_size:
            raise ValueError(
                f"min_team_size ({self.min_team_size}) must be <= max_team_size ({self.max_team_size})"
            )
        return self


class TeamMember(BaseModel):
    student_id: str
    name: str
    roles: List[str] = []
    experience_level: str = "intermediate"
    utility: Optional[StudentUtility] = None
    # Part 3 — same derivation as AbilitySummary: ability_score if the
    # student has one recorded, else a fallback derived from
    # experience_level (see app/matching.py:_student_ability_score).
    # ability_source_used names WHICH of those two actually produced the
    # value below, so a fallback is never silently indistinguishable from a
    # directly-recorded score.
    ability_score: Optional[float] = None
    ability_level: Optional[str] = None
    ability_source_used: Optional[str] = None


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
    # Accountability/cooperativeness floor policy — the rubric's stated
    # "no team may include a member at the low level" (zero_tolerance) and
    # the matcher's historical "one floor-level student per team" behavior
    # are NOT equivalent (research_alignment_audit.md Critical #1). Neither
    # is authoritatively established as "the" intended policy, so it is a
    # request-level, instructor-facing choice. Default preserves the
    # behavior this system has always shipped (feasibility + greedy spread
    # tolerate one floor-level student per team).
    floor_policy: str = "one_low_per_team"  # "one_low_per_team" | "zero_tolerance"
    # Individual-utility / fairness framework (Part 2). Utility is always
    # COMPUTED and returned for visibility (see StudentUtility/FairnessSummary
    # on the /teams/generate response) — `fairness.enabled` controls only
    # whether it is also used to CHOOSE the assignment. Default (disabled)
    # preserves matching results exactly as they were before this field
    # existed.
    fairness: FairnessConfig = Field(default_factory=FairnessConfig)
    # Ability-balance framework (Part 3). Same pattern as `fairness` above:
    # ability metrics are always COMPUTED and returned (see AbilitySummary
    # on the /teams/generate response) — `ability.enabled` controls only
    # whether it is also used to CHOOSE the assignment. Default (disabled)
    # preserves matching results exactly as they were before this field
    # existed.
    ability: AbilityConfig = Field(default_factory=AbilityConfig)
    # Familiarity / anti-clique framework (Part 4). Same pattern as
    # `fairness`/`ability` above: familiarity metrics are always COMPUTED
    # and returned (see FamiliaritySummary on the /teams/generate response)
    # from whatever FamiliarityDB edges exist — `familiarity.enabled`
    # controls only whether they are also used to CHOOSE the assignment.
    # Default (disabled) preserves matching results exactly as they were
    # before this field existed, and a roster with zero recorded
    # familiarity edges behaves identically to before Part 4 either way.
    familiarity: FamiliarityConfig = Field(default_factory=FamiliarityConfig)
    # Preference / stability framework (Part 5). Same pattern as
    # `fairness`/`ability`/`familiarity` above: preference-satisfaction and
    # stability metrics are always COMPUTED and returned (see
    # PreferenceSummary on the /teams/generate response) from whatever
    # PreferenceDB entries exist — `preference.enabled` controls only
    # whether they are also used to CHOOSE the assignment (via
    # _preference_repair_improve). Default (disabled) preserves matching
    # results exactly as they were before this field existed, and a roster
    # with zero recorded preference entries behaves identically either way.
    preference: PreferenceConfig = Field(default_factory=PreferenceConfig)
    # Part 6 — named optimization profile. Resolution precedence is
    # ALWAYS: system defaults -> named profile's overrides -> explicit
    # request-level field overrides (highest). "legacy" (the default) is
    # defined to reproduce every prior part's shipped default behavior
    # (every *.enabled flag off) — an existing client that never sends
    # `profile` at all gets EXACTLY what it always got. See
    # app.matching.OPTIMIZATION_PROFILES / resolve_profile() for the
    # registry and merge algorithm, and docs/optimization_profiles.md for
    # the full, human-readable description of every named profile
    # (purpose, effective settings, experimental status, known
    # limitations). Unknown profile names are rejected with a 422, never
    # silently substituted.
    profile: str = "legacy"


class ProfileResolution(BaseModel):
    """
    Part 6 — reports EXACTLY which optimization profile produced a given
    /teams/generate response, and its fully-resolved effective settings
    (after the system-defaults -> named-profile -> explicit-overrides
    merge — see app.matching.resolve_profile()). Returned on every
    response so a caller can always tell, after the fact, precisely what
    configuration was actually used — never left to infer it from the
    profile name alone, since explicit request overrides can change any
    individual setting.
    """
    name: str
    is_experimental: bool
    description: str
    # Full resolved TeamGenerationRequest, as a plain dict (== the JSON
    # request body that would reproduce this exact run without relying on
    # `profile` + defaults again) — NOT re-typed as TeamGenerationRequest
    # here to avoid a schema/model coupling loop; the shape is identical.
    effective_settings: dict = Field(default_factory=dict)


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
    values: List[float]
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
    