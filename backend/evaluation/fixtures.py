"""
Reusable synthetic datasets for research-alignment testing (Phase 3 of
research_alignment_audit.md).

Each `fixture_*` function returns a `FixtureCase`: the student roster, the
team_size to request, and any auxiliary research data (familiarity graph,
friend-group labels, teammate preferences) that the current `Student` model
cannot carry but the evaluation harness (`evaluation/metrics.py`) can
consume when supplied out-of-band. Every case documents, in its docstring,
which research property it is designed to expose and why.

These datasets are intentionally small (8-16 students) so that:
  (a) they stay hand-auditable — every value below was chosen deliberately,
      not generated; and
  (b) `evaluation.metrics.individual_utility`'s brute-force regret search
      stays tractable (see EvalConfig.max_exact_regret_roster_size).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.models import (
    CollaborationSignature,
    CompetenceSignature,
    ConfidenceLayer,
    MotivationLayer,
    Student,
    WorkRhythmSignature,
)


@dataclass
class FixtureCase:
    name: str
    description: str
    students: List[Student]
    team_size: int
    familiarity: Optional[Dict[Tuple[str, str], float]] = None
    friend_groups: Optional[Dict[str, str]] = None
    preferences: Optional[Dict[str, List[str]]] = None


def _s(
    id: str,
    name: str,
    skills: List[str],
    experience_level: str = "intermediate",
    accountability: str = "developing",
    cooperativeness: str = "developing",
    conflict_style: str = "collaborative",
    leadership_style: str = "facilitative",
    planning_style: str = "adaptive",
    communication_style: str = "mixed",
    availability: Optional[List[str]] = None,
    interests: Optional[List[str]] = None,
    learning_goals: Optional[List[str]] = None,
    under_determined_dims: Optional[List[str]] = None,
    ability_score: Optional[float] = None,
    ability_source: Optional[str] = None,
    skill_proficiency: Optional[Dict[str, float]] = None,
) -> Student:
    return Student(
        id=id,
        name=name,
        competence_signature=CompetenceSignature(
            skills=skills, roles=[], experience_level=experience_level,
            ability_score=ability_score, ability_source=ability_source,
            skill_proficiency=skill_proficiency or {},
        ),
        work_rhythm_signature=WorkRhythmSignature(
            planning_style=planning_style,
            communication_style=communication_style,
            execution_style="iterative",
            availability=availability or ["Mon-morning", "Wed-morning"],
        ),
        collaboration_signature=CollaborationSignature(
            conflict_style=conflict_style,
            leadership_style=leadership_style,
            accountability=accountability,
            help_seeking="proactive",
            cooperativeness=cooperativeness,
        ),
        motivation_layer=MotivationLayer(
            interests=interests or ["software"], learning_goals=learning_goals or ["growth"],
        ),
        confidence_layer=ConfidenceLayer(confidence_score=0.7),
        under_determined_dims=under_determined_dims or [],
    )


# ---------------------------------------------------------------------------
# 1. Full skill coverage vs. high behavioral compatibility
# ---------------------------------------------------------------------------
def fixture_skill_vs_behavior() -> FixtureCase:
    """
    4 'specialists' hold one unique reference skill each but clash
    behaviorally (avoidant/low-accountability vs directive/high). 4
    'generalists' overlap heavily on frontend/testing but are behaviorally
    uniform and highly compatible. Exposes whether the objective quietly
    trades skill coverage for behavioral compatibility (or vice versa) when
    both cannot be maximized for the same two teams at once.
    """
    specialists = [
        _s("sk1", "Specialist ML", ["ml"], accountability="low", cooperativeness="low",
           conflict_style="avoidant", leadership_style="emergent"),
        _s("sk2", "Specialist Devops", ["devops"], accountability="high", cooperativeness="developing",
           conflict_style="assertive", leadership_style="directive"),
        _s("sk3", "Specialist PM", ["project management"], accountability="developing", cooperativeness="low",
           conflict_style="assertive", leadership_style="directive"),
        _s("sk4", "Specialist Research", ["research"], accountability="low", cooperativeness="developing",
           conflict_style="avoidant", leadership_style="emergent"),
    ]
    generalists = [
        _s("gn1", "Generalist A", ["frontend", "testing"], accountability="high", cooperativeness="high",
           conflict_style="collaborative", leadership_style="facilitative"),
        _s("gn2", "Generalist B", ["frontend", "testing"], accountability="high", cooperativeness="high",
           conflict_style="collaborative", leadership_style="facilitative"),
        _s("gn3", "Generalist C", ["frontend", "testing"], accountability="high", cooperativeness="high",
           conflict_style="collaborative", leadership_style="facilitative"),
        _s("gn4", "Generalist D", ["frontend", "testing"], accountability="high", cooperativeness="high",
           conflict_style="collaborative", leadership_style="facilitative"),
    ]
    return FixtureCase(
        name="skill_vs_behavior",
        description="Full skill coverage requires mixing clashing specialists; behavioral compat "
                     "is maximized by 4 uniform generalists who cover almost no distinct skills.",
        students=specialists + generalists,
        team_size=4,
    )


# ---------------------------------------------------------------------------
# 2. High team averages hiding one very low-conscientiousness member
# ---------------------------------------------------------------------------
def fixture_hidden_bad_apple() -> FixtureCase:
    """
    Team-shaped input: 3 members at 'high' accountability + 1 at 'low'.
    Team MEAN accountability is 0.83 (looks strong) but the MINIMUM is the
    floor level. Tests whether mean-only reporting would hide the bad apple
    that Bell (2007) identifies as team-performance-relevant.
    """
    return FixtureCase(
        name="hidden_bad_apple",
        description="Mean accountability is high; one member sits at the floor level.",
        students=[
            _s("ba1", "High A", ["frontend"], accountability="high", cooperativeness="high"),
            _s("ba2", "High B", ["backend"], accountability="high", cooperativeness="high"),
            _s("ba3", "High C", ["testing"], accountability="high", cooperativeness="high"),
            _s("ba4", "Low D", ["design"], accountability="low", cooperativeness="developing"),
        ],
        team_size=4,
    )


# ---------------------------------------------------------------------------
# 3. Balanced teams vs. one stacked team
# ---------------------------------------------------------------------------
def fixture_stacked_team() -> FixtureCase:
    """
    12 students, team_size 4 → 3 teams. All the 'advanced' + 'high'
    accountability + 'high' cooperativeness students are deliberately
    front-loaded in the roster order to check whether the greedy builder's
    seeding concentrates strength in one team before 2-opt/maximin can
    rebalance it.
    """
    strong = [
        _s(f"st{i}", f"Strong {i}", ["backend", "ml", "algorithms"][i % 3:][:1] or ["backend"],
           experience_level="advanced", accountability="high", cooperativeness="high")
        for i in range(4)
    ]
    mid = [
        _s(f"md{i}", f"Mid {i}", ["frontend", "design", "testing"][i % 3:][:1] or ["frontend"],
           experience_level="intermediate", accountability="developing", cooperativeness="developing")
        for i in range(4)
    ]
    weak = [
        _s(f"wk{i}", f"Weak {i}", ["documentation", "communication", "research"][i % 3:][:1] or ["research"],
           experience_level="beginner", accountability="developing", cooperativeness="developing")
        for i in range(4)
    ]
    return FixtureCase(
        name="stacked_team",
        description="Strong/mid/weak ability students in roster-order blocks; checks whether "
                     "optimization spreads ability evenly across 3 teams of 4.",
        students=strong + mid + weak,
        team_size=4,
    )


# ---------------------------------------------------------------------------
# 4. Familiar classmates vs. an isolated student
# ---------------------------------------------------------------------------
def fixture_isolated_student() -> FixtureCase:
    """
    8 students who all know each other from a prior course, plus 1 transfer
    student nobody has worked with. Since the schema has no familiarity
    field, this is passed as auxiliary `familiarity` data to the harness —
    production team formation cannot see or use it at all today.
    """
    known = [_s(f"fam{i}", f"Familiar {i}", ["frontend", "backend"][i % 2:][:1] or ["frontend"])
             for i in range(8)]
    isolated = _s("iso1", "Isolated Student", ["design"])
    students = known + [isolated]
    familiarity: Dict[Tuple[str, str], float] = {}
    for a in known:
        for b in known:
            if a.id != b.id:
                familiarity[(a.id, b.id)] = 0.8
    # isolated student has zero familiarity with everyone (no entries needed —
    # lookup defaults to 0.0)
    return FixtureCase(
        name="isolated_student",
        description="8 mutually-familiar classmates + 1 student with zero prior contact with anyone.",
        students=students,
        team_size=3,
        familiarity=familiarity,
    )


# ---------------------------------------------------------------------------
# 5. A friend-group clique vs. socially mixed teams
# ---------------------------------------------------------------------------
def fixture_friend_clique() -> FixtureCase:
    """
    12 students: 4 form a tight pre-existing friend group ('clique_x'), the
    other 8 are otherwise unconnected. Auxiliary `friend_groups` +
    `familiarity` data let the harness check whether a naive
    familiarity-maximizing rule would just reassemble the clique verbatim
    (high density, but socially homogeneous) rather than mixing it.
    """
    clique = [_s(f"cq{i}", f"Clique {i}", ["frontend"]) for i in range(4)]
    others = [_s(f"ot{i}", f"Other {i}", ["backend", "ml", "design", "testing",
                                            "devops", "research", "documentation", "algorithms"][i:i+1])
              for i in range(8)]
    students = clique + others
    friend_groups = {s.id: "clique_x" for s in clique}
    friend_groups.update({s.id: f"solo_{s.id}" for s in others})
    familiarity = {
        (a.id, b.id): 0.9 for a in clique for b in clique if a.id != b.id
    }
    return FixtureCase(
        name="friend_clique",
        description="A 4-person pre-existing friend clique embedded in a 12-student roster.",
        students=students,
        team_size=4,
        familiarity=familiarity,
        friend_groups=friend_groups,
    )


# ---------------------------------------------------------------------------
# 6. Maximum total utility vs. unfair outcomes for one member
# ---------------------------------------------------------------------------
def fixture_utility_vs_fairness() -> FixtureCase:
    """
    8 students where one grouping maximizes total/average score by
    isolating a single weak-skill, low-accountability student on an
    otherwise-strong team (dragging only THAT team's average down a little)
    versus spreading the weak student's cost more evenly. Used to compute
    min-utility / worst-case regret via evaluation.metrics.individual_utility.
    """
    strong = [
        _s(f"ut{i}", f"Utility Strong {i}", ["backend", "frontend", "ml"][i % 3:][:1] or ["backend"],
           experience_level="advanced", accountability="high", cooperativeness="high")
        for i in range(6)
    ]
    weak = [
        _s("ut_weak1", "Weak Skill Low Accountability", ["documentation"],
           experience_level="beginner", accountability="low", cooperativeness="low"),
        _s("ut_weak2", "Weak Skill Two", ["research"],
           experience_level="beginner", accountability="developing", cooperativeness="developing"),
    ]
    return FixtureCase(
        name="utility_vs_fairness",
        description="6 strong students + 2 weak students; exposes whether maximizing total score "
                     "concentrates disadvantage onto whichever team gets a weak student.",
        students=strong + weak,
        team_size=4,
    )


# ---------------------------------------------------------------------------
# 7. Preference-respecting vs. unstable assignments
# ---------------------------------------------------------------------------
def fixture_preference_instability() -> FixtureCase:
    """
    8 students with explicit, deliberately-crossed teammate preferences
    (student A prefers student C, who is placed on a different team than A
    while both would be better off together). Preferences are passed as
    auxiliary data — the schema has no field for them at all today.
    """
    students = [_s(f"pr{i}", f"Pref {i}", ["frontend", "backend", "design", "testing"][i % 4:][:1] or ["frontend"])
                for i in range(8)]
    # Construct a preference ranking that a simple greedy skill/behavior
    # optimizer (blind to preference) is likely to violate: everyone's
    # #1 preference is their "work-pair", but skill-coverage greediness
    # will typically split those pairs across teams.
    pairs = [("pr0", "pr1"), ("pr2", "pr3"), ("pr4", "pr5"), ("pr6", "pr7")]
    preferences: Dict[str, List[str]] = {}
    for a, b in pairs:
        others = [s.id for s in students if s.id not in (a, b)]
        preferences[a] = [b] + others
        preferences[b] = [a] + others
    return FixtureCase(
        name="preference_instability",
        description="4 mutual best-friend pairs with skills chosen so a skill-coverage-first "
                     "optimizer tends to split each pair across different teams.",
        students=students,
        team_size=4,
        preferences=preferences,
    )


# NOTE (Part 5.6): the legacy Dict[str, List[str]] -> List[PreferenceEntry]
# adapter that used to live here (Part 5.5's `preferences_dict_to_entries`)
# has been REMOVED and its logic moved to
# evaluation.metrics.legacy_preferences_to_entries — that is now the ONE
# authoritative implementation of this conversion, since fixtures.py is
# test support, not production/evaluation infrastructure, and this
# conversion is used by production evaluation code (evaluate_teams()
# itself), not merely by tests. Callers that previously used
# `fx.preferences_dict_to_entries` should import
# `evaluation.metrics.legacy_preferences_to_entries` instead.


# ---------------------------------------------------------------------------
# 8. Uneven ability distributions
# ---------------------------------------------------------------------------
def fixture_uneven_ability() -> FixtureCase:
    """
    10 students: 7 beginners, 3 advanced — team_size 5 → 2 teams. There is
    no way to give both teams the same number of advanced students (3 does
    not divide evenly by 2 teams), so at least one team must have either 2
    advanced + 3 beginner or 1 advanced + 4 beginner. Tests how the
    optimizer resolves an ability distribution that cannot be split evenly.
    """
    advanced = [_s(f"ad{i}", f"Advanced {i}", ["backend", "ml", "algorithms"][i:i+1],
                   experience_level="advanced", accountability="high", cooperativeness="high")
                for i in range(3)]
    beginners = [_s(f"bg{i}", f"Beginner {i}", ["frontend", "design", "testing", "documentation",
                                                  "communication", "research", "devops"][i:i+1],
                    experience_level="beginner", accountability="developing", cooperativeness="developing")
                 for i in range(7)]
    return FixtureCase(
        name="uneven_ability",
        description="3 advanced + 7 beginner students across 2 teams of 5 — advanced students "
                     "cannot be split evenly.",
        students=advanced + beginners,
        team_size=5,
    )


# ---------------------------------------------------------------------------
# 9. More students than cleanly fit the requested team size
# ---------------------------------------------------------------------------
def fixture_uneven_roster_size() -> FixtureCase:
    """
    11 students, team_size 4 → floor(11/4) = 2 teams sized 4, with 3
    leftover students appended by `_greedy_build`'s leftover-distribution
    step (matching.py lines ~423-430), producing teams of unequal size
    (e.g. 4/4/3 -> distributed -> possibly 5/4/... depending on best-fit).
    Tests whether score components that are NOT size-normalized (e.g. raw
    counts) unfairly advantage/disadvantage the larger team.
    """
    students = [_s(f"rs{i}", f"Roster {i}",
                   ["frontend", "backend", "ml", "design", "testing", "devops",
                    "algorithms", "documentation", "research", "communication",
                    "project management"][i % 11:][:1] or ["frontend"])
                for i in range(11)]
    return FixtureCase(
        name="uneven_roster_size",
        description="11 students with team_size=4 (not evenly divisible) — forces leftover "
                     "distribution logic in _greedy_build().",
        students=students,
        team_size=4,
    )


# ---------------------------------------------------------------------------
# 10. Missing, tied, extreme, and invalid behavioral scores
# ---------------------------------------------------------------------------
def fixture_missing_tied_extreme() -> FixtureCase:
    """
    A mix of: a student fully under-determined on both floor dims (should
    NOT count toward floor violations or floor feasibility per
    matching.py:_is_floor_level), several students tied on every ordinal
    dimension (stresses tie-breaking in _ordinal_mean/_plurality-style
    logic upstream), and one student at the extreme low end of BOTH floor
    dimensions simultaneously (double bad-apple risk).
    """
    students = [
        _s("mx1", "Fully Unknown", ["frontend"], accountability="developing", cooperativeness="developing",
           under_determined_dims=["accountability", "cooperativeness"]),
        _s("mx2", "Tied A", ["backend"], accountability="developing", cooperativeness="developing"),
        _s("mx3", "Tied B", ["ml"], accountability="developing", cooperativeness="developing"),
        _s("mx4", "Tied C", ["design"], accountability="developing", cooperativeness="developing"),
        _s("mx5", "Double Floor", ["testing"], accountability="low", cooperativeness="low"),
        _s("mx6", "Normal High", ["devops"], accountability="high", cooperativeness="high"),
        _s("mx7", "Normal High 2", ["research"], accountability="high", cooperativeness="high"),
        _s("mx8", "Normal High 3", ["documentation"], accountability="high", cooperativeness="high"),
    ]
    return FixtureCase(
        name="missing_tied_extreme",
        description="Under-determined dims, all-tied ordinals, and a double-floor (low/low) student.",
        students=students,
        team_size=4,
    )


# ---------------------------------------------------------------------------
# 11. Greedy assignment gets stuck in a local optimum
# ---------------------------------------------------------------------------
def fixture_greedy_local_optimum() -> FixtureCase:
    """
    Constructed so the FIRST two students greedily seeded (highest
    reference-skill count, matching.py:_greedy_build seeding step) each
    grab a skill-overlapping partner early, leaving the last team forced
    into a low-skill-coverage combination that a global assignment could
    have avoided. Used to check whether 2-opt actually escapes the local
    optimum or merely fails to (documented as a known local-search
    limitation either way).
    """
    # Two "connector" students who each hold 4 distinct reference skills —
    # greedy seeds teams with them first.
    connectors = [
        _s("cn1", "Connector 1", ["frontend", "backend", "ml", "design"], experience_level="advanced"),
        _s("cn2", "Connector 2", ["testing", "devops", "algorithms", "communication"], experience_level="advanced"),
    ]
    # Students that greedily maximize marginal score against a connector
    # early, but whose combined presence leaves the last team skill-starved.
    magnets = [
        _s("mg1", "Magnet 1", ["frontend", "backend"]),
        _s("mg2", "Magnet 2", ["testing", "devops"]),
        _s("mg3", "Magnet 3", ["frontend"]),
        _s("mg4", "Magnet 4", ["testing"]),
    ]
    leftover = [
        _s("lf1", "Leftover 1", ["documentation"]),
        _s("lf2", "Leftover 2", ["documentation"]),
    ]
    return FixtureCase(
        name="greedy_local_optimum",
        description="Two high-skill 'connector' seeds each attract skill-overlapping partners, "
                     "leaving a skill-starved final team.",
        students=connectors + magnets + leftover,
        team_size=4,
    )


# ---------------------------------------------------------------------------
# 12. 2-opt improves total score but harms fairness
# ---------------------------------------------------------------------------
def fixture_2opt_harms_fairness() -> FixtureCase:
    """
    Constructed so a single pairwise swap raises the SUM of both teams'
    total scores (which `_two_opt_improve` accepts, matching.py:445-451)
    while making the post-swap minimum-scoring team's floor-dimension
    minimum worse than before the swap — i.e. total-score improvement at
    the expense of the weakest team, which 2-opt has no visibility into
    since it only compares summed pair scores, never the partition minimum.
    """
    team_a_seed = [
        _s("op1", "OptA 1", ["frontend", "backend"], accountability="high", cooperativeness="high"),
        _s("op2", "OptA 2", ["ml", "testing"], accountability="high", cooperativeness="high"),
        _s("op3", "OptA 3", ["design", "devops"], accountability="developing", cooperativeness="developing"),
        _s("op4", "OptA 4", ["research", "documentation"], accountability="developing", cooperativeness="developing"),
    ]
    team_b_seed = [
        _s("op5", "OptB 1", ["frontend"], accountability="low", cooperativeness="low"),
        _s("op6", "OptB 2", ["backend"], accountability="developing", cooperativeness="developing"),
        _s("op7", "OptB 3", ["algorithms"], accountability="developing", cooperativeness="developing"),
        _s("op8", "OptB 4", ["communication"], accountability="high", cooperativeness="developing"),
    ]
    return FixtureCase(
        name="2opt_harms_fairness",
        description="A pairwise swap that raises total combined score while worsening the "
                     "weaker team's floor-dimension minimum.",
        students=team_a_seed + team_b_seed,
        team_size=4,
    )


ALL_FIXTURES = [
    fixture_skill_vs_behavior,
    fixture_hidden_bad_apple,
    fixture_stacked_team,
    fixture_isolated_student,
    fixture_friend_clique,
    fixture_utility_vs_fairness,
    fixture_preference_instability,
    fixture_uneven_ability,
    fixture_uneven_roster_size,
    fixture_missing_tied_extreme,
    fixture_greedy_local_optimum,
    fixture_2opt_harms_fairness,
]
