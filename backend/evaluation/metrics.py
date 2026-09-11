"""
Research-alignment evaluation harness (Phase 4 of research_alignment_audit.md).

Computes the metrics enumerated in that audit's Phase 4 against team
partitions produced by the PRODUCTION algorithm (app.matching).  This module
does not change app.matching in any way — it only imports and calls it.

Design notes
------------
* Two research inputs the current `Student` model does NOT carry at all —
  familiarity/social-graph data and stated teammate preferences — are
  accepted here as OPTIONAL side-channel parameters (`familiarity`,
  `friend_groups`, `preferences`) so the harness can demonstrate what these
  metrics would compute if that data existed.  When the caller does not
  supply them, the corresponding metric reports `supported: False` with an
  explanatory `note` instead of a fabricated number.  See
  research_alignment_audit.md §4 and §7 for the research justification.
* "Ability" has no dedicated field in the schema either.  `experience_level`
  (beginner/intermediate/advanced) is used as a documented PROXY.  This is
  an assumption requiring validation, not a validated ability measure —
  every metric that uses it says so in its `note`.
* "Individual utility" is approximated as the total score of the team a
  student was placed on, because the current algorithm has no per-student
  utility function — every member of a team receives the same team-level
  score. That collapse (no individual differentiation) is itself one of the
  audit's findings; see research_alignment_audit.md Phase 2.
* All thresholds/weights live in `EvalConfig` so they are configurable
  per the audit's Phase 4 instruction, instead of being hard-coded here.
"""
from __future__ import annotations

import itertools
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from app.models import PreferenceEntry, Student
from app import matching as prod
from scoring.dimension_scorer import get_rubric

StudentId = str
Team = List[Student]


# ---------------------------------------------------------------------------
# Configuration — every threshold/weight the harness uses lives here.
# ---------------------------------------------------------------------------
@dataclass
class EvalConfig:
    reference_skills: frozenset = field(default_factory=lambda: frozenset(prod.REFERENCE_SKILLS))

    # Legacy 0-2 ordinal proxy (kept for backward compatibility with
    # ability_diversity()'s existing mean_ability/stdev_ability keys, which
    # predate app.matching.compute_ability_metrics()'s [0,1] ability_score).
    ability_dim: str = "experience_level"
    ability_order: Tuple[str, ...] = ("beginner", "intermediate", "advanced")

    # Thresholds bucketing app.matching's [0.0, 1.0] ability_score into 3
    # reporting tiers — same default as AbilityConfig.ability_level_thresholds.
    ability_level_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "developing_max": 0.4,
        "medium_max": 0.7,
    })

    conscientiousness_dim: str = "accountability"   # rubric dimension used as conscientiousness proxy
    agreeableness_dim: str = "cooperativeness"       # rubric dimension used as agreeableness proxy

    # Minimum acceptable ordinal level for the floor dimensions above.
    # "low" (index 0) is the rubric floor; this can be raised to "developing"
    # for a stricter research-aligned threshold.
    min_acceptable_level: str = "low"

    # A team counts as a "clique" if this fraction (or more) of its members
    # share the same pre-existing friend_group label. LEGACY (label-based)
    # — kept for familiarity_density()/clique_concentration()'s existing
    # side-channel-data callers. See familiarity_balance() below for the
    # Part 4 graph-based replacement, which uses the thresholds just below
    # instead of this one.
    clique_fraction_threshold: float = 0.75

    # Below this pairwise-familiarity density, a team is flagged as
    # "isolated" (not enough prior familiarity to coordinate). LEGACY —
    # same caveat as clique_fraction_threshold above.
    min_familiarity_density: float = 0.15

    # Part 4 (revised Part 4.5) — real (app.matching.compute_familiarity_metrics)
    # thresholds. Same defaults as app.models.FamiliarityConfig; NONE of
    # these are scientifically established for this population — see that
    # class's docstring.
    familiar_pair_threshold: float = 0.3
    clique_edge_threshold: float = 0.5
    familiarity_lower_bound: float = 0.15
    familiarity_upper_bound: float = 0.6
    familiarity_minimum_known_pair_coverage: float = 0.5
    familiarity_quasi_clique_density_threshold: float = 0.75

    # Part 5 — real (app.matching.compute_preference_metrics /
    # compute_stability_metrics) preference/stability thresholds. Same
    # defaults as app.models.PreferenceConfig; NONE are scientifically
    # established for this population — see that class's docstring.
    preference_high_priority_weight: float = 3.0
    preference_avoidance_violation_weight: float = 5.0
    preference_stability_definition: str = "blocking_swap"

    # Regret/utility search is exact (brute-force over alternative teams)
    # only below this roster size, since it is combinatorial. Above it,
    # the harness reports `exact: False` rather than silently sampling.
    max_exact_regret_roster_size: int = 20

    # A student's individual utility (app.matching.compute_assignment_
    # utilities) below this value counts toward students_below_floor in the
    # individual_utility() metric. Purely a reporting threshold, not
    # enforced anywhere — same default as FairnessConfig.acceptable_utility_floor.
    acceptable_utility_floor: float = 0.5

    # Weights used to build the single-number "team strength" composite
    # for between-team comparisons (criterion 5). Configurable — the
    # production algorithm has no equivalent composite today.
    strength_weights: Dict[str, float] = field(default_factory=lambda: {
        "skill_coverage": 0.25,
        "ability": 0.25,
        "conscientiousness": 0.25,
        "agreeableness": 0.25,
    })


def _rubric_dim(name: str) -> dict:
    return get_rubric()["dimensions"][name]


def _ordinal_level(value: str, allowed: Tuple[str, ...]) -> Optional[int]:
    return allowed.index(value) if value in allowed else None


def _mean(xs: List[float]) -> Optional[float]:
    return statistics.fmean(xs) if xs else None


def _stdev(xs: List[float]) -> Optional[float]:
    return statistics.pstdev(xs) if len(xs) >= 1 else None


# ---------------------------------------------------------------------------
# 1 & 2. Skill coverage
# ---------------------------------------------------------------------------
def skill_coverage(teams: List[Team], config: EvalConfig) -> dict:
    per_team = []
    for i, team in enumerate(teams):
        team_skills = set()
        for s in team:
            team_skills.update(s.competence_signature.skills)
        covered = team_skills & config.reference_skills
        missing = config.reference_skills - team_skills
        # Concentration: skills held by >1 member as a fraction of covered skills
        skill_holders: Dict[str, int] = {}
        for s in team:
            for sk in set(s.competence_signature.skills) & config.reference_skills:
                skill_holders[sk] = skill_holders.get(sk, 0) + 1
        concentrated = [sk for sk, n in skill_holders.items() if n == len(team) and len(team) > 1]
        per_team.append({
            "team_index": i,
            "coverage_rate": len(covered) / len(config.reference_skills),
            "missing_skills": sorted(missing),
            "fully_concentrated_skills": sorted(concentrated),
        })
    rates = [t["coverage_rate"] for t in per_team]
    return {
        "supported": True,
        "per_team": per_team,
        "mean_coverage_rate": _mean(rates),
        "min_coverage_rate": min(rates) if rates else None,
        "teams_with_missing_skills": sum(1 for t in per_team if t["missing_skills"]),
    }


# ---------------------------------------------------------------------------
# 3. Within-team ability diversity (WITHIN one team: range/variance of its
# members' ability). Kept on the ORIGINAL 0-2 experience_level-ordinal
# scale for full backward compatibility with every existing caller of this
# function (mean_ability/stdev_ability/all_same_level keys and their scale
# are UNCHANGED from before Part 3). For the newer [0.0, 1.0] ability_score
# scale (app.matching.compute_ability_metrics, which prefers a directly-
# recorded ability_score and only falls back to experience_level when one
# isn't set), see ability_balance() below — it reports the SAME within-team
# shape (range/variance) plus the between-team comparisons this function
# was never designed to answer.
# ---------------------------------------------------------------------------
def ability_diversity(teams: List[Team], config: EvalConfig) -> dict:
    per_team = []
    for i, team in enumerate(teams):
        levels = [
            _ordinal_level(s.competence_signature.experience_level, config.ability_order)
            for s in team
        ]
        levels = [l for l in levels if l is not None]
        per_team.append({
            "team_index": i,
            "mean_ability": _mean(levels),
            "stdev_ability": _stdev(levels),
            "range_ability": (max(levels) - min(levels)) if levels else None,
            "variance_ability": statistics.pvariance(levels) if len(levels) > 1 else (0.0 if levels else None),
            "all_same_level": len(set(levels)) <= 1 if levels else None,
        })
    return {
        "supported": True,
        "note": (
            "Uses competence_signature.experience_level (beginner/intermediate/"
            "advanced) as an ability PROXY on a 0-2 ordinal scale — the schema "
            "has no validated ability or performance measure (e.g. GPA, "
            "prior-course grade). Treat as a working hypothesis pending real "
            "outcome data. See ability_balance() for the newer [0.0, 1.0] "
            "ability_score-based view (Part 3), which prefers a directly-"
            "recorded score and reports between-team comparisons too."
        ),
        "per_team": per_team,
        "teams_all_same_level": sum(1 for t in per_team if t["all_same_level"]),
    }


# ---------------------------------------------------------------------------
# 3b. Ability balance (Part 3) — within-team range/variance AND between-team
# comparability, on the documented [0.0, 1.0] ability_score scale
# (app.models.CompetenceSignature.ability_score, falling back to
# experience_level when unset — see app.matching._student_ability_score).
# Delegates entirely to app.matching.compute_ability_metrics() rather than
# reimplementing it, matching this harness's established pattern for
# individual_utility()/constraint_violations().
# ---------------------------------------------------------------------------
def ability_balance(teams: List[Team], config: EvalConfig) -> dict:
    report = prod.compute_ability_metrics(teams, ability_level_thresholds=config.ability_level_thresholds)
    return {"supported": True, **report}


# ---------------------------------------------------------------------------
# 4 & 6. Conscientiousness / agreeableness — mean and minimum
# ---------------------------------------------------------------------------
def _dim_mean_min(teams: List[Team], dim_name: str, config: EvalConfig) -> dict:
    dim_def = _rubric_dim(dim_name)
    allowed = tuple(dim_def["allowed_values"])
    min_ok_idx = allowed.index(config.min_acceptable_level)

    per_team = []
    for i, team in enumerate(teams):
        levels = []
        unknowns = []
        for s in team:
            if dim_name in (s.under_determined_dims or []):
                unknowns.append(s.name)
                continue
            val = getattr(s.collaboration_signature, dim_name, None)
            idx = _ordinal_level(val, allowed)
            if idx is not None:
                levels.append(idx)
        below_floor = [
            s.name for s in team
            if dim_name not in (s.under_determined_dims or [])
            and _ordinal_level(getattr(s.collaboration_signature, dim_name, None), allowed) is not None
            and _ordinal_level(getattr(s.collaboration_signature, dim_name, None), allowed) < min_ok_idx + 1
            and _ordinal_level(getattr(s.collaboration_signature, dim_name, None), allowed) <= min_ok_idx
        ]
        per_team.append({
            "team_index": i,
            "mean_level": _mean(levels),
            "min_level": min(levels) if levels else None,
            "members_at_or_below_floor": below_floor,
            "unknown_members": unknowns,
        })
    means = [t["mean_level"] for t in per_team if t["mean_level"] is not None]
    mins = [t["min_level"] for t in per_team if t["min_level"] is not None]
    return {
        "supported": True,
        "allowed_values_low_to_high": list(allowed),
        "per_team": per_team,
        "global_mean_of_team_means": _mean(means),
        "global_min_of_team_mins": min(mins) if mins else None,
        "teams_with_floor_violation": sum(1 for t in per_team if t["members_at_or_below_floor"]),
    }


def conscientiousness(teams: List[Team], config: EvalConfig) -> dict:
    return _dim_mean_min(teams, config.conscientiousness_dim, config)


def agreeableness(teams: List[Team], config: EvalConfig) -> dict:
    return _dim_mean_min(teams, config.agreeableness_dim, config)


# ---------------------------------------------------------------------------
# 7 & 8. Familiarity density / clique concentration
# familiarity: Optional[Dict[Tuple[id, id], float]] — symmetric pairwise
#              familiarity strength in [0, 1]; unordered pair keys.
# friend_groups: Optional[Dict[id, str]] — pre-existing friend-group label.
# ---------------------------------------------------------------------------
def _familiarity_lookup(familiarity: Dict[Tuple[str, str], float], a: str, b: str) -> float:
    return familiarity.get((a, b), familiarity.get((b, a), 0.0))


def familiarity_density(
    teams: List[Team],
    config: EvalConfig,
    familiarity: Optional[Dict[Tuple[str, str], float]] = None,
) -> dict:
    if familiarity is None:
        return {
            "supported": False,
            "note": (
                "Student model (app/models.py) has no familiarity/prior-contact "
                "field. Would need e.g. Student.prior_teammates: List[str] or a "
                "separate FamiliarityDB edge table (student_a, student_b, "
                "strength) populated from a roster survey or LMS co-enrollment "
                "history. See research_alignment_audit.md §4."
            ),
        }
    per_team = []
    for i, team in enumerate(teams):
        if len(team) < 2:
            per_team.append({"team_index": i, "density": None, "isolated_members": []})
            continue
        pair_scores = [
            _familiarity_lookup(familiarity, a.id, b.id)
            for a, b in itertools.combinations(team, 2)
        ]
        density = _mean(pair_scores)
        isolated = [
            s.name for s in team
            if all(
                _familiarity_lookup(familiarity, s.id, o.id) == 0.0
                for o in team if o.id != s.id
            )
        ]
        per_team.append({"team_index": i, "density": density, "isolated_members": isolated})
    densities = [t["density"] for t in per_team if t["density"] is not None]
    return {
        "supported": True,
        "per_team": per_team,
        "mean_density": _mean(densities),
        "teams_below_min_density": sum(
            1 for t in per_team if t["density"] is not None and t["density"] < config.min_familiarity_density
        ),
    }


def clique_concentration(
    teams: List[Team],
    config: EvalConfig,
    friend_groups: Optional[Dict[str, str]] = None,
) -> dict:
    if friend_groups is None:
        return {
            "supported": False,
            "note": (
                "No friend-group / social-cluster data available. Would need "
                "either self-reported prior-group data at onboarding or a "
                "derived clustering over a familiarity graph. See "
                "research_alignment_audit.md §4."
            ),
        }
    per_team = []
    for i, team in enumerate(teams):
        if not team:
            continue
        labels = [friend_groups.get(s.id) for s in team]
        labels = [l for l in labels if l is not None]
        if not labels:
            per_team.append({"team_index": i, "max_group_fraction": None, "dominant_group": None})
            continue
        counts: Dict[str, int] = {}
        for l in labels:
            counts[l] = counts.get(l, 0) + 1
        dominant, n = max(counts.items(), key=lambda kv: kv[1])
        per_team.append({
            "team_index": i,
            "max_group_fraction": n / len(team),
            "dominant_group": dominant,
            "is_clique": (n / len(team)) >= config.clique_fraction_threshold,
        })
    return {
        "supported": True,
        "per_team": per_team,
        "teams_flagged_as_clique": sum(1 for t in per_team if t.get("is_clique")),
    }


# ---------------------------------------------------------------------------
# 7b/8b. Familiarity balance (Part 4) — graph-based replacement for the
# label-based clique_concentration() above and the always-zero-or-one-
# density familiarity_density() above, now backed by a REAL data source
# (the `familiarity` DB table / app.models.FamiliarityEdge) instead of a
# caller-supplied side channel. Delegates entirely to
# app.matching.compute_familiarity_metrics() rather than reimplementing it,
# matching this harness's established pattern (individual_utility(),
# ability_balance(), constraint_violations()).
#
# `edges` here is the list of FamiliarityEdge objects to evaluate against —
# pass `[]` (or omit) for "no familiarity data at all" (every metric still
# computes; every pair is simply unknown, which is a supported, meaningful
# state, not a `supported: False` short-circuit like the legacy functions
# above use for missing side-channel data).
# ---------------------------------------------------------------------------
def familiarity_balance(
    teams: List[Team],
    config: EvalConfig,
    edges: Optional[List] = None,
) -> dict:
    report = prod.compute_familiarity_metrics(
        teams,
        edges=edges or [],
        familiar_pair_threshold=config.familiar_pair_threshold,
        clique_edge_threshold=config.clique_edge_threshold,
        lower_bound=config.familiarity_lower_bound,
        upper_bound=config.familiarity_upper_bound,
        minimum_known_pair_coverage=config.familiarity_minimum_known_pair_coverage,
        quasi_clique_density_threshold=config.familiarity_quasi_clique_density_threshold,
    )
    return {"supported": True, **report}


# ---------------------------------------------------------------------------
# Part 5 (revised Part 5.6) — real (app.matching.compute_preference_metrics /
# compute_stability_metrics) preference-satisfaction and stability metrics,
# backed by app.models.PreferenceEntry. `entries` here is a List[PreferenceEntry]
# — pass `[]` (or omit) for "no preference data at all" (every metric still
# computes; every student is simply missing preferences, a supported,
# meaningful state, not a `supported: False` short-circuit).
#
# AUTHORITATIVE IMPLEMENTATION NOTICE (Part 5.6): app.matching.
# compute_preference_metrics() and compute_stability_metrics() are THE ONE
# authoritative preference-satisfaction/stability engine in this codebase.
# preference_satisfaction() (below) still accepts the legacy `preferences`
# Dict[str, List[str]] side channel and computes its OWN "top-half
# preference met" metric independently — that metric has no equivalent in
# the tier-scored engine and is UNCHANGED, kept for backward compatibility
# with test_research_alignment.py's `fraction_with_top_half_preference_met`
# assertions. blocking_pairs_preference_based() (below), in contrast, WAS a
# second, independent implementation of the SAME concept
# compute_stability_metrics()'s "blocking_pair" definition already
# covers — Part 5.6 turned it into a thin wrapper (via
# legacy_preferences_to_entries() immediately below) so there is no longer
# a second blocking-pair algorithm that could silently disagree with the
# authoritative one.
# ---------------------------------------------------------------------------
def legacy_preferences_to_entries(preferences: Optional[Dict[str, List[str]]]) -> List[PreferenceEntry]:
    """
    THE authoritative boundary adapter (Part 5.6): converts the LEGACY
    `Dict[student_id, List[ranked_target_id]]` preference shape — accepted
    by preference_satisfaction()/blocking_pairs_preference_based() below,
    and by evaluation.fixtures.FixtureCase.preferences — into the real
    app.models.PreferenceEntry rows app.matching.compute_preference_metrics()
    / compute_stability_metrics() consume. This is the ONLY place this
    conversion is implemented. (Part 5.5 introduced a copy of this inside
    evaluation.fixtures; Part 5.6 removed it from there and moved the real
    logic here, since fixtures.py is test support, not production/
    evaluation infrastructure, and this conversion is used by production
    evaluation code — evaluate_teams() itself — not merely by tests.)

    CONVERSION POLICY (each point verified against the implementation
    below, not merely asserted here):

      List order -> STRICT RANK ORDER. Earlier entries receive BETTER
      (lower-numbered) ranks: position 0 -> rank=1, position 1 -> rank=2,
      etc. Ranks are renumbered CONTIGUOUSLY among a student's VALID
      surviving targets, in their original relative order (see duplicates
      and self-targets below) — not their raw list index — so filtering
      out one invalid entry never leaves a gap in the rank sequence.

      Entries are DIRECTED preferences. Every converted entry has
      direction="prefer"; the legacy shape has no way to express "avoid",
      and this adapter never invents one.

      NO avoidance, priority, characteristic, or reason information can be
      INFERRED from a plain ranked list. Every converted entry gets
      priority="normal", target_characteristic=None, reason_category=None,
      source=None — the legacy format simply carries no signal for any of
      these, and this adapter does not guess.

      An EMPTY list (`preferences[sid] == []`) means sid submitted NO
      named-student preferences at all — sid gets ZERO PreferenceEntry
      rows, which downstream (app.matching._student_satisfaction) reads
      as satisfaction_score = None (MISSING), never 0.0. This is
      indistinguishable, after conversion, from `sid` being entirely
      ABSENT from the `preferences` dict — both are "no rows submitted",
      exactly matching PreferenceEntry's own missing-data semantics.

      UNKNOWN student IDs (a target not present in whatever roster/teams
      this is later evaluated against) are passed through UNCHANGED. This
      adapter performs NO roster validation — it has no roster reference
      to validate against. Downstream, an unknown target simply never
      matches any team's membership, contributing exactly the same
      "not satisfied" outcome a valid-but-currently-elsewhere target
      would — never a crash, never a special case. A caller wanting
      stricter validation should check the input dict's target IDs
      against a known roster before calling this adapter.

      DUPLICATE targets within one student's own list (e.g. `[b, c, b]`)
      are resolved DETERMINISTICALLY: only the FIRST (best-ranked)
      occurrence of a given target is kept; later duplicates of the same
      target are dropped rather than creating a second, contradictory-
      rank row for the same target.

      SELF-preferences (`sid` naming itself as its own target) are
      IGNORED — silently filtered out before construction, NOT raised as
      an error. This is a DELIBERATE, DOCUMENTED POLICY CHOICE (ignore,
      not reject-via-exception): constructing PreferenceEntry directly
      with a self-target raises ValueError (see its own validator), but an
      ADAPTER converting arbitrary/legacy data should not abort an entire
      student's conversion over one nonsensical entry. A caller that wants
      strict rejection instead should validate the input dict itself
      before calling this adapter.
    """
    if not preferences:
        return []
    entries: List[PreferenceEntry] = []
    for sid, ranked in preferences.items():
        seen_targets: set = set()
        rank = 0
        for target_id in ranked:
            if target_id == sid:
                continue  # self-preference: ignored, see docstring policy
            if target_id in seen_targets:
                continue  # duplicate target: keep only the first (best-ranked) occurrence
            seen_targets.add(target_id)
            rank += 1
            entries.append(PreferenceEntry(
                student_id=sid, direction="prefer", target_student_id=target_id, rank=rank,
            ))
    return entries


def preference_stability(
    teams: List[Team],
    config: EvalConfig,
    entries: Optional[List] = None,
) -> dict:
    entries = entries or []
    pref_report = prod.compute_preference_metrics(teams, entries)
    baseline = prod.AssignmentBaseline.capture(teams)
    stability_report = prod.compute_stability_metrics(
        teams, entries, baseline=baseline,
        definition=config.preference_stability_definition,
    )
    return {
        "supported": True,
        "preference_satisfaction": pref_report,
        "stability": stability_report,
    }


# ---------------------------------------------------------------------------
# 5 & 11. Between-team strength variance / team-balance score
# ---------------------------------------------------------------------------
def team_strength_comparison(teams: List[Team], config: EvalConfig) -> dict:
    sc = skill_coverage(teams, config)
    ab = ability_diversity(teams, config)
    co = conscientiousness(teams, config)
    ag = agreeableness(teams, config)

    def norm_ability(mean_ability):
        max_level = len(config.ability_order) - 1
        return mean_ability / max_level if mean_ability is not None else None

    per_team = []
    for i in range(len(teams)):
        parts = {
            "skill_coverage": sc["per_team"][i]["coverage_rate"],
            "ability": norm_ability(ab["per_team"][i]["mean_ability"]),
            "conscientiousness": (
                co["per_team"][i]["mean_level"] / (len(co["allowed_values_low_to_high"]) - 1)
                if co["per_team"][i]["mean_level"] is not None else None
            ),
            "agreeableness": (
                ag["per_team"][i]["mean_level"] / (len(ag["allowed_values_low_to_high"]) - 1)
                if ag["per_team"][i]["mean_level"] is not None else None
            ),
        }
        known = {k: v for k, v in parts.items() if v is not None}
        w = config.strength_weights
        total_w = sum(w[k] for k in known)
        strength = sum(w[k] * v for k, v in known.items()) / total_w if total_w else None
        per_team.append({"team_index": i, "components": parts, "strength": strength})

    strengths = [t["strength"] for t in per_team if t["strength"] is not None]
    metrics_for_gap = {"skill_coverage": [], "ability": [], "conscientiousness": [], "agreeableness": []}
    for t in per_team:
        for k in metrics_for_gap:
            v = t["components"][k]
            if v is not None:
                metrics_for_gap[k].append(v)

    def gap_stats(xs):
        if not xs:
            return {"variance": None, "stdev": None, "range": None, "min": None, "max": None}
        return {
            "variance": statistics.pvariance(xs) if len(xs) > 1 else 0.0,
            "stdev": statistics.pstdev(xs) if len(xs) > 1 else 0.0,
            "range": max(xs) - min(xs),
            "min": min(xs),
            "max": max(xs),
        }

    return {
        "supported": True,
        "per_team": per_team,
        "overall_strength": gap_stats(strengths),
        "per_metric_gap": {k: gap_stats(v) for k, v in metrics_for_gap.items()},
        "team_balance_score": (
            1.0 - min(1.0, gap_stats(strengths)["stdev"]) if strengths and gap_stats(strengths)["stdev"] is not None
            else None
        ),
    }


# ---------------------------------------------------------------------------
# 9. Individual preference satisfaction
# preferences: Optional[Dict[id, List[id]]] — each student's ranked list of
#              preferred teammates, most-preferred first.
# ---------------------------------------------------------------------------
def preference_satisfaction(
    teams: List[Team],
    config: EvalConfig,
    preferences: Optional[Dict[str, List[str]]] = None,
) -> dict:
    if preferences is None:
        return {
            "supported": False,
            "note": (
                "No preference data collected anywhere in the pipeline "
                "(models.py has no `preferred_teammates` / `avoid_teammates` "
                "field, and no assessment scenario asks for it). Would need "
                "a ranked-preference field on Student or a dedicated "
                "PreferenceDB table, e.g. (student_id, preferred_id, rank). "
                "See research_alignment_audit.md §7."
            ),
        }
    team_of: Dict[str, int] = {}
    for i, team in enumerate(teams):
        for s in team:
            team_of[s.id] = i

    per_student = []
    for sid, ranked in preferences.items():
        if sid not in team_of:
            continue
        my_team = team_of[sid]
        satisfied_ranks = [
            rank for rank, pref_id in enumerate(ranked)
            if team_of.get(pref_id) == my_team
        ]
        top_k = ranked[: max(1, len(ranked) // 2)] if ranked else []
        got_any_top_pref = any(team_of.get(p) == my_team for p in top_k)
        per_student.append({
            "student_id": sid,
            "n_preferences_on_team": len(satisfied_ranks),
            "best_rank_on_team": min(satisfied_ranks) if satisfied_ranks else None,
            "got_a_top_half_preference": got_any_top_pref,
        })

    rates = [1.0 if p["got_a_top_half_preference"] else 0.0 for p in per_student]
    return {
        "supported": True,
        "per_student": per_student,
        "fraction_with_top_half_preference_met": _mean(rates),
        "students_with_zero_preferences_met": sum(1 for p in per_student if p["n_preferences_on_team"] == 0),
    }


# ---------------------------------------------------------------------------
# 10. Individual utility, assignment-level fairness metrics, worst-case regret
#
# Delegates entirely to app.matching.compute_assignment_utilities() (Part 2's
# individual-utility framework) rather than reimplementing it — this harness
# evaluates the actual production formula, not a parallel approximation of
# it. Before Part 2, this function used team_total_score as a stand-in
# utility (identical for every teammate); that placeholder is gone now that
# a real per-student utility exists in app.matching.
# ---------------------------------------------------------------------------
def individual_utility(
    teams: List[Team],
    weights: dict,
    config: EvalConfig,
    utility_weights: Optional[Dict[str, float]] = None,
) -> dict:
    report = prod.compute_assignment_utilities(
        teams,
        utility_weights=utility_weights,
        acceptable_utility_floor=config.acceptable_utility_floor,
        max_exact_regret_roster_size=config.max_exact_regret_roster_size,
    )
    per_student = report["per_student"]
    summary = report["summary"]
    return {
        "supported": True,
        "exact_regret_search": summary["exact_regret_search"],
        "note": summary["note"],
        "per_student_utility": {sid: data["utility"] for sid, data in per_student.items()},
        "per_student_components": {sid: data["components"] for sid, data in per_student.items()},
        "per_student_missing_components": {sid: data["missing_components"] for sid, data in per_student.items()},
        "min_utility": summary["min_utility"],
        "max_utility": summary["max_utility"],
        "mean_utility": summary["mean_utility"],
        "total_utility": summary["total_utility"],
        "utility_range": summary["utility_range"],
        "utility_variance": summary["utility_variance"],
        "utility_stdev": summary["utility_stdev"],
        "acceptable_utility_floor": summary["acceptable_utility_floor"],
        "students_below_floor_count": summary["students_below_floor_count"],
        "students_below_floor_fraction": summary["students_below_floor_fraction"],
        "per_student_regret": {sid: data["regret"] for sid, data in per_student.items()} if summary["exact_regret_search"] else {},
        "worst_case_regret": summary["worst_case_regret"],
        "mean_regret": summary["mean_regret"],
    }


# ---------------------------------------------------------------------------
# 12. Blocking pairs / blocking moves (stability)
# ---------------------------------------------------------------------------
def blocking_moves_score_based(
    teams: List[Team],
    weights: dict,
    config: EvalConfig,
) -> dict:
    """
    Proxy stability check using team SCORE as a stand-in for preference,
    since the schema has no explicit teammate-preference data (see
    `blocking_pairs_preference_based` for the preference-aware version).

    A "blocking move" here is a swap of student i (team A) and student j
    (team B) such that i's new team scores higher than A's current score
    AND j's new team scores higher than B's current score — i.e. a Pareto
    improvement for both teams simultaneously that the algorithm did not
    find/apply. This is NOT the same as a true preference-based blocking
    pair (which requires individual desire, not team-aggregate score) —
    see `blocking_pairs_preference_based` and the audit for that distinction.
    """
    blocking = []
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            score_i = prod._total_score(teams[i], weights)
            score_j = prod._total_score(teams[j], weights)
            for a_idx, a in enumerate(teams[i]):
                for b_idx, b in enumerate(teams[j]):
                    new_i = teams[i][:a_idx] + [b] + teams[i][a_idx + 1:]
                    new_j = teams[j][:b_idx] + [a] + teams[j][b_idx + 1:]
                    ns_i = prod._total_score(new_i, weights)
                    ns_j = prod._total_score(new_j, weights)
                    if ns_i > score_i + 1e-9 and ns_j > score_j + 1e-9:
                        blocking.append({
                            "team_a": i, "student_a": a.name,
                            "team_b": j, "student_b": b.name,
                            "gain_a": round(ns_i - score_i, 4),
                            "gain_b": round(ns_j - score_j, 4),
                        })
    return {
        "supported": True,
        "is_proxy": True,
        "note": (
            "Score-based proxy: flags swaps that would raise BOTH teams' "
            "aggregate scores post-hoc. True stability requires individual "
            "preference data (see blocking_pairs_preference_based)."
        ),
        "blocking_moves": blocking,
        "n_blocking_moves": len(blocking),
        "is_2opt_stable": len(blocking) == 0,
    }


def blocking_pairs_preference_based(
    teams: List[Team],
    preferences: Optional[Dict[str, List[str]]] = None,
) -> dict:
    """
    DEPRECATED (Part 5.6) — a THIN WRAPPER around the authoritative Part 5
    stability engine (app.matching.compute_stability_metrics), no longer
    an independent implementation. Converts `preferences` via
    legacy_preferences_to_entries() (this module's boundary adapter,
    immediately above) and delegates entirely, requesting
    definition="blocking_pair" to reproduce the EXACT semantics this
    function always had (a student ranks a cross-team target strictly
    ahead of every one of their own current teammates, checked in both
    directions — see app.matching.compute_blocking_pairs, the ONE
    remaining implementation of this concept anywhere in this codebase).

    Kept only so existing callers that reference
    evaluate_teams()['blocking_pairs_preference_based'] by name keep
    working unchanged; new callers should call preference_stability() /
    compute_stability_metrics() directly instead.
    """
    if preferences is None:
        return {
            "supported": False,
            "note": (
                "Requires per-student ranked teammate/team preferences, "
                "which the schema does not currently collect. See "
                "research_alignment_audit.md §7 for the proposed field."
            ),
        }
    entries = legacy_preferences_to_entries(preferences)
    stability = prod.compute_stability_metrics(teams, entries, definition="blocking_pair")
    return {
        "supported": True,
        "is_proxy": False,
        "blocking_pairs": [{"student_a": a, "student_b": b} for a, b in stability["blocking_pairs"]],
        "n_blocking_pairs": stability["n_blocking_pairs"],
        "is_stable": stability["n_blocking_pairs"] == 0,
    }


# ---------------------------------------------------------------------------
# 13. Constraint violations
# ---------------------------------------------------------------------------
def constraint_violations(
    teams: List[Team],
    config: EvalConfig,
    floor_policy: str = prod.DEFAULT_FLOOR_POLICY,
) -> dict:
    ok, violations, under_determined_teams = prod._check_floor_violations(teams, floor_policy)
    return {
        "supported": True,
        "floor_policy": floor_policy,
        "floor_ok": ok,
        "floor_violations": violations,
        "under_determined_teams": under_determined_teams,
        "note": (
            "Reuses app.matching._check_floor_violations() directly (read-only "
            "import) so this harness evaluates exactly the constraint the "
            "production algorithm enforces, not a reimplementation of it."
        ),
    }


# ---------------------------------------------------------------------------
# 14. Runtime and scalability
# ---------------------------------------------------------------------------
def measure_runtime(
    students: List[Student],
    team_size: int,
    weights: dict,
) -> dict:
    t0 = time.perf_counter()
    teams = prod._greedy_build(students, team_size, weights)
    t1 = time.perf_counter()
    teams = prod._two_opt_improve(teams, weights)
    t2 = time.perf_counter()
    teams = prod._maximin_improve(teams, weights=weights)
    t3 = time.perf_counter()
    return {
        "supported": True,
        "n_students": len(students),
        "team_size": team_size,
        "n_teams": len(teams),
        "greedy_build_ms": round((t1 - t0) * 1000, 3),
        "two_opt_ms": round((t2 - t1) * 1000, 3),
        "maximin_ms": round((t3 - t2) * 1000, 3),
        "total_ms": round((t3 - t0) * 1000, 3),
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def evaluate_teams(
    teams: List[Team],
    weights: Optional[dict] = None,
    config: Optional[EvalConfig] = None,
    familiarity: Optional[Dict[Tuple[str, str], float]] = None,
    friend_groups: Optional[Dict[str, str]] = None,
    preferences: Optional[Dict[str, List[str]]] = None,
    floor_policy: str = prod.DEFAULT_FLOOR_POLICY,
    familiarity_edges: Optional[List] = None,
    preference_entries: Optional[List] = None,
) -> dict:
    config = config or EvalConfig()
    weights = weights or {"skill_coverage": 0.4, "behavioral_compat": 0.3,
                           "availability_overlap": 0.2, "shared_interests": 0.1}
    return {
        "skill_coverage": skill_coverage(teams, config),
        "ability_diversity": ability_diversity(teams, config),
        "ability_balance": ability_balance(teams, config),
        "conscientiousness": conscientiousness(teams, config),
        "agreeableness": agreeableness(teams, config),
        "familiarity_density": familiarity_density(teams, config, familiarity),
        "clique_concentration": clique_concentration(teams, config, friend_groups),
        "familiarity_balance": familiarity_balance(teams, config, familiarity_edges),
        "team_strength_comparison": team_strength_comparison(teams, config),
        "preference_satisfaction": preference_satisfaction(teams, config, preferences),
        "individual_utility": individual_utility(teams, weights, config),
        "blocking_moves_score_based": blocking_moves_score_based(teams, weights, config),
        "blocking_pairs_preference_based": blocking_pairs_preference_based(teams, preferences),
        "preference_stability": preference_stability(teams, config, preference_entries),
        "constraint_violations": constraint_violations(teams, config, floor_policy),
    }
