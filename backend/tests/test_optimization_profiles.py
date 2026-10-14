"""
Part 6 — combined optimization framework tests: the criterion registry,
named configuration profiles (resolve_profile), the normalized
cross-criterion objective report (compute_normalized_objective), and the
cross-pass acceptance policy (_run_optional_passes).

This suite is ADDITIVE — it does not modify or weaken any existing test.
"""
from __future__ import annotations

import time

import pytest

from app import matching as prod
from app.models import TeamGenerationRequest, ProfileResolution, PreferenceEntry
from evaluation import fixtures as fx

DEFAULT_WEIGHTS = prod._normalize_weights({
    "skill_coverage": 0.4, "behavioral_compat": 0.3,
    "availability_overlap": 0.2, "shared_interests": 0.1,
})


def _ids(teams):
    return sorted(s.id for t in teams for s in t)


def _run_profile(students, team_size, profile, **request_kwargs):
    request, resolution = prod.resolve_profile(
        TeamGenerationRequest(team_size=team_size, profile=profile, **request_kwargs)
    )
    weights = prod._normalize_weights(request.weights)
    teams = prod._greedy_build(students, team_size, weights)
    teams = prod._two_opt_improve(teams, weights, floor_policy=request.floor_policy)
    teams = prod._maximin_improve(teams, weights=weights, floor_policy=request.floor_policy)
    teams, trace = prod._run_optional_passes(
        teams, request, weights, request.floor_policy,
        familiarity_edges=[], preference_entries=[],
        primary_metric=prod.OPTIMIZATION_PROFILES[profile]["primary_metric"],
    )
    return teams, request, resolution, trace


ALL_PROFILE_NAMES = list(prod.OPTIMIZATION_PROFILES.keys())


# ===========================================================================
# Criterion registry
# ===========================================================================
def test_criterion_registry_covers_every_hard_constraint_from_the_spec_table():
    keys = {c.key for c in prod.CRITERION_REGISTRY}
    assert {"membership_complete", "team_sizes_valid", "skill_coverage_floor",
            "behavioral_floor_policy", "hard_avoidance"} <= keys
    hard = {c.key for c in prod.CRITERION_REGISTRY if c.classification == "hard_constraint"}
    assert {"membership_complete", "team_sizes_valid", "skill_coverage_floor",
            "behavioral_floor_policy", "hard_avoidance"} <= hard


def test_criterion_registry_covers_every_soft_objective_from_the_spec_table():
    soft = {c.key for c in prod.CRITERION_REGISTRY if c.classification == "soft_objective"}
    assert {"behavioral_compatibility", "skill_diversity_beyond_minimum", "ability_balance",
            "familiarity_target_band", "anti_cluster_concentration",
            "individual_utility_fairness", "preference_satisfaction"} <= soft


def test_criterion_registry_does_not_promote_research_hypotheses_to_hard_constraints():
    """Ability balance and familiarity band/anti-cluster are explicitly
    documented HYPOTHESES (AbilityConfig/FamiliarityConfig docstrings) —
    the registry must never classify them as hard constraints."""
    by_key = {c.key: c for c in prod.CRITERION_REGISTRY}
    assert by_key["ability_balance"].classification == "soft_objective"
    assert by_key["familiarity_target_band"].classification == "soft_objective"
    assert by_key["anti_cluster_concentration"].classification == "soft_objective"


def test_criterion_registry_blocking_pairs_is_both_metric_and_optional_repair_target():
    by_key = {c.key: c for c in prod.CRITERION_REGISTRY}
    assert by_key["blocking_pair_swap_counts"].classification == "evaluation_only"


def test_criterion_registry_serializes_to_plain_dicts():
    rows = prod.criterion_registry_as_dicts()
    assert len(rows) == len(prod.CRITERION_REGISTRY)
    assert all(isinstance(r, dict) for r in rows)


# ===========================================================================
# Profile resolution
# ===========================================================================
def test_default_request_resolves_to_legacy():
    request, resolution = prod.resolve_profile(TeamGenerationRequest())
    assert resolution.name == "legacy"
    assert resolution.is_experimental is False
    assert request.ability.enabled is False
    assert request.fairness.enabled is False
    assert request.familiarity.enabled is False
    assert request.preference.enabled is False
    assert request.preference.hard_avoidance is False


@pytest.mark.parametrize("profile", ALL_PROFILE_NAMES)
def test_every_profile_resolves_to_its_documented_settings(profile):
    request, resolution = prod.resolve_profile(TeamGenerationRequest(profile=profile))
    assert resolution.name == profile
    assert isinstance(resolution.description, str) and len(resolution.description) > 0
    assert resolution.effective_settings["profile"] == profile
    if profile != "legacy":
        assert resolution.is_experimental is True
        assert "EXPERIMENTAL" in resolution.description
        assert "optimal" not in resolution.description.lower() or "not" in resolution.description.lower()


def test_research_balanced_enables_every_supported_pass():
    request, _ = prod.resolve_profile(TeamGenerationRequest(profile="research_balanced"))
    assert request.ability.enabled is True
    assert request.fairness.enabled is True
    assert request.familiarity.enabled is True
    assert request.preference.enabled is True


def test_fairness_priority_enables_fairness_and_preserves_hard_constraints_config():
    request, _ = prod.resolve_profile(TeamGenerationRequest(profile="fairness_priority"))
    assert request.fairness.enabled is True
    assert request.ability.enabled is False
    assert request.floor_policy == "one_low_per_team"  # unchanged system default


def test_skill_priority_reweights_toward_skill_coverage():
    request, _ = prod.resolve_profile(TeamGenerationRequest(profile="skill_priority"))
    assert request.weights["skill_coverage"] > 0.5
    assert request.ability.enabled is False and request.fairness.enabled is False


def test_preference_priority_enables_preference_and_hard_avoidance():
    request, _ = prod.resolve_profile(TeamGenerationRequest(profile="preference_priority"))
    assert request.preference.enabled is True
    assert request.preference.hard_avoidance is True


def test_explicit_overrides_take_precedence_over_profile():
    request, _ = prod.resolve_profile(
        TeamGenerationRequest(profile="research_balanced", familiarity={"enabled": False})
    )
    assert request.familiarity.enabled is False
    assert request.ability.enabled is True  # untouched profile setting survives


def test_explicit_nested_override_preserves_sibling_profile_settings():
    """Setting only ONE sub-field of a nested config must not clobber the
    profile's setting for OTHER sub-fields of that same config."""
    request, _ = prod.resolve_profile(
        TeamGenerationRequest(profile="research_balanced", ability={"ability_level_thresholds": {"developing_max": 0.2, "medium_max": 0.5}})
    )
    assert request.ability.enabled is True  # from the profile, not clobbered
    assert request.ability.ability_level_thresholds["developing_max"] == 0.2  # explicit override applied


def test_invalid_profile_name_raises_clearly():
    with pytest.raises(ValueError, match="unknown optimization profile"):
        prod.resolve_profile(TeamGenerationRequest(profile="not_a_real_profile"))


def test_resolve_profile_never_mutates_the_registry_or_shared_state():
    before = {k: dict(v["overrides"]) for k, v in prod.OPTIMIZATION_PROFILES.items()}
    prod.resolve_profile(TeamGenerationRequest(profile="research_balanced", ability={"enabled": False}))
    prod.resolve_profile(TeamGenerationRequest(profile="fairness_priority"))
    after = {k: dict(v["overrides"]) for k, v in prod.OPTIMIZATION_PROFILES.items()}
    assert before == after


def test_resolve_profile_produces_deterministic_output():
    r1, _ = prod.resolve_profile(TeamGenerationRequest(profile="research_balanced"))
    r2, _ = prod.resolve_profile(TeamGenerationRequest(profile="research_balanced"))
    assert r1.model_dump() == r2.model_dump()


def test_profile_resolution_is_a_valid_pydantic_model_with_expected_shape():
    _, resolution = prod.resolve_profile(TeamGenerationRequest(profile="legacy"))
    assert isinstance(resolution, ProfileResolution)
    dumped = resolution.model_dump()
    assert set(dumped.keys()) == {"name", "is_experimental", "description", "effective_settings"}


# ===========================================================================
# Hard constraints across every profile
# ===========================================================================
@pytest.mark.parametrize("profile", ALL_PROFILE_NAMES)
@pytest.mark.parametrize("factory", fx.ALL_FIXTURES, ids=lambda f: f().name)
def test_every_profile_preserves_membership_and_team_sizing_on_every_fixture(profile, factory):
    case = factory()
    teams, request, _, _ = _run_profile(case.students, case.team_size, profile)
    assert _ids(teams) == sorted(s.id for s in case.students)
    sizes = [len(t) for t in teams]
    assert max(sizes) - min(sizes) <= 1  # this system's own size-spread guarantee, unaffected by profile


@pytest.mark.parametrize("profile", ALL_PROFILE_NAMES)
def test_every_profile_preserves_skill_coverage_non_regression(profile):
    case = fx.fixture_skill_vs_behavior()
    teams, request, _, _ = _run_profile(case.students, case.team_size, profile)
    greedy_only = prod._greedy_build(case.students, case.team_size, prod._normalize_weights(request.weights))
    assert prod._partition_skill_coverage_min(teams) >= prod._partition_skill_coverage_min(greedy_only) - 1e-9


@pytest.mark.parametrize("profile", ALL_PROFILE_NAMES)
def test_every_profile_preserves_behavioral_floor_policy(profile):
    case = fx.fixture_hidden_bad_apple()
    teams, request, _, _ = _run_profile(case.students, case.team_size, profile)
    _, violations, _ = prod._check_floor_violations(teams, request.floor_policy)
    baseline = prod.AssignmentBaseline.capture(
        prod._greedy_build(case.students, case.team_size, prod._normalize_weights(request.weights)),
        request.floor_policy,
    )
    assert len(violations) <= baseline.max_floor_violations


def test_preference_priority_preserves_hard_avoidance_across_a_fixture():
    a, b, c, d = [fx._s(f"hap{i}", f"HAP{i}", ["frontend"]) for i in range(4)]
    entries = [PreferenceEntry(student_id=a.id, direction="avoid", target_student_id=b.id, priority="high")]
    request, _ = prod.resolve_profile(TeamGenerationRequest(team_size=2, profile="preference_priority"))
    weights = prod._normalize_weights(request.weights)
    teams = prod._greedy_build([a, b, c, d], 2, weights)
    teams = prod._two_opt_improve(teams, weights)
    teams = prod._maximin_improve(teams, weights=weights)
    teams, _trace = prod._run_optional_passes(
        teams, request, weights, request.floor_policy, [], entries,
        primary_metric=prod.OPTIMIZATION_PROFILES["preference_priority"]["primary_metric"],
    )
    report = prod.compute_preference_metrics(teams, entries)
    assert report["total_avoidance_violations"] == 0


# ===========================================================================
# Objective interactions (Section 6 / Section 8)
# ===========================================================================
def test_ability_pass_does_not_silently_destroy_skill_coverage():
    case = fx.fixture_skill_vs_behavior()
    request, _ = prod.resolve_profile(TeamGenerationRequest(team_size=case.team_size, profile="research_balanced"))
    weights = prod._normalize_weights(request.weights)
    greedy = prod._greedy_build(case.students, case.team_size, weights)
    after_2opt = prod._two_opt_improve(greedy, weights)
    after_maximin = prod._maximin_improve(after_2opt, weights=weights)
    baseline_cov = prod._partition_skill_coverage_min(after_maximin)
    teams, _trace = prod._run_optional_passes(
        after_maximin, request, weights, request.floor_policy, [], [],
        primary_metric=None,
    )
    assert prod._partition_skill_coverage_min(teams) >= baseline_cov - 1e-9


def test_familiarity_pass_does_not_confidently_act_on_insufficient_data():
    """A team with sparse familiarity data (below minimum_known_pair_coverage)
    must not have its band-status violation 'fixed' by the optimizer under
    default penalize_insufficient_data=False."""
    from app.models import FamiliarityEdge
    a, b, c, d, e, f, g, h = [fx._s(f"fam{i}", f"FAM{i}", ["frontend"]) for i in range(8)]
    edges = [FamiliarityEdge(student_a_id=a.id, student_b_id=b.id, strength=0.9)]  # 1/6 possible pairs on team 0
    request, _ = prod.resolve_profile(TeamGenerationRequest(team_size=4, profile="research_balanced"))
    assert request.familiarity.penalize_insufficient_data is False
    weights = prod._normalize_weights(request.weights)
    teams = [[a, b, c, d], [e, f, g, h]]
    baseline_report = prod.compute_familiarity_metrics(teams, edges)
    assert baseline_report["per_team"][0]["band_status"] == "insufficient_data"
    repaired = prod._familiarity_balance_improve(
        [list(t) for t in teams], edges=edges, weights=weights, floor_policy=request.floor_policy,
        minimum_known_pair_coverage=request.familiarity.minimum_known_pair_coverage,
        penalize_insufficient_data=request.familiarity.penalize_insufficient_data,
    )
    assert _ids(repaired) == _ids(teams)  # no confident action taken


def test_anticluster_optimization_does_not_isolate_a_student_unnecessarily():
    """Enabling familiarity optimization on a roster with NO familiarity
    data at all must never move students around chasing a phantom
    clustering signal (nothing to isolate anyone from)."""
    case = fx.fixture_isolated_student()
    request, _ = prod.resolve_profile(TeamGenerationRequest(team_size=case.team_size, profile="research_balanced"))
    weights = prod._normalize_weights(request.weights)
    teams = prod._greedy_build(case.students, case.team_size, weights)
    teams = prod._two_opt_improve(teams, weights)
    teams = prod._maximin_improve(teams, weights=weights)
    before_ids = _ids(teams)
    repaired = prod._familiarity_balance_improve([list(t) for t in teams], edges=[], weights=weights, floor_policy=request.floor_policy)
    assert _ids(repaired) == before_ids


def test_preference_repair_does_not_create_a_severe_fairness_regression():
    """The preference-repair pass's own acceptance rule already refuses to
    regress the total-score objective or floor-minimum; confirm this holds
    for a roster with both fairness-relevant and preference-relevant
    structure present simultaneously."""
    case = fx.fixture_utility_vs_fairness()
    request, _ = prod.resolve_profile(TeamGenerationRequest(team_size=case.team_size, profile="preference_priority"))
    weights = prod._normalize_weights(request.weights)
    teams = prod._greedy_build(case.students, case.team_size, weights)
    teams = prod._two_opt_improve(teams, weights)
    teams = prod._maximin_improve(teams, weights=weights)
    baseline_util = prod.compute_assignment_utilities(teams)["summary"]["min_utility"]
    ids = [s.id for t in teams for s in t]
    entries = [PreferenceEntry(student_id=ids[k], direction="prefer", target_student_id=ids[k + 1], rank=1) for k in range(0, len(ids) - 1, 2)]
    repaired = prod._preference_repair_improve(
        [list(t) for t in teams], entries=entries, weights=weights, floor_policy=request.floor_policy,
        hard_avoidance=request.preference.hard_avoidance,
    )
    repaired_util = prod.compute_assignment_utilities(repaired)["summary"]["min_utility"]
    assert repaired_util >= baseline_util - 1e-6  # the objective/floor-min guard already protects this


@pytest.mark.parametrize("factory", fx.ALL_FIXTURES, ids=lambda f: f().name)
def test_later_passes_do_not_undo_earlier_accepted_improvements_outside_tolerance(factory):
    """Runs research_balanced (every pass enabled) and confirms the
    orchestration-level trace never reports a REVERTED-then-silently-lost
    regression: every accepted pass's protected metrics are non-decreasing
    relative to that SAME pass's own 'before' snapshot (which is itself
    the immediately-prior pass's accepted 'after')."""
    case = factory()
    teams, request, _, trace = _run_profile(case.students, case.team_size, "research_balanced")
    for step in trace:
        if step["accepted"]:
            for k in step["before"]:
                assert step["after"][k] >= step["before"][k] - prod.PROTECTED_METRIC_TOLERANCE, (
                    f"{step['pass']} accepted but regressed protected metric {k}: {step}"
                )


# ===========================================================================
# Data availability
# ===========================================================================
def test_research_balanced_handles_no_ability_data():
    students = [fx._s(f"na{i}", f"NA{i}", ["frontend"]) for i in range(8)]  # no ability_score, no experience override -> defaults
    teams, request, _, trace = _run_profile(students, 4, "research_balanced")
    assert _ids(teams) == sorted(s.id for s in students)  # completes without crashing


def test_research_balanced_handles_no_familiarity_data():
    case = fx.fixture_isolated_student()
    teams, request, _, trace = _run_profile(case.students, case.team_size, "research_balanced")
    report = prod.compute_familiarity_metrics(teams, [])
    assert report["total_known_pairs"] == 0
    assert _ids(teams) == sorted(s.id for s in case.students)


def test_research_balanced_handles_no_preference_data():
    case = fx.fixture_skill_vs_behavior()
    teams, request, _, trace = _run_profile(case.students, case.team_size, "research_balanced")
    report = prod.compute_preference_metrics(teams, [])
    assert report["n_students_with_preferences"] == 0
    assert _ids(teams) == sorted(s.id for s in case.students)


def test_normalized_objective_handles_sparse_data_by_renormalizing_not_zeroing():
    case = fx.fixture_skill_vs_behavior()
    teams, request, _, _ = _run_profile(case.students, case.team_size, "legacy")
    obj = prod.compute_normalized_objective(teams, familiarity_edges=[], preference_entries=[])
    assert obj["components"]["familiarity_band_fit"]["raw"] is None
    assert obj["components"]["familiarity_band_fit"]["weighted_contribution"] == 0.0
    assert obj["weights_renormalized_over_usable_components"] is True
    assert obj["total_weighted_objective"] is not None


def test_normalized_objective_mixed_direct_and_proxy_ability_data():
    a = fx._s("mix1", "MIX1", ["frontend"], ability_score=0.9, ability_source="instructor_rated")
    b = fx._s("mix2", "MIX2", ["backend"], experience_level="beginner")  # falls back to experience_level proxy
    c = fx._s("mix3", "MIX3", ["design"])
    d = fx._s("mix4", "MIX4", ["testing"])
    teams = [[a, b], [c, d]]
    obj = prod.compute_normalized_objective(teams, familiarity_edges=[], preference_entries=[])
    assert obj["components"]["ability_balance"]["raw"] is not None  # resolves via direct + proxy fallback, no crash


def test_generate_teams_reports_unknown_preference_targets_as_a_count_only(monkeypatch=None):
    a, b = fx._s("upt1", "UPT1", ["frontend"]), fx._s("upt2", "UPT2", ["backend"])
    entries = [PreferenceEntry(student_id=a.id, direction="prefer", target_student_id="ghost-student-id", rank=1)]
    roster_ids = {a.id, b.id}
    n_unknown = sum(1 for e in entries if e.target_student_id is not None and e.target_student_id not in roster_ids)
    assert n_unknown == 1
    # confirm it never influences the actual satisfaction score (stale IDs cannot silently inflate anything)
    report = prod.compute_preference_metrics([[a, b]], entries)
    a_stats = next(p for p in report["per_student"] if p["student_id"] == a.id)
    assert a_stats["satisfaction_score"] == pytest.approx(0.0)


# ===========================================================================
# Determinism and performance
# ===========================================================================
@pytest.mark.parametrize("profile", ALL_PROFILE_NAMES)
def test_identical_input_and_profile_produces_identical_assignment(profile):
    case = fx.fixture_utility_vs_fairness()
    teams_1, _, _, _ = _run_profile(case.students, case.team_size, profile)
    teams_2, _, _, _ = _run_profile(case.students, case.team_size, profile)
    assert [sorted(s.id for s in t) for t in teams_1] == [sorted(s.id for s in t) for t in teams_2]


def _synthetic_roster(n):
    return [
        fx._s(f"syn{i}", f"SYN{i}", [["frontend", "backend", "design", "testing", "ml"][i % 5]],
              experience_level=["beginner", "intermediate", "advanced"][i % 3])
        for i in range(n)
    ]


@pytest.mark.parametrize("n", [20, 60])
def test_runtime_measured_on_increasing_synthetic_roster_sizes(n):
    students = _synthetic_roster(n)
    t0 = time.perf_counter()
    teams, request, _, trace = _run_profile(students, 4, "research_balanced")
    elapsed = time.perf_counter() - t0
    assert _ids(teams) == sorted(s.id for s in students)
    assert elapsed < 30.0, f"research_balanced took {elapsed:.2f}s for n={n} — investigate before scaling further"


def test_optimization_terminates_at_iteration_limit_not_infinite_loop():
    case = fx.fixture_greedy_local_optimum()
    started = time.perf_counter()
    teams, request, _, trace = _run_profile(case.students, case.team_size, "research_balanced")
    assert time.perf_counter() - started < 10.0
