"""
Tests for the L4 deterministic scorer and MCQ adapter.

Run with:
    cd backend && venv/bin/pytest tests/test_dimension_scorer.py -v
"""
from __future__ import annotations

import pytest

from scoring.dimension_scorer import SignalUnit, score, get_rubric
from scoring.mcq_adapter import responses_to_signals, get_scenarios_for_client


# ---------------------------------------------------------------------------
# Minimal inline rubric for unit tests — avoids dependency on the YAML file
# for the core scoring logic tests
# ---------------------------------------------------------------------------
_MINI_RUBRIC = {
    "schema_version": "test",
    "dimensions": {
        "conflict_style": {
            "type": "categorical",
            "group": "collaboration",
            "composition_rule": "soft_compatibility",
            "allowed_values": ["avoidant", "assertive", "collaborative"],
            "default_value": "collaborative",
            "matching_weight": 0.0,
        },
        "accountability": {
            "type": "ordinal",
            "group": "collaboration",
            "composition_rule": "high_mean_with_floor",
            "allowed_values": ["low", "developing", "high"],
            "default_value": "developing",
            "matching_weight": 0.35,
        },
        "cooperativeness": {
            "type": "ordinal",
            "group": "collaboration",
            "composition_rule": "high_mean_with_floor",
            "allowed_values": ["low", "developing", "high"],
            "default_value": "developing",
            "matching_weight": 0.35,
        },
        "confidence_score": {
            "type": "derived",
            "group": "confidence",
            "matching_weight": 0.0,
        },
    },
}


# ===========================================================================
# 1. DETERMINISM — exact same input always yields exact same output
# ===========================================================================

def test_scorer_determinism():
    signals = [
        SignalUnit("conflict_style", "collaborative", 1.0),
        SignalUnit("accountability", "high", 1.0),
        SignalUnit("cooperativeness", "developing", 1.0),
    ]
    result_a = score(signals, _MINI_RUBRIC)
    result_b = score(signals, _MINI_RUBRIC)
    assert result_a == result_b, "scorer is non-deterministic"


def test_scorer_determinism_different_order():
    """Signal order must not change the output for the same set."""
    sig_ab = [SignalUnit("conflict_style", "collaborative"), SignalUnit("accountability", "high")]
    sig_ba = [SignalUnit("accountability", "high"), SignalUnit("conflict_style", "collaborative")]
    # Both should yield the same labels (order of signals within a trait doesn't matter)
    r_ab = score(sig_ab, _MINI_RUBRIC)
    r_ba = score(sig_ba, _MINI_RUBRIC)
    assert r_ab["conflict_style"] == r_ba["conflict_style"]
    assert r_ab["accountability"] == r_ba["accountability"]


# ===========================================================================
# 2. CATEGORICAL — plurality vote
# ===========================================================================

def test_categorical_plurality_winner():
    signals = [
        SignalUnit("conflict_style", "collaborative"),
        SignalUnit("conflict_style", "collaborative"),
        SignalUnit("conflict_style", "assertive"),
    ]
    result = score(signals, _MINI_RUBRIC)
    assert result["conflict_style"] == "collaborative"


def test_categorical_single_signal():
    signals = [SignalUnit("conflict_style", "avoidant")]
    result = score(signals, _MINI_RUBRIC)
    assert result["conflict_style"] == "avoidant"


def test_categorical_weighted_vote():
    signals = [
        SignalUnit("conflict_style", "assertive", weight=3.0),
        SignalUnit("conflict_style", "collaborative", weight=1.0),
        SignalUnit("conflict_style", "collaborative", weight=1.0),
    ]
    result = score(signals, _MINI_RUBRIC)
    assert result["conflict_style"] == "assertive"


# ===========================================================================
# 3. ORDINAL — weighted mean → nearest level
# ===========================================================================

def test_ordinal_all_high():
    signals = [SignalUnit("accountability", "high"), SignalUnit("accountability", "high")]
    assert score(signals, _MINI_RUBRIC)["accountability"] == "high"


def test_ordinal_all_low():
    signals = [SignalUnit("accountability", "low"), SignalUnit("accountability", "low")]
    assert score(signals, _MINI_RUBRIC)["accountability"] == "low"


def test_ordinal_weighted_mean_rounds_correctly():
    # high=2, low=0 → weight 2:1 → mean = (2*2 + 1*0)/3 = 1.33 → rounds to 1 → "developing"
    signals = [
        SignalUnit("accountability", "high", weight=2.0),
        SignalUnit("accountability", "low", weight=1.0),
    ]
    assert score(signals, _MINI_RUBRIC)["accountability"] == "developing"


def test_ordinal_equal_high_low_rounds_to_mid():
    # high=2, low=0 → equal weight → mean=1 → "developing"
    signals = [
        SignalUnit("accountability", "high"),
        SignalUnit("accountability", "low"),
    ]
    assert score(signals, _MINI_RUBRIC)["accountability"] == "developing"


# ===========================================================================
# 4. UNKNOWN / missing dimension tracking
# ===========================================================================

def test_unknown_dim_listed_when_no_signals():
    signals = [SignalUnit("conflict_style", "collaborative")]
    result = score(signals, _MINI_RUBRIC)
    # accountability and cooperativeness have no signals
    assert "accountability" in result["_unknown_dims"]
    assert "cooperativeness" in result["_unknown_dims"]
    assert "conflict_style" not in result["_unknown_dims"]


def test_known_dim_not_listed_as_unknown():
    signals = [SignalUnit("accountability", "high")]
    result = score(signals, _MINI_RUBRIC)
    assert "accountability" not in result["_unknown_dims"]


def test_confidence_score_is_mean_coverage():
    # All three non-derived dims have signals → full coverage
    signals = [
        SignalUnit("conflict_style", "collaborative"),
        SignalUnit("accountability", "high"),
        SignalUnit("cooperativeness", "high"),
    ]
    result = score(signals, _MINI_RUBRIC)
    assert result["confidence_score"] > 0.0
    assert result["confidence_score"] <= 1.0


def test_confidence_score_zero_when_no_signals():
    result = score([], _MINI_RUBRIC)
    assert result["confidence_score"] == 0.0


# ===========================================================================
# 5. MCQ ADAPTER
# ===========================================================================

class _FakeResponse:
    def __init__(self, scenario_id, option_id):
        self.scenario_id = scenario_id
        self.option_id = option_id
        self.response_text = ""


def test_mcq_adapter_emits_signals_for_known_option():
    responses = [_FakeResponse("missed_deadline", "opt_c")]  # collaborative + cooperativeness=high
    signals = responses_to_signals(responses)
    traits = {s.trait for s in signals}
    assert "conflict_style" in traits
    assert "cooperativeness" in traits
    cstyle = next(s for s in signals if s.trait == "conflict_style")
    assert cstyle.value == "collaborative"


def test_mcq_adapter_unknown_scenario_skipped():
    responses = [_FakeResponse("nonexistent_scenario", "opt_a")]
    signals = responses_to_signals(responses)
    assert signals == []


def test_mcq_adapter_unknown_option_skipped():
    responses = [_FakeResponse("missed_deadline", "opt_z")]
    signals = responses_to_signals(responses)
    assert signals == []


def test_mcq_adapter_old_internal_id_skipped():
    # The old internal IDs ("collaborative", "confrontational") are no longer
    # valid anonymous IDs — they must not produce signals.
    responses = [_FakeResponse("missed_deadline", "collaborative")]
    signals = responses_to_signals(responses)
    assert signals == []


def test_get_scenarios_for_client_anonymises_option_ids():
    scenarios = get_scenarios_for_client()
    assert len(scenarios) == 6
    for s in scenarios:
        assert "id" in s and "title" in s and "description" in s
        assert len(s["options"]) == 4
        for opt in s["options"]:
            assert opt["id"].startswith("opt_")
            # Must not contain any raw trait value as option ID
            assert opt["id"] not in (
                "avoidant", "assertive", "collaborative",
                "directive", "facilitative", "emergent",
                "low", "developing", "high", "confrontational",
            )


# ===========================================================================
# 6. BAD-APPLE PREVENTION — tested through the matching module
#    (lives in test_assessment.py; imported here for discoverability)
# ===========================================================================

def test_scorer_does_not_impute_defaults_as_evidence():
    """
    A student with no signals for a dimension must have that dimension
    in _unknown_dims — the default_value must NOT appear as a real signal.
    """
    signals = [SignalUnit("conflict_style", "collaborative")]
    result = score(signals, _MINI_RUBRIC)
    # accountability has no signal → must be unknown, even though default_value = "developing"
    assert "accountability" in result["_unknown_dims"]
    # The value field is populated with default for display but not trusted:
    assert result["accountability"] == _MINI_RUBRIC["dimensions"]["accountability"]["default_value"]


# ===========================================================================
# 7. REAL RUBRIC integration smoke test
# ===========================================================================

def test_score_with_real_rubric_returns_all_dimensions():
    rubric = get_rubric()
    signals = [
        SignalUnit("conflict_style", "collaborative"),
        SignalUnit("accountability", "high"),
        SignalUnit("cooperativeness", "high"),
        SignalUnit("planning_style", "planner"),
        SignalUnit("leadership_style", "facilitative"),
        SignalUnit("communication_style", "async"),
        SignalUnit("execution_style", "methodical"),
        SignalUnit("help_seeking", "proactive"),
    ]
    result = score(signals, rubric)
    assert result["conflict_style"] == "collaborative"
    assert result["accountability"] == "high"
    assert result["cooperativeness"] == "high"
    assert isinstance(result["confidence_score"], float)
    assert result["_unknown_dims"] == []
