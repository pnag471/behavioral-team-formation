"""
Tests for the interview-to-signal adapter (scoring/interview_adapter.py).

Uses the real rubric (get_rubric()), so a rubric change that breaks the
adapter's assumptions shows up here.

Run with:
    cd backend && python3 -m pytest tests/test_interview_adapter.py -q
"""
from __future__ import annotations

import random
from dataclasses import replace

import pytest

from scoring.dimension_scorer import SignalUnit, get_rubric, score
from scoring.interview_adapter import (
    MAX_ITEMS_PER_INDICATOR,
    EvidenceItem,
    evidence_to_signals,
    floor_dimensions,
    normalize_text,
)

RUBRIC = get_rubric()

# Student turns only (no interviewer text). Curly quotes and odd whitespace on
# purpose, to exercise the normalization.
STUDENT_TEXT = (
    "Last spring I made a shared calendar for our group the first day.\n"
    "I always   reply to the group chat within an hour.\n"
    "When we disagreed about the database, I told them straight out I didn’t agree.\n"
    "I usually just start coding and see what works.\n"
    "Honestly I missed the deadline once and didn't tell anyone.\n"
    "I check in with everyone before our weekly meeting.\n"
    "I check in with everyone before our weekly meeting, every single week.\n"
    "I also review my teammates' pull requests the same day.\n"
    "I helped my teammate debug her code at midnight.\n"
)


def item(quote, dimension, indicator, polarity="supporting", strength="medium", confidence=0.8):
    return EvidenceItem(quote=quote, dimension=dimension, indicator=indicator,
                        polarity=polarity, strength=strength, confidence=confidence)


def run(items, text=STUDENT_TEXT, **kw):
    return evidence_to_signals(items, text, RUBRIC, **kw)


def only_decision(decisions):
    assert len(decisions) == 1
    return decisions[0]


# A realistic mixed batch used by the invariance and ordering tests.
MIXED = [
    item("I made a shared calendar for our group the first day", "planning_style", "planner", strength="strong"),
    item("I always reply to the group chat within an hour", "communication_style", "sync"),
    item("I told them straight out I didn't agree", "conflict_style", "assertive", strength="strong"),
    item("I usually just start coding and see what works", "execution_style", "exploratory"),
    item("I usually just start coding", "planning_style", "spontaneous", strength="weak"),
    item("I missed the deadline once and didn't tell anyone", "accountability", "low", strength="strong"),
    item("I check in with everyone before our weekly meeting", "cooperativeness", "high"),
    item("I also review my teammates' pull requests the same day", "cooperativeness", "high", strength="strong"),
    item("I helped my teammate debug her code at midnight", "cooperativeness", "high", strength="strong"),
    item("I always reply to the group chat within an hour", "accountability", "high", polarity="contradicting"),
    item("this quote is not in the transcript", "leadership_style", "directive"),
    item("I made a shared calendar", "planning_style", "chaotic"),
]


# ===========================================================================
# 1. Each rejection / exclusion reason fires
# ===========================================================================

@pytest.mark.parametrize("bad, reason", [
    (item("I made a shared calendar", "vibes", "planner"), "unknown_dimension"),
    (item("I made a shared calendar", "confidence_score", "planner"), "unknown_dimension"),  # derived
    (item("I made a shared calendar", "conflict_style", "confrontational"), "indicator_not_allowed"),
    (item("I made a shared calendar", "accountability", "medium"), "indicator_not_allowed"),
    (item("I made a shared calendar", "planning_style", "planner", strength="very_strong"), "unknown_strength"),
    (item("I made a shared calendar", "planning_style", "planner", polarity="neutral"), "unknown_polarity"),
    (item("I made a detailed Gantt chart", "planning_style", "planner"), "not_verbatim"),
    (item("   ", "planning_style", "planner"), "not_verbatim"),
])
def test_each_rejection_reason(bad, reason):
    signals, decisions = run([bad])
    d = only_decision(decisions)
    assert signals == []
    assert d.status == "rejected"
    assert d.reason == reason
    assert d.decision == f"rejected:{reason}"
    assert d.needs_review is True
    assert d.source_ref is None


def test_interviewer_text_is_not_verbatim():
    """Only student turns count: a quote that exists only in the interviewer's text is rejected."""
    interviewer_line = "Tell me about a time your group missed a deadline"
    signals, decisions = run([item(interviewer_line, "accountability", "low")])
    assert signals == []
    assert only_decision(decisions).reason == "not_verbatim"


def test_verbatim_normalizes_case_whitespace_and_quotes():
    # Transcript has a curly apostrophe in "didn’t" and a run of spaces in "always   reply".
    quotes = [
        "i TOLD them straight out i didn't agree",
        "I always reply to the group chat",
        "I always\n\treply   to the group chat",
    ]
    for q in quotes:
        _, decisions = run([item(q, "conflict_style", "assertive")])
        assert only_decision(decisions).status == "accepted", q


def test_verbatim_does_not_strip_punctuation_or_fuzzy_match():
    # Missing word -> not a substring -> rejected (no fuzzy matching).
    _, decisions = run([item("I told them out I didn't agree", "conflict_style", "assertive")])
    assert only_decision(decisions).reason == "not_verbatim"


def test_contradicting_is_excluded_not_signal():
    signals, decisions = run([item("I made a shared calendar", "planning_style", "planner",
                                   polarity="contradicting")])
    d = only_decision(decisions)
    assert signals == []
    assert (d.status, d.reason) == ("excluded", "contradicting")
    # planning_style is not a floor dimension -> no review flag
    assert d.needs_review is False


def test_contradicting_on_floor_dimension_needs_review():
    floors = floor_dimensions(RUBRIC)
    assert floors == ["accountability", "cooperativeness"]  # derived from the rubric, not hardcoded in the adapter
    for dim in floors:
        _, decisions = run([item("I always reply to the group chat", dim, "high", polarity="contradicting")])
        d = only_decision(decisions)
        assert (d.status, d.reason, d.needs_review) == ("excluded", "contradicting", True)


def test_validation_runs_before_contradiction_check():
    _, decisions = run([item("not in transcript", "accountability", "high", polarity="contradicting")])
    assert only_decision(decisions).decision == "rejected:not_verbatim"


def test_accepted_item_shape():
    signals, decisions = run([item("I made a shared calendar", "planning_style", "planner")])
    d = only_decision(decisions)
    assert d.status == "accepted" and d.reason is None and d.decision == "accepted"
    assert d.needs_review is False
    expected_start = normalize_text(STUDENT_TEXT).find("i made a shared calendar")
    assert d.quote_start == expected_start
    assert signals == [SignalUnit(trait="planning_style", value="planner", weight=1.0,
                                  source_ref=f"planning_style:planner:{expected_start}")]
    assert d.source_ref == signals[0].source_ref


# ===========================================================================
# 2. Strength -> weight comes from the rubric
# ===========================================================================

def test_rubric_strength_weights_are_the_provisional_values():
    assert {k: v["weight"] for k, v in RUBRIC["evidence_strength"].items()} == {
        "strong": 1.5, "medium": 1.0, "weak": 0.5,
    }


@pytest.mark.parametrize("strength", ["strong", "medium", "weak"])
def test_strength_maps_to_rubric_weight(strength):
    signals, _ = run([item("I made a shared calendar", "planning_style", "planner", strength=strength)])
    assert len(signals) == 1
    assert signals[0].weight == RUBRIC["evidence_strength"][strength]["weight"]


# ===========================================================================
# 3. Confidence never affects signals or scores
# ===========================================================================

def test_confidence_invariance():
    low = [replace(i, confidence=0.0) for i in MIXED]
    high = [replace(i, confidence=1.0) for i in MIXED]
    none = [replace(i, confidence=None) for i in MIXED]

    s_low, d_low = run(low)
    s_high, d_high = run(high)
    s_none, _ = run(none)

    assert s_low == s_high == s_none
    assert score(s_low, RUBRIC) == score(s_high, RUBRIC)
    # Same statuses, reasons and review flags (only the stored confidence differs).
    strip = lambda ds: [(d.item.quote, d.item.dimension, d.decision, d.needs_review, d.quote_start)
                        for d in ds]
    assert strip(d_low) == strip(d_high)


def test_confidence_threshold_is_off_by_default_and_only_touches_needs_review():
    items = [replace(i, confidence=0.0) for i in MIXED]
    s_default, d_default = run(items)
    accepted_default = [d for d in d_default if d.status == "accepted"]
    assert accepted_default and not any(d.needs_review for d in accepted_default)

    s_on, d_on = run(items, confidence_review_threshold=0.5)
    assert s_on == s_default  # signals unchanged
    assert all(d.needs_review for d in d_on if d.status == "accepted")


# ===========================================================================
# 4. Deterministic order, independent of LLM output order
# ===========================================================================

def test_shuffled_input_gives_identical_output():
    base_signals, base_decisions = run(MIXED)
    assert base_signals  # sanity: the batch produces signals
    for seed in range(20):
        shuffled = list(MIXED)
        random.Random(seed).shuffle(shuffled)
        signals, decisions = run(shuffled)
        assert signals == base_signals, f"seed {seed}"
        assert decisions == base_decisions, f"seed {seed}"


def test_signals_sorted_by_dimension_then_transcript_position():
    signals, decisions = run(MIXED)
    starts = {d.source_ref: d.quote_start for d in decisions if d.status == "accepted"}
    keys = [(s.trait, starts[s.source_ref], s.value) for s in signals]
    assert keys == sorted(keys)


def test_one_decision_per_input_item():
    _, decisions = run(MIXED)
    assert len(decisions) == len(MIXED)
    assert sorted(map(repr, (d.item for d in decisions))) == sorted(map(repr, MIXED))


def test_mixed_batch_end_to_end_score():
    signals, _ = run(MIXED)
    scored = score(signals, RUBRIC)
    assert scored["planning_style"] == "planner"         # strong planner (1.5) beats weak spontaneous (0.5)
    assert scored["conflict_style"] == "assertive"
    assert scored["accountability"] == "low"             # the contradicting "high" item is excluded
    assert scored["cooperativeness"] == "high"
    assert "leadership_style" in scored["_unknown_dims"]  # only evidence was not verbatim


# ===========================================================================
# 5. Dedupe and cap
# ===========================================================================

def test_substring_duplicate_is_dropped():
    long_q = "I check in with everyone before our weekly meeting, every single week"
    short_q = "I check in with everyone before our weekly meeting"
    signals, decisions = run([item(short_q, "cooperativeness", "high"),
                              item(long_q, "cooperativeness", "high")])
    assert len(signals) == 1
    by_quote = {d.item.quote: d for d in decisions}
    assert by_quote[long_q].status == "accepted"
    assert by_quote[short_q].decision == "excluded:duplicate"
    assert by_quote[short_q].needs_review is False


def test_identical_quote_twice_keeps_one_the_stronger():
    q = "I made a shared calendar"
    signals, decisions = run([item(q, "planning_style", "planner", strength="weak"),
                              item(q, "planning_style", "planner", strength="strong")])
    assert len(signals) == 1 and signals[0].weight == 1.5
    assert sorted(d.decision for d in decisions) == ["accepted", "excluded:duplicate"]


def test_substring_is_not_duplicate_across_indicators_or_dimensions():
    q_long = "I usually just start coding and see what works"
    q_short = "I usually just start coding"
    signals, _ = run([item(q_long, "execution_style", "exploratory"),
                      item(q_short, "planning_style", "spontaneous"),
                      item(q_short, "execution_style", "iterative")])
    assert len(signals) == 3


def test_cap_per_dimension_indicator():
    quotes = [
        "I helped my teammate debug her code at midnight",          # latest in transcript
        "I always reply to the group chat within an hour",          # earliest
        "I also review my teammates' pull requests the same day",
        "I check in with everyone before our weekly meeting, every single week",
    ]
    signals, decisions = run([item(q, "cooperativeness", "high", strength="strong") for q in quotes])
    assert MAX_ITEMS_PER_INDICATOR == 2
    assert len(signals) == MAX_ITEMS_PER_INDICATOR
    accepted = sorted(d.item.quote for d in decisions if d.status == "accepted")
    # Default selection rule: the earliest in the transcript survive.
    assert accepted == sorted([quotes[1], quotes[3]])
    assert sorted(d.decision for d in decisions) == ["accepted", "accepted",
                                                     "excluded:over_cap", "excluded:over_cap"]


def test_cap_is_per_indicator_not_per_dimension():
    q = ["I always reply to the group chat within an hour",
         "I check in with everyone before our weekly meeting, every single week",
         "I also review my teammates' pull requests the same day",
         "I helped my teammate debug her code at midnight"]
    items = [item(q[0], "cooperativeness", "high"), item(q[1], "cooperativeness", "high"),
             item(q[2], "cooperativeness", "developing"), item(q[3], "cooperativeness", "developing")]
    signals, _ = run(items)
    assert len(signals) == 4


def test_contradicting_items_do_not_count_toward_cap():
    q = ["I always reply to the group chat within an hour",
         "I check in with everyone before our weekly meeting, every single week",
         "I also review my teammates' pull requests the same day"]
    items = [item(q[0], "cooperativeness", "high", polarity="contradicting"),
             item(q[1], "cooperativeness", "high"),
             item(q[2], "cooperativeness", "high")]
    signals, _ = run(items)
    assert len(signals) == 2
