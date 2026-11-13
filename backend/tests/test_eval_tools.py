"""Tests for the offline eval tools: order independence, cohort generator, quality eval."""
import hashlib
import json
import random
from pathlib import Path

import pytest

from app import matching as prod
from app.models import (
    CollaborationSignature,
    CompetenceSignature,
    Student,
    TeamGenerationRequest,
)
from eval import quality_eval as qe
from eval.cohort import record_to_student
from eval.generate_cohort import generate_cohort, to_json

BACKEND = Path(__file__).resolve().parents[1]


def _canonical(teams):
    return tuple(sorted(tuple(sorted(s.id for s in t)) for t in teams))


def _sample_roster():
    with open(BACKEND / "sample_students.json") as f:
        return [Student(**raw) for raw in json.load(f)]


def _synthetic_roster(n=24, seed=7):
    return [record_to_student(r) for r in generate_cohort(n, seed, low_scorer_fraction=0.15)["students"]]


# ---------------------------------------------------------------------------
# Team formation does not depend on input order
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("roster", [_sample_roster, _synthetic_roster], ids=["sample", "synthetic24"])
def test_form_teams_is_input_order_independent(roster):
    students = roster()
    partitions = set()
    for k in range(30):
        order = list(students)
        random.Random(k).shuffle(order)
        teams, _ = prod.form_teams(order, TeamGenerationRequest(team_size=4), [], [])
        partitions.add(_canonical(teams))
    assert len(partitions) == 1


# ---------------------------------------------------------------------------
# Cohort generator
# ---------------------------------------------------------------------------

# Hashes of the generator's output before --low-mode existed.
@pytest.mark.parametrize("kwargs, expected", [
    (dict(n=24, seed=7), "93ecd6285e4cb2a079b4bc7efeec968ebb9a9ef5d466e4d5a1598c57db14e1c3"),
    (dict(n=40, seed=3, low_scorer_fraction=0.15), "978836c25db11a20e6405839d0858396126db70b42720288ca4315ebb2fc7ce1"),
    (dict(n=22, seed=11, low_scorer_fraction=0.25, with_skills=False),
     "8dd46af08f07c13acc8818294e06a01408395ac2e82602332b7cd37e84810ef8"),
])
def test_default_cohort_output_is_unchanged(kwargs, expected):
    assert hashlib.sha256(to_json(generate_cohort(**kwargs)).encode()).hexdigest() == expected


def test_partial_low_mode_is_deterministic():
    a = to_json(generate_cohort(40, 3, low_scorer_fraction=0.25, low_mode="partial"))
    b = to_json(generate_cohort(40, 3, low_scorer_fraction=0.25, low_mode="partial"))
    assert a == b


def test_partial_low_scorers_are_low_on_one_or_two_ordinal_traits():
    cohort = generate_cohort(80, 5, low_scorer_fraction=0.25, low_mode="partial")
    ordinal = ("accountability", "cooperativeness", "safety_contribution")
    low = [s for s in cohort["students"] if s["low_scorer"]]
    assert low
    for s in low:
        at_floor = sorted(t for t in ordinal if s["traits"][t] == "low")
        assert at_floor == s["low_traits"]
        assert 1 <= len(at_floor) <= 2
    for s in cohort["students"]:
        if not s["low_scorer"]:
            assert all(s["traits"][t] != "low" for t in ordinal)


# ---------------------------------------------------------------------------
# Quality metrics on hand-made teams
# ---------------------------------------------------------------------------

def _student(sid, acc, coop, skills):
    return Student(
        id=sid, name=sid,
        competence_signature=CompetenceSignature(skills=skills),
        collaboration_signature=CollaborationSignature(accountability=acc, cooperativeness=coop),
    )


def test_team_metrics_on_known_teams():
    ref = sorted(prod.REFERENCE_SKILLS)
    team_a = [
        _student("a1", "low", "high", [ref[0]]),
        _student("a2", "high", "high", []),
        _student("a3", "developing", "developing", []),
        _student("a4", "high", "low", []),
    ]
    team_b = [_student(f"b{i}", "high", "high", ref[i::4]) for i in range(4)]

    m = qe.team_metrics([team_a, team_b])

    assert m["teams_with_floor_student"] == 1
    assert m["max_floor_students_on_one_team"] == 2
    assert m["max_same_trait_floor_on_one_team"] == 1
    assert m["accountability_min_team_mean"] == pytest.approx(1.25)
    assert m["accountability_mean_of_team_means"] == pytest.approx(1.625)
    assert m["cooperativeness_min_team_mean"] == pytest.approx(1.25)
    assert m["cooperativeness_mean_of_team_means"] == pytest.approx(1.625)
    assert m["mean_coverage"] == pytest.approx((1 / len(ref) + 1) / 2)
    assert m["worst_team_score"] == pytest.approx(
        min(prod._total_score(t, prod._DEFAULT_WEIGHTS) for t in (team_a, team_b)))


def test_floor_student_count_ignores_non_floor_traits():
    students = [
        _student("f1", "low", "high", []),
        _student("f2", "high", "low", []),
        _student("f3", "high", "high", []),
    ]
    assert [qe.is_floor_student(s) for s in students] == [True, True, False]


def test_cohort_reports_actual_floor_students():
    out = qe.run_cohort((0.25, 40, 3))
    cohort = generate_cohort(40, 3, low_scorer_fraction=0.25, low_mode="partial")
    expected = sum(
        1 for s in cohort["students"]
        if s["traits"]["accountability"] == "low" or s["traits"]["cooperativeness"] == "low"
    )
    assert out["n_floor_students"] == expected
    assert expected < cohort["n_low_scorers"]


def test_sort_and_chunk_puts_weakest_together():
    students = [_student(f"s{i}", lvl, lvl, []) for i, lvl in
                enumerate(["low", "low", "high", "high", "developing", "developing", "high", "high"])]
    teams = qe.form_sort_and_chunk(students, "k")
    assert [s.id for s in teams[0]] == ["s0", "s1", "s4", "s5"]


def test_behavior_only_really_has_zero_skill_weight():
    assert qe.effective_weights("behavior_only")["skill_coverage"] == 0.0
    assert qe.effective_weights("ours")["skill_coverage"] > 0.0


def test_floor_check_is_restored_after_no_floor_method():
    original = prod._is_floor_level
    qe.form_no_floor(_synthetic_roster(), "k")
    assert prod._is_floor_level is original


# ---------------------------------------------------------------------------
# Quality eval is deterministic
# ---------------------------------------------------------------------------

def test_quality_eval_output_is_identical_across_runs(tmp_path):
    qe.write_outputs(qe.evaluate([8, 10], seeds=2, workers=1), tmp_path / "a")
    qe.write_outputs(qe.evaluate([8, 10], seeds=2, workers=1), tmp_path / "b")
    for name in ("quality_results.json", "quality_table.csv", "quality_table.md"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
