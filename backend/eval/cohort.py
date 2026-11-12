"""
Shared cohort helpers: which trait dimensions are real, and how a synthetic
cohort record maps onto the real app.models.Student.

Trait names are NOT hardcoded here. They are read from the single source of
truth, backend/traits/rubric_v1.yaml, via scoring.dimension_scorer.get_rubric()
(the same loader app/matching.py uses).

SHAPE NOTE: real students do NOT carry trait scores in [0, 1]. Every trait
dimension in the rubric is a LABEL drawn from that dimension's
`allowed_values` (e.g. accountability: low | developing | high), stored on
Student.work_rhythm_signature or Student.collaboration_signature according to
the dimension's rubric `group`. The synthetic generator therefore emits labels,
not floats, so the cohort exercises the real matcher exactly as production
data would.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from app.models import (
    CollaborationSignature,
    CompetenceSignature,
    Student,
    WorkRhythmSignature,
)
from scoring.dimension_scorer import get_rubric

# Rubric groups that hold per-student trait labels, mapped to the Student
# sub-object that stores them (mirrors app/matching.py:_get_student_dim).
_GROUP_TO_SIGNATURE = {
    "work_rhythm": "work_rhythm_signature",
    "collaboration": "collaboration_signature",
}


def trait_dimensions() -> Dict[str, dict]:
    """
    Every scoreable trait dimension in rubric_v1.yaml, in rubric file order.

    Excludes `type: derived` dimensions. The only one today is
    confidence_score, which the rubric defines as scorer MEASUREMENT
    confidence (evidence coverage), explicitly "NOT a student personality
    trait", and which lives on Student.confidence_layer, not a signature.

    Includes safety_contribution and stress_response even though the rubric
    marks them "collected, not scored" (composition_rule: null,
    matching_weight: 0.0): they are real dimensions on real students, so a
    realistic cohort carries them. They cannot affect team formation.
    """
    dims = get_rubric()["dimensions"]
    out = {}
    for name, d in dims.items():
        if d.get("type") == "derived":
            continue
        if d.get("group") not in _GROUP_TO_SIGNATURE:
            raise ValueError(
                f"rubric dimension {name!r} has group {d.get('group')!r}; "
                "eval/cohort.py does not know which Student field stores it"
            )
        out[name] = d
    return out


def record_to_student(rec: dict) -> Student:
    """Convert one synthetic cohort record into a real app.models.Student."""
    dims = trait_dimensions()
    sig_fields: Dict[str, dict] = {"work_rhythm_signature": {}, "collaboration_signature": {}}
    for name, value in rec["traits"].items():
        if name not in dims:
            raise ValueError(f"student {rec['id']}: unknown trait {name!r} (not in rubric_v1.yaml)")
        if value not in dims[name]["allowed_values"]:
            raise ValueError(f"student {rec['id']}: {name}={value!r} not in allowed_values")
        sig_fields[_GROUP_TO_SIGNATURE[dims[name]["group"]]][name] = value

    skills = rec.get("skills") or {}
    return Student(
        id=rec["id"],
        name=rec.get("name", rec["id"]),
        competence_signature=CompetenceSignature(
            # The real matcher reads skill NAMES (skill_coverage); the per-skill
            # [0,1] numbers go to the real skill_proficiency field, which only
            # optional (default-off) passes read.
            skills=sorted(skills),
            skill_proficiency=dict(skills),
        ),
        work_rhythm_signature=WorkRhythmSignature(**sig_fields["work_rhythm_signature"]),
        collaboration_signature=CollaborationSignature(**sig_fields["collaboration_signature"]),
    )


def load_cohort(path: str | Path) -> List[Student]:
    with open(path) as f:
        data = json.load(f)
    return [record_to_student(r) for r in data["students"]]
