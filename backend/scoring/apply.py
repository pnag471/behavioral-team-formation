"""
Shared helper: write score() output onto a Student row.

Maps each scored dimension to the Student JSON field named by its rubric
`group`, so the field list lives in the rubric instead of being hand-written
in each route. Used by the interview path (app/conversation.py).

The MCQ path (app/behavioral_analysis.py) still writes its own field list and
is deliberately NOT switched over in this change (plan TODO-10), so MCQ
behavior stays exactly as it was. Switching it would also make MCQ write
safety_contribution and stress_response, which it does not write today.

No DB session, no commit: this only sets attributes on the row it is given.
"""
from __future__ import annotations

from typing import Any, Optional

from scoring.dimension_scorer import get_rubric

# rubric `group` -> StudentDB JSON column
GROUP_TO_FIELD = {
    "work_rhythm": "work_rhythm_signature",
    "collaboration": "collaboration_signature",
}


def apply_scored_to_student(
    student_row: Any,
    scored: dict,
    unknown: list[str],
    rubric: Optional[dict] = None,
) -> None:
    """
    Write scored labels, coverage and unknown dimensions onto `student_row`.

    - Every non-derived rubric dimension goes into the signature field for its
      group. Existing keys in that field (availability, unavailable_times, ...)
      are kept; only the dimension keys are overwritten.
    - Unknown dimensions still get score()'s display default, exactly like the
      MCQ path. `under_determined_dims` is what tells the matcher to ignore them.
    - confidence_layer is REPLACED with the coverage under both names: the new
      `measurement_coverage` and the legacy `confidence_score` key, same value,
      so existing readers keep working (plan TODO-11). Callers that add more
      keys (e.g. consistency_flags) do so after this call.

    JSON columns are reassigned (not mutated in place) so SQLAlchemy sees the change.
    """
    rubric = rubric or get_rubric()

    updates: dict[str, dict[str, Any]] = {}
    for dim, dim_def in rubric["dimensions"].items():
        if dim_def.get("type") == "derived":
            continue
        group = dim_def.get("group")
        if group not in GROUP_TO_FIELD:
            raise ValueError(f"rubric dimension {dim!r} has group {group!r}, which maps to no Student field")
        updates.setdefault(GROUP_TO_FIELD[group], {})[dim] = scored[dim]

    for field, values in updates.items():
        setattr(student_row, field, {**(getattr(student_row, field, None) or {}), **values})

    coverage = scored["confidence_score"]
    student_row.confidence_layer = {
        "measurement_coverage": coverage,
        "confidence_score": coverage,
    }
    student_row.under_determined_dims = list(unknown)
