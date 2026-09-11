"""
Part 5 — student team-preference API.

PRIVACY DESIGN (this module's central concern — see also
app.models.PreferenceEntry's docstring and PreferenceStudentStats'):

This app has no authentication/authorization layer anywhere (every existing
endpoint, including app/familiarity.py, trusts the caller-supplied IDs
outright) — that is a pre-existing, repo-wide limitation this module does
not attempt to solve. What this module DOES control, and controls
deliberately, is the SHAPE of the API surface:

  1. There is NO "list every student's preferences" endpoint. Contrast
     app.familiarity.list_familiarity(), which intentionally dumps every
     FamiliarityEdge (mutual, symmetric, comparatively low-stakes data) —
     preference data is directional and can include avoidance requests
     tied to safety/conflict, so this module never offers an equivalent
     bulk read. The full PreferenceDB table is read ONLY server-side, by
     app.matching.generate_teams() (for scoring/optimization) — never
     returned verbatim through this router.
  2. GET /preferences/{student_id} returns ONLY that student's own
     OUTGOING entries (what they said about others) — never any INCOMING
     information (who named them as a prefer/avoid target, or at what
     rank/priority). A student cannot use this API to learn who avoided
     them or how another student ranked them.
  3. `reason_category` (free-form context for a preference, e.g.
     "prior_conflict") round-trips through this API only back to the
     submitting student — see `_entry_to_public_model` — and is never
     passed into app/explainability.py's team-explanation prompts (that
     module does not import or reference preference data at all).
  4. Every metric/report this system generates from preference data (see
     app.models.PreferenceStudentStats, PreferenceSummary) is aggregate —
     counts, booleans, scores — never named targets. The richer,
     name-bearing dict app.matching.compute_preference_metrics() returns
     internally is consumed by generate_teams() and tests, never
     serialized onto a public response.
"""
from __future__ import annotations
import uuid
from typing import List, Optional

from fastapi import APIRouter, HTTPException

from app.models import PreferenceEntry

router = APIRouter(prefix="/preferences", tags=["preferences"])


def _db_entry_to_model(row) -> PreferenceEntry:
    return PreferenceEntry(
        student_id=row.student_id,
        direction=row.direction,
        target_student_id=row.target_student_id,
        target_characteristic=row.target_characteristic,
        rank=row.rank,
        priority=row.priority or "normal",
        reason_category=row.reason_category,
        source=row.source,
    )


@router.post("", response_model=PreferenceEntry)
def upsert_preference(entry: PreferenceEntry):
    """
    Create or update ONE of `entry.student_id`'s own preference statements.
    A second POST with the same (student_id, target_student_id, direction)
    — or, for a characteristic target, the same (student_id,
    target_characteristic, direction) — UPDATES the existing row (rank/
    priority/reason_category/source all overwritten with the new values)
    rather than creating a duplicate, mirroring
    app.familiarity.upsert_familiarity's upsert semantics. Unlike
    familiarity, there is no canonical-pair step: preferences are
    directional, so (student_id, target) identifies a row on its own.

    Part 5.5: rejects a 'prefer' entry for a target the same student
    already has an 'avoid' entry for, and vice versa — app.matching's
    blocking-pair diagnostic never reconciles a contradictory prefer+avoid
    pair for the same target (see _named_prefer_tier's docstring), so the
    write path is where this is caught instead.
    """
    from app.database import SessionLocal, PreferenceDB

    db = SessionLocal()
    try:
        if entry.target_student_id is not None:
            opposite_direction = "avoid" if entry.direction == "prefer" else "prefer"
            contradiction = db.query(PreferenceDB).filter(
                PreferenceDB.student_id == entry.student_id,
                PreferenceDB.direction == opposite_direction,
                PreferenceDB.target_student_id == entry.target_student_id,
            ).first()
            if contradiction:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"student {entry.student_id!r} already has a {opposite_direction!r} entry "
                        f"for target {entry.target_student_id!r} — remove it first "
                        f"(DELETE /preferences/{{student_id}}/{opposite_direction}/{{target_student_id}}) "
                        f"before submitting a contradictory {entry.direction!r} entry for the same target"
                    ),
                )

        query = db.query(PreferenceDB).filter(
            PreferenceDB.student_id == entry.student_id,
            PreferenceDB.direction == entry.direction,
        )
        if entry.target_student_id is not None:
            query = query.filter(PreferenceDB.target_student_id == entry.target_student_id)
            existing = query.first()
        else:
            # Characteristic targets are matched by exact dict equality —
            # scanned in Python since JSON-column equality isn't portably
            # queryable across backends; PreferenceDB rows per student are
            # small (a handful of entries), so this is cheap.
            existing = next(
                (r for r in query.filter(PreferenceDB.target_student_id.is_(None)).all()
                 if r.target_characteristic == entry.target_characteristic),
                None,
            )

        if existing:
            existing.rank = entry.rank
            existing.priority = entry.priority
            existing.reason_category = entry.reason_category
            existing.source = entry.source
        else:
            db.add(PreferenceDB(
                id=str(uuid.uuid4()),
                student_id=entry.student_id,
                direction=entry.direction,
                target_student_id=entry.target_student_id,
                target_characteristic=entry.target_characteristic,
                rank=entry.rank,
                priority=entry.priority,
                reason_category=entry.reason_category,
                source=entry.source,
            ))
        db.commit()
        return entry
    finally:
        db.close()


@router.get("/{student_id}", response_model=List[PreferenceEntry])
def list_own_preferences(student_id: str):
    """
    ALL of `student_id`'s own OUTGOING preference entries (what they said
    about others) — see this module's docstring for why there is
    deliberately no way, through this API, to learn who named `student_id`
    as a target.
    """
    from app.database import SessionLocal, PreferenceDB

    db = SessionLocal()
    try:
        rows = db.query(PreferenceDB).filter(PreferenceDB.student_id == student_id).all()
        return [_db_entry_to_model(r) for r in rows]
    finally:
        db.close()


@router.delete("/{student_id}/{direction}/{target_student_id}")
def delete_preference(student_id: str, direction: str, target_student_id: str):
    """Remove one of `student_id`'s own named-student preference entries
    (a student changing their mind, or withdrawing an avoidance request).
    Characteristic-target entries are not addressable by this path — they
    have no natural single-string identifier — and can only be replaced via
    a fresh POST."""
    from app.database import SessionLocal, PreferenceDB

    if direction not in ("prefer", "avoid"):
        raise HTTPException(status_code=422, detail="direction must be 'prefer' or 'avoid'")

    db = SessionLocal()
    try:
        existing = db.query(PreferenceDB).filter(
            PreferenceDB.student_id == student_id,
            PreferenceDB.direction == direction,
            PreferenceDB.target_student_id == target_student_id,
        ).first()
        if not existing:
            raise HTTPException(status_code=404, detail="no matching preference entry")
        db.delete(existing)
        db.commit()
        return {"deleted": True}
    finally:
        db.close()
