from __future__ import annotations
import uuid
from typing import List, Tuple

from fastapi import APIRouter, HTTPException

from app.models import FamiliarityEdge

router = APIRouter(prefix="/familiarity", tags=["familiarity"])


def canonical_pair(a: str, b: str) -> Tuple[str, str]:
    """
    Familiarity is symmetric (see FamiliarityEdge's docstring) — this
    canonicalizes an unordered pair to a fixed (lexicographically smaller,
    larger) order so (A, B) and (B, A) always resolve to the SAME storage
    key. Used both when writing (upsert target) and reading (graph
    construction) so a caller never has to remember which order a pair was
    originally recorded in.
    """
    return (a, b) if a <= b else (b, a)


def _db_edge_to_model(row) -> FamiliarityEdge:
    return FamiliarityEdge(
        student_a_id=row.student_a_id,
        student_b_id=row.student_b_id,
        strength=row.strength,
        source=row.source,
    )


@router.get("", response_model=List[FamiliarityEdge])
def list_familiarity():
    """All known familiarity edges. A pair absent from this list is
    UNKNOWN familiarity, not confirmed zero — see FamiliarityEdge."""
    from app.database import SessionLocal, FamiliarityDB

    db = SessionLocal()
    try:
        rows = db.query(FamiliarityDB).all()
        return [_db_edge_to_model(r) for r in rows]
    finally:
        db.close()


@router.post("", response_model=FamiliarityEdge)
def upsert_familiarity(edge: FamiliarityEdge):
    """
    Create or update the familiarity edge for (student_a_id, student_b_id).
    Self-relationships are rejected by FamiliarityEdge's own validator
    before this ever runs. A second POST for the same unordered pair
    UPDATES the existing row (last write wins) rather than creating a
    conflicting duplicate — the canonical pair ordering plus the DB's
    unique constraint on (student_a_id, student_b_id) is what makes "the
    same unordered pair" well-defined regardless of argument order.
    """
    from app.database import SessionLocal, FamiliarityDB

    a, b = canonical_pair(edge.student_a_id, edge.student_b_id)
    db = SessionLocal()
    try:
        existing = (
            db.query(FamiliarityDB)
            .filter(FamiliarityDB.student_a_id == a, FamiliarityDB.student_b_id == b)
            .first()
        )
        if existing:
            existing.strength = edge.strength
            existing.source = edge.source
        else:
            db.add(FamiliarityDB(
                id=str(uuid.uuid4()),
                student_a_id=a,
                student_b_id=b,
                strength=edge.strength,
                source=edge.source,
            ))
        db.commit()
        return FamiliarityEdge(student_a_id=a, student_b_id=b, strength=edge.strength, source=edge.source)
    finally:
        db.close()


@router.get("/{student_id}", response_model=List[FamiliarityEdge])
def list_familiarity_for_student(student_id: str):
    """All known edges involving `student_id`, in either canonical slot."""
    from app.database import SessionLocal, FamiliarityDB

    db = SessionLocal()
    try:
        rows = (
            db.query(FamiliarityDB)
            .filter(
                (FamiliarityDB.student_a_id == student_id)
                | (FamiliarityDB.student_b_id == student_id)
            )
            .all()
        )
        return [_db_edge_to_model(r) for r in rows]
    finally:
        db.close()
