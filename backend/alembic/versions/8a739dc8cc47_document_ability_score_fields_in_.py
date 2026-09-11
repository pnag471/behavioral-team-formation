"""document ability_score/ability_source/skill_proficiency fields in competence_signature

Revision ID: 8a739dc8cc47
Revises: fdfcfe5ac73b
Create Date: 2026-09-01 00:00:00.000000

Part 3 (research-alignment): CompetenceSignature gained three new fields —
`ability_score` (Optional[float], [0.0, 1.0]), `ability_source`
(Optional[str]), and `skill_proficiency` (Dict[str, float] in [0.0, 1.0]).
See app/models.py:CompetenceSignature for the full field documentation and
app/matching.py:_student_ability_score for how a missing value is resolved.

This is a NO-OP migration at the SQL level, and deliberately so:
`students.competence_signature` (see StudentDB in app/database.py) is a
JSON column, not individual typed columns — it has been schemaless at the
database layer since the initial migration (b7957d736061). Every existing
row's JSON blob (created before this revision) simply lacks the three new
keys; Pydantic's CompetenceSignature model supplies their documented
defaults (`None`, `None`, `{}`) on read, so old records deserialize
without any backfill, ALTER TABLE, or data migration required. Validation
(range-checking, distinguishing "missing" from "assessed at 0.0") happens
entirely at the Pydantic layer, not the database layer, exactly as it
already does for every other field inside this same JSON column
(experience_level, skill_depth, skill_no_gos, ...).

This revision exists purely so the migration history accurately records
WHEN this schema extension was introduced and WHY no upgrade()/downgrade()
step was needed — a discoverable audit trail entry, not a placeholder
left unfinished. See research_alignment_audit.md's Part 3 section for the
full rationale, including why a denormalized top-level ability_score
column on StudentDB was considered and deliberately NOT added (it would
require a second write path to keep in sync with competence_signature,
which no other competence-related field in this schema does).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '8a739dc8cc47'
down_revision: Union[str, Sequence[str], None] = 'fdfcfe5ac73b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """No schema change — see module docstring. competence_signature is a
    JSON column; the new ability fields are validated by Pydantic, not the
    database, so no ALTER TABLE is applicable or necessary."""
    pass


def downgrade() -> None:
    """No schema change to reverse — see upgrade()."""
    pass
