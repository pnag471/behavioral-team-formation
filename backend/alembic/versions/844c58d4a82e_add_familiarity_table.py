"""add familiarity table

Revision ID: 844c58d4a82e
Revises: 8a739dc8cc47
Create Date: 2026-09-01 00:00:00.000000

Part 4 (research-alignment): prior familiarity between two students is a
RELATIONSHIP, not a property of one student — represented as its own edge
table (`familiarity`) rather than a column on `students`, unlike Part 3's
ability fields (which were scalar per-student data and fit inside the
existing `competence_signature` JSON column with no schema change needed).

A row's ABSENCE means "unknown" (nobody has recorded anything about that
pair); a row with strength=0.0 means "confirmed no prior contact" — see
app/models.py:FamiliarityEdge for the full scale documentation.
Familiarity is symmetric (see that same docstring for the directionality
decision), so rows are stored with a canonical (student_a_id < student_b_id)
ordering; the unique constraint below enforces "at most one row per
unordered pair" at the database layer, not just in application code
(app.familiarity.canonical_pair() + upsert_familiarity()).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '844c58d4a82e'
down_revision: Union[str, Sequence[str], None] = '8a739dc8cc47'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'familiarity',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('student_a_id', sa.String(), nullable=False),
        sa.Column('student_b_id', sa.String(), nullable=False),
        sa.Column('strength', sa.Float(), nullable=False),
        sa.Column('source', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('student_a_id', 'student_b_id', name='uq_familiarity_pair'),
    )
    op.create_index(op.f('ix_familiarity_student_a_id'), 'familiarity', ['student_a_id'], unique=False)
    op.create_index(op.f('ix_familiarity_student_b_id'), 'familiarity', ['student_b_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_familiarity_student_b_id'), table_name='familiarity')
    op.drop_index(op.f('ix_familiarity_student_a_id'), table_name='familiarity')
    op.drop_table('familiarity')
