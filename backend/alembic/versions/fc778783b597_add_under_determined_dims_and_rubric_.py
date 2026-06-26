"""add under_determined_dims and rubric versioning fields

Revision ID: fc778783b597
Revises: fe4544b29b36
Create Date: 2026-06-26 03:50:14.811773

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'fc778783b597'
down_revision: Union[str, Sequence[str], None] = 'fe4544b29b36'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    from sqlalchemy import inspect
    from alembic import op
    import sqlalchemy as sa
    
    bind = op.get_bind()
    inspector = inspect(bind)
    tables = inspector.get_table_names()
    
    if 'behavioral_signatures' in tables:
        existing_cols = [c['name'] for c in inspector.get_columns('behavioral_signatures')]
        with op.batch_alter_table('behavioral_signatures') as batch_op:
            if 'rubric_version' not in existing_cols:
                batch_op.add_column(sa.Column('rubric_version', sa.String(), nullable=True))
            if 'bank_version' not in existing_cols:
                batch_op.add_column(sa.Column('bank_version', sa.String(), nullable=True))
            if 'scorer_version' not in existing_cols:
                batch_op.add_column(sa.Column('scorer_version', sa.String(), nullable=True))

    if 'students' in tables:
        existing_cols = [c['name'] for c in inspector.get_columns('students')]
        if 'under_determined_dims' not in existing_cols:
            with op.batch_alter_table('students') as batch_op:
                batch_op.add_column(sa.Column('under_determined_dims', sa.JSON(), nullable=True))


def downgrade() -> None:
    pass