"""Change datasets.row_count from INTEGER to BIGINT

Revision ID: 0006
Revises: 0005
Create Date: 2026-04-16
"""

from alembic import op
import sqlalchemy as sa


revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "datasets",
        "row_count",
        type_=sa.BigInteger(),
        existing_nullable=True,
    )


def downgrade() -> None:
    op.alter_column(
        "datasets",
        "row_count",
        type_=sa.Integer(),
        existing_nullable=True,
    )
