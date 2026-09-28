"""add source to training_runs (internal | external, for imported runs e.g. Colab)

Revision ID: 0014
Revises: 0013
Create Date: 2026-09-01
"""
from alembic import op
import sqlalchemy as sa

revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # nullable=False with a server_default is a metadata-only change on Postgres 11+ (no table
    # rewrite, no long-held lock) — safe to run while a training worker is actively updating
    # other columns on this table.
    op.add_column(
        "training_runs",
        sa.Column("source", sa.Text(), nullable=False, server_default="internal"),
    )


def downgrade() -> None:
    op.drop_column("training_runs", "source")
