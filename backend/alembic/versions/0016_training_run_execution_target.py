"""add execution_target to training_runs (local | colab)

Revision ID: 0016
Revises: 0015
Create Date: 2026-09-02
"""
from alembic import op
import sqlalchemy as sa

revision = "0016"
down_revision = "0015"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # nullable=False with a server_default is a metadata-only change on Postgres 11+ — safe to
    # run while a training worker is actively updating other columns on this table (same
    # reasoning as 0014's `source` column).
    op.add_column(
        "training_runs",
        sa.Column("execution_target", sa.Text(), nullable=False, server_default="local"),
    )


def downgrade() -> None:
    op.drop_column("training_runs", "execution_target")
