"""add last_heartbeat_at and error_message to training_runs (R-13)

Revision ID: 0017
Revises: 0016
Create Date: 2026-09-28
"""
from alembic import op
import sqlalchemy as sa

revision = "0017"
down_revision = "0016"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Both nullable with no default: a metadata-only ALTER on Postgres, safe while workers are
    # updating other columns of this table. Workers started before this migration never write
    # last_heartbeat_at, and the reaper ignores NULL heartbeats, so their runs are unaffected.
    op.add_column("training_runs", sa.Column("last_heartbeat_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("training_runs", sa.Column("error_message", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("training_runs", "error_message")
    op.drop_column("training_runs", "last_heartbeat_at")
