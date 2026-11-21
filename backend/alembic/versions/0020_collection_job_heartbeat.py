"""add last_heartbeat_at to collection_jobs (collection-job liveness, data/scheduler.py)

Revision ID: 0020
Revises: 0019
Create Date: 2026-11-07
"""
from alembic import op
import sqlalchemy as sa

revision = "0020"
down_revision = "0019"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Nullable, no default: metadata-only ALTER on Postgres. Workers started before this migration
    # never write it; the reaper judges such jobs by their Redis enqueue lock instead.
    op.add_column("collection_jobs", sa.Column("last_heartbeat_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column("collection_jobs", "last_heartbeat_at")
