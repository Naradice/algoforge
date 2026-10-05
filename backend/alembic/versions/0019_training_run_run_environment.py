"""add run_environment to training_runs (code + package fingerprint per run)

Revision ID: 0019
Revises: 0018
Create Date: 2026-10-05
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "0019"
down_revision = "0018"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Nullable, no default: metadata-only ALTER on Postgres. Workers started before this
    # migration don't write it, and existing runs stay NULL (their environment is unknown).
    op.add_column("training_runs", sa.Column("run_environment", JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("training_runs", "run_environment")
