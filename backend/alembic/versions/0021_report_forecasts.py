"""create report_forecasts table (analyst forecast ranges & directions)

Revision ID: 0021
Revises: 0020
Create Date: 2026-11-07
"""
from alembic import op
import sqlalchemy as sa

revision = "0021"
down_revision = "0020"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "report_forecasts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("datasource_id", sa.Integer(), sa.ForeignKey("datasources.id", ondelete="SET NULL"), nullable=True),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("symbol", sa.Text(), nullable=False, server_default="USDJPY"),
        sa.Column("horizon", sa.Text(), nullable=False, server_default="Daily"),
        sa.Column("report_date", sa.Date(), nullable=False),
        sa.Column("analyst", sa.Text(), nullable=True),
        sa.Column("direction", sa.Text(), nullable=True),
        sa.Column("expected_low", sa.Float(), nullable=True),
        sa.Column("expected_high", sa.Float(), nullable=True),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("raw_artifact_path", sa.Text(), nullable=True),
        sa.Column("extracted_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("source", "symbol", "horizon", "report_date", name="uq_report_forecasts_source_sym_hor_date"),
    )
    op.create_index("ix_report_forecasts_datasource_id", "report_forecasts", ["datasource_id"])
    op.create_index("ix_report_forecasts_source", "report_forecasts", ["source"])
    op.create_index("ix_report_forecasts_symbol", "report_forecasts", ["symbol"])
    op.create_index("ix_report_forecasts_report_date", "report_forecasts", ["report_date"])


def downgrade() -> None:
    op.drop_table("report_forecasts")
