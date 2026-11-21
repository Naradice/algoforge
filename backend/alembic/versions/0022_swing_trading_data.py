"""create swing trading tables (large holding reports, daily margin records, regulations, market history)

Revision ID: 0022
Revises: 0021
Create Date: 2026-11-09
"""
from alembic import op
import sqlalchemy as sa

revision = "0022"
down_revision = "0021"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. large_holding_reports
    op.create_table(
        "large_holding_reports",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("doc_id", sa.Text(), unique=True, nullable=False),
        sa.Column("issuer_code", sa.Text(), nullable=True),
        sa.Column("issuer_name", sa.Text(), nullable=True),
        sa.Column("filer_name", sa.Text(), nullable=True),
        sa.Column("base_date", sa.Date(), nullable=True),
        sa.Column("submit_date", sa.Date(), nullable=True),
        sa.Column("doc_type", sa.Text(), nullable=True),
        sa.Column("holding_ratio", sa.Float(), nullable=True),
        sa.Column("prev_holding_ratio", sa.Float(), nullable=True),
        sa.Column("ratio_change", sa.Float(), nullable=True),
        sa.Column("total_shares_held", sa.Float(), nullable=True),
        sa.Column("total_outstanding_shares", sa.Float(), nullable=True),
        sa.Column("purpose_of_holding", sa.Text(), nullable=True),
        sa.Column("joint_holders_json", sa.JSON(), nullable=True),
        sa.Column("extracted_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_large_holding_reports_doc_id", "large_holding_reports", ["doc_id"])
    op.create_index("ix_large_holding_reports_issuer_code", "large_holding_reports", ["issuer_code"])
    op.create_index("ix_large_holding_reports_filer_name", "large_holding_reports", ["filer_name"])
    op.create_index("ix_large_holding_reports_base_date", "large_holding_reports", ["base_date"])

    # 2. large_holding_trades
    op.create_table(
        "large_holding_trades",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("report_id", sa.Integer(), sa.ForeignKey("large_holding_reports.id", ondelete="CASCADE"), nullable=False),
        sa.Column("doc_id", sa.Text(), nullable=False),
        sa.Column("issuer_code", sa.Text(), nullable=True),
        sa.Column("trade_date", sa.Date(), nullable=True),
        sa.Column("security_type", sa.Text(), nullable=True),
        sa.Column("quantity", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("ratio", sa.Float(), nullable=True),
        sa.Column("market_type", sa.Text(), nullable=True),
        sa.Column("trade_type", sa.Text(), nullable=True),
        sa.Column("unit_price", sa.Float(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
    )
    op.create_index("ix_large_holding_trades_report_id", "large_holding_trades", ["report_id"])
    op.create_index("ix_large_holding_trades_doc_id", "large_holding_trades", ["doc_id"])
    op.create_index("ix_large_holding_trades_issuer_code", "large_holding_trades", ["issuer_code"])
    op.create_index("ix_large_holding_trades_trade_date", "large_holding_trades", ["trade_date"])

    # 3. daily_margin_records
    op.create_table(
        "daily_margin_records",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("as_of_date", sa.Date(), nullable=False),
        sa.Column("code", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("market", sa.Text(), nullable=True),
        sa.Column("issue_type", sa.Text(), nullable=True),
        sa.Column("restrictions", sa.JSON(), nullable=True),
        sa.Column("isin", sa.Text(), nullable=True),
        sa.Column("sell_shares", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("sell_change", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("sell_ratio", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("buy_shares", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("buy_change", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("buy_ratio", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("ratio", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("negotiable_sell_shares", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("negotiable_sell_change", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("standardized_sell_shares", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("standardized_sell_change", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("negotiable_buy_shares", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("negotiable_buy_change", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("standardized_buy_shares", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("standardized_buy_change", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("as_of_date", "code", name="uq_daily_margin_date_code"),
    )
    op.create_index("ix_daily_margin_records_as_of_date", "daily_margin_records", ["as_of_date"])
    op.create_index("ix_daily_margin_records_code", "daily_margin_records", ["code"])

    # 4. margin_regulation_records
    op.create_table(
        "margin_regulation_records",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("code", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("event_date", sa.Date(), nullable=False),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("code", "event_date", "event_type", name="uq_margin_reg_code_date_type"),
    )
    op.create_index("ix_margin_regulation_records_code", "margin_regulation_records", ["code"])
    op.create_index("ix_margin_regulation_records_event_date", "margin_regulation_records", ["event_date"])

    # 5. market_margin_weekly_records
    op.create_table(
        "market_margin_weekly_records",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("date", sa.Date(), unique=True, nullable=False),
        sa.Column("standardized_sell_shares", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("standardized_buy_shares", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("negotiable_sell_shares", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("negotiable_buy_shares", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("total_sell_shares", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("total_buy_shares", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("margin_ratio", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("date", name="uq_market_margin_weekly_date"),
    )
    op.create_index("ix_market_margin_weekly_records_date", "market_margin_weekly_records", ["date"])


def downgrade() -> None:
    op.drop_table("market_margin_weekly_records")
    op.drop_table("margin_regulation_records")
    op.drop_table("daily_margin_records")
    op.drop_table("large_holding_trades")
    op.drop_table("large_holding_reports")
