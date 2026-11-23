"""create pdf_documents and pdf_extracted_tables tables

Revision ID: 0023
Revises: 0022
Create Date: 2026-11-23
"""
from alembic import op
import sqlalchemy as sa

revision = "0023"
down_revision = "0022"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. pdf_documents
    op.create_table(
        "pdf_documents",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("doc_id", sa.Text(), unique=True, nullable=False),
        sa.Column("sec_code", sa.Text(), nullable=True),
        sa.Column("filer_name", sa.Text(), nullable=True),
        sa.Column("doc_type", sa.Text(), nullable=True),
        sa.Column("doc_type_code", sa.Text(), nullable=True),
        sa.Column("category", sa.Text(), nullable=False, server_default="prospectus"),
        sa.Column("filing_date", sa.Date(), nullable=True),
        sa.Column("file_path", sa.Text(), nullable=True),
        sa.Column("file_size", sa.BigInteger(), nullable=True),
        sa.Column("sha256", sa.Text(), nullable=True),
        sa.Column("num_pages", sa.Integer(), nullable=True),
        sa.Column("extraction_status", sa.Text(), nullable=False, server_default="pending"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("extracted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_pdf_documents_doc_id", "pdf_documents", ["doc_id"])
    op.create_index("ix_pdf_documents_sec_code", "pdf_documents", ["sec_code"])
    op.create_index("ix_pdf_documents_filer_name", "pdf_documents", ["filer_name"])
    op.create_index("ix_pdf_documents_doc_type_code", "pdf_documents", ["doc_type_code"])
    op.create_index("ix_pdf_documents_category", "pdf_documents", ["category"])
    op.create_index("ix_pdf_documents_filing_date", "pdf_documents", ["filing_date"])

    # 2. pdf_extracted_tables
    op.create_table(
        "pdf_extracted_tables",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("document_id", sa.Integer(), sa.ForeignKey("pdf_documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("doc_id", sa.Text(), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("table_index", sa.Integer(), nullable=False),
        sa.Column("table_title", sa.Text(), nullable=True),
        sa.Column("row_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("col_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("headers_json", sa.JSON(), nullable=True),
        sa.Column("data_json", sa.JSON(), nullable=True),
        sa.Column("html", sa.Text(), nullable=False, server_default=""),
        sa.Column("bbox_json", sa.JSON(), nullable=True),
        sa.Column("meta_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_pdf_extracted_tables_document_id", "pdf_extracted_tables", ["document_id"])
    op.create_index("ix_pdf_extracted_tables_doc_id", "pdf_extracted_tables", ["doc_id"])
    op.create_index("ix_pdf_extracted_tables_table_title", "pdf_extracted_tables", ["table_title"])


def downgrade() -> None:
    op.drop_table("pdf_extracted_tables")
    op.drop_table("pdf_documents")
