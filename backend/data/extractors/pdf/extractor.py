"""High-level PDF Table Extractor module for AlgoForge.

Extracts structured tables from PDF documents into pandas DataFrames and HTML tables.
Designed for iterative improvements as new financial report formats and layout anomalies are encountered.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from pdfminer.high_level import extract_pages

from data.extractors.pdf.geometry import (
    retrieve_table_elements,
    find_table_regions,
    is_in_bb,
)
from data.extractors.pdf.table_builder import (
    retrieve_text_elements,
    texts_ele_to_df,
    assign_grid_coordinates,
    build_dataframe_from_grid,
    dataframe_to_html,
)
from data.extractors.pdf.models import ExtractedPdfTable

logger = logging.getLogger("extractors.pdf")


class PdfTableExtractor:
    """Extracts tables from financial and corporate PDF reports."""

    def __init__(
        self,
        line_margin: float = 2.0,
        grid_margin: float = 2.0,
        min_lines_for_table: int = 2,
        min_table_cells: int = 2,
    ):
        self.line_margin = line_margin
        self.grid_margin = grid_margin
        self.min_lines_for_table = min_lines_for_table
        self.min_table_cells = min_table_cells

    def extract_tables(
        self,
        pdf_path: str | Path,
        page_numbers: Sequence[int] | None = None,
    ) -> list[ExtractedPdfTable]:
        """Extract all structured tables from specified pages (1-indexed) of a PDF.

        Args:
            pdf_path: Path to the PDF file.
            page_numbers: Optional list of 1-based page numbers to extract. If None, extracts all pages.

        Returns:
            List of ExtractedPdfTable objects.
        """
        p = Path(pdf_path)
        if not p.exists():
            raise FileNotFoundError(f"PDF file not found: {p}")

        extracted_tables: list[ExtractedPdfTable] = []

        try:
            pages = list(extract_pages(str(p)))
        except Exception as exc:
            logger.error(f"Failed to read PDF pages from {p}: {exc}", exc_info=True)
            return []

        for p_idx, page in enumerate(pages):
            page_num = p_idx + 1
            if page_numbers and page_num not in page_numbers:
                continue

            try:
                page_tables = self._extract_tables_from_page(page, page_num)
                extracted_tables.extend(page_tables)
            except Exception as exc:
                logger.warning(
                    f"Error extracting tables on page {page_num} of {p.name}: {exc}. "
                    "Skipping page to preserve robustness."
                )

        return extracted_tables

    def _extract_tables_from_page(self, page: Any, page_num: int) -> list[ExtractedPdfTable]:
        """Extract all candidate tables from a single layout page."""
        tables: list[ExtractedPdfTable] = []
        lines_df = retrieve_table_elements(page)
        all_texts = retrieve_text_elements(page)

        if lines_df.empty:
            return tables

        # Find candidate table regions (enclosed grids and horizontal-rule open tables)
        regions = find_table_regions(
            lines_df,
            margin=self.line_margin,
            min_lines=self.min_lines_for_table,
        )

        for table_idx, bb in enumerate(regions, start=1):
            w = bb[2] - bb[0]
            h = bb[3] - bb[1]

            if w < 10.0 or h < 10.0:
                continue

            # Gather text elements enclosed within this table's bounding box
            table_texts = [t for t in all_texts if is_in_bb(bb, t.bbox, margin=self.line_margin)]
            if not table_texts:
                continue

            txt_df = texts_ele_to_df(table_texts)
            if txt_df.empty or len(txt_df) < self.min_table_cells:
                continue

            # Build 2D grid matrix
            grid_df, row_count, col_count = assign_grid_coordinates(txt_df, margin=self.grid_margin)
            df = build_dataframe_from_grid(grid_df, row_count, col_count)

            if not df.empty and len(df) >= 1 and len(df.columns) >= 2:
                html_str = dataframe_to_html(df)
                t_obj = ExtractedPdfTable(
                    page_number=page_num,
                    table_index=table_idx,
                    bbox=bb,
                    df=df,
                    html=html_str,
                    metadata={"text_count": len(table_texts)},
                )
                tables.append(t_obj)

        return tables

    def extract_tables_as_dataframe(
        self,
        pdf_path: str | Path,
        page_numbers: Sequence[int] | None = None,
    ) -> list[pd.DataFrame]:
        """Extract tables directly as a list of pandas DataFrames."""
        tables = self.extract_tables(pdf_path, page_numbers=page_numbers)
        return [t.df for t in tables]

    def extract_tables_as_html(
        self,
        pdf_path: str | Path,
        page_numbers: Sequence[int] | None = None,
    ) -> list[str]:
        """Extract tables directly as clean HTML strings for consumption by table_parser."""
        tables = self.extract_tables(pdf_path, page_numbers=page_numbers)
        return [t.html for t in tables]
