"""Data models for PDF table extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import pandas as pd


@dataclass
class ExtractedPdfTable:
    """Represents a structured table extracted from a PDF page."""

    page_number: int
    table_index: int
    bbox: tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max)
    df: pd.DataFrame
    html: str
    row_count: int = 0
    col_count: int = 0
    headers: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.df is not None and not self.df.empty:
            self.row_count = len(self.df)
            self.col_count = len(self.df.columns)
            if not self.headers:
                self.headers = [str(c) for c in self.df.columns.tolist()]
        if not self.html and self.df is not None and not self.df.empty:
            self.html = self.df.to_html(index=False, border=1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "table_index": self.table_index,
            "bbox": self.bbox,
            "row_count": self.row_count,
            "col_count": self.col_count,
            "headers": self.headers,
            "html": self.html,
            "data": self.df.to_dict(orient="records") if self.df is not None else [],
            "metadata": self.metadata,
        }
