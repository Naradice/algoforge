"""Table assembly from layout texts and geometry grid."""

from __future__ import annotations

import html
import re
from typing import Any
import numpy as np
import pandas as pd
from pdfminer.layout import LTTextContainer

from data.extractors.pdf.geometry import is_in_bb


def texts_ele_to_df(texts_ele: list[Any]) -> pd.DataFrame:
    """Convert LTTextContainer elements into a DataFrame with bbox and stripped text."""
    items = []
    for text_ele in texts_ele:
        if isinstance(text_ele, LTTextContainer):
            t = text_ele.get_text().strip()
            if t:
                items.append([*text_ele.bbox, t])
        elif hasattr(text_ele, "bbox"):
            t = getattr(text_ele, "get_text", lambda: "")().strip()
            if t:
                items.append([*text_ele.bbox, t])
    cols = ["x_min", "y_min", "x_max", "y_max", "text"]
    if not items:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(items, columns=cols)


def retrieve_text_elements(page: Any) -> list[LTTextContainer]:
    """Extract all text container objects from a PDF page."""
    texts = []
    for item in page:
        if isinstance(item, LTTextContainer):
            texts.append(item)
    return texts


def h_overlap(bbox1: pd.Series | tuple, bbox2: pd.Series | tuple, margin: float = 1.0) -> bool:
    """Check if two bounding boxes overlap horizontally."""
    x1_min, x1_max = bbox1["x_min"], bbox1["x_max"]
    x2_min, x2_max = bbox2["x_min"], bbox2["x_max"]
    return not (x1_max < x2_min - margin or x2_max < x1_min - margin)


def v_overlap(bbox1: pd.Series | tuple, bbox2: pd.Series | tuple, margin: float = 1.0) -> bool:
    """Check if two bounding boxes overlap vertically."""
    y1_min, y1_max = bbox1["y_min"], bbox1["y_max"]
    y2_min, y2_max = bbox2["y_min"], bbox2["y_max"]
    return not (y1_max < y2_min - margin or y2_max < y1_min - margin)


def assign_grid_coordinates(text_df: pd.DataFrame, margin: float = 2.0) -> tuple[pd.DataFrame, int, int]:
    """Assign discrete row (y) and column (x) integer indices to text bounding boxes.

    y=0 is the top-most row (descending y_min).
    x=0 is the left-most column (ascending x_min).
    """
    if text_df.empty:
        return text_df, 0, 0

    df = text_df.copy()

    # Assign column indices based on horizontal clustering
    df = df.sort_values(by="x_min").reset_index(drop=True)
    x = 0
    x_coords = [0]
    target_text = df.iloc[0]
    for i in range(1, len(df)):
        next_text = df.iloc[i]
        if h_overlap(target_text, next_text, margin=margin):
            x_coords.append(x)
        else:
            x += 1
            x_coords.append(x)
            target_text = next_text
    df["x"] = x_coords

    # Assign row indices based on vertical clustering (top down)
    df = df.sort_values(by="y_max", ascending=False).reset_index(drop=True)
    y = 0
    y_coords = [0]
    target_text = df.iloc[0]
    for i in range(1, len(df)):
        next_text = df.iloc[i]
        if v_overlap(target_text, next_text, margin=margin):
            y_coords.append(y)
        else:
            y += 1
            y_coords.append(y)
            target_text = next_text
    df["y"] = y_coords

    row_count = df["y"].max() + 1
    col_count = df["x"].max() + 1
    return df, row_count, col_count


def build_dataframe_from_grid(
    text_grid_df: pd.DataFrame, row_count: int, col_count: int
) -> pd.DataFrame:
    """Construct a rectangular pandas DataFrame from text items with (x, y) coordinates."""
    if row_count <= 0 or col_count <= 0:
        return pd.DataFrame()

    matrix = [["" for _ in range(col_count)] for _ in range(row_count)]

    for _, row in text_grid_df.iterrows():
        r = int(row["y"])
        c = int(row["x"])
        val = str(row["text"]).strip()
        if 0 <= r < row_count and 0 <= c < col_count:
            if matrix[r][c]:
                matrix[r][c] += " " + val
            else:
                matrix[r][c] = val

    df = pd.DataFrame(matrix)
    # Remove completely empty rows or columns
    df = df.replace("", np.nan).dropna(how="all").dropna(axis=1, how="all").fillna("")
    df.reset_index(drop=True, inplace=True)
    return df


def dataframe_to_html(df: pd.DataFrame, table_class: str = "pdf-extracted-table") -> str:
    """Format DataFrame as a clean, semantic HTML <table>."""
    if df.empty:
        return ""

    rows = []
    rows.append(f'<table class="{table_class}" border="1">')
    
    # Use first row as header if it looks like a header, or general tr/td
    for i, row in df.iterrows():
        tag = "th" if i == 0 else "td"
        cells = [f"<{tag}>{html.escape(str(c))}</{tag}>" for c in row]
        rows.append(f"  <tr>{''.join(cells)}</tr>")
    rows.append("</table>")
    return "\n".join(rows)
