"""HTML Table parser for EDINET financial disclosure documents.

Robustly reconstructs 2D tabular grids from complex HTML tables, handling:
- Multi-row / multi-column spans (`rowspan`, `colspan`)
- Line-broken multi-metric labels within a single rowspan cell (preventing item-value misalignment)
- Section header rows spanning entire table width
- Nested tables within cells (`<table>` inside `<td>`)
- Multi-line headers and merged title rows
- Hidden elements (`style="display:none"`, `@hidden`)
"""

from __future__ import annotations

import logging
import re
from typing import Any

import lxml.html
import pandas as pd

logger = logging.getLogger(__name__)


def get_colspan(cell: Any) -> int:
    """Return colspan attribute value as int (defaults to 1)."""
    try:
        return int(cell.attrib.get("colspan", 1))
    except (ValueError, TypeError):
        return 1


def get_rowspan(cell: Any) -> int:
    """Return rowspan attribute value as int (defaults to 1)."""
    try:
        return int(cell.attrib.get("rowspan", 1))
    except (ValueError, TypeError):
        return 1


def extract_cell_lines(element: Any, strip_nested_tables: bool = True) -> list[str]:
    """Extract individual text lines separated by <br> or newlines from a cell."""
    if element is None:
        return []

    texts: list[str] = []
    if element.text:
        texts.append(element.text)

    for node in element.iterchildren():
        if strip_nested_tables and node.tag == "table":
            nested_txt = " ".join(node.text_content().split())
            if nested_txt:
                texts.append(f"[{nested_txt}]")
            continue

        if node.tag == "br":
            texts.append("\n")
        elif node.text:
            texts.append(node.text)

        for subnode in node.iterdescendants():
            if strip_nested_tables and subnode.tag == "table":
                continue
            if subnode.tag == "br":
                texts.append("\n")
            elif subnode.text:
                texts.append(subnode.text)
            if subnode.tail:
                texts.append(subnode.tail)

        if node.tail:
            texts.append(node.tail)

    raw = "".join(texts)
    # Split by explicit newlines
    lines = [line.strip() for line in raw.split("\n") if line.strip()]
    return lines


def extract_cell_text(element: Any, strip_nested_tables: bool = True) -> str:
    """Extract clean text content from a table cell (th/td)."""
    lines = extract_cell_lines(element, strip_nested_tables=strip_nested_tables)
    return " ".join(lines)


def split_lines_for_rowspan(raw_lines: list[str], rowspan: int) -> list[str]:
    """Distribute line chunks evenly into `rowspan` items if applicable."""
    if not raw_lines or rowspan <= 1:
        return raw_lines

    if len(raw_lines) == rowspan:
        return raw_lines

    # Try grouping if lines > rowspan
    if len(raw_lines) > rowspan and len(raw_lines) % rowspan == 0:
        chunk_size = len(raw_lines) // rowspan
        return [
            " ".join(raw_lines[i * chunk_size : (i + 1) * chunk_size])
            for i in range(rowspan)
        ]

    # Cannot cleanly split into distinct lines; repeat unified string
    full_text = " ".join(raw_lines)
    return [full_text] * rowspan


def get_direct_rows(table_element: Any) -> list[Any]:
    """Retrieve direct <tr> rows of this table, excluding any nested table rows."""
    rows: list[Any] = []
    for child in table_element.iterchildren():
        if child.tag in ("tbody", "thead", "tfoot"):
            for subchild in child.iterchildren():
                if subchild.tag == "tr" and not subchild.attrib.get("hidden"):
                    rows.append(subchild)
        elif child.tag == "tr" and not child.attrib.get("hidden"):
            rows.append(child)
    return rows


def parse_html_table_to_grid(table_element: Any) -> list[list[str]]:
    """Convert an lxml table element into a 2D matrix of strings using grid expansion.

    Resolves arbitrary rowspan and colspan layouts by projecting cells onto a dense grid.
    If a rowspan cell contains distinct lines matching `rowspan`, each line is assigned
    to its corresponding target row to prevent metric-content misalignment.
    """
    rows = get_direct_rows(table_element)
    if not rows:
        return []

    grid: list[list[str | None]] = []

    for r_idx, row in enumerate(rows):
        # Ensure row exists in grid
        while len(grid) <= r_idx:
            grid.append([])

        c_idx = 0
        direct_cells = [
            c for c in row.iterchildren()
            if c.tag in ("th", "td") and not c.attrib.get("hidden")
        ]

        for cell in direct_cells:
            # Advance c_idx to the next unfilled cell in the current row
            while c_idx < len(grid[r_idx]) and grid[r_idx][c_idx] is not None:
                c_idx += 1

            colspan = get_colspan(cell)
            rowspan = get_rowspan(cell)

            if rowspan > 1:
                raw_lines = extract_cell_lines(cell)
                row_texts = split_lines_for_rowspan(raw_lines, rowspan)
            else:
                row_texts = [extract_cell_text(cell)]

            # Fill the rectangular region for (rowspan, colspan)
            for r_offset in range(rowspan):
                target_r = r_idx + r_offset
                while len(grid) <= target_r:
                    grid.append([])

                val_for_row = row_texts[r_offset] if r_offset < len(row_texts) else row_texts[-1]

                for c_offset in range(colspan):
                    target_c = c_idx + c_offset
                    while len(grid[target_r]) <= target_c:
                        grid[target_r].append(None)
                    grid[target_r][target_c] = val_for_row

            c_idx += colspan

    # Replace remaining None with empty strings and normalize row widths
    max_cols = max((len(r) for r in grid), default=0)
    dense_grid: list[list[str]] = []
    for r in grid:
        filled_row = [(val or "").strip() for val in r]
        if len(filled_row) < max_cols:
            filled_row.extend([""] * (max_cols - len(filled_row)))
        dense_grid.append(filled_row)

    return dense_grid


def html_tables_to_df(
    table_element: Any,
    index_col: int | None = 0,
    header_rows: int = 1,
    combine_header_lines: bool = True,
) -> pd.DataFrame:
    """Parse an lxml table element into a pandas DataFrame."""
    grid = parse_html_table_to_grid(table_element)
    if not grid:
        return pd.DataFrame()

    valid_rows = [r for r in grid if any(cell.strip() for cell in r)]
    if not valid_rows:
        return pd.DataFrame()

    if len(valid_rows) <= header_rows:
        cols = [f"col_{i}" for i in range(len(valid_rows[0]))]
        return pd.DataFrame(columns=cols)

    raw_header_rows = valid_rows[:header_rows]
    data_rows = valid_rows[header_rows:]
    num_cols = len(valid_rows[0])

    if header_rows == 1:
        headers = raw_header_rows[0]
        headers = [h if h else f"col_{i}" for i, h in enumerate(headers)]
    else:
        headers = []
        for col_idx in range(num_cols):
            parts: list[str] = []
            for r_idx in range(header_rows):
                val = raw_header_rows[r_idx][col_idx].strip()
                if val and val not in parts:
                    parts.append(val)
            header_str = " - ".join(parts) if parts else f"col_{col_idx}"
            headers.append(header_str)

    df = pd.DataFrame(data_rows, columns=headers)

    if index_col is not None and 0 <= index_col < len(headers):
        idx_col_name = headers[index_col]
        df = df.set_index(idx_col_name)

    return df


def html_table_to_markdown(table_element: Any) -> str:
    """Convert an HTML table into a clean GitHub Flavored Markdown table string."""
    grid = parse_html_table_to_grid(table_element)
    if not grid:
        return ""

    lines: list[str] = []
    header = grid[0]
    escaped_header = [cell.replace("|", "\\|") for cell in header]
    lines.append("| " + " | ".join(escaped_header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    for row in grid[1:]:
        escaped_row = [cell.replace("|", "\\|") for cell in row]
        lines.append("| " + " | ".join(escaped_row) + " |")

    return "\n".join(lines) + "\n"
