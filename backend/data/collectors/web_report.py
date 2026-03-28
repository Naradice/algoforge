"""
Web report collector — scrapes structured data (economic calendars, financial statements)
using Playwright. Replaces the Node.js cyclic_downloader for Python-native scraping.

Datasource config shape:
    {
        "url": "https://example.com/data",
        "selector": "table.data",        # CSS selector for the data table
        "date_col": "Date",               # column name containing the date
        "scrape_script": null,            # optional: path to a custom scrape script
        "format": "table" | "json",       # how to parse the scraped content
    }

TODO Phase 1: implement Playwright scraping.
Currently raises NotImplementedError — will be implemented when finance/cyclic_downloader
is ported or sidecar file-watcher is set up.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ARTIFACT_STORE = Path(os.getenv("ARTIFACT_STORE_PATH", "artifacts"))


@dataclass
class CollectResult:
    artifact_path: str
    row_count: int
    from_ts: datetime
    to_ts: datetime


def collect(datasource_id: int, config: dict) -> CollectResult:  # noqa: ARG001
    """
    TODO Phase 1: implement Playwright-based web scraping.
    """
    raise NotImplementedError(
        "web_report collector not yet implemented. "
        "Either run cyclic_downloader as a sidecar or implement Playwright scraping in Phase 1."
    )
