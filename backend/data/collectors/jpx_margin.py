"""Collector for JPX Margin Trading Data from public endpoints."""

from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup
import requests

from data.extractors.margin import (
    DailyMarginIssue,
    MarginRegulationEvent,
    MarketMarginRecord,
    parse_daily_margin_announcements,
    parse_daily_margin_excel,
    parse_market_margin_excel,
)

logger = logging.getLogger(__name__)

JPX_BASE_URL = "https://www.jpx.co.jp"
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


class JPXMarginCollector:
    """Client for fetching and parsing daily margin publications from JPX."""

    def __init__(self, session: requests.Session | None = None, timeout: int = 15):
        self.session = session or requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)
        self.timeout = timeout

    def fetch_daily_margin_announcements(self) -> dict[str, list[MarginRegulationEvent]]:
        """Fetch latest daily margin designation & lifting announcements.

        Returns
        -------
        dict[str, list[MarginRegulationEvent]]
            {'designated': [...], 'lifted': [...]}
        """
        url = urljoin(JPX_BASE_URL, "/markets/equities/margin-daily/index.html")
        logger.info("Fetching daily margin announcements from %s", url)
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()

        return parse_daily_margin_announcements(resp.content)

    def fetch_latest_daily_margin_balances(self) -> tuple[str, list[DailyMarginIssue]]:
        """Discover and download the latest daily margin excel file (mtdaily.xlsx).

        Returns
        -------
        tuple[str, list[DailyMarginIssue]]
            (file_url, list of parsed DailyMarginIssue)
        """
        index_url = urljoin(JPX_BASE_URL, "/markets/statistics-equities/margin/index.html")
        logger.info("Discovering latest daily margin excel link from %s", index_url)
        resp = self.session.get(index_url, timeout=self.timeout)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.content, "html.parser")
        excel_href = None
        for a in soup.find_all("a"):
            href = a.get("href", "")
            if "mtdaily.xlsx" in href:
                excel_href = href
                break

        if not excel_href:
            raise RuntimeError(f"Could not find mtdaily.xlsx link on {index_url}")

        download_url = urljoin(index_url, excel_href)
        logger.info("Downloading daily margin excel from %s", download_url)
        excel_resp = self.session.get(download_url, timeout=self.timeout)
        excel_resp.raise_for_status()

        issues = parse_daily_margin_excel(excel_resp.content)
        return download_url, issues

    def fetch_market_margin_history(self) -> list[MarketMarginRecord]:
        """Download and parse market-wide margin history excel."""
        url_05 = urljoin(JPX_BASE_URL, "/markets/statistics-equities/margin/05.html")
        logger.info("Discovering market margin history link from %s", url_05)
        resp = self.session.get(url_05, timeout=self.timeout)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.content, "html.parser")
        excel_href = None
        for a in soup.find_all("a"):
            href = a.get("href", "")
            if href.endswith(".xlsx") and "tvdivq" in href:
                excel_href = href
                break

        if not excel_href:
            raise RuntimeError(f"Could not find market margin history excel link on {url_05}")

        download_url = urljoin(url_05, excel_href)
        logger.info("Downloading market margin history from %s", download_url)
        excel_resp = self.session.get(download_url, timeout=self.timeout)
        excel_resp.raise_for_status()

        return parse_market_margin_excel(excel_resp.content)
