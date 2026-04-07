"""
Web report collector — downloads financial reports (PDFs, HTML, audio) from
financial institution websites. Config schema mirrors cyclic_downloader/source.json.

Datasource config shape:
    {
        "url": "https://www.mizuho-sc.com/market/report.html",
        "ext": "pdf",
        "subfolder": "みずほ証券",
        "filename": "{YYMMDD}_digest.pdf",
        "type": "load",              # load | goto_load | goto_download | load_rep
        "unique": "segment",         # segment | checksum | text
        "interval_days": 1,          # null = download once only
        "custom": null               # null = direct download; list = multi-step (see below)
    }

Custom step shape (link_parse / element_parse):
    [
        {
            "type": "link_parse",
            "targets": [
                {
                    "filename": "{YYYYMMDD}.pdf",
                    "value": "/path/to/.*\\.pdf",   # regex matched against href
                    "ext": "pdf",
                    "unique": "segment",
                    "type": "load",
                    "interval_days": 1,
                    "custom": null                  # recursive
                }
            ]
        },
        {
            "type": "element_parse",
            "targets": [
                {
                    "selector": "div.card a",
                    "value": ".*some-pattern",
                    "filename": "{YYYYMMDD}.html",
                    "ext": "html",
                    "unique": "checksum",
                    "type": "goto_load",
                    "interval_days": 1,
                    "custom": null
                }
            ]
        }
    ]

Filename placeholders (resolved against today's date + target URL):
    {YYYYMMDD}  {YYMMDD}  {YYYYMM}  {YYMM}  {filename}  {basefilename}
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

logger = logging.getLogger(__name__)

ARTIFACT_STORE = Path(os.getenv("ARTIFACT_STORE_PATH", "artifacts"))

# Match cyclic_downloader's User-Agent so Akamai/CDN bot-checks pass
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/119.0.0.0 Safari/537.36"
)
# Launch arg + init script that hides the automation fingerprint
_STEALTH_ARGS = [
    "--disable-blink-features=AutomationControlled",
    # Required when running as root inside Docker
    "--no-sandbox",
    "--disable-setuid-sandbox",
    # Prevents crashes due to /dev/shm size limits in Docker
    "--disable-dev-shm-usage",
]
_STEALTH_SCRIPT = "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"


def _launch_browser(p):
    """
    Launch browser with stealth settings.
    Tries the real system Chrome first (most authentic fingerprint, bypasses Akamai/CDN).
    Falls back to Playwright's bundled Chromium if Chrome is not installed.
    """
    launch_kwargs = dict(
        headless=True,
        args=_STEALTH_ARGS,
    )
    try:
        browser = p.chromium.launch(channel="chrome", **launch_kwargs)
        logger.info("[browser] launched real Chrome (channel=chrome)")
        return browser
    except Exception:
        pass
    try:
        browser = p.chromium.launch(channel="msedge", **launch_kwargs)
        logger.info("[browser] launched Microsoft Edge (channel=msedge)")
        return browser
    except Exception:
        pass
    browser = p.chromium.launch(**launch_kwargs)
    logger.info("[browser] launched bundled Chromium (Chrome/Edge not found)")
    return browser


def _stealth_context(browser):
    """Create a browser context with bot-detection countermeasures applied."""
    ctx = browser.new_context(
        user_agent=_UA,
        viewport={"width": 1280, "height": 800},
        extra_http_headers={
            "Accept-Language": "ja-JP,ja;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        },
    )
    ctx.add_init_script(_STEALTH_SCRIPT)
    return ctx


# ── Link preview (called synchronously from an executor thread) ───────────────


def preview_links(url: str, ext: str | None = None) -> list[dict]:
    """
    Navigate to *url* with a headless browser, extract every <a href> link,
    and return them annotated with whether they match *ext*.

    Returns a list of dicts: {href, text, matches_ext, filename}
    Raises RuntimeError if Playwright / Chromium is not installed.
    """
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    except ImportError as exc:
        raise RuntimeError(
            "playwright not installed. Run: pip install playwright && playwright install chromium"
        ) from exc

    logger.info("[preview_links] starting scan: %s (ext=%s)", url, ext)

    with sync_playwright() as p:
        try:
            browser = _launch_browser(p)
        except Exception as exc:
            raise RuntimeError(
                "Playwright Chromium not installed. Run: playwright install chromium"
            ) from exc

        ctx = _stealth_context(browser)
        page = ctx.new_page()
        try:
            # First wait: DOM ready
            try:
                response = page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                status = response.status if response else "?"
                logger.info("[preview_links] page loaded — HTTP %s, title=%r", status, page.title())
            except PWTimeout:
                logger.warning("[preview_links] domcontentloaded timed out — proceeding anyway")

            # Second wait: let JS finish rendering links (up to 10 s extra)
            try:
                page.wait_for_load_state("networkidle", timeout=10_000)
                logger.info("[preview_links] networkidle reached")
            except PWTimeout:
                logger.info("[preview_links] networkidle timed out — extracting links with current DOM")

            raw = page.evaluate(
                """() => Array.from(document.querySelectorAll('a[href]')).map(a => ({
                    href: a.href,
                    text: a.textContent.trim().slice(0, 100)
                }))"""
            )
            logger.info("[preview_links] raw <a> tags extracted: %d", len(raw))
        finally:
            page.close()
            ctx.close()
            browser.close()

    # Deduplicate by href, skip non-http links
    seen: set[str] = set()
    links: list[dict] = []
    ext_lower = (ext or "").lower().lstrip(".")

    for item in raw:
        href: str = item.get("href") or ""
        if not href.startswith("http") or href in seen:
            continue
        seen.add(href)

        path = urlparse(href).path.lower()
        matches = bool(ext_lower and path.endswith(f".{ext_lower}"))
        filename = Path(urlparse(href).path).name or ""

        links.append({
            "href": href,
            "text": item.get("text") or "",
            "filename": filename,
            "matches_ext": matches,
        })

    logger.info(
        "[preview_links] done — %d total links, %d match .%s",
        len(links), sum(1 for l in links if l["matches_ext"]), ext or ""
    )
    # Return matching links first, then the rest; cap total at 300
    matched = [l for l in links if l["matches_ext"]]
    others  = [l for l in links if not l["matches_ext"]]
    return (matched + others)[:300]


def _annotate_links(raw: list[dict], base_url: str, ext_lower: str) -> list[dict]:
    """Deduplicate raw {href, text} list and annotate with matches_ext / filename."""
    seen: set[str] = set()
    links: list[dict] = []
    for item in raw:
        href: str = item.get("href") or ""
        if not href.startswith("http") or href in seen:
            continue
        seen.add(href)
        path = urlparse(href).path.lower()
        matches = bool(ext_lower and path.endswith(f".{ext_lower}"))
        links.append({
            "href": href,
            "text": item.get("text") or "",
            "filename": Path(urlparse(href).path).name or "",
            "matches_ext": matches,
        })
    matched = [l for l in links if l["matches_ext"]]
    others  = [l for l in links if not l["matches_ext"]]
    return (matched + others)[:300]


def test_fetch(url: str, fetch_type: str, ext: str | None = None) -> dict:
    """
    Test whether *url* is accessible using *fetch_type*, exactly as the collector would.

    Returns:
        {
            "success": bool,
            "status_code": int | None,
            "content_type": str | None,
            "size_bytes": int | None,
            "title": str | None,        # goto_load only
            "links": [...],             # populated for HTML responses
            "link_matches": int,
            "error": str | None,
        }
    """
    result: dict = {
        "fetch_type": fetch_type,
        "success": False,
        "status_code": None,
        "content_type": None,
        "size_bytes": None,
        "title": None,
        "links": [],
        "link_matches": 0,
        "error": None,
    }
    ext_lower = (ext or "").lower().lstrip(".")
    logger.info("[test_fetch] %s  type=%s  ext=%s", url, fetch_type, ext_lower or "-")

    # ── HTTP-only methods ─────────────────────────────────────────────────────
    if fetch_type in ("load", "load_rep"):
        try:
            import httpx
            r = httpx.get(url, headers=_HEADERS, follow_redirects=True, timeout=20)
            ct = r.headers.get("content-type", "").split(";")[0].strip()
            size = int(r.headers.get("content-length") or 0) or len(r.content)
            result.update(status_code=r.status_code, content_type=ct, size_bytes=size)
            if r.status_code >= 400:
                result["error"] = f"HTTP {r.status_code}"
            else:
                result["success"] = True
                if "html" in ct:
                    import re as _re
                    hrefs = _re.findall(r'href=["\']([^"\']+)["\']', r.text)
                    raw = [{"href": _abs_url(h, url), "text": ""} for h in hrefs
                           if not h.startswith(("javascript:", "mailto:", "tel:", "#"))]
                    result["links"] = _annotate_links(raw, url, ext_lower)
                    result["link_matches"] = sum(1 for l in result["links"] if l["matches_ext"])
        except Exception as exc:
            result["error"] = str(exc)
        logger.info("[test_fetch] result: %s", result)
        return result

    # ── Browser-based methods ─────────────────────────────────────────────────
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    except ImportError as exc:
        result["error"] = "playwright not installed"
        return result

    with sync_playwright() as p:
        try:
            browser = _launch_browser(p)
        except Exception as exc:
            result["error"] = str(exc)
            return result

        ctx = _stealth_context(browser)
        page = ctx.new_page()
        try:
            if fetch_type == "goto_download":
                # Navigate to origin first so fetch() runs with a proper Origin header
                # (fetch from about:blank sends Origin: null which many servers reject)
                try:
                    _parsed = urlparse(url)
                    _origin = f"{_parsed.scheme}://{_parsed.netloc}/"
                    try:
                        page.goto(_origin, wait_until="domcontentloaded", timeout=30_000)
                    except Exception:
                        page.goto("about:blank")
                    info = page.evaluate(
                        """async (u) => {
                            const r = await fetch(u);
                            const buf = await r.arrayBuffer();
                            return {
                                status: r.status,
                                contentType: r.headers.get('content-type') || '',
                                size: buf.byteLength,
                            };
                        }""",
                        url,
                    )
                    ct = info["contentType"].split(";")[0].strip()
                    result.update(
                        status_code=info["status"],
                        content_type=ct,
                        size_bytes=info["size"],
                    )
                    if info["status"] >= 400:
                        result["error"] = f"HTTP {info['status']}"
                    else:
                        result["success"] = True
                except Exception as exc:
                    result["error"] = str(exc)

            else:  # goto_load
                try:
                    response = page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                    try:
                        page.wait_for_load_state("networkidle", timeout=10_000)
                    except PWTimeout:
                        pass
                    status = response.status if response else None
                    ct = (response.headers.get("content-type", "") if response else "").split(";")[0].strip()
                    result.update(status_code=status, content_type=ct, title=page.title())
                    if (status or 0) >= 400:
                        result["error"] = f"HTTP {status}"
                    else:
                        result["success"] = True
                        raw = page.evaluate(
                            """() => Array.from(document.querySelectorAll('a[href]')).map(a => ({
                                href: a.href, text: a.textContent.trim().slice(0, 100)
                            }))"""
                        )
                        result["links"] = _annotate_links(raw, url, ext_lower)
                        result["link_matches"] = sum(1 for l in result["links"] if l["matches_ext"])
                except PWTimeout:
                    result["error"] = "Page load timed out"
                except Exception as exc:
                    result["error"] = str(exc)
        finally:
            page.close()
            ctx.close()
            browser.close()

    logger.info("[test_fetch] result: success=%s status=%s ct=%s size=%s",
                result["success"], result["status_code"], result["content_type"], result["size_bytes"])
    return result


_STATE_FILE = "last_check_dates.json"
_CHECKSUM_DIR = "checksums"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/119.0.0.0 Safari/537.36"
    )
}


@dataclass
class CollectResult:
    artifact_path: str
    row_count: int
    from_ts: datetime
    to_ts: datetime


# ── Public entry point ────────────────────────────────────────────────────────


def collect(datasource_id: int, config: dict, *, force: bool = False) -> CollectResult:
    url: str = config["url"]
    ext: str = config.get("ext", "pdf")
    subfolder: str = config.get("subfolder", f"src_{datasource_id}")
    filename_tpl: str = config.get("filename", f"{{YYYYMMDD}}.{ext}")
    fetch_type: str = config.get("type", "load")
    unique = config.get("unique", "segment")
    interval_days = config.get("interval_days", 1)
    custom: list | None = config.get("custom")

    reports_root = ARTIFACT_STORE / "web_reports"
    out_dir = reports_root / subfolder
    out_dir.mkdir(parents=True, exist_ok=True)
    (reports_root / _CHECKSUM_DIR).mkdir(parents=True, exist_ok=True)

    state_file = reports_root / _STATE_FILE
    if not force and interval_days is not None and not _should_run(url, interval_days, state_file):
        logger.info("[collect] skipping %s — interval_days=%s not elapsed", url, interval_days)
        files = [f for f in out_dir.rglob("*") if f.is_file()]
        now = datetime.now(tz=timezone.utc)
        return CollectResult(
            artifact_path=f"web_reports/{subfolder}",
            row_count=len(files),
            from_ts=now,
            to_ts=now,
        )

    # browser_state: [context | None, playwright_cm | None, browser | None]
    browser_state: list = [None, None, None]
    files_downloaded = 0

    try:
        if custom is None:
            filename = _resolve_filename(filename_tpl, url, ext)
            dest = out_dir / filename
            if _save_file(url, dest, fetch_type, unique, reports_root, browser_state):
                files_downloaded += 1
        else:
            files_downloaded = _process_custom(url, custom, out_dir, reports_root, browser_state, force=force)
    finally:
        _close_browser(browser_state)

    _update_last_check(url, state_file)

    now = datetime.now(tz=timezone.utc)
    return CollectResult(
        artifact_path=f"web_reports/{subfolder}",
        row_count=files_downloaded,
        from_ts=now,
        to_ts=now,
    )


# ── Custom multi-step processing ──────────────────────────────────────────────


def _process_custom(
    page_url: str, steps: list, out_dir: Path, reports_root: Path, browser_state: list,
    *, force: bool = False,
) -> int:
    total = 0
    for step in steps:
        step_type = step.get("type")
        targets = step.get("targets", [])

        if step_type == "link_parse":
            links = _extract_links(page_url, browser_state)
            for target in targets:
                pattern = target.get("value", "")
                for link in links:
                    if re.search(pattern, link.get("href") or ""):
                        href = _abs_url(link["href"], page_url)
                        total += _handle_target(href, link.get("text", ""), target, out_dir, reports_root, browser_state, force=force)

        elif step_type == "element_parse":
            for target in targets:
                selector = target.get("selector", "a")
                pattern = target.get("value", "")
                for val in _extract_elements(page_url, selector, browser_state):
                    if re.search(pattern, val or ""):
                        abs_val = _abs_url(val, page_url)
                        total += _handle_target(abs_val, val, target, out_dir, reports_root, browser_state, force=force)

    return total


def _handle_target(
    url: str,
    link_text: str,
    target: dict,
    out_dir: Path,
    reports_root: Path,
    browser_state: list,
    *,
    force: bool = False,
) -> int:
    inner_custom = target.get("custom")
    if inner_custom:
        return _process_custom(url, inner_custom, out_dir, reports_root, browser_state, force=force)

    interval_days = target.get("interval_days", 1)
    state_file = reports_root / _STATE_FILE
    if not force and interval_days is not None and not _should_run(url, interval_days, state_file):
        return 0

    ext = target.get("ext", "pdf")
    filename = _resolve_filename(target.get("filename", "{filename}"), url, ext)
    dest = out_dir / filename
    fetch_type = target.get("type", "load")
    unique = target.get("unique", "segment")

    ok = _save_file(url, dest, fetch_type, unique, reports_root, browser_state, link_text=link_text)
    if ok and interval_days is not None:
        _update_last_check(url, state_file)
    return 1 if ok else 0


# ── File saving ───────────────────────────────────────────────────────────────


def _save_file(
    url: str,
    dest: Path,
    fetch_type: str,
    unique,
    reports_root: Path,
    browser_state: list,
    *,
    link_text: str = "",
) -> bool:
    checksum_file = reports_root / _CHECKSUM_DIR / (_url_key(url) + ".txt")
    dest.parent.mkdir(parents=True, exist_ok=True)

    if fetch_type == "load":
        try:
            data = _http_binary(url)
        except Exception as exc:
            logger.warning("[save_file] load failed for %s: %s", url, exc)
            return False
        if not _dedup_ok(unique, dest, data, checksum_file, link_text):
            return False
        dest.write_bytes(data)
        _dedup_save(unique, data, checksum_file, link_text)
        return True

    if fetch_type in ("goto_load", "goto_download"):
        if fetch_type == "goto_load":
            if unique == "segment" and dest.exists():
                return False
            try:
                _playwright_save_pdf(url, dest, browser_state)
            except Exception as exc:
                logger.warning("[save_file] goto_load failed for %s: %s", url, exc)
                return False
            return dest.exists()
        else:
            try:
                data = _playwright_fetch_binary(url, browser_state)
            except Exception as exc:
                logger.warning("[save_file] goto_download failed for %s: %s", url, exc)
                return False
            if not _dedup_ok(unique, dest, data, checksum_file, link_text):
                return False
            dest.write_bytes(data)
            _dedup_save(unique, data, checksum_file, link_text)
            return True

    if fetch_type == "load_rep":
        try:
            html = _http_text(url)
        except Exception:
            return False
        data = html.encode()
        if not _dedup_ok(unique, dest, data, checksum_file, link_text):
            return False
        if dest.suffix not in (".html", ".htm"):
            dest = dest.with_suffix(".html")
        dest.write_bytes(data)
        _dedup_save(unique, data, checksum_file, link_text)
        return True

    return False


# ── Playwright helpers ────────────────────────────────────────────────────────


def _init_browser(state: list) -> object:
    """Lazily initialise a stealth browser context. Returns the context (not the browser)."""
    if state[0] is not None:
        return state[0]  # already a context
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "playwright not installed. Run: pip install playwright && playwright install chromium"
        ) from exc
    pw_cm = sync_playwright()
    p = pw_cm.__enter__()
    try:
        browser = _launch_browser(p)
    except Exception as exc:
        pw_cm.__exit__(type(exc), exc, exc.__traceback__)
        raise RuntimeError(
            "No browser available. Run: playwright install chromium  (or install Chrome/Edge)"
        ) from exc
    ctx = _stealth_context(browser)
    state[0] = ctx      # callers do state[0].new_page() → context page
    state[1] = pw_cm
    state[2] = browser  # kept for cleanup
    return ctx


def _close_browser(state: list) -> None:
    # state layout: [context | None, pw_cm | None, browser | None]
    for i in (0, 2):  # close context, then browser
        if len(state) > i and state[i] is not None:
            try:
                state[i].close()
            except Exception:
                pass
    if state[1] is not None:
        try:
            state[1].__exit__(None, None, None)
        except Exception:
            pass


def _playwright_save_pdf(url: str, dest: Path, state: list) -> None:
    from playwright.sync_api import TimeoutError as PWTimeout

    browser = _init_browser(state)
    page = browser.new_page()
    try:
        try:
            response = page.goto(url, wait_until="networkidle", timeout=60_000)
        except PWTimeout:
            response = None
        if response is not None and response.status >= 400:
            raise RuntimeError(f"HTTP {response.status} navigating to {url}")
        page.pdf(path=str(dest), format="A4", print_background=False)
    finally:
        page.close()


def _playwright_fetch_binary(url: str, state: list) -> bytes:
    from playwright.sync_api import TimeoutError as PWTimeout

    browser = _init_browser(state)
    page = browser.new_page()
    try:
        # Navigate to the origin first so fetch() runs with a proper Origin header
        # (fetch from about:blank sends Origin: null which many servers reject)
        parsed = urlparse(url)
        origin_url = f"{parsed.scheme}://{parsed.netloc}/"
        try:
            page.goto(origin_url, wait_until="domcontentloaded", timeout=30_000)
        except Exception:
            page.goto("about:blank")
        data = page.evaluate(
            """async (u) => {
                const r = await fetch(u);
                const buf = await r.arrayBuffer();
                return Array.from(new Uint8Array(buf));
            }""",
            url,
        )
        return bytes(data)
    finally:
        page.close()


def _extract_links(url: str, state: list) -> list[dict]:
    from playwright.sync_api import TimeoutError as PWTimeout

    browser = _init_browser(state)
    page = browser.new_page()
    try:
        try:
            page.goto(url, wait_until="networkidle", timeout=60_000)
        except PWTimeout:
            pass
        return page.evaluate(
            """() => Array.from(document.querySelectorAll('a[href]')).map(a => ({
                href: a.getAttribute('href'),
                text: a.textContent.trim()
            }))"""
        )
    finally:
        page.close()


def _extract_elements(url: str, selector: str, state: list) -> list[str]:
    from playwright.sync_api import TimeoutError as PWTimeout

    browser = _init_browser(state)
    page = browser.new_page()
    try:
        try:
            page.goto(url, wait_until="networkidle", timeout=60_000)
        except PWTimeout:
            pass
        values = page.evaluate(
            """(sel) => Array.from(document.querySelectorAll(sel)).map(el => {
                const tag = el.tagName.toLowerCase();
                if (tag === 'a') return el.getAttribute('href') || el.textContent;
                if (['img','video','audio','source'].includes(tag)) return el.getAttribute('src');
                return el.textContent.trim();
            })""",
            selector,
        )
        return [v for v in values if v]
    finally:
        page.close()


# ── HTTP helpers ──────────────────────────────────────────────────────────────


def _http_binary(url: str) -> bytes:
    import httpx
    r = httpx.get(url, headers=_HEADERS, follow_redirects=True, timeout=30)
    r.raise_for_status()
    return r.content


def _http_text(url: str) -> str:
    import httpx
    r = httpx.get(url, headers=_HEADERS, follow_redirects=True, timeout=30)
    r.raise_for_status()
    return r.text


# ── Deduplication ─────────────────────────────────────────────────────────────


def _dedup_ok(unique, dest: Path, data: bytes, checksum_file: Path, link_text: str) -> bool:
    if isinstance(unique, dict):
        return True  # selector-based: allow (requires browser eval; skip for now)

    if unique == "segment":
        return not dest.exists()

    if unique in ("checksum", "text"):
        payload = link_text.encode() if unique == "text" else data
        new_hash = hashlib.sha256(payload).hexdigest()
        if checksum_file.exists() and checksum_file.read_text().strip() == new_hash:
            # Skip only if the file is also still on disk — re-download if it was deleted
            return not dest.exists()
        return True

    return True


def _dedup_save(unique, data: bytes, checksum_file: Path, link_text: str) -> None:
    if unique in ("checksum", "text"):
        payload = link_text.encode() if unique == "text" else data
        checksum_file.parent.mkdir(parents=True, exist_ok=True)
        checksum_file.write_text(hashlib.sha256(payload).hexdigest())


# ── State (interval tracking) ─────────────────────────────────────────────────


def _load_state(state_file: Path) -> dict:
    if state_file.exists():
        try:
            return json.loads(state_file.read_text())
        except Exception:
            pass
    return {}


def _should_run(url: str, interval_days: int, state_file: Path) -> bool:
    last_str = _load_state(state_file).get(url)
    if not last_str:
        return True
    try:
        return (date.today() - date.fromisoformat(last_str)).days >= interval_days
    except ValueError:
        return True


def _update_last_check(url: str, state_file: Path) -> None:
    state = _load_state(state_file)
    state[url] = date.today().isoformat()
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state, indent=2, ensure_ascii=False))


# ── Filename / URL utilities ──────────────────────────────────────────────────


def _resolve_filename(template: str, url: str, ext: str) -> str:
    today = date.today()
    parsed = urlparse(url)
    url_filename = Path(parsed.path).name or f"report.{ext}"
    url_base = Path(parsed.path).stem or "report"

    name = (
        template
        .replace("{YYYYMMDD}", today.strftime("%Y%m%d"))
        .replace("{YYMMDD}", today.strftime("%y%m%d"))
        .replace("{YYYYMM}", today.strftime("%Y%m"))
        .replace("{YYMM}", today.strftime("%y%m"))
        .replace("{filename}", url_filename)
        .replace("{basefilename}", url_base)
    )
    # Ensure file has an extension
    if not Path(name).suffix:
        name = f"{name}.{ext}"
    return name


def _abs_url(href: str, base_url: str) -> str:
    return urljoin(base_url, href)


def _url_key(url: str) -> str:
    parsed = urlparse(url)
    key = f"{parsed.hostname or ''}_{parsed.path}".replace("/", "_").strip("_")
    return re.sub(r"[^\w\-]", "_", key)[:120]
