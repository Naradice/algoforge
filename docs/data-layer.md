# Data Layer

## Concepts

```
Datasource  →  CollectionJob  →  (Celery worker)  →  Dataset  →  Artifact (partitioned parquet)
                                                           ↓
                                                  DataCharacteristics
```

A **Datasource** describes *where* data comes from and *how* to fetch it.
A **Dataset** is the result of one collection run — a parquet file plus metadata.
**DataCharacteristics** are statistical properties computed from a Dataset.

---

## Datasource Types

### `ohlc_download`
Downloads OHLCV candles from a market data provider.

```json
{
  "type": "ohlc_download",
  "config": {
    "client": "yfinance",
    "symbol": "EURUSD=X",
    "timeframe": "H1",
    "from_ts": "2020-01-01",
    "to_ts": ""
  }
}
```

Supported clients: `yfinance` (free, no credentials), `vantage` (Alpha Vantage API key required).

Timeframes: `M1`, `M5`, `M15`, `M30`, `H1`, `H4`, `D1`.

### `ddm_simulation`
Generates synthetic OHLCV data using the Deterministic Dealer Model v3.

```json
{
  "type": "ddm_simulation",
  "config": {
    "num_agent": "50",
    "initial_price": "100",
    "length": "50000",
    "tick_time": "1",
    "timeframe": "M1",
    "seed": "42"
  }
}
```

Key parameters:
- `length`: number of ticks to simulate
- `tick_time`: simulated seconds per tick (controls time compression)
- `timeframe`: OHLC aggregation period
- `seed`: reproducibility seed

Rule of thumb for row count: `rows ≈ length × tick_time / (timeframe_seconds)`.
Example: 50000 ticks × 1s / 60s = ~833 M1 bars.

### `manual_upload`
Upload a CSV or Parquet file directly via `POST /datasets/upload`.

### `web_report`
Downloads financial reports (PDFs, HTML, audio) from institution websites using Playwright.
Config schema mirrors `cyclic_downloader/source.json`.

```json
{
  "type": "web_report",
  "config": {
    "url": "https://www.mizuho-sc.com/market/report.html",
    "ext": "pdf",
    "subfolder": "みずほ証券",
    "filename": "{YYMMDD}_digest.pdf",
    "type": "goto_download",
    "unique": "text",
    "interval_days": 1,
    "custom": [
      {
        "type": "link_parse",
        "targets": [
          {
            "value": "global_market_digest\\.pdf",
            "ext": "pdf",
            "filename": "{YYMMDD}_global_market_digest.pdf",
            "type": "goto_download",
            "unique": "text",
            "interval_days": 1
          }
        ]
      }
    ]
  }
}
```

| Field | Description |
|-------|-------------|
| `url` | Landing page URL |
| `ext` | File type: `pdf`, `html`, `mp3`, `txt` |
| `subfolder` | Output directory under `artifacts/web_reports/` |
| `filename` | Filename template — see placeholders below |
| `type` | Fetch method: `load` (HTTP), `goto_load` (browser→PDF), `goto_download` (browser fetch, bypasses CDN), `load_rep` (HTTP+save HTML) |
| `unique` | Dedup: `segment` (skip if file exists), `checksum` (skip if content unchanged), `text` (skip if link text unchanged) |
| `interval_days` | Min days between downloads; `null` = download once only |
| `custom` | Optional list of `link_parse` / `element_parse` steps for multi-level scraping |

**Filename placeholders:** `{YYYYMMDD}` `{YYMMDD}` `{YYYYMM}` `{YYMM}` `{filename}` `{basefilename}`

**Prerequisite:** Playwright Chromium must be installed in the worker environment:
```bash
playwright install chromium
```

### `economic_calendar`
Downloads historical economic indicator releases (CPI, NFP, unemployment, interest rate decisions) from Alpha Vantage or FRED.

```json
{
  "type": "economic_calendar",
  "config": {
    "source": "alpha_vantage",
    "api_key": "YOUR_KEY",
    "indicators": ["CPI", "NONFARM_PAYROLL", "UNEMPLOYMENT", "FEDERAL_FUNDS_RATE"],
    "interval": "monthly",
    "from_ts": "2020-01-01",
    "to_ts": ""
  }
}
```

**Alpha Vantage indicators** (`source: "alpha_vantage"`, free key at alphavantage.co):
`CPI`, `NONFARM_PAYROLL`, `UNEMPLOYMENT`, `FEDERAL_FUNDS_RATE`, `REAL_GDP`, `REAL_GDP_PER_CAPITA`, `RETAIL_SALES`, `DURABLES`, `TREASURY_YIELD`, `INFLATION`

**FRED series** (`source: "fred"`, free key at fred.stlouisfed.org):
Use short names like `CPI`, `NONFARM_PAYROLL`, `UNEMPLOYMENT`, `FEDERAL_FUNDS_RATE`, `GDP`, `RETAIL_SALES`, `DURABLES`, `TREASURY_YIELD` — or raw FRED series IDs (`CPIAUCSL`, `PAYEMS`, `UNRATE`, `FEDFUNDS`, `GDP`, etc.).

**Output schema** (long format, partitioned parquet):

| Column | Type | Description |
|--------|------|-------------|
| `datetime` | timestamp UTC | Release date (index) |
| `indicator` | string | Indicator name (e.g. `CPI`) |
| `value` | float | Released value |
| `unit` | string | Unit string from provider (e.g. `percent`, `thousands of persons`) |

Note: Free Alpha Vantage keys are limited to 5 API calls/minute — the collector adds a 0.5 s delay between indicator fetches.

---

## Collection Flow

1. Create a datasource: `POST /datasources`
2. One of:
   - **Immediate run**: `POST /datasources/{id}/collect` — finds or creates a job, enqueues immediately
   - **Manual job**: `POST /collection-jobs` with `datasource_id` → `POST /collection-jobs/{id}/run`
   - **Scheduled job**: `POST /collection-jobs` with `schedule_cron` (e.g. `"0 * * * *"` for hourly) — executed by Celery Beat
3. Celery worker (on the `collection` queue) picks up `run_collection_job`, runs the appropriate collector
4. The collector streams output to disk partition-by-partition — memory usage stays bounded regardless of dataset size
5. On success: a new `Dataset` is created with `status: ready`
6. On failure: `CollectionJob.last_error` is updated, `status: error`

Multiple collection jobs run in parallel in separate OS processes (up to the queue concurrency limit). Jobs do not share memory and a crash in one does not affect others.

---

## Dataset Characteristics

Characteristics describe the statistical properties of a time series. They help compare datasets and guide model/strategy selection.

Trigger: `POST /datasets/{id}/characteristics/compute`
Retrieve: `GET /datasets/{id}/characteristics`

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| `hurst_exponent` | Long-range dependence | > 0.5: trending, < 0.5: mean-reverting, = 0.5: random walk |
| `acf_lag1` | Autocorrelation at lag 1 | High: momentum effect |
| `kurtosis` | Fat tails | > 3: more extreme moves than Gaussian |
| `volatility` | Annualised std of returns | Baseline risk measure |
| `skewness` | Return distribution asymmetry | Negative: more downside risk |
| `adf_statistic` | Augmented Dickey-Fuller test | Low p-value: stationary |

The `CHARACTERISTIC_REGISTRY` in `backend/data/characteristics.py` is extensible — add new analysis functions by decorating them with `@register_characteristic`.

---

## Adding a New Datasource Type

1. Create `backend/data/collectors/my_source.py` with a `collect(datasource_id, config) → CollectResult` function. Write output as partitioned parquet using `pyarrow.parquet.ParquetWriter` — do not accumulate rows in memory.
2. Add the type string to `backend/data/models.py` (comment in `Datasource.type`)
3. Add the dispatch case in `backend/celery_worker.py` `run_collection_job()`
4. Add a default config in `web/app/data/new/page.tsx` `TYPE_CONFIGS`
5. Optionally add a schema to `GET /datasource-config/types/{type_name}`

---

## Artifact Store

All dataset artifacts are stored under `artifacts/datasets/src_{datasource_id}/` as **date-partitioned Parquet directories**.
Path root: `ARTIFACT_STORE_PATH` env var (default: `backend/artifacts/`).

### Partition layout

```
artifacts/datasets/src_5/ddm_ticks/
  year=2024/month=01/day=15/part-000.parquet   # ~200–500 MB each
  year=2024/month=01/day=15/part-001.parquet
  year=2024/month=01/day=16/part-000.parquet
  ...
```

The Dataset record stores a relative path to the **directory** (e.g. `datasets/src_5/ddm_ticks`). The backend resolves it with `Path(ARTIFACT_STORE_PATH) / dataset.artifact_path`.

### Reading

All reads go through the PyArrow dataset API:

```python
import pyarrow.dataset as ds

dataset = ds.dataset(artifact_path, format="parquet", partitioning="hive")

# Date-range filter — only scans the relevant partition files
table = dataset.to_table(filter=(ds.field("year") == 2024) & (ds.field("month") == 1))
```

This means backtest and training jobs only read the partition files they need — no full-scan of a 100 GB file.

### Downloading artifacts

- `GET /datasets/{id}/download` exports the dataset as CSV.
- `GET /datasets/{id}/artifact` downloads the original stored artifact:
  - file-backed datasets are returned directly
  - directory-backed partitioned datasets are packaged as a zip archive

The MCP tool `get_dataset_download(dataset_id)` returns the artifact path plus the API download URL for the full dataset contents.

### Writing

Collectors write partitions incrementally using `pyarrow.parquet.ParquetWriter`. New partitions are flushed to disk as data arrives; the full dataset never needs to fit in memory.

Before starting a new collection run the existing partition directory is cleared to avoid mixing old and new data (stale artifact accumulation bug).
