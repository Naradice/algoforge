# Data Layer

## Concepts

```
Datasource  →  CollectionJob  →  (arq worker)  →  Dataset  →  Artifact (parquet)
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

### `web_report` *(not yet implemented)*
Scrapes structured data from a web page (earnings tables, economic calendars, etc.) using Playwright.

### `economic_calendar` *(planned)*
Fetches scheduled economic events (NFP, CPI, interest rate decisions) from a provider.

---

## Collection Flow

1. Create a datasource: `POST /datasources`
2. One of:
   - **Immediate run**: `POST /datasources/{id}/collect` — finds or creates a job, enqueues immediately
   - **Manual job**: `POST /collection-jobs` with `datasource_id` → `POST /collection-jobs/{id}/run`
   - **Scheduled job**: `POST /collection-jobs` with `schedule_cron` (e.g. `"0 * * * *"` for hourly)
3. arq worker picks up `run_collection_job`, runs the appropriate collector
4. On success: a new `Dataset` is created with `status: ready`
5. On failure: `CollectionJob.last_error` is updated, `status: error`

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

1. Create `backend/data/collectors/my_source.py` with a `collect(datasource_id, config) → CollectResult` function
2. Add the type string to `backend/data/models.py` (comment in `Datasource.type`)
3. Add the dispatch case in `backend/arq_worker.py` `run_collection_job()`
4. Add a default config in `web/app/data/new/page.tsx` `TYPE_CONFIGS`
5. Optionally add a schema to `GET /datasource-config/types/{type_name}`

---

## Artifact Store

All parquet files are stored under `artifacts/datasets/src_{datasource_id}/`.
Path: `ARTIFACT_STORE_PATH` env var (default: `backend/artifacts/`).

File naming: `ddm_M1_len50000.parquet`, `ohlc_EURUSD_H1_20200101_20240101.parquet`, etc.

The Dataset record stores a relative path. The backend resolves it with `Path(ARTIFACT_STORE_PATH) / dataset.artifact_path`.
