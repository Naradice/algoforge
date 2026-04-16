# Architecture

## Three-Layer Design

AlgoForge separates concerns into three independent but connected layers. Each layer has its own API, database models, worker tasks, and UI pages. Layers communicate only through the HTTP API and shared artifact paths — there are no direct cross-layer imports in the backend.

```
┌─────────────────────────────────────────────────────────────────┐
│  STRATEGY LAYER                                                  │
│  Strategy definition → execution (backtest / paper / live)      │
│  Uses: datasets (from Data), deployed models (from ML)          │
├─────────────────────────────────────────────────────────────────┤
│  ML MODEL LAYER                                                  │
│  Train → validate → deploy prediction models                    │
│  Uses: datasets (from Data)                                      │
├─────────────────────────────────────────────────────────────────┤
│  DATA LAYER                                                      │
│  Collect → store → analyse time-series and documents            │
│  Independent of Strategy and ML layers                          │
└─────────────────────────────────────────────────────────────────┘
```

### Cross-layer relationships

| Consumer | Depends on | Via |
|----------|------------|-----|
| Strategy execution | Dataset | `dataset_id` in StrategyRun → reads artifact parquet |
| Strategy condition | Deployed model | `model_id` in condition definition → calls `/models/{id}/predict` |
| ML training | Dataset | `dataset_id` in TrainingRun → reads artifact parquet |
| ML validation | Dataset | `dataset_id` in ModelValidation → reads artifact parquet |

All cross-layer calls go through the HTTP API or artifact file paths. No Python imports cross layer boundaries.

---

## Backend Structure

```
backend/
├── main.py                  # FastAPI app, router registration, exception handler
├── database.py              # SQLAlchemy async engine, Base, get_db dependency
├── celery_worker.py         # Celery task definitions (collection, training, backtest, validation)
├── celery_app.py            # Celery application instance, queue definitions, task routing
├── events.py                # In-process EventBus for SSE streaming
├── log_writer.py            # Async log writer (buffer → DB)
├── schemas.py               # DataResponse[T] envelope, Meta pagination
├── pagination.py            # Pagination dependency
│
├── strategy/
│   ├── router.py            # /strategies/* endpoints
│   ├── config_router.py     # /strategy-config/* (handlers, event types, brokers)
│   ├── service.py           # Business logic
│   ├── repository.py        # DB queries
│   ├── models.py            # ORM + Pydantic schemas
│   ├── executor.py          # Dispatches to backtest or live runner
│   ├── engine/backtest.py   # Bar-by-bar backtest engine
│   └── live_runner.py       # Paper/live runner (paper implemented, live placeholder)
│
├── model/
│   ├── router.py            # /models/* endpoints
│   ├── training_runs_router.py  # /training-runs/* endpoints
│   ├── config_router.py     # /model-config/* (architectures, schemas)
│   ├── service.py           # Business logic
│   ├── repository.py        # DB queries
│   ├── models.py            # ORM + Pydantic schemas
│   ├── training.py          # Training loop (PyTorch)
│   └── inference.py         # Model loading, prediction
│
├── data/
│   ├── router.py            # /datasources/*, /datasets/*, /collection-jobs/* endpoints
│   ├── service.py           # Business logic
│   ├── repository.py        # DB queries
│   ├── models.py            # ORM + Pydantic schemas
│   ├── characteristics.py   # Hurst, ACF, kurtosis, volatility analysis registry
│   └── collectors/
│       ├── ohlc.py              # OHLC download via finance_client (yfinance / Alpha Vantage)
│       ├── ddm_simulator.py     # Deterministic Dealer Model synthetic data
│       ├── economic_calendar.py # Economic indicators (FRED, Alpha Vantage)
│       └── web_report.py        # Financial report scraping (PDF, HTML)
│
├── webhooks/
│   ├── router.py            # /webhooks/* endpoints
│   ├── models.py            # ORM + Pydantic schemas
│   └── dispatcher.py        # HMAC-signed webhook dispatch
│
├── mcp_server/
│   ├── __init__.py          # FastMCP app, resources
│   └── tools/
│       ├── data.py          # Data MCP tools
│       ├── model.py         # Model MCP tools
│       ├── strategy.py      # Strategy MCP tools
│       └── logs.py          # Log MCP tools
│
└── alembic/
    └── versions/
        ├── 0001_initial_schema.py
        ├── 0002_training_run_stop_and_metrics.py
        └── 0003_missing_schema_columns.py
```

---

## Frontend Structure

```
web/app/
├── layout.tsx               # Root layout with sidebar nav
├── page.tsx                 # Redirects → /dashboard
├── dashboard/page.tsx       # Summary counts
├── data/
│   ├── page.tsx             # List datasources + datasets
│   ├── new/page.tsx         # Create datasource form
│   ├── datasources/[id]/
│   │   ├── page.tsx         # Datasource detail + collection jobs
│   │   └── new-job/page.tsx # Create collection job
│   └── datasets/[id]/page.tsx  # Dataset detail (preview, characteristics)
├── model/
│   ├── page.tsx             # List models
│   ├── new/page.tsx         # Create model form
│   ├── compare/page.tsx     # Side-by-side training run comparison
│   └── [id]/
│       ├── page.tsx         # Model detail (training runs, deploy)
│       └── training-runs/[run_id]/page.tsx  # Training run detail (loss chart)
├── strategy/
│   ├── page.tsx             # List strategies
│   ├── new/page.tsx         # Create strategy form
│   └── [id]/
│       ├── page.tsx         # Strategy detail (runs, definition editor)
│       └── runs/[run_id]/page.tsx  # Run detail (equity, trades, chat)
└── settings/
    ├── page.tsx             # Settings index
    ├── api-keys/page.tsx    # API key management (stub)
    ├── brokers/page.tsx     # Broker connections
    └── webhooks/page.tsx    # Webhook registration
```

---

## Key Design Decisions

### Response envelope
All API responses use `DataResponse[T]`:
```json
{ "data": <T>, "meta": { "total": N, "page": N, "page_size": N } }
```
The frontend `fetcher` automatically unwraps `.data`. Use `fetcherWithMeta` when pagination counts are needed.

### Async jobs via Celery
Long-running operations (collection, training, backtest) are enqueued to Redis and processed by Celery workers. The HTTP endpoint returns 202 Accepted immediately. The UI polls via SWR `refreshInterval` or subscribes to SSE for live updates.

Jobs are routed to named queues with independent worker concurrency:

| Queue | Default concurrency | Job types |
|-------|-------------------|-----------|
| `collection` | 3 | DDM simulation, OHLC download, CSV ingest |
| `characteristics` | 12 | Statistical analysis (Hurst, ACF, kurtosis, …) |
| `training` | 2 | PyTorch model training, validation |
| `backtest` | 5 | Strategy backtest runs |

Each Celery worker is a **separate OS process** (prefork pool). A blocking or OOM-failing job in one process cannot affect jobs running in other processes. Concurrency limits are tuned to disk I/O capacity for collection (100 GB+ writes) and RAM/GPU availability for training.

Celery Beat handles scheduled collection jobs (replaces arq cron scheduling).

### SSE streaming
Running jobs publish events to an in-process `EventBus`. Clients subscribe via:
- `GET /strategies/{id}/runs/{run_id}/events` — strategy run events
- `GET /training-runs/{run_id}/events` — training epoch events

### Artifact store
Dataset artifacts use **date-partitioned Parquet** directories. PyTorch checkpoints are stored as single `.pt` files. All paths are stored in DB as relative strings. Default location: `backend/artifacts/`.

Dataset layout:
```
artifacts/datasets/src_{id}/
  year=2024/month=01/day=15/part-000.parquet
  year=2024/month=01/day=15/part-001.parquet
  year=2024/month=01/day=16/part-000.parquet
  ...
```

Partition files are sized to ~200–500 MB each. The PyArrow dataset API is used for all reads — it prunes partitions by date range before scanning, so backtest and training jobs only read the files they actually need. The DDM simulator streams writes partition-by-partition via `pyarrow.parquet.ParquetWriter`, never buffering the full dataset in memory.

### finance_client integration

The data layer depends on [`finance_client`](../../finance_client/) — the shared provider library located at `../finance_client/` relative to this repo. It supplies the download logic for OHLC data sources so AlgoForge does not re-implement provider-specific API handling.

**Import pattern in collectors:**
```python
from finance_client.yfinance import download_ohlc as yf_download_ohlc
from finance_client.vantage import download_ohlc as vantage_download_ohlc
```

Each `download_ohlc()` function instantiates the provider client with `data_only=True`, which skips account/risk-manager initialisation (those subsystems are only needed for live trading). The collector then filters the result to the requested date range and writes a parquet artifact.

**Timeframe mapping** — AlgoForge uses string timeframes (`"M1"`, `"H1"`, `"D1"`, …); `finance_client` uses integer frame constants (`Frame.MIN1`, `Frame.H1`, `Frame.D1`, …). `collectors/ohlc.py` owns the `_FRAME_MAP` translation.

**H4 note** — yfinance has no native 4-hour interval. The collector downloads H1 bars and resamples with `df.resample("4h")`.

**Adding a new provider:**
1. Add `data_only` to the provider's `__init__` chain in `finance_client` and pass it to `super()`
2. Create `finance_client/<provider>/downloader.py` with a `download_ohlc()` function
3. Export it from `finance_client/<provider>/__init__.py`
4. Add a branch to `collect()` in `backend/data/collectors/ohlc.py`

### Strategy definition format
Stored as JSONB in `strategies.definition`. Schema is intentionally flexible — the executor reads it at runtime. See [strategy-layer.md](strategy-layer.md) for the schema.
