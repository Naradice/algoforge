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
├── arq_worker.py            # Background job functions (collection, training, execution)
├── arq_pool.py              # Redis/arq connection pool, enqueue helper
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
│       ├── ohlc_downloader.py   # yfinance / Alpha Vantage download
│       ├── ddm_simulator.py     # Deterministic Dealer Model synthetic data
│       └── web_report.py        # Playwright web scraping (NOT IMPLEMENTED)
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

### Async jobs via arq
Long-running operations (collection, training, backtest) are enqueued to Redis and processed by the arq worker. The HTTP endpoint returns 202 Accepted immediately. The UI polls via SWR `refreshInterval` or subscribes to SSE for live updates.

### SSE streaming
Running jobs publish events to an in-process `EventBus`. Clients subscribe via:
- `GET /strategies/{id}/runs/{run_id}/events` — strategy run events
- `GET /training-runs/{run_id}/events` — training epoch events

### Artifact store
Parquet files (datasets) and PyTorch checkpoints (models) are stored under `artifacts/`. Path stored in DB as relative string. Default location: `backend/artifacts/`.

### Strategy definition format
Stored as JSONB in `strategies.definition`. Schema is intentionally flexible — the executor reads it at runtime. See [strategy-layer.md](strategy-layer.md) for the schema.
