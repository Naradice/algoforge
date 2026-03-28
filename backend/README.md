# AlgoForge Backend

FastAPI application — REST API, WebSocket, background jobs (arq), and MCP server.

## Requirements

- Python 3.12+
- PostgreSQL 16
- Redis 7 (optional — set `ALGOFORGE_NO_REDIS=1` to skip)

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env    # fill in DATABASE_URL, REDIS_URL, GOOGLE_API_KEY
alembic upgrade head    # create/migrate the database
```

## Running

```bash
# API server
uvicorn main:app --reload --port 8000

# Background job worker (separate terminal)
python -m arq arq_worker.WorkerSettings
```

API docs: http://localhost:8000/docs
MCP endpoint: http://localhost:8000/mcp

## Project structure

```
backend/
├── main.py               FastAPI app, CORS, router registration, startup hooks
├── database.py           SQLAlchemy async engine + session factory + Base
├── arq_worker.py         arq WorkerSettings + all job functions
├── arq_pool.py           get_arq_pool() + enqueue() helper used by routers
├── events.py             Event bus — InProcessEventBus (dev) / RedisEventBus (prod)
├── auth.py               JWT auth helpers (currently bypassed by ALGOFORGE_NO_AUTH)
├── loger.py              StructuredLogger — writes to logs table via ContextVar
├── log_writer.py         Async batch writer that flushes to DB every 50 ms
├── log_models.py         Log ORM model (separate to avoid circular imports)
├── logs_router.py        GET /logs, GET /logs/summary — log query endpoints
├── ws_router.py          WebSocket /ws/strategies/{id}/runs/{run_id}/chat
├── webhook_models.py     Webhook payload models (Phase 5+)
│
├── alembic/              Database migrations
│   └── versions/
│       └── 0001_initial_schema.py
│
├── data/                 Data management layer
│   ├── models.py         ORM: Datasource, CollectionJob, Dataset, DataCharacteristics
│   ├── repository.py     DB queries
│   ├── service.py        Business logic
│   ├── router.py         REST endpoints
│   ├── characteristics.py  ACF, Hurst, fat-tail, QQ, diffusion, seasonality analysis
│   └── collectors/
│       ├── ohlc.py       yfinance + Alpha Vantage downloader → parquet
│       ├── ddm_simulator.py  Synthetic tick data (DDM v3)
│       └── web_report.py     Playwright stub (not implemented)
│
├── model/                ML model layer
│   ├── models.py         ORM: MLModel, TrainingRun, TrainingCheckpoint, ModelValidation
│   ├── repository.py
│   ├── service.py
│   ├── router.py         REST endpoints + inference endpoint
│   ├── inference.py      predict() with in-process model cache
│   ├── validation.py     validate_supervised() + validate_gan()
│   ├── architectures/
│   │   ├── __init__.py   build_model() + TRAINING_DEFAULTS
│   │   ├── transformer.py  Seq2SeqTransformer
│   │   ├── lstm.py         LSTM encoder-decoder
│   │   └── gan.py          TimeGAN (generator + discriminator)
│   └── trainers/
│       ├── __init__.py   get_trainer_fns() dispatch
│       ├── dataset.py    OHLCWindowDataset (obs/pred windows, normalisation)
│       ├── supervised.py train_epoch + eval_epoch for LSTM/Transformer
│       └── gan_trainer.py  GAN adversarial training loop
│
├── strategy/             Strategy layer
│   ├── models.py         ORM: Strategy, StrategyRun, StrategyEvent, Trade, RunMetric, StrategyRunChat
│   ├── repository.py
│   ├── service.py        Business logic + stop_run()
│   ├── router.py         REST endpoints
│   ├── executor.py       execute_strategy_run() arq job — backtest + paper dispatch
│   ├── live_runner.py    run_paper() — yfinance polling loop
│   ├── chat_agent.py     stream_response() — Gemini chat with strategy context
│   ├── handlers/         ConditionHandler ABC + HANDLER_REGISTRY
│   ├── analysis/         (reserved for future statistical analysis)
│   ├── events/           (reserved for event schema definitions)
│   └── engine/           Core execution engine
│       ├── indicators.py   apply_indicators() — MACD, RSI, ATR, EMA, SMA, BB, Slope
│       ├── conditions.py   evaluate_conditions() — comparison, ml_signal, llm_signal
│       ├── backtest.py     run_backtest() — bar-by-bar simulator + metrics
│       ├── ml_condition.py evaluate_ml_condition() — calls model.inference.predict()
│       └── llm_condition.py  evaluate_llm_condition() — Gemini API with bar cache
│
└── mcp_server/           FastMCP server (mounted at /mcp)
    ├── __init__.py       FastMCP("AlgoForge") instance
    └── tools/
        ├── logs.py       get_run_logs, search_logs, get_log_summary
        ├── strategy.py   list_strategies, get_run_metrics, get_run_trades, …
        ├── model.py      list_models, get_model_validations, compare_model_runs, …
        └── data.py       list_datasets, get_dataset_characteristics, …
```

## Background jobs

All heavy work runs in the arq worker process, not the API process.

| Job | Trigger | Description |
|-----|---------|-------------|
| `run_collection_job` | `POST /collection-jobs/{id}/run` | Download/simulate OHLC data |
| `compute_characteristics` | `POST /datasets/{id}/characteristics/compute` | Statistical analysis |
| `train_model` | `POST /models/{id}/training-runs` | ML model training loop |
| `validate_model` | (internal) | Post-training validation metrics |
| `execute_strategy_run` | `POST /strategies/{id}/runs` | Backtest or paper trading |

Workers share the Redis queue but are separate processes. The API enqueues via `arq_pool.enqueue()`.

## Database schema

17 tables in 5 groups:

| Group | Tables |
|-------|--------|
| Auth | `users`, `api_keys` |
| Data | `datasources`, `collection_jobs`, `datasets`, `data_characteristics` |
| Model | `ml_models`, `training_runs`, `training_checkpoints`, `model_validations` |
| Strategy | `strategies`, `strategy_runs`, `strategy_events`, `trades`, `run_metrics`, `strategy_run_chats` |
| Logging | `logs` |

Migrations are managed with Alembic. Run `alembic upgrade head` to apply.

## Strategy definition format

Strategies are stored as JSONB in `strategies.definition`. Full format:

```json
{
  "symbol": "AAPL",
  "timeframe": "1d",
  "indicators": [
    {"id": "macd", "type": "macd", "params": {"fast": 12, "slow": 26, "signal_period": 9}},
    {"id": "rsi",  "type": "rsi",  "params": {"period": 14}},
    {"id": "atr",  "type": "atr",  "params": {"period": 14}}
  ],
  "entry": {
    "direction": "buy",
    "conditions": [
      {"left": "macd_line",   "op": ">",  "right": "macd_signal"},
      {"left": "rsi",         "op": "<",  "right": 70},
      {"type": "ml_signal",   "model_id": 1, "direction": "buy", "step": 1},
      {"type": "llm_signal",  "direction": "buy", "lookback": 10}
    ],
    "logic": "and"
  },
  "exit": {
    "conditions": [
      {"left": "macd_line", "op": "<", "right": "macd_signal"}
    ],
    "logic": "or"
  },
  "risk": {
    "sl_pct": 0.02,
    "tp_pct": 0.04,
    "position_size": 1.0
  }
}
```

### Condition types

| Type | Fields | Description |
|------|--------|-------------|
| Comparison (default) | `left`, `op`, `right` | Compare column to column or constant |
| `ml_signal` | `model_id`, `direction`, `step`, `min_confidence` | Deployed model prediction direction |
| `llm_signal` | `direction`, `lookback`, `model`, `columns`, `cache` | Gemini market signal |

### Supported indicators

| Type | Output columns | Parameters |
|------|---------------|------------|
| `macd` | `{id}_line`, `{id}_signal`, `{id}_hist` | `fast`, `slow`, `signal_period` |
| `rsi` | `{id}` | `period` |
| `atr` | `{id}` | `period` |
| `ema` | `{id}` | `period`, `column` |
| `sma` | `{id}` | `period`, `column` |
| `bb` | `{id}_upper`, `{id}_middle`, `{id}_lower` | `period`, `std` |
| `slope` | `{id}` | `period` |

## Logging

All components log via `StructuredLogger` from `loger.py`. Logs are written to the `logs` table with:
- Automatic run ID propagation via `contextvars.ContextVar`
- Batch writes flushed every 50 ms (up to 200 rows)
- Retention: DEBUG 7 days, INFO 90 days, WARNING 1 year, ERROR permanent

Query logs via `GET /api/v1/logs` or MCP tools.

## Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL async URL | required |
| `REDIS_URL` | Redis URL | `redis://localhost:6379/0` |
| `ARTIFACT_STORE_PATH` | Root directory for parquet/model files | `artifacts` |
| `GOOGLE_API_KEY` | Gemini API key | — |
| `LLM_MODEL` | Gemini model name | `gemini-2.0-flash` |
| `ALGOFORGE_NO_REDIS` | `1` = use in-process event bus | `0` |
| `ALGOFORGE_NO_AUTH` | `1` = skip auth checks | `0` |
| `ALGOFORGE_DEBUG` | `1` = DEBUG log level | `0` |
| `PAPER_CHECK_INTERVAL_S` | Paper trading candle poll interval | `60` |

## Code style

Black, line length 150, target Python 3.12. Run `black .` before committing.
