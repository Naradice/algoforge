# AlgoForge

Unified algorithmic trading platform — strategy backtesting, ML model training, paper/live execution, and AI-assisted analysis.

## Architecture

```
algoforge/
├── backend/      FastAPI + arq — REST API, background jobs, MCP server
├── web/          Next.js 14 — dashboard UI
├── ml_worker/    Python 3.8 arq worker — RL model training (PFRL/gym)
└── infra/        Docker Compose — postgres, redis, services
```

### Component overview

| Component | Role | Port |
|-----------|------|------|
| `backend` | REST API, WebSocket, MCP server, arq worker | 8000 |
| `web` | Next.js dashboard | 3000 |
| `ml_worker` | RL training jobs (Python 3.8) | — |
| PostgreSQL | Primary database | 5432 |
| Redis | arq job queue, event bus | 6379 |

### Three layers

```
Strategy layer   — define conditions, run backtests, paper/live trade
Model layer      — train & deploy Transformer / LSTM / TimeGAN / RL models
Data layer       — collect OHLC data, run DDM simulation, compute characteristics
```

Each layer follows the same structure: `router → service → repository → ORM models`.

## Quick start

### Docker (recommended)

```bash
cd infra
cp .env.example .env          # edit GOOGLE_API_KEY etc.
docker compose up --build
```

Open http://localhost:3000.

### Local development

Prerequisites: Python 3.12, Node 20, PostgreSQL 16, Redis 7.

```bash
# 1. Start infrastructure
cd infra && docker compose up postgres redis -d

# 2. Backend
cd backend
pip install -r requirements.txt
cp .env.example .env           # edit as needed
alembic upgrade head
uvicorn main:app --reload --port 8000

# 3. arq worker (separate terminal)
cd backend
python -m arq arq_worker.WorkerSettings

# 4. Frontend
cd web
npm install
npm run dev
```

## Feature overview

### Data management (`/data`)
- Download OHLC data from yfinance or Alpha Vantage
- Simulate synthetic tick data using DDM (Directional Change + DDM v3)
- Compute dataset characteristics: ACF, Hurst exponent, fat tails, seasonality

### ML models (`/model`)
- Architectures: **Seq2Seq Transformer**, **LSTM**, **TimeGAN**
- Training with epoch-level progress tracking and checkpoint saving
- Validation metrics: directional accuracy, MAE, RMSE, Sharpe proxy (supervised); ACF match, Hurst diff, kurtosis (GAN)
- One-click deploy; cached inference for low-latency prediction

### Strategy backtesting & trading (`/strategy`)
- JSON strategy definition with composable conditions:
  - **Comparison** — `{"left": "macd_line", "op": ">", "right": "macd_signal"}`
  - **ML signal** — `{"type": "ml_signal", "model_id": 1, "direction": "buy"}`
  - **LLM signal** — `{"type": "llm_signal", "direction": "buy", "lookback": 10}`
- Built-in indicators: MACD, RSI, ATR, EMA, SMA, Bollinger Bands, Slope
- Backtest mode: bar-by-bar simulation with SL/TP, metrics, trade log
- Paper mode: live yfinance polling loop, stoppable via API
- Per-run metrics: win rate, total PnL, Sharpe ratio, max drawdown, profit factor
- AI chat: Gemini-powered assistant with full strategy context (WebSocket)

### MCP server (`/mcp`)
Exposes 16 tools for Claude Desktop and other MCP clients:

| Domain | Tools |
|--------|-------|
| Logs | `get_run_logs`, `search_logs`, `get_log_summary` |
| Strategy | `list_strategies`, `get_strategy_definition`, `get_strategy_runs`, `get_run_metrics`, `get_run_trades` |
| Model | `list_models`, `get_model_training_runs`, `get_model_validations`, `compare_model_runs` |
| Data | `list_datasets`, `get_dataset_characteristics`, `list_datasources`, `get_dataset_preview` |

**Claude Desktop config** (`claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "algoforge": { "url": "http://localhost:8000/mcp" }
  }
}
```

## Environment variables

See `backend/.env.example` for all variables. Key ones:

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | localhost/algoforge |
| `REDIS_URL` | Redis connection string | localhost:6379 |
| `ARTIFACT_STORE_PATH` | Where model/dataset files are saved | `../artifacts` |
| `GOOGLE_API_KEY` | Gemini API key (for LLM conditions + chat) | — |
| `ALGOFORGE_NO_REDIS` | Use in-process event bus instead of Redis | `0` |
| `PAPER_CHECK_INTERVAL_S` | Paper trading poll interval (seconds) | `60` |
| `LLM_MODEL` | Gemini model name | `gemini-2.0-flash` |

## API reference

Interactive docs available at http://localhost:8000/docs (Swagger UI).

Main endpoints:

```
GET  /api/v1/health
GET  /api/v1/datasets
POST /api/v1/collection-jobs/{id}/run
POST /api/v1/datasets/{id}/characteristics/compute

GET  /api/v1/models
POST /api/v1/models
POST /api/v1/models/{id}/training-runs
POST /api/v1/models/{id}/predict
POST /api/v1/models/{id}/deploy

GET  /api/v1/strategies
POST /api/v1/strategies
POST /api/v1/strategies/{id}/runs
POST /api/v1/strategies/{id}/runs/{run_id}/stop
GET  /api/v1/strategies/{id}/runs/{run_id}/metrics
GET  /api/v1/strategies/{id}/runs/{run_id}/trades

GET  /api/v1/logs
WS   /api/v1/ws/strategies/{id}/runs/{run_id}/chat
```

## Project layout

```
algoforge/
├── backend/
│   ├── main.py               App entry point
│   ├── arq_worker.py         Background job definitions
│   ├── arq_pool.py           Job enqueue helper
│   ├── database.py           SQLAlchemy async engine
│   ├── events.py             In-process / Redis event bus
│   ├── ws_router.py          WebSocket endpoints
│   ├── logs_router.py        Log query endpoints
│   ├── mcp_server/           FastMCP tools
│   ├── data/                 Data layer (models, router, collectors)
│   ├── model/                ML layer (architectures, trainers, inference)
│   └── strategy/             Strategy layer (engine, executor, live runner)
├── web/
│   ├── app/                  Next.js App Router pages
│   └── components/           Shared UI components
├── ml_worker/                Python 3.8 RL worker
└── infra/
    └── docker-compose.yml
```

## Related projects

This project imports (as editable installs) from sibling directories:
- `finance_client/` — broker/data client abstractions, risk management, indicators
- `trade_strategy/` — strategy execution framework, signal generation
