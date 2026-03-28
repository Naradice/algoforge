# AlgoForge — Documentation Index

AlgoForge is a three-layer algorithmic trading platform designed for both human use (web UI) and AI-driven automation (MCP / API).

## Documents

| File | Description |
|------|-------------|
| [architecture.md](architecture.md) | System design, layer overview, data flow |
| [roadmap.md](roadmap.md) | Phased feature and UX improvement plan |
| [strategy-layer.md](strategy-layer.md) | Strategy definition, events, conditions, execution |
| [data-layer.md](data-layer.md) | Data sources, collection, characteristics |
| [model-layer.md](model-layer.md) | ML model lifecycle, training, deployment |
| [mcp-guide.md](mcp-guide.md) | Using the platform as an AI agent via MCP |

## Quick Start

```bash
# Backend
cd algoforge/backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Worker (separate terminal)
python -m arq arq_worker.WorkerSettings

# Frontend
cd algoforge/web
npm install && npm run dev
```

## Platform Overview

```
┌─────────────────────────────────────────────────────────┐
│                   Web UI (Next.js)                       │
│   Dashboard · Strategy · ML Model · Data · Settings     │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP / SSE / WebSocket
┌────────────────────▼────────────────────────────────────┐
│              FastAPI Backend  :8000                      │
│   /api/v1/{strategy,model,data,logs,webhooks}           │
│   /mcp  (MCP endpoint for AI agents)                    │
└──────┬────────────────────────────────┬─────────────────┘
       │ arq (Redis queue)              │ PostgreSQL
┌──────▼───────────┐          ┌─────────▼───────────────┐
│   arq Worker     │          │  PostgreSQL + JSONB      │
│ • collection     │          │  strategies, models,     │
│ • training       │          │  datasets, runs, trades  │
│ • backtest       │          └─────────────────────────┘
│ • validation     │
└──────────────────┘
       │ artifacts (parquet / checkpoints)
┌──────▼───────────┐
│  File Store      │
│  artifacts/      │
└──────────────────┘
```
