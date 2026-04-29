# AlgoForge — Documentation Index

AlgoForge is a three-layer algorithmic trading platform designed for both human use (web UI) and AI-driven automation (MCP / API).


Initial Request
---
please help to create consolidate web app for projects we have in this folder: agentic_trade, cyclic_downloader, finance_client, stocknet, trade_strategy and
trade_viewer.
Some of them have backend, scripts, fronend. I want to use them from web UI, keeping modularity especialy finance_client.
My current assumption is as followings.
We have 3 layers. Trade Strategy Management, ML Model Management, Data Management.

Trade Strategy Management
It has API to manage (CRUD) trade strategy. Human or AI agent will use the API to continue improving their trade.
A strategy consists of events and conditions. An event is like time step to judge a condition (e.g. every 1 min with ohlc data), economic event, technical indicator update, etc. A       
condition output true/false based on an event data. The condition is handled by logically, agentic AI, AI model, traditional ML etc. So we can do the same as what we are doing on the    
trade_strategy or agentic_trade with events and conditions. Human or AI agent can receive strategy results periodicaly or when they want it with API response or webhook.
On the web, we have a UI which use the API to manage strategies and also a UI to show events and market data for the strategy. If a straegy has an event for a user input, the user can   
send a query from chat ui of the UI.
So AI agent and human can manage trade strategy through API and UI. The human can see and use it on UI.

ML Model Management                    
It has API to manage ML model. Human or AI will use the API to develop better ML models.                                                                                                  
We have web UI to manage models and view validation results. Human or AI agent can receive validation results priodically or when they want it with API response or webhook.              
Trained ML model can be used by a strategy or on data management.
      
Data Management
We have several options to register data.  1. data generation with ML model or deterministic agents, etc. 2. data collection like cyclic download or web scraping. We can register many   
types of data. Time series data of FX, Stock, Bond or Crypt Currency, Market report published by a company, Staements, and so on. We can analyse charcteristics of registerd data. Now we have fixed validation on stocknet web app, but it should be more extensionable.
 
We want to maintain loose coupling, but we want to avoid duplication of development efforts. We want to design a user interface that is easy for humans to use, but we also want to make  
it available as an MCP to promote AI-driven automation. Feel free to suggest improvements to the layer structure or integration, or even propose additional features. Please organize     
useful documentation under the “docs” directory to help us build a web app that integrates these projects.
---

## Documents

| File | Description |
|------|-------------|
| [architecture.md](architecture.md) | System design, layer overview, data flow |
| [roadmap.md](roadmap.md) | Phased feature and UX improvement plan |
| [strategy-layer.md](strategy-layer.md) | Strategy definition, events, conditions, execution |
| [strategy-execution-migration.md](strategy-execution-migration.md) | Canonical execution model and migration plan for retiring `trade_strategy` |
| [data-layer.md](data-layer.md) | Data sources, collection, characteristics |
| [model-layer.md](model-layer.md) | ML model lifecycle, training, deployment |
| [mcp-guide.md](mcp-guide.md) | Using the platform as an AI agent via MCP |

## Quick Start

```bash
# Backend
cd algoforge/backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Celery workers (separate terminals — one per queue group)
celery -A celery_worker worker -Q collection      -c 3  --loglevel=info
celery -A celery_worker worker -Q characteristics -c 12 --loglevel=info
celery -A celery_worker worker -Q training        -c 2  --loglevel=info
celery -A celery_worker worker -Q backtest        -c 5  --loglevel=info

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
       │ Celery (Redis broker)          │ PostgreSQL
┌──────▼───────────────────────┐  ┌─────▼───────────────────┐
│   Celery Workers             │  │  PostgreSQL + JSONB      │
│                              │  │  strategies, models,     │
│  [collection]     c=3        │  │  datasets, runs, trades  │
│  [characteristics] c=12      │  └─────────────────────────┘
│  [training]       c=2        │
│  [backtest]       c=5        │
└──────────────────────────────┘
       │ artifacts (partitioned parquet / checkpoints)
┌──────▼───────────────────────┐
│  File Store                  │
│  artifacts/                  │
│  datasets/src_N/             │
│    year=YYYY/month=MM/day=DD/│
│      part-NNN.parquet        │
└──────────────────────────────┘
```
