# AlgoForge Roadmap

This roadmap is organized by the three platform layers plus cross-cutting concerns. Items are grouped into phases: **Now** (known gaps, quick wins), **Next** (planned features), and **Later** (longer-horizon ideas).

---

## Data Layer

### Now
- [x] Fix `web_report` collector — Playwright scraper for earnings tables, economic calendars
- [x] Improve collection error display in the UI: show `last_error` text inline on the datasource card instead of just "error" status
- [x] Add dataset delete — remove parquet file and DB record together
- [x] Empty state on the datasets list (currently blank when no datasets exist)

### Next
- [x] `economic_calendar` datasource type — fetch scheduled events (NFP, CPI, rate decisions) from a provider
- [x] Dataset tagging / search — filter datasets by symbol, timeframe, source type
- [x] Incremental collection — append new bars to an existing dataset instead of re-downloading everything
- [x] Dataset merge — economic calendar events auto-discovered and overlaid on the run price chart by timestamp alignment; full UI-driven merge deferred to later
- [x] Characteristic auto-compute on collection completion — auto-enqueued on the characteristics queue after every successful collection (except economic_calendar datasources)

### Later
- [ ] Streaming ingest — write live ticks to a dataset in real time for perpetual paper-data feeds
- [ ] Data quality monitor — alert when a dataset drifts (stationarity change, gap detected)

---

## ML Model Layer

### Now
- [x] Show training loss chart on the model detail page — exists on training run detail page (polling-based; SSE cross-process gap noted)
- [x] Hyperparameter search results page — search form on model detail page; compare page shows overlaid val_loss curves per run
- [x] Empty state on the models list

### Next
- [x] Model comparison page wired to live data — now also joins model size (`num_params`) and validation performance metrics against training-data characteristics (Hurst, periodicity, entropy, regime changes, ...) in a "Data × Model Analysis" scatter plot
- [ ] Validation job UI — trigger `POST /models/{id}/validations` from the model detail page and show results
- [ ] Sharpe proxy and directional accuracy displayed on the model card (currently only shown in raw JSON)
- [ ] Architecture selection guidance — tooltip explaining when to use LSTM vs Transformer vs RL agent
- [ ] Export trained model weights (download checkpoint as `.pt` file)

### Later
- [ ] Online learning — continue training a deployed model on new data without full retraining
- [ ] Ensemble inference — average predictions from multiple deployed models
- [ ] AutoML mode — agent-driven hyperparameter search that stops when val_loss converges

---

## Strategy Layer

### Now
- [x] Equity curve chart on the run detail page — OHLC candlestick + indicator overlays + trade markers + oscillator sub-panels + equity curve
- [x] Trade table on the run detail page — individual trades with entry/exit/SL/TP/PnL/reason; exit_reason persisted via migration 0004
- [x] Chat panel on the run detail page — Gemini AI wired end-to-end; bubble layout with thinking indicator and auto-scroll
- [x] Empty state on the strategies list — also improved runs sub-list in strategy detail and training runs in model detail

### Next
- [ ] Version history UI — show `GET /strategies/{id}/versions` diff view alongside run metrics to correlate definition changes with performance
- [ ] Paper trading dashboard — live equity, open position, live log stream for a running paper strategy
- [ ] `rule_engine` condition handler — multi-condition AND/OR trees (currently only `comparison`, `ml_signal`, `llm_signal`)
- [ ] Run comparison view — overlay equity curves from multiple runs on one chart
- [ ] Stop-loss / take-profit visualisation on trade chart

### Later
- [ ] Live trading mode — connect MT5 / Coincheck broker, submit real orders (backend placeholder exists)
- [ ] `agentic` condition handler — AI agent picks action autonomously at each bar
- [ ] `user_input` condition handler — pause execution and wait for human decision via chat UI
- [ ] `economic_event` trigger — fire strategy logic on NFP / CPI release

---

## Cross-Cutting / Infrastructure

### Now
- [ ] **Celery migration** — replace arq with Celery + prefork workers; separate queues for `collection`, `characteristics`, `training`, `backtest` with independent concurrency limits
- [ ] **Partitioned Parquet** — replace flat per-dataset parquet files with date-partitioned directories (`year=/month=/day=/`); update all collectors to stream-write via `ParquetWriter`; update all readers to use PyArrow dataset API with partition pruning
- [ ] **Celery Beat** — replace arq cron scheduling with Celery Beat for scheduled collection jobs
- [ ] Process manager — `docker-compose` service definitions for each Celery worker queue
- [ ] Unified error toast in the frontend — currently some API errors are silently swallowed
- [ ] Loading skeletons on detail pages (strategy, model, dataset) — currently blank while SWR loads

### Next
- [ ] Authentication — JWT-based login so the platform is not open by default
- [ ] Settings → API Keys page (currently a stub) — store Alpha Vantage key, broker credentials
- [ ] Webhook registration UI polish — show last fired timestamp, last status, delivery history
- [ ] Flower monitoring UI — optional Celery task dashboard (task history, worker status, queue depths)
- [ ] Dark mode — design system uses Tailwind; straightforward to add `dark:` variants
- [ ] Mobile-responsive layout — sidebar nav collapses to hamburger menu

### Later
- [ ] Multi-user support — workspaces scoped per user
- [ ] Notifications — email / Slack alerts on run completion, training error, collection failure
- [ ] Hosted deployment guide — Docker Compose with Postgres, Redis, backend, worker, frontend

---

## MCP / AI Agent Experience

### Now
- [ ] MCP server test coverage — integration tests for each tool using a test DB
- [ ] `algoforge://dashboard` resource returns live counts (currently stubbed)

### Next
- [ ] Webhook → agent bridge — POST webhook events directly to an agent endpoint with HMAC verification example
- [ ] Agent session persistence — allow an agent to resume an in-progress optimization loop after restart
- [ ] MCP tool: `get_equity_curve(run_id)` — expose equity data to agents for programmatic analysis

### Later
- [ ] Hosted MCP endpoint with API key authentication
- [ ] Prompt library — curated agent prompts for common workflows (find best dataset, optimize Sharpe, etc.)

---

## Phase Summary

| Phase | Focus | Key Deliverables |
|-------|-------|-----------------|
| **Phase 1** (now) | Fix known gaps | Equity curve chart, trade table, chat UI, Celery migration, partitioned parquet, collection error display |
| **Phase 2** (next) | Complete core UX | Model comparison, paper trading dashboard, version history diff, rule_engine handler |
| **Phase 3** (next) | Data richness | Incremental collection, economic_calendar, web_report collector, dataset merge |
| **Phase 4** (later) | Live trading | MT5/Coincheck live mode, user_input condition, agentic handler |
| **Phase 5** (later) | Platform hardening | Auth, multi-user, hosted deployment, notifications |
