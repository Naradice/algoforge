# AlgoForge Roadmap

This roadmap is organized by the three platform layers plus cross-cutting concerns. Items are grouped into phases: **Now** (known gaps, quick wins), **Next** (planned features), and **Later** (longer-horizon ideas).

---

## Data Layer

### Now
- [ ] Fix `web_report` collector (currently raises `NotImplementedError`) — Playwright scraper for earnings tables, economic calendars
- [ ] Improve collection error display in the UI: show `last_error` text inline on the datasource card instead of just "error" status
- [ ] Add dataset delete — remove parquet file and DB record together
- [ ] Empty state on the datasets list (currently blank when no datasets exist)

### Next
- [ ] `economic_calendar` datasource type — fetch scheduled events (NFP, CPI, rate decisions) from a provider
- [ ] Dataset tagging / search — filter datasets by symbol, timeframe, source type
- [ ] Incremental collection — append new bars to an existing dataset instead of re-downloading everything
- [ ] Dataset merge — combine two datasets (e.g. OHLC + fundamentals) by timestamp alignment
- [ ] Characteristic auto-compute on collection completion (currently manual trigger)

### Later
- [ ] Streaming ingest — write live ticks to a dataset in real time for perpetual paper-data feeds
- [ ] Data quality monitor — alert when a dataset drifts (stationarity change, gap detected)

---

## ML Model Layer

### Now
- [ ] Show training loss chart on the model detail page (SSE epoch events are published; chart not wired)
- [ ] Hyperparameter search results page — currently `compare_runs` MCP tool works but no UI
- [ ] Empty state on the models list

### Next
- [ ] Model comparison page wired to live data — `web/app/model/compare/page.tsx` exists but needs API integration
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
- [ ] Equity curve chart on the run detail page — `equity_curve` is stored but not rendered
- [ ] Trade table on the run detail page — individual trades with entry/exit/PnL
- [ ] Chat panel on the run detail page — `POST /strategies/{id}/runs/{run_id}/chat` is implemented but no UI yet
- [ ] Empty state on the strategies list
- [ ] Strategy definition editor validation — highlight invalid JSON before submit

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
- [ ] Process manager for arq worker — `Procfile` or `docker-compose` so worker starts alongside the API server
- [ ] Unified error toast in the frontend — currently some API errors are silently swallowed
- [ ] Loading skeletons on detail pages (strategy, model, dataset) — currently blank while SWR loads

### Next
- [ ] Authentication — JWT-based login so the platform is not open by default
- [ ] Settings → API Keys page (currently a stub) — store Alpha Vantage key, broker credentials
- [ ] Webhook registration UI polish — show last fired timestamp, last status, delivery history
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
| **Phase 1** (now) | Fix known gaps | Equity curve chart, trade table, chat UI, arq process manager, collection error display |
| **Phase 2** (next) | Complete core UX | Model comparison, paper trading dashboard, version history diff, rule_engine handler |
| **Phase 3** (next) | Data richness | Incremental collection, economic_calendar, web_report collector, dataset merge |
| **Phase 4** (later) | Live trading | MT5/Coincheck live mode, user_input condition, agentic handler |
| **Phase 5** (later) | Platform hardening | Auth, multi-user, hosted deployment, notifications |
