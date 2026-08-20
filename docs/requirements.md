# AlgoForge — Requirements for External Autonomous Consumers

> **What this is:** a living backlog of concrete gaps found in AlgoForge's *existing* implementation
> while designing external, autonomous MCP clients — starting with the
> [Research Agent Service](research-agent-service.md). Unlike `roadmap.md` (feature ideas, UX polish),
> every entry here was found by reading actual backend code and confirming the gap is real, not
> assumed. Append new entries as they're discovered; don't remove entries when fixed — move them to
> the "Resolved" section at the bottom with the commit/date.
>
> Each entry: what's missing, where (file:line), why it blocks or weakens an external autonomous
> consumer, and a proposed fix. Priority is from the Research Agent Service's point of view, not a
> general AlgoForge roadmap priority.

---

## Open

### R-2. Webhook delivery is global and unscoped — P1

**Evidence:** `dispatch()` (`backend/webhooks/dispatcher.py:18-24`) fires to **every** active
registration whose `events` list contains the event type — there's no filtering by resource id.
`WebhookRegistration` (`backend/webhooks/models.py`) has no scoping field beyond `events: list[str]`.

**Impact:** once R-1 is fixed, any registered webhook receives every matching event platform-wide —
including jobs started by a human via the UI or by a different concurrent Research Session. The
consumer must filter client-side using whatever ids are in the payload, and since the payload
structure doesn't exist yet (dispatch is never called), there's a real risk it ships without the
correlating id.

**Proposed fix:**
1. When implementing R-1, guarantee the payload always includes the primary resource id
   (`training_run_id` / `run_id` / `collection_job_id`) plus its parent (`model_id` / `strategy_id` /
   `datasource_id`).
2. Nice-to-have, not blocking: let a webhook registration optionally filter by resource id(s), not
   just event type, to cut noise for consumers only interested in their own jobs.

---

### R-4. No idempotency-key support on job-creating MCP tools — P1

**Evidence:** none of `start_training_run` (`mcp_server/tools/model.py:203`),
`start_hyperparameter_search` (`model.py:388`), `collect_data` (`mcp_server/tools/data.py:251`), or
`start_strategy_run` (`mcp_server/tools/strategy.py:268`) accept a client-supplied dedup/idempotency
token.

**Impact:** if a caller loses the response to a job-creation call (timeout, dropped connection — the
exact failure mode an autonomous, long-running agent will eventually hit) it cannot safely retry
without risking a duplicate job. A duplicate `start_training_run` silently double-spends the caller's
`budget.max_training_runs`/`max_wall_clock_minutes` and occupies queue capacity twice for what was
meant to be one action.

**Proposed fix:** accept an optional `idempotency_key: str` on these tools. A repeated call with the
same key (scoped to the same model/datasource/strategy id) returns the existing job instead of
creating a new one.

---

### R-5. No MCP tool to introspect queue/worker load — P2

**Evidence:** none of `mcp_server/tools/{data,model,strategy,logs}.py` expose queue depth or worker
concurrency. Concurrency (`collection`=3, `characteristics`=12, `training`=2, `backtest`=5) is only
documented statically in `architecture.md`, not queryable at runtime.

**Impact:** the Research Agent Service's concurrency self-throttling
([research-agent-service.md §5.2](research-agent-service.md#52-同時実行数の管理web設定から変更可能))
has to hard-code AlgoForge's queue concurrency as a manually-entered setting rather than reading it
live. If ops changes worker concurrency, the external throttle silently drifts out of sync and either
under-utilizes or over-saturates the real queues.

**Proposed fix:** add `get_queue_status() -> {queue_name: {concurrency, active, reserved}}` backed by
Celery's `inspect()` API.

---

### R-6. `deploy_model` has no concurrency guard — P1 (upgraded from P2)

**Evidence:** `backend/model/service.py:44-51` — `deploy_model()` does a plain read-then-write on the
model row with no version check / optimistic lock.

**Impact:** if two callers (e.g. two concurrent Research Sessions investigating the same model, or a
Research Session racing a human clicking "deploy" in the UI) call `deploy_model` with different
`training_run_id` near-simultaneously, the loser's deploy is silently overwritten with no error and no
way to detect the race afterward. [research-agent-service.md §5.3(B)](research-agent-service.md#b-実行中セッション間のリソース書き込み競合の防止)
adds a client-side `resource_claims` lock around `deploy_model` as a stopgap, but that only works if
every caller cooperates with it — it can't protect against a human clicking "deploy" in the AlgoForge
UI directly, or any other MCP client that doesn't know about the Research Agent Service's claim table.
The real fix has to live in AlgoForge itself. Priority raised from P2 → P1 because the Research Agent
Service's design now directly depends on this as the backstop, not just a nice-to-have.

**Proposed fix:** add a `version` column to `ml_models`; require callers that care to pass the version
they last read, reject on mismatch.

---

### R-7. `create_preprocessed_dataset` not exposed via MCP — P2 (already known)

**Evidence:** `docs/mcp-guide.md` already documents this: "Preprocessed datasets aren't created via
MCP yet (no `create_preprocessed_dataset` tool) — create one through the UI... or `POST
/preprocessed-datasets`."

**Impact:** the Research Agent Service must bypass MCP and call AlgoForge's REST API directly for this
one operation if it wants to autonomously create preprocessing recipes (research-agent-service.md
§2).

**Proposed fix:** add the MCP tool wrapping `POST /preprocessed-datasets`.

---

### R-8. `validation.completed` / `validation.error` has no event type at all — P2

**Evidence:** `validate_model` (`backend/celery_worker.py:1036`) runs as an async Celery task, but no
event type for it exists in the documented webhook event list (`run.completed`, `run.error`,
`training.completed`, `training.error`, `collection.completed`, `collection.error` — `mcp-guide.md`).
Even once R-1 is fixed, validation completion still wouldn't be observable without adding this.

**Impact:** the Agent Loop's `Evaluate` step (research-agent-service.md §5) wants to know when
`get_model_validations` results are ready; without this event it must poll.

**Proposed fix:** add `validation.completed` / `validation.error` as event types and dispatch them
from `_validate_model` in `celery_worker.py` alongside the R-1 fix.

---

### R-9. `get_equity_curve(run_id)` not exposed via MCP — P2 (already tracked)

**Evidence:** `docs/roadmap.md` → "MCP / AI Agent Experience → Next".

**Impact:** report generation (research-agent-service.md §7) wants equity-curve-level detail for
charts; currently has to be reconstructed client-side from `get_run_trades` + `get_run_metrics`.

**Proposed fix:** already tracked in `roadmap.md`; no new action beyond prioritizing it if the Research
Agent Service becomes the reason to build it sooner.

---

## Resolved

### R-1. Webhooks are registered but never fired — P0 (blocker) — resolved 2026-07-31

`dispatch(db, event_type, payload)` is now called from every existing status="completed"/"error"
transition point: `celery_worker.py::_train_model`/`_run_arima_training` (`training.completed`/
`training.error`), `celery_worker.py::_run_collection_job` (`collection.completed`/
`collection.error`), `strategy/executor.py::_run` (`run.completed`/`run.error`, backtest + live/
unknown-mode error paths), and `strategy/live_runner.py::run_paper` (`run.completed` on stop).
Each dispatch call is placed before the transaction's `db.commit()` so the webhook fire and the
status write land in the same commit. Per R-2's proposed fix item 1, every payload includes the
primary resource id (`training_run_id`/`run_id`/`collection_job_id`) plus its parent
(`model_id`/`strategy_id`/`datasource_id`) — R-2's per-registration resource-id *filtering* (item 2,
a noise-reduction nice-to-have) is still open.

Not covered by this pass — pre-existing gaps, unchanged: `_TrainingResolutionError`
(training_run_not_found/model_not_found/dataset_not_found_or_no_artifact) and the analogous
run_not_found/strategy_not_found early returns in `strategy/executor.py::_run` don't write a
status="error" transition at all today, so there's nothing to hang a dispatch off without inventing
new behavior — left as-is. `validate_model` has no `validation.*` event type yet — tracked
separately as R-8.

### R-10. `POST /webhooks` 500s against real Postgres — P0 (blocker) — resolved 2026-08-12

**Evidence:** `webhooks/models.py`'s `WebhookRegistration.events` was declared `sa.JSON`, but
`alembic/versions/0001_initial_schema.py` created the actual column as `ARRAY(sa.Text)` (Postgres
`text[]`) — confirmed by querying `information_schema.columns` against a live local instance.
Every `POST /webhooks` against real Postgres raised an unhandled exception (asyncpg rejecting a
JSON-typed bind against a `text[]` column) → bare 500, no detail in the response body. Silently
masked by the test suite: SQLite has no `ARRAY` type at all, so `tests/test_webhooks.py` always
exercised the JSON fallback path and passed regardless of which type the model declared — this
gap predates R-1/R-3 and was never caught by them; found live, by study_manager's own webhook
self-registration hitting a real AlgoForge instance (`algoforge_client.register_webhook_if_needed`
→ `httpx.HTTPStatusError`), not by any AlgoForge-side test.

**Impact:** R-1 (webhook dispatch firing) is only useful if a webhook can be *registered* in the
first place. Registration was unreachable against any real (Postgres-backed) deployment — a
consumer would silently fall back to polling forever, defeating R-1's fix and
research-agent-service.md §5's "Webhook駆動が基本" premise, with no visible error on the AlgoForge
side (client-side, study_manager only saw a generic 500 and its own httpx traceback).

**Fix:** `events: Mapped[list] = mapped_column(sa.ARRAY(sa.Text).with_variant(sa.JSON(), "sqlite"), ...)`
— keeps the real Postgres column type (no migration needed, the column was always `text[]`) while
still letting SQLite-backed tests compile the table via the JSON variant. `tests/test_webhooks.py`
(5/5) and the full suite still pass (the only new failure vs. baseline is
`test_arima_trainer.py::test_large_series_stays_fast_and_correct`, a pre-existing timing-flaky
test unrelated to this change — 32.4s against a 30s cap). Not re-verified against the user's live
Postgres instance in this pass (by request — the SQLite-level fix + root-cause confirmation via
direct `information_schema` query against that same live DB was judged sufficient; the running
AlgoForge process would need a restart to pick up the code change).

**Residual risk:** `secret_hash` is a leftover nullable column in the live schema with no
corresponding ORM field (migration history suggests an earlier hashed-secret design was replaced
by the plaintext `secret` column in `0003_missing_schema_columns.py`, which never dropped the
old column). Dead weight, not a correctness bug — flagged here in case it's ever mistaken for
something that needs populating.

### R-3. `/mcp` endpoint has no authentication — P0 for any non-localhost use — resolved 2026-07-31

Added `mcp_server/auth_middleware.py::MCPAuthMiddleware`, a raw ASGI wrapper around the FastMCP
sub-app (mounted in `main.py`) since it isn't a FastAPI router and can't use
`Depends(require_api_key)`. It checks the same `api_keys` table/hash scheme as `auth.py`, expects
`Authorization: Bearer <key>`, and returns 401 JSON otherwise. Bypassed under
`ALGOFORGE_NO_AUTH=1`, matching `auth.optional_auth`'s existing dev-only escape hatch. Added
`scripts/create_api_key.py` to issue keys (`python scripts/create_api_key.py --name
research-agent-service`) since the Settings → API Keys UI is still a roadmap stub.
