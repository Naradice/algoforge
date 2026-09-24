# Handoff — Jev replication investigation (as of 2026-09-25)

Written for: whoever (human or a future Claude session) picks this investigation back up —
assumes no memory of the conversation that produced it, only this repo and
`docs/jev-replication.md`.

## 1. What this project actually is

Two things are being tested at once, and both matter:

1. **The stated research question**: can TypeSafe's "Jev" model (a non-Transformer, calibrated
   typed-decision model, architecture undisclosed — `docs.typesafe.ai`) be functionally
   reproduced with known architectures, starting with BERT? Full spec/plan/results:
   **`docs/jev-replication.md`** — that file is the authoritative experiment log, kept current
   through every phase. This handoff does not repeat its content, only points at it.
2. **The actual point of doing it this way**: this whole investigation is a touchstone for
   running autonomous research *between* `study_manager` (a separate FastAPI service — sibling
   directory, not this repo, no git) and AlgoForge (this repo), via AlgoForge's MCP server.
   study_manager submits a research question → BRIEFING generates a brief → human approves →
   study_manager's own Agent Loop drives `create_model`/`start_training_run`/etc. against a real,
   running AlgoForge instance autonomously. **The standing instruction from the user is to keep
   using study_manager for this, not to just run experiments by hand.**

## 2. Current phase status (see jev-replication.md for full detail/numbers)

- **Phase 0** (spec extraction from typesafe.ai docs): done.
- **Phase 1** (Model A = `jev_bert_v1`, BERT + one typed-decision head per fixed question, all
  concatenated into one sequence with `[Q_<id>]` marker tokens): done, run for real via
  study_manager's Agent Loop, full success after a hyperparameter search.
- **Phase 2** (scale question count N, test Jev's own "Speculative Fan-Out" flat-latency-in-N
  claim): Model A tested at N=3/N=6 — **latency 59.0ms → 137.5ms (2.33x)**, clearly *not* flat.
  N=9 attempted repeatedly, always crashed; root-caused to system-wide virtual memory exhaustion
  from ~24h of concurrent long-running celery workers (this investigation's own, plus an
  unrelated concurrent "Five Axes of Scaling" experiment sharing the same machine) — not a code
  bug, though one real bug (`resize_token_embeddings`'s `mean_resizing=True` segfaulting this
  environment at 9 new tokens) was found and fixed along the way regardless.
- **Phase 2 follow-up** (Model B = `jev_cross_attn_v1`, encode state once, each question
  cross-attends independently, batched): built after discussing with the user *why* Model A
  can't be flat (self-attention couples every question's cost to every other's) and what
  would fix it (decouple state encoding from question evaluation). Tested at N=3/N=6 —
  **latency 81.2ms → 107.8ms mean (1.33x) / ~flat by median** — markedly closer to Jev's claim,
  and better accuracy/calibration too. N=9 for Model B not yet attempted.
- **N=9 (both models): still deferred**, blocked by the same shared-machine memory constraint.
  **Important update**: as of this handoff, the machine appears to have been restarted since the
  last work session — free virtual memory is back up to ~32GB (was ~5GB), and *no servers or
  celery workers are currently running at all* (see §4). The memory blocker may simply be gone
  now — this is the natural point to retry N=9, starting with Model B (it needs less memory than
  Model A at the same N, since its state length doesn't grow with N).

## 3. What's committed vs. not

- **AlgoForge (this repo)**: everything from this investigation is committed and pushed to
  `master`, currently at `8f82e7a`. Relevant files:
  - `backend/data/collectors/llm_typed_decisions.py` — synthetic dataset collector,
    `QUESTION_POOL` (9 fixed questions), `question_ids` config selects a subset.
  - `backend/model_core/architectures/jev_bert.py` — Model A.
  - `backend/model_core/architectures/jev_cross_attn.py` — Model B.
  - `backend/model_core/trainers/typed_decision.py` — Model A's dataset/loss/calibration/latency.
  - `backend/model_core/trainers/cross_attn_typed_decision.py` — Model B's (reuses Model A's
    loss/calibration math, doesn't duplicate it).
  - `backend/celery_worker.py` — `_run_typed_decision_training` (Model A),
    `_run_cross_attn_training` (Model B), both dispatched from `_train_model`.
  - `backend/model_core/architectures/__init__.py`, `backend/model_core/trainers/__init__.py` —
    registration/exports for both.
  - `docs/jev-replication.md` — the full experiment log (read this for actual numbers/results).
  - `docs/requirements.md` — R-12 (the `/mcp` unreachable + no-commit bugs, resolved), R-11
    (Agent Loop's `collect` decision can't create a new datasource, still open).
  - **⚠️ This repo also has a large amount of unrelated, uncommitted work from a different,
    concurrent "Five Axes of Scaling" investigation** (many modified/untracked files under
    `backend/`, plus `docs/research-review-five-axes-of-scaling.md` etc.). **Do not touch, stage,
    or commit any of it** — it belongs to separate, ongoing work. Every commit made for this
    investigation was staged file-by-file for exactly this reason; keep doing that.
- **study_manager (sibling directory, no git)**: changes are plain file edits, not committed
  anywhere — they only exist on this machine. Files touched:
  - `backend/agent_loop.py` — `_TRAIN_ARCHITECTURES` and `_DECIDE_SYSTEM_PROMPT` include both
    `jev_bert_v1` and `jev_cross_attn_v1` (this is the *only* place the Agent Loop's Decide step
    learns an architecture exists — adding one to AlgoForge alone does nothing here).
  - `backend/main.py` — `load_dotenv()` added.
  - `backend/database.py` — `poolclass=NullPool` for the SQLite engine.
  - `backend/scheduler.py` — `get_queue_limit()`.
  - `README.md` — running status log, including a full account of every infra bug found.
  - `study_manager.db` (SQLite) — has real session history; sessions 15/16/17 are `paused`
    (orphaned N=9 attempts / a wrong-architecture retry) and can be ignored or cleaned up — the
    README documents that orphaned backtest/collect-mid-wait sessions aren't auto-recoverable,
    and these were `train` decisions whose training runs died with their worker, so resuming
    them would just re-hang the same way.

## 4. Infrastructure: everything is currently DOWN

As of this handoff (2026-09-25), neither AlgoForge's API (`:8000`) nor study_manager's API
(`:8100`) is reachable, and no celery workers or uvicorn processes are running — consistent with
the machine having been restarted since the last active work. **Postgres/Redis status was not
confirmed either** — check those first. To get back to a working state:

1. Confirm Postgres and Redis are up (AlgoForge's `.env`: `postgresql+asyncpg://...`,
   `redis://localhost:6379/0`).
2. Start AlgoForge: `cd algoforge/backend && uvicorn main:app --port 8000` (plus whatever the
   other concurrent experiment normally needs — check with the user before assuming only this
   investigation's needs matter).
3. Start at least one celery worker on AlgoForge's `training` queue and one on `collection`:
   `celery -A celery_worker worker -Q training --pool=solo --loglevel=info` /
   `-Q collection`. **Known gotcha**: this queue is shared with the unrelated concurrent
   experiment's own workers/tasks — a fresh worker can and will pick up *their* queued tasks
   too, and a worker that's been running a long time will NOT have picked up any code changes
   made after it started (Python module caching — there's no hot reload). If a training run
   errors instantly (`Could not open Parquet input source...` or similar) right after you've
   changed `celery_worker.py`/`model_core`, that's almost always this — restart the worker(s).
   Killing/restarting a worker **you started** is fine; never touch a worker that's mid-task on
   the other experiment's own work.
4. Start study_manager: `cd study_manager/backend && uvicorn main:app --port 8100`. **No
   `--reload`** — same caching gotcha applies to `agent_loop.py` changes; restart after editing.

## 5. Bugs found and fixed this investigation (don't re-discover these)

- AlgoForge's `/mcp` endpoint was completely unreachable + every MCP write silently rolled back
  (R-12, `docs/requirements.md`) — fixed in `main.py`/`database.py`/`mcp_server/tools/*.py`.
- study_manager's SQLite async engine self-deadlocked when `acquire_queue_slot(s)` was called
  from inside an already-open DB session — fixed with `NullPool` + reordering in `agent_loop.py`.
- `jev_bert_v1` / `jev_cross_attn_v1` missing from `agent_loop.py`'s architecture whitelist — the
  Decide LLM will never choose (or will be rejected for choosing) an architecture not listed
  there, regardless of what AlgoForge itself supports.
- study_manager never loaded `.env` — silent `OPENAI_API_KEY` failures.
- `acquire_queue_slots(queue, n)` had no cap vs. the queue's configured concurrency limit — an
  oversized `search_grid` deadlocked forever instead of failing; now fails fast.
- BRIEFING's LLM call frequently emits `budget.max_llm_cost_usd: 0` even when nothing in the
  raw question mentions cost — this silently triggers an immediate `budget_exceeded` on the very
  first cycle. **Always inspect/PATCH a brief's budget before approving** if this matters (it did,
  repeatedly, in this investigation).
- Running 3 Agent Loop sessions fully concurrently hits a *different* SQLite issue than the one
  above (`database is locked`, a real multi-writer collision, not the same single-session
  deadlock) — run sessions one at a time, not in parallel, until this is properly fixed.
- `JevBertModel`/`JevCrossAttnModel`'s `resize_token_embeddings` call needs
  `mean_resizing=False` — this environment's transformers/torch build segfaults with the
  (newer) default `mean_resizing=True` at 9+ new tokens (fine at 3/6). Both files already have
  this fix; don't remove it.
- `jev_cross_attn.py`'s `forward_batched` originally mean-pooled over padded query positions
  when batching different-length questions together, diluting shorter questions' output —
  fixed with an explicit `q_len_mask` before any real training used it. If you add a third
  model with its own batched multi-question path, check for the same class of bug.

## 6. Known-open, not fixed

- Report-wording bug: study_manager's generated Markdown report always says "Success criteria
  were met: [...]" regardless of the actual `success_criteria_met` boolean — cosmetic, the
  underlying Evaluate logic is correct.
- R-11: Agent Loop's `collect` decision can only run collection for an *existing* datasource, not
  create a new one — every new dataset in this investigation (71/72/73) was created directly via
  REST, not through the Agent Loop.
- No MCP tool exposes AlgoForge's actual live training-queue depth — study_manager's queue-slot
  tracking is self-tracked/in-process only, and has no visibility into the unrelated concurrent
  experiment's own queue usage. This is *why* the queue-fairness and worker-staleness problems in
  §5 keep recurring — there's no way for study_manager to know a worker exists, is fresh, or is
  busy with someone else's task before dispatching to it.

## 7. Operational lessons (process, not code)

- **`ScheduleWakeup` does not work outside `/loop` mode.** It appeared to schedule a check-in
  but silently did nothing for hours at a time in this session (confirmed by its own tool
  description: it's for `/loop` dynamic-pacing). Do not use it to wait on a long-running
  training job. What *does* work reliably: `Bash` with `run_in_background: true` and a polling
  loop that exits on the target condition — this produced a real task-notification every single
  time it was used correctly in this investigation. A single call is capped around 10 minutes;
  for longer waits, let it time out and immediately relaunch another one from the next turn
  rather than trying to make one call span hours.
- Adding more dedicated celery workers does not fix a system-wide memory shortage (a real
  point the user raised and was correct about) — check `Get-CimInstance Win32_OperatingSystem`'s
  `FreePhysicalMemory`/`FreeVirtualMemory` before assuming more workers will help anything.
- The auto-mode permission classifier blocks direct writes to AlgoForge's shared Postgres DB and
  to AlgoForge's `/training-runs/{id}/stop` endpoint ("Modify Shared Resources"), but does allow
  study_manager's own `/sessions/{id}/pause` endpoint. If a training run is truly orphaned (its
  worker died mid-task) there is currently no clean way to force it to a terminal state on the
  AlgoForge side — the practical workaround used here was to leave the orphaned run alone and
  pause the study_manager session waiting on it, then submit a fresh research question instead.

## 8. Suggested immediate next steps

1. Bring the stack back up (§4) and confirm the other concurrent experiment's own state wasn't
   affected by whatever caused the restart — this isn't this investigation's call to make alone.
2. Re-check free memory once everything relevant is running again; if there's real headroom,
   retry N=9 for Model B first via a fresh study_manager research question (datasets 71/72/73
   already exist — dataset 73 is the N=9 one), same pattern as the N=3/N=6 submissions
   documented in `jev-replication.md`'s Phase 2 follow-up section.
3. If N=9 succeeds for both models, `jev-replication.md` gets its first real 3-point scaling
   comparison — worth writing up as its own conclusion rather than folding into the existing
   two-point sections.
4. Beyond N=9: `jev-replication.md`'s own phase plan continues with dynamic (per-example)
   questions (Phase 3), scaling state length with a long-context encoder swap-in (Phase 4),
   state-update/iterative loops, GPT/recurrent baselines, and finally isolating architecture vs.
   RLCD training-objective contribution — nothing in those phases has been started.
