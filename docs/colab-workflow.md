# Colab Workflow — Training Tiny Models on Google Colab's CPU Runtime

Some architectures/hyperparams (a tiny LSTM, a short obs_len, a small dataset) don't need this
machine's GPU or even its CPU tied up for the duration — they finish fine on a bare Colab CPU
runtime. This doc is the practical guide for that path: pick a candidate, export its data,
generate a notebook, run it, watch it, and register the result back into algoforge.

Related design docs: [data-layer.md](data-layer.md) (dataset artifacts),
[model-layer.md](model-layer.md) (TrainingRun lifecycle) — this doc is the operational
"how do I actually run one" companion to both.

## Overview

```
┌─────────────┐   ① export+upload    ┌──────────────┐   ② generate   ┌─────────────────┐
│  algoforge   │ ───────────────────▶ │ Google Drive │ ─────────────▶ │ notebooks/*.ipynb │
│  dataset     │  snapshot (hash-     │  (public dl  │   (a thin      │ (Git-tracked)     │
│  (mutable)   │  stamped, immutable) │   link)      │   driver that  │                   │
└─────────────┘                      └──────────────┘   pip installs └────────┬──────────┘
                                                          model_core at        │
                                                          this commit)         │ ③ run
                                                                                ▼
┌──────────────┐   ⑤ registers as     ┌──────────────────┐   downloads   ┌───────────────┐
│  algoforge    │ ◀──────────────────  │ best.pt +         │ ◀───────────  │  Colab CPU    │
│  TrainingRun  │   source="external"  │ metrics.json      │               │  runtime      │
│  (/model/     │                      └──────────────────┘               └───────────────┘
│   compare)    │
└──────────────┘
```

`model_core` (`backend/model_core/`) is algoforge's model architectures + training loop,
packaged standalone (no FastAPI/SQLAlchemy/Celery dependency — see its own `__init__.py`). Both
this backend's own celery worker and a generated notebook `pip install` and `import` the exact
same package at the exact same git commit — not a copy of its source text — so a local run and a
Colab run are guaranteed to execute identical code, not just code that looked the same when the
notebook was generated.

Everything on the left runs on this machine (or in the `colab-cli` container); everything on
the right runs on Google's infrastructure. Nothing in between requires this machine to stay on
or the algoforge backend to be reachable from outside — that's the whole point.

## One-time setup

1. **Google Drive OAuth** (for uploading dataset snapshots) — see
   `data/gdrive_export.py`'s module docstring. Needs `GDRIVE_OAUTH_CLIENT_JSON` and
   `GDRIVE_SNAPSHOT_FOLDER_ID` in `backend/.env`, plus one interactive browser login (the
   resulting token then caches at `GDRIVE_OAUTH_TOKEN_PATH` and every later call is
   non-interactive). **Use a personal Gmail account's own OAuth login, not a service account** —
   service accounts have no Drive storage quota and fail with `storageQuotaExceeded` even on a
   shared folder, confirmed live; that only works around a Google Workspace Shared Drive.
2. **`colab-cli` container** — see `infra/colab-cli/README.md`. One-time interactive login
   (`docker compose exec -it colab-cli colab status`, follow the printed OAuth URL) which also
   caches into a Docker volume.

Both logins are things only a human can do (a real browser round-trip) — do them once, then
everything below is scriptable.

## Picking a candidate

Good fit for this path: an architecture/size/hyperparam combination that would finish in
minutes-to-low-hours on a CPU. `docs/model-layer.md`'s architecture table and the "Row cap"
section are the relevant references — a `lstm`/`decoder_only` at small `hidden_dim` on a dataset
under a few hundred thousand rows is squarely in range; a multi-million-row `seq2seq_transformer`
sweep is not (send that to this machine's GPU worker instead, via the normal
`start_training_run` MCP tool / `/models/{id}/training-runs` API).

**Current generator scope**: any architecture `build_model()` accepts works — the notebook
dispatches via `model_core.trainers.get_trainer_fns`/`get_default_criterion` generically
(verified end-to-end for `lstm`, `decoder_only`, `timegan`, `vae`) rather than hardcoding a
whitelist. Excluded: `model_core.architectures.NON_GRADIENT_ARCHITECTURES` (`rl_agent`/`ar`/
`ma`/`arma` — not `torch.nn.Module`-based). `split_mode` (`"chronological"`/`"regime_controlled"`)
`token_level` (`"diff"`/`"quantize_diff"`/`"cluster"`/`"digits"`/`"sax"`), and an inline
`preprocessing` recipe (indicators/clustering) are all supported and verified end-to-end
(`token_level="cluster"` and a `preprocessing` recipe each trigger their own extra `pip
install` cell automatically — `scikit-learn` for the former, `finance_client` plus its actual
runtime deps for the latter). `preprocessed_dataset_id` (a *saved* recipe, as opposed to one
given inline in hyperparams) is not supported yet -- see `model/colab_trainer.py`'s
`check_colab_supported`, which still rejects it and points the caller at inline `preprocessing`
instead.

**Training-control hyperparams**: `optimizer` (`adam`/`adamw`/`sgd`) with `beta1`/`beta2`/
`momentum`/`weight_decay`, `disable_lr_scheduler`, `shuffle`, `lr_warmup_epochs`,
`early_stop_patience`, and `divergence_factor` are all wired into the generated notebook
identically to `celery_worker.py`'s `_train_model` and verified end-to-end (`divergence_factor`
and `early_stop_patience` each confirmed to actually fire, not just fail to error). `max_steps`
step-based training (an epoch-free research mode -- see `docs/model-layer.md`) is not
implemented here.

**Hyperparameter search** (`POST /training-runs/search` / MCP `start_hyperparameter_search`):
`execution_target="colab"` runs every combination the grid expands into on Colab — each is
validated against `check_colab_supported` individually at creation time, so one unsupported
combination rejects the whole search up front rather than failing partway through. Verified
that a 4-combination grid creates all four `TrainingRun`s with `execution_target="colab"`
correctly, and that a `NON_GRADIENT_ARCHITECTURES` model rejects the whole search with no rows
left behind. Actual concurrency across the resulting runs depends on how many `colab` queue
workers are running (see the parallelization note in Step 3 below) — the dev setup's single
`--pool=solo` worker runs them one at a time regardless of how many get enqueued at once.

## Step 1 — Export a dataset snapshot

```
cd algoforge/backend
python -m scripts.export_dataset_snapshot --dataset-id <DATASET_ID> --upload-gdrive
```

Prints a `snapshot_id` — the dataset's current contents, frozen and hash-stamped, since a live
dataset can keep changing under incremental collection (see data-layer.md) and a disconnected
Colab runtime can't reach it anyway. Reuse the same `snapshot_id` for every notebook that should
train on the exact same data (comparing runs against a moving dataset defeats the comparison —
see model-layer.md's "Comparing training runs" methodology notes).

## Step 2 — Generate the notebook

```
python -m scripts.generate_colab_notebook \
    --model-id <MODEL_ID> --model-name <MODEL_NAME> \
    --snapshot-id <SNAPSHOT_ID> \
    --hyperparams-json '{"obs_len":60,"pred_len":10,"epochs":30,"lr":0.001}' \
    --out ../notebooks/<MODEL_NAME>.ipynb
```

`--model-id` loads the architecture and its structural config (`hidden_dim`, `num_layers`, ...)
straight from the `MLModel` record — kept deliberately separate from `--hyperparams-json`
(`obs_len`/`epochs`/`lr`/etc.), the same split `celery_worker.py`'s own `_train_model` keeps
between `model_config` and `hp`. If you don't have (or want) a DB record, pass
`--architecture`/`--model-config-json` directly instead.

The resulting `.ipynb` records the algoforge commit it was generated from and installs
`model_core` from that exact commit at run time (Step 3) — not a copy of its source — so commit
the notebook and it's independently reproducible from GitHub alone: anyone can open
`https://colab.research.google.com/github/<org>/algoforge/blob/<ref>/notebooks/<MODEL_NAME>.ipynb`
directly in a browser too, no `colab-cli` needed, if a human wants to just click through it.

Repeat Steps 1–2 once per (model, dataset, hyperparams) combination you want to try — this is
the natural point to queue up several candidates before moving to Step 3.

## Step 3 — Run it, watch it, retrieve it

```
cd algoforge/infra/colab-cli
docker compose exec colab-cli colab new -s <MODEL_NAME>
docker compose exec colab-cli colab exec -s <MODEL_NAME> \
    -f /notebooks/<MODEL_NAME>.ipynb --timeout 3600   # default timeout is 30s -- always raise it
```

`colab exec` blocks until the notebook finishes (or the timeout fires) — its exit code is the
completion signal for that terminal. **To check on it from elsewhere while it's still running**,
open a second `docker compose exec`:

```
docker compose exec colab-cli colab status -s <MODEL_NAME>
docker compose exec colab-cli colab log -s <MODEL_NAME>
```

**`colab log` only shows which cells have executed (their code), not their printed output** —
confirmed live, so it does *not* surface the notebook's own `"epoch N/M: train=... val=..."`
print lines. `colab status` at least tells you whether the session is still running. The
`colab exec` command's own captured stdout (which `model/colab_trainer.py` logs unconditionally
when running this automatically) is currently the only place those progress lines are visible —
which means, for a manual run, they're genuinely only visible once `colab exec` returns and
prints everything it buffered, not while it's in progress.

Once it finishes:

```
docker compose exec colab-cli colab download -s <MODEL_NAME> /content/best.pt /notebooks/<MODEL_NAME>_best.pt
docker compose exec colab-cli colab download -s <MODEL_NAME> /content/metrics.json /notebooks/<MODEL_NAME>_metrics.json
docker compose exec colab-cli colab stop -s <MODEL_NAME>
```

### Getting a completion notification instead of babysitting a terminal

Since `colab exec` is a single blocking command, running it via Claude Code with
`run_in_background: true` turns "check on Colab periodically" into "get told when it's done" —
ask your Claude Code session to run the `colab exec` command above in the background for a given
notebook (or a whole list of them, one after another), and it will surface a notification the
moment each one exits, instead of you polling `colab status` by hand. This is the practical
answer to "how do I know when it's done" for this workflow.

## Step 4 — Register the result with algoforge

```
cd algoforge/backend
python -m scripts.import_training_run --model-id <MODEL_ID> --dataset-id <DATASET_ID> \
    --checkpoint ../notebooks/<MODEL_NAME>_best.pt \
    --metadata ../notebooks/<MODEL_NAME>_metrics.json \
    --notebook-url https://colab.research.google.com/github/<org>/algoforge/blob/<ref>/notebooks/<MODEL_NAME>.ipynb
```

This creates a `TrainingRun(source="external")` with the notebook URL, git commit, and dataset
snapshot sha256 recorded in `hyperparams._external_ref` — it then shows up in `/model/compare`
and the model detail page exactly like a run this machine's own celery worker trained, so
Sharpe/val_loss/param-count comparisons work across both without special-casing.

## Running several (model, dataset) candidates at once

Repeat Steps 1–4 per candidate. Two things to know before parallelizing across multiple
`colab new -s <name>` sessions concurrently: how many concurrent Colab runtimes your account's
tier actually allows, and whether this container's single `colab-cli` process handles concurrent
`exec` calls to different sessions cleanly — neither has been tested here. Running them
sequentially (or a couple at a time) is the safe default until that's checked.

## Live progress and stopping (execution_target="colab" only)

The manual CLI flow above (Steps 3–4) has no live progress — `colab exec` is a single blocking
call. But a run started with `execution_target="colab"` (i.e. going through
`model/colab_trainer.py`'s `run_colab_training`, not the manual CLI steps by hand) gets both,
via the same mechanism: while `colab_runner.exec_notebook` is blocked, a concurrent task
(`_poll_and_maybe_stop`) polls the notebook's own `progress.json` (written every epoch — see
`model/notebook_export.py`) every 20s via `colab download`, reflecting `current_epoch`/`val_loss`
onto the `TrainingRun` live — the same fields `GET /training-runs/{id}/status` and the UI's
training-run table already show for a local run. The same task also checks
`TrainingRun.stop_requested` each cycle; `POST /training-runs/{id}/stop` (the same endpoint that
stops a local run) sets it, and once seen, the task grabs the latest checkpoint it can and calls
`colab stop` on the session — confirmed live that this reliably makes the blocked `colab exec`
call exit non-zero within the poll interval, which is what lets the run actually be interrupted
rather than just flagged for later. A stopped run ends up `status="completed"` (matching a
stopped local run's own convention) with whatever `best_epoch`/`val_loss` progress.json last
recorded, but with zero `TrainingRunMetric` rows (only the bulk metrics.json import on normal
completion writes those, to avoid double-writing what polling already wrote to `TrainingRun`
itself) — an accurate reflection of a run that didn't finish normally, not a bug.

Verified live end to end: a 300-epoch run, `stop_requested` set mid-run, stopped within one poll
cycle at epoch ~236 and registered as `completed` with `best_epoch=229` and
`hyperparams._external_ref.stopped_early=true`.

## Known limitations (be aware before relying on this unattended)

- No push notification from Colab back to algoforge or this session for anything *other* than
  the polling above — the Colab side has no way to reach this machine's backend directly. See
  `docs/research-agent-service.md`'s and `algoforge/docs/mcp-guide.md`'s webhook design for why
  that's true generally.
- `colab exec`'s exit code reliability for an in-notebook failure (vs. only
  CLI/connection-level failures) is unconfirmed.
- Whether a long `colab exec` run risks Colab's own idle/runtime-length limits the same way the
  browser UI does is unconfirmed (the CLI advertises an automatic keep-alive daemon against
  idle-out specifically).
- No duplicate-execution lock for `run_colab_training` the way `_train_model` has (see
  `model/colab_trainer.py`'s module docstring) — unlikely to matter given Redis's 12h broker
  visibility_timeout, but flagged for a very large `colab_timeout_seconds`.
