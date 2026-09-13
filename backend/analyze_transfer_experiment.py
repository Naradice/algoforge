"""Compare N conditions (e.g. A: USDJPY from scratch, B: DDM pretrain -> USDJPY fine-tune,
C: synthetic-mixture pretrain -> USDJPY fine-tune) of TrainingRuns produced by
submit_transfer_experiment.py.

Usage:
    python analyze_transfer_experiment.py <A_run_id> [<A_run_id> ...] \\
        -- <B_run_id> [<B_run_id> ...] \\
        [-- <C_run_id> [<C_run_id> ...] ...]

Each `--`-separated group is one condition, labelled A, B, C, ... in the order given. Two
groups (the original A-vs-B usage) still work exactly as before; a third (or more) group is
just another condition compared the same way.

Reports, per the user's Phase 2 spec (final val_loss/R2, initial-post-fine-tune performance,
steps-to-best convergence speed, and cross-seed mean/std for each condition):
  - val_loss at the first validation check (proxy for "fine-tuning start" performance)
  - best val_loss and the step count it was reached at (best_epoch * val_every_steps)
  - final val_loss (last recorded checkpoint)
  - R^2 = 1 - best_val_loss / target_variance, using the fine-tune dataset's own val-split
    target variance (same target the runs were trained on: vol_{VOL_PERIOD}, z-scored) so it's
    comparable to the R^2 figures already reported elsewhere in this investigation
  - cross-seed mean/std of best val_loss within each condition

A paired significance check (bootstrap + Diebold-Mariano-style, following
backend/ddm_step5_significance.py's approach) is left as a follow-up if two conditions'
numbers turn out close enough that "did one actually beat the other" isn't visually obvious
from this table alone -- with only 3 seeds per condition a full resampling test needs the
per-window losses, not just the summary stats this script pulls.
"""
from __future__ import annotations

import asyncio
import string
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Load backend/.env before any OHLCWindowDataset construction below reads ARTIFACT_STORE_PATH --
# celery_worker.py does this at import time (so a real TrainingRun always has it set), but this
# script never imports celery_worker, so without this it silently falls back to
# OHLCWindowDataset's own default ("artifacts", not this project's actual "../artifacts"),
# loading nothing from an empty/wrong directory instead of raising -- caught live as a
# "Mean of empty slice" warning and an all-NaN R^2 in _target_variance below.
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

import numpy as np


async def _fetch_run_and_metrics(db, run_id: int):
    from sqlalchemy import select
    from model.models import TrainingRun, TrainingRunMetric

    run = (await db.execute(select(TrainingRun).where(TrainingRun.id == run_id))).scalar_one()
    metrics = (await db.execute(
        select(TrainingRunMetric)
        .where(TrainingRunMetric.training_run_id == run_id)
        .order_by(TrainingRunMetric.epoch)
    )).scalars().all()
    return run, metrics


async def _target_variance(dataset_id: int, hyperparams: dict) -> float:
    """Variance of the SAME (val-split, z-scored) target the runs were trained on -- lets
    best_val_loss (an MSE) convert to an R^2 comparable across conditions/datasets."""
    from model_core.trainers import OHLCWindowDataset

    ds = OHLCWindowDataset(
        artifact_path=(await _dataset_artifact_path(dataset_id)),
        obs_len=hyperparams["obs_len"], pred_len=hyperparams["pred_len"],
        feature_cols=hyperparams["feature_cols"], tgt_feature_cols=hyperparams["tgt_feature_cols"],
        normalize=hyperparams["normalize"], preprocessing=hyperparams.get("preprocessing"),
        val_split=hyperparams.get("val_split", 0.2), split_mode=hyperparams.get("split_mode", "chronological"),
        require_contiguous=hyperparams.get("require_contiguous", False),
        max_rows=hyperparams.get("max_rows"),
    )
    ds.eval()
    _, tgt = ds[0:len(ds)]
    return float(np.nanvar(tgt.numpy()))


async def _dataset_artifact_path(dataset_id: int) -> str:
    import database
    from sqlalchemy import select
    from data.models import Dataset

    async with database.async_session_factory() as db:
        dataset = (await db.execute(select(Dataset).where(Dataset.id == dataset_id))).scalar_one()
        return dataset.artifact_path


def _summarize(run, metrics: list) -> dict:
    val_every_steps = run.hyperparams.get("val_every_steps")
    first = metrics[0] if metrics else None
    best = next((m for m in metrics if m.epoch == run.best_epoch), metrics[-1] if metrics else None)
    last = metrics[-1] if metrics else None
    return {
        "run_id": run.id,
        "seed": run.hyperparams.get("seed"),
        "status": run.status,
        "warm_started": bool(run.hyperparams.get("warm_start_checkpoint")),
        "val_loss_at_start": first.val_loss if first else None,
        "best_val_loss": best.val_loss if best else None,
        "best_step": (best.epoch * val_every_steps) if (best and val_every_steps) else None,
        "final_val_loss": last.val_loss if last else None,
    }


async def analyze(condition_groups: list[tuple[str, list[int]]]) -> None:
    """condition_groups: [(label, [run_id, ...]), ...] -- one entry per `--`-separated group."""
    import database

    target_var = None
    all_summaries: list[tuple[str, list[dict]]] = []

    async with database.async_session_factory() as db:
        for label, run_ids in condition_groups:
            summaries = []
            for run_id in run_ids:
                run, metrics = await _fetch_run_and_metrics(db, run_id)
                if target_var is None:
                    target_var = await _target_variance(run.dataset_id, run.hyperparams)
                summaries.append(_summarize(run, metrics))
            all_summaries.append((label, summaries))

    def _report(label: str, summaries: list[dict]) -> None:
        print(f"\n=== Condition {label} ===")
        for s in summaries:
            r2 = (1 - s["best_val_loss"] / target_var) if (s["best_val_loss"] is not None and target_var) else None
            print(
                f"  run={s['run_id']} seed={s['seed']} status={s['status']} "
                f"warm_started={s['warm_started']} "
                f"val_loss@start={s['val_loss_at_start']:.6f} "
                f"best_val_loss={s['best_val_loss']:.6f} (R2={r2:.4f}) @step={s['best_step']} "
                f"final_val_loss={s['final_val_loss']:.6f}"
                if s["best_val_loss"] is not None and s["val_loss_at_start"] is not None and s["final_val_loss"] is not None
                else f"  run={s['run_id']} seed={s['seed']} status={s['status']} -- incomplete, no metrics yet"
            )
        best_losses = [s["best_val_loss"] for s in summaries if s["best_val_loss"] is not None]
        if best_losses:
            print(f"  cross-seed best_val_loss: mean={np.mean(best_losses):.6f} std={np.std(best_losses):.6f}")

    for label, summaries in all_summaries:
        _report(label, summaries)

    means = {}
    for label, summaries in all_summaries:
        best = [s["best_val_loss"] for s in summaries if s["best_val_loss"] is not None]
        if best:
            means[label] = float(np.mean(best))

    if len(means) >= 2:
        print()
        baseline_label = next(iter(means))
        for label, mean_val in means.items():
            marker = "" if label == baseline_label else (
                f" ({'better' if mean_val < means[baseline_label] else 'worse or equal'} than {baseline_label})"
            )
            print(f"Mean best_val_loss: {label}={mean_val:.6f}{marker}")


def main():
    args = sys.argv[1:]
    if "--" not in args:
        raise SystemExit(__doc__)

    groups: list[list[int]] = [[]]
    for arg in args:
        if arg == "--":
            groups.append([])
        else:
            groups[-1].append(int(arg))

    if len(groups) < 2 or any(not g for g in groups):
        raise SystemExit(__doc__)

    labels = list(string.ascii_uppercase[: len(groups)])
    condition_groups = list(zip(labels, groups))
    asyncio.run(analyze(condition_groups))


if __name__ == "__main__":
    main()
