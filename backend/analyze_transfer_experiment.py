"""Compare condition A (USDJPY from scratch) vs condition B (DDM pretrain -> USDJPY fine-tune)
TrainingRuns produced by submit_transfer_experiment.py.

Usage:
    python analyze_transfer_experiment.py <condition_a_run_id> [<condition_a_run_id> ...] \\
        -- <condition_b_finetune_run_id> [<condition_b_finetune_run_id> ...]

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
backend/ddm_step5_significance.py's approach) is left as a follow-up if the two conditions'
numbers turn out close enough that "did B actually beat A" isn't visually obvious from this
table alone -- with only 3 seeds per condition a full resampling test needs the per-window
losses, not just the summary stats this script pulls.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

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


async def analyze(condition_a_ids: list[int], condition_b_ids: list[int]) -> None:
    import database

    async with database.async_session_factory() as db:
        a_summaries, b_summaries = [], []
        target_var = None
        for run_id in condition_a_ids:
            run, metrics = await _fetch_run_and_metrics(db, run_id)
            if target_var is None:
                target_var = await _target_variance(run.dataset_id, run.hyperparams)
            a_summaries.append(_summarize(run, metrics))
        for run_id in condition_b_ids:
            run, metrics = await _fetch_run_and_metrics(db, run_id)
            b_summaries.append(_summarize(run, metrics))

    def _report(label: str, summaries: list[dict]) -> None:
        print(f"\n=== {label} ===")
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

    _report("Condition A (USDJPY from scratch)", a_summaries)
    _report("Condition B (DDM pretrain -> USDJPY fine-tune)", b_summaries)

    a_best = [s["best_val_loss"] for s in a_summaries if s["best_val_loss"] is not None]
    b_best = [s["best_val_loss"] for s in b_summaries if s["best_val_loss"] is not None]
    if a_best and b_best:
        print(f"\nMean best_val_loss: A={np.mean(a_best):.6f}  B={np.mean(b_best):.6f}  "
              f"(B {'better' if np.mean(b_best) < np.mean(a_best) else 'worse or equal'} than A)")


def main():
    args = sys.argv[1:]
    if "--" not in args:
        raise SystemExit(__doc__)
    sep = args.index("--")
    condition_a_ids = [int(x) for x in args[:sep]]
    condition_b_ids = [int(x) for x in args[sep + 1:]]
    if not condition_a_ids or not condition_b_ids:
        raise SystemExit(__doc__)
    asyncio.run(analyze(condition_a_ids, condition_b_ids))


if __name__ == "__main__":
    main()
