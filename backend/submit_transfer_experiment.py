"""DDM -> USDJPY synthetic-pretraining transfer experiment (priority (1) of the user's staged
research plan: condition A [USDJPY from scratch] vs condition B [DDM pretrain -> USDJPY
fine-tune], fine-tune budget/model/data/seeds fully matched).

Two phases, run as separate CLI invocations:

    python submit_transfer_experiment.py prepare-data
        Collects the USDJPY and DDM datasets needed (no Celery worker required -- calls each
        collector's collect() directly, then registers the resulting Dataset row, the same way
        celery_worker.py's run_collection_job would). Prints the resulting dataset ids; paste
        them into USDJPY_DATASET_ID / DDM_DATASET_ID below before running smoke/full.

    python submit_transfer_experiment.py smoke
        1 seed, small max_steps. Submits one condition-A run and one condition-B pretrain+
        fine-tune pair via the real TrainingRun/Celery pipeline (requires a `training`-queue
        Celery worker running -- see CLAUDE.md). Verifies warm_start_checkpoint actually loads
        (condition B's fine-tune run should NOT start from the same initial val_loss as
        condition A) before committing to the full run.

    python submit_transfer_experiment.py full [seed ...]
        N_SEEDS_FULL (default 3) condition-A runs + 1 pretrain + N_SEEDS_FULL condition-B
        fine-tune runs, at the real step budget.

Follows backend/submit_regime_training.py's pattern: same create_training_run + enqueue
("train_model", run.id) path the API router uses, so results show up in the real training
pipeline (TrainingRun/TrainingCheckpoint/model-compare), not a disconnected reimplementation.

Both DDM and USDJPY pretraining/fine-tuning use the SAME target task (close -> vol_{PERIOD},
the "return -> realized volatility" convention established across all five datasets in the
"Five Axes of Scaling" investigation, docs/research-seed-five-axes-of-scaling.md) via the
existing preprocessing.py "volatility" indicator + OHLCWindowDataset's tgt_feature_cols -- no
new dataset-prep code needed, just the right hyperparams.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---------------------------------------------------------------------------
# Fill these in after `prepare-data` prints the registered dataset ids.
# ---------------------------------------------------------------------------
# dataset id=29 "USDJPY" (M1, 2,239,284 rows, 2016-08-31 -> 2022-08-31) -- a real, full-history
# dataset already registered in this project (the same one the "Five Axes of Scaling"
# investigation's usdjpy_1min_volatility.parquet derives from), NOT the ~1-month yfinance
# snippet `prepare-data` used to collect fresh (dataset id=51, 29,611 rows) -- yfinance's M1
# endpoint is provider-limited to recent history only; there was no need to hit it at all once
# this was found. Note: dataset 29's own derived sibling (id=30, "usdjpy_1min_volatility") has
# precomputed rv/log_rv columns, but their exact formula isn't documented anywhere in this repo
# and empirically isn't just a plain rolling-std-of-returns at any clean window (correlation
# ~0.91 at best against several tried, never 1.0) -- reusing an unverified legacy column would
# also break target-formula consistency with DDM's side. Using our own preprocessing
# "volatility" indicator (see BASE_HP below) on dataset 29's raw OHLC instead keeps the target
# definition IDENTICAL between DDM and USDJPY, which matters more for a fair transfer comparison
# than reusing whatever the original undocumented rv/log_rv computation was.
USDJPY_DATASET_ID: int | None = 29
DDM_DATASET_ID: int | None = 52
# Condition C's mixture pretrain dataset -- filled in by `prepare-mixture-data`.
MIXTURE_DATASET_ID: int | None = 53
# Phase 4 component ablation: D1-D4 = DDM 240K + one component at 60K (see conversation --
# isolates "does adding this one component to DDM help" from C's "all four at once" result,
# though not that component's synergy with the others). Filled in by `prepare-ablation-data`.
D1_DATASET_ID: int | None = 54  # DDM 240K + Sine 60K
D2_DATASET_ID: int | None = 55  # DDM 240K + Delay 60K
D3_DATASET_ID: int | None = 56  # DDM 240K + XOR 60K
D4_DATASET_ID: int | None = 57  # DDM 240K + LFSR 60K
# Phase 4a Sine dose-response sweep (user-requested, following D1/D2's confirmed "smooth
# structure helps" finding): DDM 300K (=B) -> DDM 240K+Sine 60K (=D1) -> ... -> Sine 300K,
# holding total volume fixed at DDM_PRETRAIN_ROWS throughout, varying only the DDM/Sine split.
# E1/E2/E3 fill in the middle of the curve; E4 is the pure-Sine endpoint. Filled in by
# `prepare-dose-response-data`.
E1_DATASET_ID: int | None = 58  # DDM 180K + Sine 120K
E2_DATASET_ID: int | None = 59  # DDM 120K + Sine 180K
E3_DATASET_ID: int | None = 60  # DDM 60K + Sine 240K
E4_DATASET_ID: int | None = 61  # Sine 300K (pure)
# Phase 4b: does the DDM x Sine interaction found in D1/E1-E4 generalize to Delay (D2's other
# "dramatic effect" component from the D1-D4 screening), or is it Sine-specific? Mirrors E4's
# structure exactly: F1 is the pure-Delay endpoint (0 DDM) -- D2 (DDM 240K + Delay 60K, already
# run) and B (DDM 300K, no Delay/Sine) are reused as the other two corners of the same 2x2 the
# Sine investigation used (DDM alone / component alone / DDM+component together).
F1_DATASET_ID: int | None = 62  # Delay 300K (pure)
# Phase 5b (user-requested): "what is Sine actually doing" -- representation analysis (Phase 5)
# found DDM+Sine/DDM+Delay build a genuinely new, deep (layer 2-3) volatility-specific
# representation neither DDM alone nor the component alone has. Now test whether this needs
# smoothness per se, or specifically a deterministic/reproducible generator (both sine and delay
# are deterministic; xor/lfsr, which show no effect, are not smooth). G1 substitutes an AR(1)
# process (smooth, autocorrelated, but stochastic and non-periodic) for Sine, same
# DDM_240K+component_60K design as D1-D4.
G1_DATASET_ID: int | None = 63  # DDM 240K + AR(1) 60K
# Phase 5c (user-requested): G1 showed AR(1) -- smooth+autocorrelated but STOCHASTIC -- does NOT
# reproduce Sine/Delay's transfer effect. The candidates that separate Sine/Delay (deterministic)
# from AR1 (stochastic) overlap heavily (determinism, noise-free, closed-form recurrence, exact
# reproducibility) -- user's priority order to disentangle them:
#   H1: a SECOND deterministic+continuous+chaotic generator (Lorenz system), independent of
#       Mackey-Glass's specific delay-differential-equation structure -- if it ALSO shows the
#       dramatic effect, "deterministic continuous dynamics" (not periodicity, not Delay's
#       specific recurrence) is the common factor.
#   H2: an AR(1)-shaped recurrence with the SAME smoothing structure (same ar_phi) but
#       DETERMINISTIC (quasi-periodic multi-sine) forcing instead of iid noise -- isolates
#       whether it's specifically the injected per-step randomness that blocks plain ar1's effect.
H1_DATASET_ID: int | None = 64  # DDM 240K + Lorenz 60K
H2_DATASET_ID: int | None = 65  # DDM 240K + AR(1)-forced 60K
# Phase 5c-timescale (user-requested): H2's result was NOT a clean positive or negative -- across
# 3 (then 5, after extending) fine-tune seeds on the SAME DDM240K+ar1_forced60K checkpoint, some
# seeds converged to baseline (~0.83) and others broke through to a dramatically-improved,
# genuinely-converged plateau (~0.61) -- a seed-dependent bimodal/basin-selection pattern never
# seen in D1/D2 (always dramatic) or G1/H1 (always baseline). Rather than read further into that
# one ambiguous data point, isolate the characteristic-recurrence-TIMESCALE axis directly: swap
# ar1_forced's 5-superposed-period forcing (whose measured recurrence_lag=140 emerged indirectly)
# for a SINGLE forcing period, so the resulting series' recurrence timescale is controlled and
# measured directly -- then vary it (70 / 140 / 280 bars) to see whether transfer strength (and
# now also transfer PROBABILITY across seeds, given H2's instability finding) depends on the
# timescale's specific value once "a detectable recurrence exists at all" is held fixed.
I1_DATASET_ID: int | None = 68  # DDM 240K + AR(1)-forced(period=70) 60K
I2_DATASET_ID: int | None = 69  # DDM 240K + AR(1)-forced(period=140) 60K
I3_DATASET_ID: int | None = 70  # DDM 240K + AR(1)-forced(period=280) 60K
# Phase 5c-richness (user-requested): I1/I2 (1 period, 0/5 each) ruled out "140 is a special
# timescale" as H2's (5-period mixture, 2/3 unstable transfer) explanation. J1/J2 vary the period
# COUNT directly (1=I1/I2 -> 2 -> 3 -> 5=H2) over the SAME 47-157 span H2 used, to see whether
# transfer probability rises with the number of superposed periods, and where (if anywhere) it
# starts. K1 holds count fixed at 5 (like H2) but swaps in 5 DIFFERENT periods from a similar
# range -- separates "having 5 periods" from "H2's specific period values" per the user's request.
J1_DATASET_ID: int | None = 74  # DDM 240K + AR(1)-forced(2 periods: 47,157) 60K
J2_DATASET_ID: int | None = 75  # DDM 240K + AR(1)-forced(3 periods: 47,97,157) 60K
K1_DATASET_ID: int | None = 76  # DDM 240K + AR(1)-forced(5 DIFFERENT periods: 53,79,103,131,149) 60K
# Phase 5c-period-structure (user-requested): K1 (same count/range/all-prime as H2, but clean
# negative) narrowed the difference down to H2's exact gap sequence 24,26,30,30 (a repeated
# 30-gap / 3-term AP among 97,127,157) -- see analyze_period_structure.py. L1 reproduces that
# EXACT gap sequence shifted +10 to different absolute values (57,81,107,137,167), isolating
# "having the repeated-gap structure" from "H2's specific absolute period values".
L1_DATASET_ID: int | None = 77  # DDM 240K + AR(1)-forced(5 periods, H2's gap pattern +10: 57,81,107,137,167) 60K
# Phase 5d-dominant-scale (user-requested): the only axis that cleanly separated {sine,delay}
# (transfer) from every negative control (ar1/lorenz/ar1_forced/xor/lfsr) in the full 7-generator
# re-check was the GENERATED SERIES' OWN dominant spectral period landing near 50 bars (sine=50.0,
# delay=51.3; every negative off by 3x-50x). M1/M2 move Sine's period away from 50 in both
# directions (shorter, longer) to test whether transfer specifically requires matching this scale.
M1_DATASET_ID: int | None = 78  # DDM 240K + Sine(period=15) 60K
M2_DATASET_ID: int | None = 79  # DDM 240K + Sine(period=200) 60K
# Phase 6 Lyapunov dial (handoff open thread): Delay (~0.009/bar, transfers) vs Lorenz
# (~0.017/bar, doesn't) differ in far more than their Lyapunov exponent. Time-rescaling moves the
# per-bar exponent with each attractor held fixed (characterize_lyapunov_dial.py measured it):
#   N1: Lorenz at lorenz_dt=0.01 -> 0.0090/bar, matched to Delay's 0.0089
#   N2: Delay at stride=2        -> 0.0209/bar, above Lorenz's 0.0169
# Per-bar exponent decides -> N1 transfers, N2 doesn't. Generator identity decides -> reverse.
N1_DATASET_ID: int | None = 80  # DDM 240K + Lorenz(dt=0.01) 60K
N2_DATASET_ID: int | None = 81  # DDM 240K + Delay(stride=2) 60K
# Filled in by `prepare-data`/first submit -- the shared decoder_only MLModel both conditions'
# runs are created under (same architecture config = same warm-started weight shapes).
ML_MODEL_ID: int | None = None

VOL_PERIOD = 20  # bars; also fixes the tgt_feature_cols column name "vol_{VOL_PERIOD}"

# Only used by `prepare-data`'s (now unnecessary, kept only as a documented fallback -- see
# USDJPY_DATASET_ID's comment above) fresh yfinance collection path.
USDJPY_SYMBOL = "USDJPY=X"  # yfinance forex ticker format (plain "USDJPY" resolves to no data)
USDJPY_TIMEFRAME = "M1"

# Row cap applied to BOTH datasets via BASE_HP's max_rows (DDM's 300k-row dataset is unaffected;
# this caps USDJPY's 2.24M-row real history to a recent, still-substantial 1M-row / ~2-year
# slice instead of silently falling back to OHLCWindowDataset's own 50,000-row default -- see
# CLAUDE.md's "Always pass max_rows explicitly" standing lesson).
MAX_ROWS = 1_000_000

# Pretraining budget is explicitly NOT required to match condition A's data volume -- only the
# fine-tune side must match (user's own Phase 2 spec) -- picked independently, generous since
# it's "free".
#
# Generated as many SHORT independent runs concatenated together, NOT one long continuous run:
# a single DDMv3 trajectory long enough to cover DDM_PRETRAIN_ROWS candles (num_agent=300-500,
# tried directly against ddm_simulator.collect()'s wall-clock-bounded simulate_stream) reliably
# hit "DDM simulation stalled ... Agent prices likely diverged" -- WMA-feedback price divergence
# over a long single horizon, the same failure mode already on record in project memory ("DDM
# Simulator -- small agent count WMA divergence", "DDM is non-stationary over large data
# volumes"). generate_regime_datasets.py's established fix is exactly this: many short, freshly-
# seeded runs (divergence risk resets each run) concatenated with timestamp gaps, relying on
# OHLCWindowDataset's require_contiguous to keep windows from crossing a run boundary.
DDM_PRETRAIN_ROWS = 300_000
DDM_CANDLES_PER_RUN = 5_000
DDM_N_RUNS = DDM_PRETRAIN_ROWS // DDM_CANDLES_PER_RUN
DDM_TRADES_PER_CANDLE = 20
DDM_GAP_SECONDS = 86_400  # 1 day -- far outside the 60s M1 stride, unambiguous gap
DDM_NUM_AGENT = 500

# Condition C (priority (2)): DDM + Sine + Delay(Mackey-Glass) + XOR(temporal) + LFSR(8-bit),
# each contributing an equal share so the mixture's TOTAL row count equals DDM_PRETRAIN_ROWS --
# see _build_synthetic_mixture_data's docstring for why (isolating "more structural diversity"
# from "more data" is the whole point of comparing B vs C, per the user's own Phase 3/4 note).
MIXTURE_ROWS_PER_SOURCE = DDM_PRETRAIN_ROWS // 5  # 60_000 at current DDM_PRETRAIN_ROWS

# Phase 4 component ablation (user-requested): D1-D4 each replace ABLATION_COMPONENT_ROWS worth
# of condition B's all-DDM pretrain with one synthetic component, keeping the SAME total
# DDM_PRETRAIN_ROWS -- e.g. D1 = DDM_240K + Sine_60K vs B's DDM_300K, so "does adding Sine help"
# is isolated from "is there just less DDM now" (there's less DDM, but the same total volume).
ABLATION_COMPONENT_ROWS = MIXTURE_ROWS_PER_SOURCE  # 60_000
ABLATION_DDM_ROWS = DDM_PRETRAIN_ROWS - ABLATION_COMPONENT_ROWS  # 240_000

SEEDS_FULL = [42, 43, 44]

# Fine-tune-side budget -- IDENTICAL across condition A and condition B's fine-tune runs. Start
# conservative; recalibrate after `smoke` reports real wall-clock-per-step on this machine.
FINETUNE_MAX_STEPS = 20_000
FINETUNE_VAL_EVERY_STEPS = 1_000
FINETUNE_EARLY_STOP_PATIENCE_CHECKS = 5

# Pretrain-side budget -- independent of the fine-tune budget above (see DDM_PRETRAIN_ROWS note).
PRETRAIN_MAX_STEPS = 40_000
PRETRAIN_VAL_EVERY_STEPS = 2_000

# Extended fine-tune budget for the low-DDM-volume end of the Sine dose-response sweep (E3, E4):
# E3's original 20000-step fine-tune runs showed 2 of 3 seeds still actively improving (not
# early-stopped, not plateaued) when the budget ran out, while D1/E1/E2 all converged cleanly
# well within 20000 steps -- so the 20000-step number likely understated E3's true transfer
# performance rather than reflecting a real effect gap. Matches the pretrain budget so the
# early-stop patience (still FINETUNE_EARLY_STOP_PATIENCE_CHECKS=5 checks) has enough room to
# actually trigger. Applied to E3 (rerun) and E4 (planned from the start) only -- D1/E1/E2 keep
# their original 20000-step results since those already converged.
EXTENDED_FINETUNE_MAX_STEPS = 40_000
EXTENDED_FINETUNE_VAL_EVERY_STEPS = 2_000

MODEL_CONFIG = {
    # Pre-LN + small LayerScale: the fix (see uncommitted backend/model_core/architectures/
    # decoder_only.py diff / attention_no_layernorm_diagnostic.py) for the collapse plain
    # post-LN causes on a volatility-like target that needs magnitude information preserved.
    "layernorm_mode": "pre",
    "layerscale_init": 1e-2,
    # "last" (causal pooling), not "mean" -- the Five Axes investigation found the attention/
    # order mechanism itself (not just distributional/pairwise statistics) is what separates
    # Transformer from a permutation-invariant baseline on this exact target; mean-pooling with
    # full attention would give that mechanism away for free instead of testing it.
    "pooling": "last",
    "d_model": 64,
    "nhead": 4,
    "num_layers": 4,
    "dim_feedforward": 256,
    "dropout": 0.1,
}

BASE_HP = {
    "obs_len": 60,
    "pred_len": 1,
    "feature_cols": ["close"],
    "tgt_feature_cols": [f"vol_{VOL_PERIOD}"],
    "preprocessing": {"indicators": [{"type": "volatility", "period": VOL_PERIOD, "column": "close"}]},
    # zscore (not "none"): DDM's and USDJPY's raw vol_{PERIOD} scales differ by construction
    # (different price/tick conventions) -- standardizing per-dataset puts warm-started weights
    # on a comparable target scale, the same rationale as dataset.py's "returns_zscore" input
    # transform. Without this, most of any transfer effect (or lack of one) could just be a
    # trivial output-scale mismatch the fine-tune run has to relearn regardless of pretraining.
    "normalize": "zscore",
    # "regime_controlled" (not "chronological"): the first full run on the real 1M-row USDJPY
    # dataset (TrainingRuns 1432-1438) showed every condition-A seed's best val_loss landing at
    # the very first checkpoint then degrading monotonically -- diagnosed as a real train/val
    # regime shift, not noise (chronological split puts validation mostly in 2022's USDJPY
    # volatility surge, a different regime than the 2019-2021 training period). This sanity-check
    # rerun (user-requested, see conversation) stratifies train/val by the target's own value
    # distribution instead of by time (OHLCWindowDataset's regime_controlled mode -- deciles of
    # each window's target level, split within each decile), so a genuine transfer effect isn't
    # confounded with "which condition happens to handle the 2022 regime better." Trade-off
    # (documented on the mode itself in dataset.py): no longer a walk-forward evaluation, an
    # in-distribution one -- appropriate for isolating the pretraining question, not a claim this
    # is now a deployable forecast.
    "split_mode": "regime_controlled",
    "require_contiguous": True,
    "max_rows": MAX_ROWS,
    "val_split": 0.2,
    "batch_size": 64,
    "disable_lr_scheduler": True,
    # NOTE: vol_{PERIOD}'s first (PERIOD-1) rows are NaN (rolling std warm-up) -- preprocessing.py
    # doesn't drop them and OHLCWindowDataset's require_contiguous gap-mask is timestamp-gap-based,
    # not NaN-based, so a handful of early windows can carry a NaN target. Verified directly (see
    # conversation) that with max_rows=1,000,000 on both the USDJPY and DDM datasets used here,
    # zero windows in either the train or val split actually carry a NaN target -- the leading
    # NaN block is a negligible ~19 rows out of hundreds of thousands and gets diluted away.
}


def _finetune_hp(seed: int, warm_start_checkpoint: str | None) -> dict:
    hp = {
        **BASE_HP,
        "seed": seed,
        "max_steps": FINETUNE_MAX_STEPS,
        "val_every_steps": FINETUNE_VAL_EVERY_STEPS,
        "early_stop_patience_checks": FINETUNE_EARLY_STOP_PATIENCE_CHECKS,
    }
    if warm_start_checkpoint:
        hp["warm_start_checkpoint"] = warm_start_checkpoint
    return hp


def _pretrain_hp(seed: int, max_steps: int, val_every_steps: int) -> dict:
    return {
        **BASE_HP,
        "seed": seed,
        "max_steps": max_steps,
        "val_every_steps": val_every_steps,
        # No early stopping on the pretrain side -- it's not the comparison being measured, and
        # early-stopping the representation-building phase early is not obviously desirable here.
    }


def _simulate_ddm_segment(total_rows: int, candles_per_run: int = DDM_CANDLES_PER_RUN,
                           cursor_ts=None, seed_offset: int = 1000):
    """n_runs = total_rows // candles_per_run independent, freshly-seeded DDMv3 (v3_shock
    params) runs of candles_per_run candles each, concatenated with a timestamp gap between
    every pair -- see DDM_PRETRAIN_ROWS's comment for why not one long run. Returns
    (combined_df, from_ts, to_ts). Mirrors generate_regime_datasets.py's simulate/trades_to_ohlc/
    build_combined, which established this exact pattern for the same reason (avoiding DDMv3's
    long-horizon WMA divergence). seed_offset lets condition C's smaller DDM slice use a
    disjoint seed range from condition B's full-size pretrain, so they're not literally the same
    trajectories truncated."""
    import numpy as np
    import pandas as pd
    from data.collectors.ddm_simulator import DDMv3

    n_runs = total_rows // candles_per_run
    n_trades_per_run = candles_per_run * DDM_TRADES_PER_CANDLE
    cursor_ts = cursor_ts if cursor_ts is not None else pd.Timestamp("2000-01-03 00:00:00", tz="UTC")
    blocks = []
    for run_idx in range(n_runs):
        seed = seed_offset + run_idx
        np.random.seed(seed)
        model = DDMv3(
            num_agent=DDM_NUM_AGENT, max_volatility=0.02, min_volatility=0.01, wma=5,
            dealer_sensitive_min=-3.5, dealer_sensitive_max=-1.5,
            exogenous_shock_probability=0.0015, exogenous_shock_size=0.3,
        )
        prices = model.simulate(n_trades=n_trades_per_run)["price"].values

        n = (len(prices) // DDM_TRADES_PER_CANDLE) * DDM_TRADES_PER_CANDLE
        grouped = prices[:n].reshape(-1, DDM_TRADES_PER_CANDLE)
        ohlc = pd.DataFrame({
            "open": grouped[:, 0], "high": grouped.max(axis=1), "low": grouped.min(axis=1),
            "close": grouped[:, -1], "volume": float(DDM_TRADES_PER_CANDLE),
        }).iloc[:candles_per_run]

        idx = cursor_ts + pd.to_timedelta(np.arange(len(ohlc)) * 60, unit="s")
        ohlc.index = idx
        ohlc.index.name = "datetime"
        blocks.append(ohlc)
        cursor_ts = idx[-1] + pd.Timedelta(seconds=DDM_GAP_SECONDS)
        print(f"  DDM segment run {run_idx + 1}/{n_runs}: {len(ohlc)} candles")

    combined = pd.concat(blocks)
    return combined, combined.index[-1] + pd.Timedelta(seconds=DDM_GAP_SECONDS)


def _simulate_ddm_pretrain_data():
    """DDM-only pretrain dataset (DDM_PRETRAIN_ROWS candles) -- see _simulate_ddm_segment."""
    combined, _next_cursor = _simulate_ddm_segment(DDM_PRETRAIN_ROWS)
    return combined, combined.index[0].to_pydatetime(), combined.index[-1].to_pydatetime()


def _generate_synthetic_segment(function: str, length: int, seed: int, cursor_ts, **extra_config):
    """One synthetic_function.py series (sine/delay/xor/lfsr), as a flat-candle OHLC DataFrame
    re-indexed to start at cursor_ts -- calls the collector's own _generate_series directly
    (not collect(), which writes its own standalone file/datasource-keyed artifact; here we only
    want the raw values to fold into one combined mixture dataset, same reasoning as
    _simulate_ddm_segment building its own OHLC rather than going through ddm_simulator.collect()).
    Returns (df, next_cursor_ts)."""
    import numpy as np
    import pandas as pd
    from data.collectors.synthetic_function import _generate_series

    period = float(extra_config.get("period", 50))
    amplitude = float(extra_config.get("amplitude", 1.0))
    freq_ratio = float(extra_config.get("freq_ratio", 5))
    tau = float(extra_config.get("tau", 17))
    lfsr_bits = int(extra_config.get("lfsr_bits", 8))
    stride = int(extra_config.get("stride", 1))
    lorenz_dt = float(extra_config.get("lorenz_dt", 0.02))
    ar_phi = float(extra_config.get("ar_phi", 0.98))
    ar_sigma = float(extra_config.get("ar_sigma", 1.0))
    base_price = float(extra_config.get("base_price", 100.0))
    forced_periods = extra_config.get("forced_periods")

    values = base_price + _generate_series(
        function, length, period, amplitude, freq_ratio, tau=tau, lfsr_bits=lfsr_bits,
        ar_phi=ar_phi, ar_sigma=ar_sigma, seed=seed, forced_periods=forced_periods,
        stride=stride, lorenz_dt=lorenz_dt,
    )
    idx = cursor_ts + pd.to_timedelta(np.arange(length) * 60, unit="s")
    df = pd.DataFrame({
        "open": values, "high": values, "low": values, "close": values,
        "volume": np.ones(length),
    }, index=idx)
    df.index.name = "datetime"
    next_cursor = idx[-1] + pd.Timedelta(seconds=DDM_GAP_SECONDS)
    print(f"  synthetic segment {function!r}: {length} candles")
    return df, next_cursor


# Fixed per-component seeds/config so the same underlying synthetic sub-data is reused (not
# re-drawn) across every condition that includes that component -- e.g. condition C's 60K-row
# Sine segment (seed=2001) and D1's 60K-row Sine segment are byte-identical; only the DDM row
# count and overall composition differ between conditions. DDM's seed_offset is likewise shared
# (3000, 3001, 2, ...) so a smaller DDM allocation (e.g. condition C's 60K = seeds 3000-3011) is
# always a strict prefix of a larger one (e.g. D1-D4's 240K = seeds 3000-3047) -- not required
# for correctness, just keeps "which DDM trajectories are in this mixture" interpretable.
_SYNTHETIC_COMPONENT_CONFIG = {
    "sine": {"seed": 2001, "period": 50, "amplitude": 1.0},
    "delay": {"seed": 2002, "tau": 17},
    "xor": {"seed": 2003, "amplitude": 1.0},
    "lfsr": {"seed": 2004, "lfsr_bits": 8, "amplitude": 1.0},
    # ar1: Phase 5b (user-requested) "what is Sine actually doing" follow-up -- a smooth,
    # strongly-autocorrelated (ar_phi=0.98) but STOCHASTIC, non-periodic process, to test whether
    # the DDM+X transfer interaction needs smoothness per se or specifically a deterministic/
    # reproducible generator like sine/delay.
    "ar1": {"seed": 2005, "ar_phi": 0.98, "ar_sigma": 1.0},
    # lorenz/ar1_forced: Phase 5c determinism-isolation controls (see H1_DATASET_ID/H2_DATASET_ID
    # above) -- both fully deterministic, "seed" kept only for dict-shape consistency (unused).
    "lorenz": {"seed": 2006},
    "ar1_forced": {"seed": 2007, "ar_phi": 0.98, "amplitude": 1.0},
    # ar1_forced_p70/p140/p280: Phase 5c-timescale (user-requested) -- H2 (ar1_forced, 5-period
    # mixture, measured recurrence_lag=140) gave an unstable/seed-dependent result rather than a
    # clean positive or negative, so instead of reading more into that one data point, vary the
    # characteristic recurrence timescale DIRECTLY and cleanly: a single forcing period instead
    # of 5 superposed ones, so the resulting series' ACF recurrence lag equals `forced_periods[0]`
    # by construction rather than emerging indirectly. "function" overrides the dict key so all
    # three reuse the same underlying "ar1_forced" generator (see _build_mixture_data's loop).
    "ar1_forced_p70": {
        "seed": 2008, "ar_phi": 0.98, "amplitude": 1.0, "forced_periods": (70.0,), "function": "ar1_forced",
    },
    "ar1_forced_p140": {
        "seed": 2009, "ar_phi": 0.98, "amplitude": 1.0, "forced_periods": (140.0,), "function": "ar1_forced",
    },
    "ar1_forced_p280": {
        "seed": 2010, "ar_phi": 0.98, "amplitude": 1.0, "forced_periods": (280.0,), "function": "ar1_forced",
    },
    # ar1_forced_p2/p3/rand5: Phase 5c-richness (user-requested) -- I1/I2 (single clean period)
    # both gave clean 0/5 negatives at the SAME nominal timescales H2 (5-period mixture, 2/3
    # unstable transfer) used, ruling out "140 is a special timescale" as H2's explanation. This
    # richness sweep varies period COUNT directly (1=I1/I2, 2, 3, 5=H2) to test whether transfer
    # probability rises with the number of superposed periods. p2/p3 span the SAME 47-157 range
    # as H2's 5-period set (endpoints/midpoint included) so bandwidth stays comparable across the
    # sweep -- only the count of distinct timescales varies. rand5 holds count FIXED at 5 (same as
    # H2) but swaps in 5 different periods from a similar range/spread -- same amplitude weighting
    # (amplitude/n) and same spectral bandwidth as H2's set, isolating "having 5 periods" from
    # "H2's SPECIFIC period values" per the user's explicit request. NOTE: the existing
    # amplitude/n weighting (unchanged, kept consistent with H2/I1/I2's own formula) means total
    # forcing variance is NOT held exactly constant across the richness sweep (p2/p3/p5) -- it
    # decreases roughly as 1/n -- flagged to the user rather than silently changed.
    "ar1_forced_p2": {
        "seed": 2011, "ar_phi": 0.98, "amplitude": 1.0, "forced_periods": (47.0, 157.0), "function": "ar1_forced",
    },
    "ar1_forced_p3": {
        "seed": 2012, "ar_phi": 0.98, "amplitude": 1.0, "forced_periods": (47.0, 97.0, 157.0), "function": "ar1_forced",
    },
    "ar1_forced_rand5": {
        "seed": 2013, "ar_phi": 0.98, "amplitude": 1.0,
        "forced_periods": (53.0, 79.0, 103.0, 131.0, 149.0), "function": "ar1_forced",
    },
    # ar1_forced_l1: Phase 5c-period-structure (user-requested) -- analyze_period_structure.py
    # found the ONE clean mechanical difference between H2 (transfers, 2/3) and K1 (clean
    # negative, 0/5, same count/range/all-prime as H2): H2's adjacent-period gaps are
    # 24,26,30,30 -- the last three periods (97,127,157) form an exact 3-term arithmetic
    # progression (common difference 30), the only repeated value among all 10 pairwise
    # differences. K1 has no such repeat (10/10 distinct gaps). l1 reproduces H2's EXACT gap
    # sequence (24,26,30,30) shifted by +10 to different absolute period values (57,81,107,
    # 137,167) -- isolates "having this repeated-gap/AP structure" from "H2's specific absolute
    # period values".
    "ar1_forced_l1": {
        "seed": 2014, "ar_phi": 0.98, "amplitude": 1.0,
        "forced_periods": (57.0, 81.0, 107.0, 137.0, 167.0), "function": "ar1_forced",
    },
    # sine_p15/sine_p200: Phase 5d-dominant-scale (user-requested) -- characterize_synthetic_
    # generators.py's full 7-generator re-check found that NONE of periodicity/recurrence/
    # entropy/chaos/boundedness cleanly separates {sine, delay} (transfer) from every negative
    # control (ar1, lorenz, ar1_forced, xor, lfsr) -- except one: sine and delay's own dominant
    # spectral period lands almost exactly at 50 bars (sine=50.0, delay=51.3), while every
    # negative's dominant period is off by 3x-50x. M1/M2 move Sine's period AWAY from 50 in both
    # directions (shorter and longer) while keeping it a perfectly clean, deterministic,
    # single-tone periodic signal -- isolates "matches DDM's own ~50-bar characteristic
    # timescale" from "is periodic at all".
    "sine_p15": {"seed": 2001, "period": 15, "amplitude": 1.0, "function": "sine"},
    "sine_p200": {"seed": 2001, "period": 200, "amplitude": 1.0, "function": "sine"},
    # Phase 6 Lyapunov dial -- see N1/N2_DATASET_ID. Same seeds as the base lorenz/delay entries
    # (both are deterministic, so seed is irrelevant anyway).
    "lorenz_dt0.01": {"seed": 2006, "lorenz_dt": 0.01, "function": "lorenz"},
    "delay_s2": {"seed": 2002, "tau": 17, "stride": 2, "function": "delay"},
}


def _build_mixture_data(component_rows: dict):
    """General mixture builder: component_rows maps a component name ("ddm", "sine", "delay",
    "xor", "lfsr", or "ar1") to how many candles of it to include, e.g. {"ddm": 240_000, "sine": 60_000}
    for condition D1 (Phase 4's component-ablation design -- see conversation). Components are
    concatenated in a fixed order (ddm, sine, delay, xor, lfsr, ar1 -- whichever are present) with a
    timestamp gap between every segment, same as _build_synthetic_mixture_data (condition C's
    all-five-equal special case, now just one call to this with all five keys at
    MIXTURE_ROWS_PER_SOURCE each). Returns (combined_df, from_ts, to_ts)."""
    import pandas as pd

    cursor_ts = pd.Timestamp("2000-01-03 00:00:00", tz="UTC")
    blocks = []

    if component_rows.get("ddm"):
        ddm_df, cursor_ts = _simulate_ddm_segment(
            component_rows["ddm"], candles_per_run=DDM_CANDLES_PER_RUN, cursor_ts=cursor_ts, seed_offset=3000
        )
        blocks.append(ddm_df)

    for name in (
        "sine", "delay", "xor", "lfsr", "ar1", "lorenz", "ar1_forced",
        "ar1_forced_p70", "ar1_forced_p140", "ar1_forced_p280",
        "ar1_forced_p2", "ar1_forced_p3", "ar1_forced_rand5", "ar1_forced_l1",
        "sine_p15", "sine_p200", "lorenz_dt0.01", "delay_s2",
    ):
        rows = component_rows.get(name)
        if not rows:
            continue
        cfg = dict(_SYNTHETIC_COMPONENT_CONFIG[name])
        function = cfg.pop("function", name)
        df, cursor_ts = _generate_synthetic_segment(function, rows, cursor_ts=cursor_ts, **cfg)
        blocks.append(df)

    if not blocks:
        raise ValueError("component_rows must specify at least one component with rows > 0")

    combined = pd.concat(blocks)
    return combined, combined.index[0].to_pydatetime(), combined.index[-1].to_pydatetime()


def _build_synthetic_mixture_data():
    """Condition C: Sine + Delay(Mackey-Glass) + XOR(temporal) + LFSR(8-bit) + DDM(v3_shock),
    each MIXTURE_ROWS_PER_SOURCE candles. Total rows == DDM_PRETRAIN_ROWS (condition B's
    pretrain volume) by construction: the whole point of comparing B vs C is "same total
    pretraining volume, different composition" (per the user's own Phase 3/4 methodology note --
    separate 'more DDM data' from 'more structural diversity'), not "C also has more data than
    B." Thin wrapper over _build_mixture_data -- kept as its own function since it's referenced
    by name in docs/commit history and the already-registered dataset it produced (id=53)."""
    return _build_mixture_data({
        "ddm": MIXTURE_ROWS_PER_SOURCE, "sine": MIXTURE_ROWS_PER_SOURCE,
        "delay": MIXTURE_ROWS_PER_SOURCE, "xor": MIXTURE_ROWS_PER_SOURCE, "lfsr": MIXTURE_ROWS_PER_SOURCE,
    })


def _read_cached_yfinance_ohlc(symbol: str, timeframe: str):
    """Read finance_client's on-disk yfinance CSV cache directly, bypassing
    YahooClient.download()/CSVClient._get_ohlc_from_client()'s length=None bulk-retrieval path --
    that path is built around the CSV client's simulation-stepping API (self._step_index,
    "CSV data exhausted") and returns only 1 row for a fresh index=0 request instead of the
    full cached history. ohlc.collect() still has to be called first (see prepare_datasets) to
    make YahooClient.__init__'s incremental __get_rates() actually populate/refresh this cache
    file on disk -- that part works fine and is the only part actually needed."""
    import pandas as pd
    from finance_client import frames as Frame
    from data.collectors.ohlc import _FRAME_MAP

    frame = _FRAME_MAP.get(timeframe, Frame.D1)
    frame_str = Frame.to_str(frame)
    cache_path = Path.cwd() / "data_source" / "yfinance" / f"yfinance_{symbol}_{frame_str}.csv"
    df = pd.read_csv(cache_path, parse_dates=["Datetime"], index_col="Datetime")
    df.index.name = "datetime"
    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].sort_index()
    return df


# ---------------------------------------------------------------------------
# Phase 1: data prep -- no Celery worker required, calls collect() directly and registers the
# Dataset row the same way celery_worker.py's run_collection_job does for a non-incremental,
# non-DDM-streaming collector.
# ---------------------------------------------------------------------------

async def prepare_datasets() -> None:
    import database
    from data.models import Datasource, Dataset

    async with database.async_session_factory() as db:
        usdjpy_source = Datasource(
            name="USDJPY M1 (transfer experiment)",
            type="ohlc_download",
            config={"client": "yfinance", "symbol": USDJPY_SYMBOL, "timeframe": USDJPY_TIMEFRAME},
        )
        db.add(usdjpy_source)
        await db.flush()
        await db.refresh(usdjpy_source)

        await db.commit()
        usdjpy_source_id = usdjpy_source.id

    print(f"Created Datasource id={usdjpy_source_id} (USDJPY)")

    from data.collectors import ohlc

    print("Collecting USDJPY via yfinance (M1 intraday history is provider-limited -- expect "
          "only the last several weeks, not years)...")
    try:
        ohlc.collect(usdjpy_source_id, {
            "client": "yfinance", "symbol": USDJPY_SYMBOL, "timeframe": USDJPY_TIMEFRAME,
        })
    except RuntimeError:
        # "No data returned" can fire on collect()'s own (broken) length=None read; the on-disk
        # cache it refreshed as a side effect is what we actually read below.
        pass
    usdjpy_df = _read_cached_yfinance_ohlc(USDJPY_SYMBOL, USDJPY_TIMEFRAME)

    ddm_combined, ddm_from_ts, ddm_to_ts = _simulate_ddm_pretrain_data()

    async with database.async_session_factory() as db:
        import os
        store = Path(os.getenv("ARTIFACT_STORE_PATH", "../artifacts")).resolve()
        usdjpy_artifact_rel = f"datasets/src_{usdjpy_source_id}/{USDJPY_SYMBOL.replace('/', '_')}_{USDJPY_TIMEFRAME}.parquet"
        (store / usdjpy_artifact_rel).parent.mkdir(parents=True, exist_ok=True)
        usdjpy_df.to_parquet(store / usdjpy_artifact_rel)

        usdjpy_ds = Dataset(
            datasource_id=usdjpy_source_id, name="USDJPY M1 (transfer experiment)",
            symbol=USDJPY_SYMBOL, timeframe=USDJPY_TIMEFRAME,
            from_ts=usdjpy_df.index[0].to_pydatetime(), to_ts=usdjpy_df.index[-1].to_pydatetime(),
            row_count=len(usdjpy_df), artifact_path=usdjpy_artifact_rel,
            status="ready",
        )
        db.add(usdjpy_ds)

        ddm_artifact_rel = "datasets/derived/ddm_v3_shock_transfer_pretrain.parquet"
        (store / ddm_artifact_rel).parent.mkdir(parents=True, exist_ok=True)
        ddm_combined.to_parquet(store / ddm_artifact_rel)

        ddm_ds = Dataset(
            datasource_id=None, name="DDM v3_shock (transfer experiment pretrain)",
            symbol="DDM-SYNTH", timeframe="M1",
            from_ts=ddm_from_ts, to_ts=ddm_to_ts,
            row_count=len(ddm_combined), artifact_path=ddm_artifact_rel,
            status="ready",
        )
        db.add(ddm_ds)
        await db.commit()
        await db.refresh(usdjpy_ds)
        await db.refresh(ddm_ds)

    print(f"USDJPY dataset id={usdjpy_ds.id} row_count={usdjpy_ds.row_count} "
          f"span={usdjpy_ds.from_ts} .. {usdjpy_ds.to_ts}")
    print(f"DDM dataset id={ddm_ds.id} row_count={ddm_ds.row_count} "
          f"span={ddm_ds.from_ts} .. {ddm_ds.to_ts}")
    print("\nPaste these into USDJPY_DATASET_ID / DDM_DATASET_ID at the top of this file.")


async def prepare_mixture_data() -> None:
    """Condition C's pretraining dataset: Sine + Delay + XOR + LFSR + DDM, MIXTURE_ROWS_PER_SOURCE
    candles each -- see _build_synthetic_mixture_data. No Celery worker or Datasource row needed
    (same direct-register pattern as the DDM-only dataset in prepare_datasets)."""
    import os

    import database
    from data.models import Dataset

    mixture_combined, mixture_from_ts, mixture_to_ts = _build_synthetic_mixture_data()

    async with database.async_session_factory() as db:
        store = Path(os.getenv("ARTIFACT_STORE_PATH", "../artifacts")).resolve()
        artifact_rel = "datasets/derived/synthetic_mixture_transfer_pretrain.parquet"
        (store / artifact_rel).parent.mkdir(parents=True, exist_ok=True)
        mixture_combined.to_parquet(store / artifact_rel)

        mixture_ds = Dataset(
            datasource_id=None, name="Synthetic mixture (transfer experiment pretrain, condition C)",
            symbol="MIXED-SYNTH", timeframe="M1",
            from_ts=mixture_from_ts, to_ts=mixture_to_ts,
            row_count=len(mixture_combined), artifact_path=artifact_rel,
            status="ready",
        )
        db.add(mixture_ds)
        await db.commit()
        await db.refresh(mixture_ds)

    print(f"Mixture dataset id={mixture_ds.id} row_count={mixture_ds.row_count} "
          f"span={mixture_ds.from_ts} .. {mixture_ds.to_ts}")
    print("\nPaste this into MIXTURE_DATASET_ID at the top of this file.")


async def _register_mixture_dataset(name: str, artifact_name: str, component_rows: dict) -> int:
    """Builds one _build_mixture_data(component_rows) result and registers it as a Dataset --
    shared by prepare_ablation_data for D1-D4. Returns the new Dataset's id."""
    import os

    import database
    from data.models import Dataset

    combined, from_ts, to_ts = _build_mixture_data(component_rows)

    async with database.async_session_factory() as db:
        store = Path(os.getenv("ARTIFACT_STORE_PATH", "../artifacts")).resolve()
        artifact_rel = f"datasets/derived/{artifact_name}.parquet"
        (store / artifact_rel).parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(store / artifact_rel)

        ds = Dataset(
            datasource_id=None, name=name, symbol="MIXED-SYNTH", timeframe="M1",
            from_ts=from_ts, to_ts=to_ts, row_count=len(combined), artifact_path=artifact_rel,
            status="ready",
        )
        db.add(ds)
        await db.commit()
        await db.refresh(ds)

    print(f"{name}: dataset id={ds.id} row_count={ds.row_count} span={ds.from_ts} .. {ds.to_ts}")
    return ds.id


async def prepare_ablation_data() -> None:
    """Phase 4 component ablation (user-requested): D1-D4 = DDM ABLATION_DDM_ROWS + one
    component at ABLATION_COMPONENT_ROWS, isolating "does adding this one component to DDM
    help" one at a time, before condition C's "all four at once" result is decomposed further.
    No Celery worker needed -- same direct-register pattern as prepare_mixture_data."""
    d1 = await _register_mixture_dataset(
        "D1: DDM 240K + Sine 60K (ablation)", "ablation_d1_ddm_sine",
        {"ddm": ABLATION_DDM_ROWS, "sine": ABLATION_COMPONENT_ROWS},
    )
    d2 = await _register_mixture_dataset(
        "D2: DDM 240K + Delay 60K (ablation)", "ablation_d2_ddm_delay",
        {"ddm": ABLATION_DDM_ROWS, "delay": ABLATION_COMPONENT_ROWS},
    )
    d3 = await _register_mixture_dataset(
        "D3: DDM 240K + XOR 60K (ablation)", "ablation_d3_ddm_xor",
        {"ddm": ABLATION_DDM_ROWS, "xor": ABLATION_COMPONENT_ROWS},
    )
    d4 = await _register_mixture_dataset(
        "D4: DDM 240K + LFSR 60K (ablation)", "ablation_d4_ddm_lfsr",
        {"ddm": ABLATION_DDM_ROWS, "lfsr": ABLATION_COMPONENT_ROWS},
    )
    print(f"\nPaste these into D1_DATASET_ID={d1} / D2_DATASET_ID={d2} / "
          f"D3_DATASET_ID={d3} / D4_DATASET_ID={d4} at the top of this file.")


async def prepare_dose_response_data() -> None:
    """Phase 4a Sine dose-response sweep (user-requested): E1-E3 fill in the DDM/Sine ratio
    curve between B (DDM 300K, all-DDM) and D1 (DDM 240K + Sine 60K, already confirmed
    dramatic); E4 is the pure-Sine endpoint (no DDM at all). Same total DDM_PRETRAIN_ROWS
    volume throughout -- only the split changes. No Celery worker needed."""
    e1 = await _register_mixture_dataset(
        "E1: DDM 180K + Sine 120K (dose-response)", "dose_response_e1_ddm180_sine120",
        {"ddm": 180_000, "sine": 120_000},
    )
    e2 = await _register_mixture_dataset(
        "E2: DDM 120K + Sine 180K (dose-response)", "dose_response_e2_ddm120_sine180",
        {"ddm": 120_000, "sine": 180_000},
    )
    e3 = await _register_mixture_dataset(
        "E3: DDM 60K + Sine 240K (dose-response)", "dose_response_e3_ddm60_sine240",
        {"ddm": 60_000, "sine": 240_000},
    )
    e4 = await _register_mixture_dataset(
        "E4: Sine 300K pure (dose-response)", "dose_response_e4_sine300",
        {"sine": DDM_PRETRAIN_ROWS},
    )
    print(f"\nPaste these into E1_DATASET_ID={e1} / E2_DATASET_ID={e2} / "
          f"E3_DATASET_ID={e3} / E4_DATASET_ID={e4} at the top of this file.")


async def prepare_interaction_data() -> None:
    """Phase 4b (user-requested): does the DDM x Sine interaction found via D1/E1-E4 (DDM alone
    -> baseline, Sine alone -> baseline, DDM+Sine together -> dramatic effect) generalize to
    Delay, the other component D1-D4's screening found a "dramatic effect" for? Only the missing
    corner needs building -- DDM alone is condition B (DDM_DATASET_ID, already run) and DDM+Delay
    is D2 (D2_DATASET_ID, already run); this adds F1 = pure Delay (0 DDM), D2's mixture composition
    with the DDM component simply omitted so it's directly comparable to D2 the same way E4 was to
    D1. No Celery worker needed."""
    f1 = await _register_mixture_dataset(
        "F1: Delay 300K pure (interaction check)", "interaction_f1_delay300",
        {"delay": DDM_PRETRAIN_ROWS},
    )
    print(f"\nPaste this into F1_DATASET_ID={f1} at the top of this file.")


async def prepare_mechanism_data() -> None:
    """Phase 5b (user-requested): G1 = DDM 240K + AR(1) 60K, testing whether the DDM+X transfer
    interaction (Phase 5's representation analysis) needs smoothness per se or specifically a
    deterministic/reproducible generator like sine/delay. Same ABLATION_DDM_ROWS/
    ABLATION_COMPONENT_ROWS split as D1-D4. No Celery worker needed."""
    g1 = await _register_mixture_dataset(
        "G1: DDM 240K + AR(1) 60K (mechanism check)", "mechanism_g1_ddm_ar1",
        {"ddm": ABLATION_DDM_ROWS, "ar1": ABLATION_COMPONENT_ROWS},
    )
    print(f"\nPaste this into G1_DATASET_ID={g1} at the top of this file.")


async def prepare_determinism_data() -> None:
    """Phase 5c (user-requested): G1 (AR(1)) showed no effect, isolating that Sine/Delay's shared
    "smoothness" isn't the operative property. H1/H2 split the overlapping determinism-adjacent
    candidates (deterministic / noise-free / closed-form recurrence / exact reproducibility) --
    see H1_DATASET_ID/H2_DATASET_ID's comments for what each isolates. Same ABLATION_DDM_ROWS/
    ABLATION_COMPONENT_ROWS split as D1-D4/G1. No Celery worker needed."""
    h1 = await _register_mixture_dataset(
        "H1: DDM 240K + Lorenz 60K (determinism control)", "determinism_h1_ddm_lorenz",
        {"ddm": ABLATION_DDM_ROWS, "lorenz": ABLATION_COMPONENT_ROWS},
    )
    h2 = await _register_mixture_dataset(
        "H2: DDM 240K + AR(1)-forced 60K (determinism control)", "determinism_h2_ddm_ar1forced",
        {"ddm": ABLATION_DDM_ROWS, "ar1_forced": ABLATION_COMPONENT_ROWS},
    )
    print(f"\nPaste these into H1_DATASET_ID={h1} / H2_DATASET_ID={h2} at the top of this file.")


async def prepare_timescale_data() -> None:
    """Phase 5c-timescale (user-requested): H2 (ar1_forced, 5-superposed-period forcing) gave a
    seed-dependent, unstable result instead of a clean positive/negative -- see I1/I2/I3's
    comments at the top of this file. I1/I2/I3 swap ar1_forced's 5-period mixture for a SINGLE
    forcing period (70 / 140 / 280 bars respectively), giving each a directly-controlled
    characteristic recurrence timescale instead of one that emerges indirectly. Same
    ABLATION_DDM_ROWS/ABLATION_COMPONENT_ROWS split as D1-D4/G1/H1/H2. No Celery worker needed."""
    i1 = await _register_mixture_dataset(
        "I1: DDM 240K + AR(1)-forced(period=70) 60K (timescale control)", "timescale_i1_ddm_ar1forced_p70",
        {"ddm": ABLATION_DDM_ROWS, "ar1_forced_p70": ABLATION_COMPONENT_ROWS},
    )
    i2 = await _register_mixture_dataset(
        "I2: DDM 240K + AR(1)-forced(period=140) 60K (timescale control)", "timescale_i2_ddm_ar1forced_p140",
        {"ddm": ABLATION_DDM_ROWS, "ar1_forced_p140": ABLATION_COMPONENT_ROWS},
    )
    i3 = await _register_mixture_dataset(
        "I3: DDM 240K + AR(1)-forced(period=280) 60K (timescale control)", "timescale_i3_ddm_ar1forced_p280",
        {"ddm": ABLATION_DDM_ROWS, "ar1_forced_p280": ABLATION_COMPONENT_ROWS},
    )
    print(f"\nPaste these into I1_DATASET_ID={i1} / I2_DATASET_ID={i2} / I3_DATASET_ID={i3} "
          f"at the top of this file.")


async def prepare_richness_data() -> None:
    """Phase 5c-richness (user-requested): I1/I2 (single clean period, 0/5 each) ruled out "140
    is a special timescale" as H2's (5-superposed-period mixture, 2/3 unstable transfer)
    explanation. J1/J2/K1 test what IS different about H2's forcing:
      J1/J2: vary period COUNT directly (1=I1/I2 -> 2 -> 3 -> 5=H2), same 47-157 span H2 used, to
             see whether transfer probability rises with the number of superposed periods.
      K1:    holds count fixed at 5 (like H2) but swaps in 5 DIFFERENT periods from a similar
             range/spread -- separates "having 5 periods" from "H2's specific period values".
    Same ABLATION_DDM_ROWS/ABLATION_COMPONENT_ROWS split as D1-D4/G1/H1/H2/I1-I3. No Celery
    worker needed."""
    j1 = await _register_mixture_dataset(
        "J1: DDM 240K + AR(1)-forced(2 periods: 47,157) 60K (richness control)",
        "richness_j1_ddm_ar1forced_p2",
        {"ddm": ABLATION_DDM_ROWS, "ar1_forced_p2": ABLATION_COMPONENT_ROWS},
    )
    j2 = await _register_mixture_dataset(
        "J2: DDM 240K + AR(1)-forced(3 periods: 47,97,157) 60K (richness control)",
        "richness_j2_ddm_ar1forced_p3",
        {"ddm": ABLATION_DDM_ROWS, "ar1_forced_p3": ABLATION_COMPONENT_ROWS},
    )
    k1 = await _register_mixture_dataset(
        "K1: DDM 240K + AR(1)-forced(5 different periods: 53,79,103,131,149) 60K (richness control)",
        "richness_k1_ddm_ar1forced_rand5",
        {"ddm": ABLATION_DDM_ROWS, "ar1_forced_rand5": ABLATION_COMPONENT_ROWS},
    )
    print(f"\nPaste these into J1_DATASET_ID={j1} / J2_DATASET_ID={j2} / K1_DATASET_ID={k1} "
          f"at the top of this file.")


async def prepare_period_structure_data() -> None:
    """Phase 5c-period-structure (user-requested): L1 reproduces H2's exact adjacent-gap sequence
    (24,26,30,30 -- a repeated 30-gap / 3-term AP among its last three periods) shifted +10 to
    different absolute period values (57,81,107,137,167), isolating "having the repeated-gap/AP
    structure" from "H2's specific absolute period values" -- see analyze_period_structure.py for
    the full feature comparison that motivated this. Same ABLATION_DDM_ROWS/
    ABLATION_COMPONENT_ROWS split as D1-D4/G1/H1/H2/I1-I3/J1-J2/K1. No Celery worker needed."""
    l1 = await _register_mixture_dataset(
        "L1: DDM 240K + AR(1)-forced(H2 gap pattern +10: 57,81,107,137,167) 60K (period-structure control)",
        "period_structure_l1_ddm_ar1forced",
        {"ddm": ABLATION_DDM_ROWS, "ar1_forced_l1": ABLATION_COMPONENT_ROWS},
    )
    print(f"\nPaste this into L1_DATASET_ID={l1} at the top of this file.")


async def prepare_dominant_scale_data() -> None:
    """Phase 5d-dominant-scale (user-requested): M1/M2 move Sine's period away from 50 (the
    dominant spectral period shared by sine/delay, the only axis found to discriminate them from
    every negative control -- see characterize_synthetic_generators.py) in both directions --
    M1 shorter (period=15), M2 longer (period=200) -- while keeping the signal a perfectly clean,
    deterministic, single-tone periodic sine. Same ABLATION_DDM_ROWS/ABLATION_COMPONENT_ROWS
    split as D1-D4/G1/H1/H2/I1-I3/J1-J2/K1/L1. No Celery worker needed."""
    m1 = await _register_mixture_dataset(
        "M1: DDM 240K + Sine(period=15) 60K (dominant-scale control)",
        "dominant_scale_m1_ddm_sine_p15",
        {"ddm": ABLATION_DDM_ROWS, "sine_p15": ABLATION_COMPONENT_ROWS},
    )
    m2 = await _register_mixture_dataset(
        "M2: DDM 240K + Sine(period=200) 60K (dominant-scale control)",
        "dominant_scale_m2_ddm_sine_p200",
        {"ddm": ABLATION_DDM_ROWS, "sine_p200": ABLATION_COMPONENT_ROWS},
    )
    print(f"\nPaste these into M1_DATASET_ID={m1} / M2_DATASET_ID={m2} at the top of this file.")


async def prepare_lyapunov_dial_data() -> None:
    """Phase 6 Lyapunov dial: N1 (Lorenz, dt=0.01) and N2 (Delay, stride=2) -- see
    N1/N2_DATASET_ID. Same ABLATION_DDM_ROWS/ABLATION_COMPONENT_ROWS split as D1-M2."""
    n1 = await _register_mixture_dataset(
        "N1: DDM 240K + Lorenz(dt=0.01) 60K (Lyapunov dial)",
        "lyapunov_dial_n1_ddm_lorenz_dt001",
        {"ddm": ABLATION_DDM_ROWS, "lorenz_dt0.01": ABLATION_COMPONENT_ROWS},
    )
    n2 = await _register_mixture_dataset(
        "N2: DDM 240K + Delay(stride=2) 60K (Lyapunov dial)",
        "lyapunov_dial_n2_ddm_delay_s2",
        {"ddm": ABLATION_DDM_ROWS, "delay_s2": ABLATION_COMPONENT_ROWS},
    )
    print(f"\nPaste these into N1_DATASET_ID={n1} / N2_DATASET_ID={n2} at the top of this file.")


# ---------------------------------------------------------------------------
# Phase 2: submit TrainingRuns
# ---------------------------------------------------------------------------

async def _ensure_model() -> int:
    if ML_MODEL_ID is not None:
        return ML_MODEL_ID
    import database
    from model.models import MLModel

    async with database.async_session_factory() as db:
        model = MLModel(name="decoder_only (transfer experiment)", architecture="decoder_only", config=MODEL_CONFIG)
        db.add(model)
        await db.commit()
        await db.refresh(model)
        print(f"Created MLModel id={model.id} -- paste into ML_MODEL_ID at the top of this file.")
        return model.id


async def _submit(model_id: int, dataset_id: int, hyperparams: dict, execution_target: str = "local") -> int:
    import database
    from model.repository import model_repo
    from celery_app import enqueue

    async with database.async_session_factory() as db:
        run = await model_repo.create_training_run(
            db, model_id=model_id, dataset_id=dataset_id,
            preprocessed_dataset_id=None, hyperparams=hyperparams, execution_target=execution_target,
        )
        await db.commit()
        run_id = run.id
    task_name = "colab_train_model" if execution_target == "colab" else "train_model"
    await enqueue(task_name, run_id)
    print(f"Submitted TrainingRun id={run_id} dataset_id={dataset_id} execution_target={execution_target} hyperparams={hyperparams}")
    return run_id


def _require_dataset_ids() -> None:
    if USDJPY_DATASET_ID is None or DDM_DATASET_ID is None:
        raise SystemExit(
            "USDJPY_DATASET_ID / DDM_DATASET_ID are not set -- run `prepare-data` first, "
            "then paste the printed dataset ids into this file."
        )


async def run_condition_a(seeds: list[int], max_steps: int, val_every_steps: int, execution_target: str = "local") -> list[int]:
    """Condition A: USDJPY from scratch."""
    model_id = await _ensure_model()
    run_ids = []
    for seed in seeds:
        hp = _finetune_hp(seed, warm_start_checkpoint=None)
        hp["max_steps"], hp["val_every_steps"] = max_steps, val_every_steps
        run_ids.append(await _submit(model_id, USDJPY_DATASET_ID, hp, execution_target))
    return run_ids


async def run_pretrain_then_finetune(
    pretrain_dataset_id: int,
    seeds: list[int], pretrain_seed: int,
    pretrain_max_steps: int, pretrain_val_every_steps: int,
    finetune_max_steps: int, finetune_val_every_steps: int,
    execution_target: str = "local",
) -> tuple[int, list[int]]:
    """Shared by condition B (pretrain_dataset_id=DDM_DATASET_ID) and condition C
    (pretrain_dataset_id=MIXTURE_DATASET_ID): pretrain (single seed) on pretrain_dataset_id ->
    USDJPY fine-tune (one run per seed in `seeds`).

    Waits for the pretrain run to reach a terminal status before submitting fine-tune runs, since
    they need its best.pt checkpoint path. Requires a Celery worker on the matching queue
    (`training` for execution_target="local", `colab` for "colab") to actually be running --
    this function only submits/polls, it does not run the training itself.
    """
    import database
    from sqlalchemy import select
    from model.models import TrainingRun

    model_id = await _ensure_model()
    pretrain_hp = _pretrain_hp(pretrain_seed, pretrain_max_steps, pretrain_val_every_steps)
    pretrain_run_id = await _submit(model_id, pretrain_dataset_id, pretrain_hp, execution_target)

    print(f"Waiting for pretrain run {pretrain_run_id} to complete "
          f"(requires a `{execution_target}`-queue Celery worker running)...")
    while True:
        async with database.async_session_factory() as db:
            run = (await db.execute(select(TrainingRun).where(TrainingRun.id == pretrain_run_id))).scalar_one()
            if run.status in ("completed", "error", "stopped"):
                break
        await asyncio.sleep(10)

    if run.status != "completed":
        raise RuntimeError(f"Pretrain run {pretrain_run_id} ended with status={run.status!r}, not proceeding to fine-tune")

    checkpoint_path = f"models/{model_id}/training_{pretrain_run_id}/best.pt"
    finetune_run_ids = []
    for seed in seeds:
        hp = _finetune_hp(seed, warm_start_checkpoint=checkpoint_path)
        hp["max_steps"], hp["val_every_steps"] = finetune_max_steps, finetune_val_every_steps
        finetune_run_ids.append(await _submit(model_id, USDJPY_DATASET_ID, hp, execution_target))
    return pretrain_run_id, finetune_run_ids


async def run_finetune_only(
    checkpoint_path: str, seeds: list[int],
    finetune_max_steps: int, finetune_val_every_steps: int,
    execution_target: str = "local",
) -> list[int]:
    """Fine-tune-only re-run against an EXISTING, already-completed pretrain checkpoint --
    used to extend a condition's fine-tune step budget without re-running the (deterministic,
    unaffected-by-finetune-budget) pretrain phase. checkpoint_path is artifact-store-relative,
    e.g. "models/135/training_1478/best.pt" (same format run_pretrain_then_finetune builds)."""
    model_id = await _ensure_model()
    finetune_run_ids = []
    for seed in seeds:
        hp = _finetune_hp(seed, warm_start_checkpoint=checkpoint_path)
        hp["max_steps"], hp["val_every_steps"] = finetune_max_steps, finetune_val_every_steps
        finetune_run_ids.append(await _submit(model_id, USDJPY_DATASET_ID, hp, execution_target))
    return finetune_run_ids


async def _run_smoke() -> None:
    seeds = [42]
    print("=== condition A (smoke) ===")
    await run_condition_a(seeds, max_steps=1000, val_every_steps=200)
    print("=== condition B (smoke) ===")
    await run_pretrain_then_finetune(
        DDM_DATASET_ID, seeds, pretrain_seed=42,
        pretrain_max_steps=1000, pretrain_val_every_steps=200,
        finetune_max_steps=1000, finetune_val_every_steps=200,
    )


async def _run_full(seeds: list[int]) -> None:
    # Both conditions local: the colab split (condition B via execution_target="colab", see
    # docs/colab-workflow.md) is implemented and verified via a local cell-extraction proxy test,
    # but blocked for a REAL run right now by an expired Google Drive OAuth token that needs an
    # interactive browser re-login (docs/colab-workflow.md "One-time setup" step 1) -- pass
    # execution_target="colab" to run_pretrain_then_finetune once that's done, to parallelize
    # condition A/B across the `training`/`colab` queues instead of running both serially here.
    print("=== condition A (full, local) ===")
    await run_condition_a(seeds, max_steps=FINETUNE_MAX_STEPS, val_every_steps=FINETUNE_VAL_EVERY_STEPS)
    print("=== condition B (full, local) ===")
    await run_pretrain_then_finetune(
        DDM_DATASET_ID, seeds, pretrain_seed=42,
        pretrain_max_steps=PRETRAIN_MAX_STEPS, pretrain_val_every_steps=PRETRAIN_VAL_EVERY_STEPS,
        finetune_max_steps=FINETUNE_MAX_STEPS, finetune_val_every_steps=FINETUNE_VAL_EVERY_STEPS,
    )


async def _run_condition_c(seeds: list[int]) -> None:
    """Priority (2), minimal 3-condition design (user-requested): condition C only -- A and B
    already exist from the regime_controlled sanity-check rerun (TrainingRuns 1439-1441 and
    1442-1445). Pretrains on the synthetic mixture (MIXTURE_DATASET_ID) instead of DDM alone,
    then fine-tunes on USDJPY with the exact same budget/seeds/architecture as A and B so the
    three conditions are directly comparable."""
    if MIXTURE_DATASET_ID is None:
        raise SystemExit(
            "MIXTURE_DATASET_ID is not set -- run `prepare-mixture-data` first, "
            "then paste the printed dataset id into this file."
        )
    print("=== condition C (mixture pretrain -> USDJPY fine-tune) ===")
    await run_pretrain_then_finetune(
        MIXTURE_DATASET_ID, seeds, pretrain_seed=42,
        pretrain_max_steps=PRETRAIN_MAX_STEPS, pretrain_val_every_steps=PRETRAIN_VAL_EVERY_STEPS,
        finetune_max_steps=FINETUNE_MAX_STEPS, finetune_val_every_steps=FINETUNE_VAL_EVERY_STEPS,
    )


_D_DATASET_IDS = {
    "d1": lambda: D1_DATASET_ID, "d2": lambda: D2_DATASET_ID,
    "d3": lambda: D3_DATASET_ID, "d4": lambda: D4_DATASET_ID,
    # Phase 4a Sine dose-response sweep -- same lookup dict/CLI plumbing as D1-D4, since
    # _run_condition_d/_run_replicate_pretrain don't care which phase a dataset belongs to.
    "e1": lambda: E1_DATASET_ID, "e2": lambda: E2_DATASET_ID,
    "e3": lambda: E3_DATASET_ID, "e4": lambda: E4_DATASET_ID,
    "f1": lambda: F1_DATASET_ID,
    "g1": lambda: G1_DATASET_ID,
    "h1": lambda: H1_DATASET_ID,
    "h2": lambda: H2_DATASET_ID,
    "i1": lambda: I1_DATASET_ID, "i2": lambda: I2_DATASET_ID, "i3": lambda: I3_DATASET_ID,
    "j1": lambda: J1_DATASET_ID, "j2": lambda: J2_DATASET_ID, "k1": lambda: K1_DATASET_ID,
    "l1": lambda: L1_DATASET_ID,
    "m1": lambda: M1_DATASET_ID, "m2": lambda: M2_DATASET_ID,
    "n1": lambda: N1_DATASET_ID, "n2": lambda: N2_DATASET_ID,
}
_D_LABELS = {
    "d1": "D1 (DDM 240K + Sine 60K)", "d2": "D2 (DDM 240K + Delay 60K)",
    "d3": "D3 (DDM 240K + XOR 60K)", "d4": "D4 (DDM 240K + LFSR 60K)",
    "e1": "E1 (DDM 180K + Sine 120K)", "e2": "E2 (DDM 120K + Sine 180K)",
    "e3": "E3 (DDM 60K + Sine 240K)", "e4": "E4 (Sine 300K pure)",
    "f1": "F1 (Delay 300K pure)",
    "g1": "G1 (DDM 240K + AR(1) 60K)",
    "h1": "H1 (DDM 240K + Lorenz 60K)",
    "h2": "H2 (DDM 240K + AR(1)-forced 60K)",
    "i1": "I1 (DDM 240K + AR(1)-forced period=70 60K)",
    "i2": "I2 (DDM 240K + AR(1)-forced period=140 60K)",
    "i3": "I3 (DDM 240K + AR(1)-forced period=280 60K)",
    "j1": "J1 (DDM 240K + AR(1)-forced 2 periods 60K)",
    "j2": "J2 (DDM 240K + AR(1)-forced 3 periods 60K)",
    "k1": "K1 (DDM 240K + AR(1)-forced 5 different periods 60K)",
    "l1": "L1 (DDM 240K + AR(1)-forced H2 gap pattern +10 60K)",
    "m1": "M1 (DDM 240K + Sine period=15 60K)",
    "m2": "M2 (DDM 240K + Sine period=200 60K)",
    "n1": "N1 (DDM 240K + Lorenz dt=0.01 60K)",
    "n2": "N2 (DDM 240K + Delay stride=2 60K)",
}
_PREPARE_HINT = {
    "d1": "prepare-ablation-data", "d2": "prepare-ablation-data",
    "d3": "prepare-ablation-data", "d4": "prepare-ablation-data",
    "e1": "prepare-dose-response-data", "e2": "prepare-dose-response-data",
    "e3": "prepare-dose-response-data", "e4": "prepare-dose-response-data",
    "f1": "prepare-interaction-data",
    "g1": "prepare-mechanism-data",
    "h1": "prepare-determinism-data", "h2": "prepare-determinism-data",
    "i1": "prepare-timescale-data", "i2": "prepare-timescale-data", "i3": "prepare-timescale-data",
    "j1": "prepare-richness-data", "j2": "prepare-richness-data", "k1": "prepare-richness-data",
    "l1": "prepare-period-structure-data",
    "m1": "prepare-dominant-scale-data", "m2": "prepare-dominant-scale-data",
    "n1": "prepare-lyapunov-dial-data", "n2": "prepare-lyapunov-dial-data",
}


async def _run_condition_d(which: str, seeds: list[int]) -> None:
    """Phase 4 component ablation / Phase 4a Sine dose-response (user-requested): one of
    D1-D4/E1-E4 -- pretrain on the corresponding mixture dataset, then fine-tune on USDJPY with
    the exact same budget/seeds/architecture as A/B/C. Run each condition separately
    (`condition-d d1`, `condition-d e2`, etc.) rather than batched -- each is already a
    multi-hour job on this machine's single local worker; running them as separate invocations
    means a crash/restart partway through only loses the one in flight, not the whole batch."""
    dataset_id = _D_DATASET_IDS[which]()
    if dataset_id is None:
        raise SystemExit(
            f"{which.upper()}_DATASET_ID is not set -- run `{_PREPARE_HINT[which]}` first, "
            "then paste the printed dataset ids into this file."
        )
    # 0-DDM conditions (E4: pure Sine, F1: pure Delay) get the extended fine-tune budget from the
    # start: E3 (DDM 60K, the next point up in DDM volume) needed it -- 2 of 3 fine-tune seeds
    # hadn't converged at FINETUNE_MAX_STEPS -- and zero-DDM conditions are at least as likely to.
    zero_ddm = which in ("e4", "f1")
    finetune_max_steps = EXTENDED_FINETUNE_MAX_STEPS if zero_ddm else FINETUNE_MAX_STEPS
    finetune_val_every_steps = EXTENDED_FINETUNE_VAL_EVERY_STEPS if zero_ddm else FINETUNE_VAL_EVERY_STEPS
    print(f"=== condition {_D_LABELS[which]} pretrain -> USDJPY fine-tune "
          f"(finetune_max_steps={finetune_max_steps}) ===")
    await run_pretrain_then_finetune(
        dataset_id, seeds, pretrain_seed=42,
        pretrain_max_steps=PRETRAIN_MAX_STEPS, pretrain_val_every_steps=PRETRAIN_VAL_EVERY_STEPS,
        finetune_max_steps=finetune_max_steps, finetune_val_every_steps=finetune_val_every_steps,
    )


async def _run_replicate_pretrain(which: str, pretrain_seed: int, finetune_seeds: list[int]) -> None:
    """Pretrain-seed replication (user-requested robustness check on D1's surprising ~39% loss
    reduction, which so far rests on a SINGLE pretrain seed): re-pretrain the SAME mixture
    composition/dataset with a different pretrain_seed (model weight-init/shuffle seed -- the
    underlying synthetic data itself, e.g. D1's DDM+Sine rows, is unchanged, since
    _build_mixture_data's per-component seeds are fixed constants, not this seed), then
    fine-tune at finetune_seeds. If repeated pretrain seeds all land near the original result,
    the effect is a property of the composition; if they scatter back toward B/C's range, the
    original checkpoint was a lucky draw. The existing pretrain_seed=42 row (D1: TrainingRuns
    1450->1451) is NOT resubmitted here -- reused as-is, per the user's own instruction."""
    dataset_id = _D_DATASET_IDS[which]()
    if dataset_id is None:
        raise SystemExit(f"{which.upper()}_DATASET_ID is not set -- run `{_PREPARE_HINT[which]}` first.")
    print(f"=== {_D_LABELS[which]} pretrain seed={pretrain_seed} replication "
          f"-> USDJPY fine-tune {finetune_seeds} ===")
    await run_pretrain_then_finetune(
        dataset_id, finetune_seeds, pretrain_seed=pretrain_seed,
        pretrain_max_steps=PRETRAIN_MAX_STEPS, pretrain_val_every_steps=PRETRAIN_VAL_EVERY_STEPS,
        finetune_max_steps=FINETUNE_MAX_STEPS, finetune_val_every_steps=FINETUNE_VAL_EVERY_STEPS,
    )


async def _run_colab_smoke() -> None:
    """Minimal REAL-Colab check (not the extracted-cells local proxy already validated) -- one
    small DDM pretrain run via execution_target="colab", to confirm the actual colab-cli/Drive
    round trip works end-to-end (session creation, dataset snapshot export+download, the
    generated notebook's own execution, checkpoint retrieval) before committing to the full run's
    4-job colab sequence. Requires a `colab`-queue Celery worker running."""
    model_id = await _ensure_model()
    hp = _pretrain_hp(seed=42, max_steps=300, val_every_steps=100)
    run_id = await _submit(model_id, DDM_DATASET_ID, hp, execution_target="colab")
    print(f"Submitted colab smoke run {run_id} -- poll its status via the API or DB; "
          f"see docs/colab-workflow.md Step 3 for what to expect.")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"

    if mode == "prepare-data":
        asyncio.run(prepare_datasets())
        return
    if mode == "prepare-mixture-data":
        asyncio.run(prepare_mixture_data())
        return
    if mode == "prepare-ablation-data":
        asyncio.run(prepare_ablation_data())
        return
    if mode == "prepare-dose-response-data":
        asyncio.run(prepare_dose_response_data())
        return
    if mode == "prepare-interaction-data":
        asyncio.run(prepare_interaction_data())
        return
    if mode == "prepare-mechanism-data":
        asyncio.run(prepare_mechanism_data())
        return
    if mode == "prepare-determinism-data":
        asyncio.run(prepare_determinism_data())
        return
    if mode == "prepare-timescale-data":
        asyncio.run(prepare_timescale_data())
        return
    if mode == "prepare-richness-data":
        asyncio.run(prepare_richness_data())
        return
    if mode == "prepare-period-structure-data":
        asyncio.run(prepare_period_structure_data())
        return
    if mode == "prepare-dominant-scale-data":
        asyncio.run(prepare_dominant_scale_data())
        return
    if mode == "prepare-lyapunov-dial-data":
        asyncio.run(prepare_lyapunov_dial_data())
        return

    _require_dataset_ids()

    if mode == "smoke":
        # Single asyncio.run() for the whole run, not one per condition -- database.
        # async_session_factory's pooled connections bind to whichever event loop created them,
        # so a separate asyncio.run() per condition leaves condition B trying to reuse a
        # connection whose loop already closed (same pitfall submit_regime_training.py's
        # submit_many() docstring already documents).
        asyncio.run(_run_smoke())
    elif mode == "colab-smoke":
        asyncio.run(_run_colab_smoke())
    elif mode == "full":
        seeds = [int(s) for s in sys.argv[2:]] if len(sys.argv) > 2 else SEEDS_FULL
        asyncio.run(_run_full(seeds))
    elif mode == "condition-c":
        seeds = [int(s) for s in sys.argv[2:]] if len(sys.argv) > 2 else SEEDS_FULL
        asyncio.run(_run_condition_c(seeds))
    elif mode == "condition-d":
        if len(sys.argv) < 3 or sys.argv[2] not in _D_DATASET_IDS:
            raise SystemExit("usage: condition-d <d1|d2|d3|d4|e1|e2|e3|e4|f1|g1|h1|h2|i1|i2|i3|j1|j2|k1|l1|m1|m2> [seed ...]")
        which = sys.argv[2]
        seeds = [int(s) for s in sys.argv[3:]] if len(sys.argv) > 3 else SEEDS_FULL
        asyncio.run(_run_condition_d(which, seeds))
    elif mode == "replicate-pretrain":
        if len(sys.argv) < 4 or sys.argv[2] not in _D_DATASET_IDS:
            raise SystemExit(
                "usage: replicate-pretrain <d1|d2|d3|d4|e1|e2|e3|e4|f1|g1|h1|h2|i1|i2|i3|j1|j2|k1|l1|m1|m2> "
                "<pretrain_seed> [finetune_seed ...]"
            )
        which = sys.argv[2]
        pretrain_seed = int(sys.argv[3])
        finetune_seeds = [int(s) for s in sys.argv[4:]] if len(sys.argv) > 4 else [42]
        asyncio.run(_run_replicate_pretrain(which, pretrain_seed, finetune_seeds))
    elif mode == "extend-finetune":
        # Re-run fine-tune ONLY (reusing an existing pretrain checkpoint) at
        # EXTENDED_FINETUNE_MAX_STEPS instead of FINETUNE_MAX_STEPS -- see the constant's
        # docstring: used for E3 (and E4) after E3's original 20000-step fine-tune runs showed
        # 2 of 3 seeds still improving, not converged, when the budget ran out.
        if len(sys.argv) < 3:
            raise SystemExit("usage: extend-finetune <checkpoint_path> [seed ...]")
        checkpoint_path = sys.argv[2]
        seeds = [int(s) for s in sys.argv[3:]] if len(sys.argv) > 3 else SEEDS_FULL
        run_ids = asyncio.run(run_finetune_only(
            checkpoint_path, seeds,
            finetune_max_steps=EXTENDED_FINETUNE_MAX_STEPS,
            finetune_val_every_steps=EXTENDED_FINETUNE_VAL_EVERY_STEPS,
        ))
        print(f"Submitted extended-finetune runs: {run_ids}")
    else:
        raise SystemExit(
            f"unknown mode {mode!r}, expected 'prepare-data', 'prepare-mixture-data', "
            f"'prepare-ablation-data', 'prepare-dose-response-data', 'prepare-interaction-data', "
            f"'prepare-mechanism-data', 'prepare-determinism-data', 'prepare-timescale-data', "
            f"'prepare-richness-data', 'prepare-period-structure-data', 'prepare-dominant-scale-data', "
            f"'smoke', 'colab-smoke', "
            f"'full', 'condition-c', 'condition-d <d1|d2|d3|d4|e1|e2|e3|e4|f1|g1|h1|h2|i1|i2|i3|j1|j2|k1|l1|m1|m2>', "
            f"'replicate-pretrain <d1|d2|d3|d4> <pretrain_seed> [finetune_seed ...]', or "
            f"'extend-finetune <checkpoint_path> [seed ...]'"
        )


if __name__ == "__main__":
    main()
