# Research log — DDM + synthetic pretraining → USDJPY volatility transfer

Phases 1–5 (conditions B, D1–M2) are summarized in the handoff memo
(https://claude.ai/artifact/NBvbCFuwMtB5nDdXf1WSfh). This log records work from Phase 6 on.
All runs use `backend/submit_transfer_experiment.py` (`BASE_HP`: decoder-only Transformer,
obs_len 60 → `vol_20`, regime_controlled split, 40K-step pretrain, 20K-step fine-tune) and are
compared with `backend/analyze_transfer_experiment.py`. Lower `best_val_loss` is better; the
basin thresholds are < 0.70 transfer, > 0.78 baseline.

## Phase 6 — Lyapunov dial (2026-09-25)

**Question.** The handoff proposed that transfer requires the synthetic segment's per-bar
Lyapunov exponent to be low: Delay (0.009/bar) transfers, Lorenz (0.017/bar) does not. But
Delay and Lorenz differ in much more than their exponent.

**Design.** Time-rescale each attractor so only the per-bar exponent moves
(`lorenz_dt`, delay `stride`; see `docs/data-layer.md`). Measured by
`backend/characterize_lyapunov_dial.py`:

| Condition | Dataset | Component | Per-bar exponent |
|---|---|---|---|
| N1 | 80 | Lorenz, `lorenz_dt=0.01` | 0.0090 (= Delay) |
| N2 | 81 | Delay, `stride=2` | 0.0209 (> Lorenz) |

Prediction if the exponent is the dial: N1 transfers, N2 does not.

**Result — the prediction fails. Both transfer, under two pretrain seeds.**

| Condition | Pretrain seed | Fine-tune runs | Mean best val_loss | Transfer |
|---|---|---|---|---|
| B baseline (DDM only) | 42 | 1443–1445 | 0.837 | 0/3 |
| H1 Lorenz dt=0.02 | 42 | 1502–1504 | 0.826 | 0/3 |
| **N1 Lorenz dt=0.01** | 42 | 1581–1585 | **0.615** | 5/5 |
| **N1 Lorenz dt=0.01** | 99 | 1593–1595 | **0.577** | 3/3 |
| D2 Delay stride=1 | 42 | 1459–1461 | 0.516 | 3/3 |
| **N2 Delay stride=2** | 42 | 1586–1590 | **0.324** | 5/5 |
| **N2 Delay stride=2** | 99 | 1596–1598 | **0.420** | 3/3 |

- The per-bar Lyapunov exponent is not the dial: N2 diverges faster per bar than Lorenz yet
  transfers.
- Generator identity is not the dial either: the same Lorenz attractor fails at dt=0.02 and
  transfers at dt=0.01.
- N2 beats D2 under both pretrain seeds (0.32 / 0.42 vs 0.52), but D2 has one pretrain seed
  only, and the pretrain-seed spread for N2 (0.10) is large — treat "stride=2 is better" as
  suggestive.
- Budget caveat: in about half of these fine-tunes the best checkpoint is at step 19–20K, so
  losses are likely still falling; transfer/no-transfer is unaffected.

**Model-free checks that also weaken the handoff's mechanism** (no training involved):

- kNN R² for `vol_20` 20 bars past the window is ≥ 0.97 for Lorenz dt=0.02 and 1.0 for LFSR,
  both non-transferring. "Future volatility is predictable from the window" does not separate
  the groups as stated.
- Rosenstein's estimator reads 0.009/bar for a clean sine (period 15), so exponents near 0.01
  are not reliably distinguishable from zero with it.
- Synthetic/DDM median `vol_20` ratio: transferring 0.37–26×, non-transferring 7.9–130×. Scale
  overlaps between groups, so it is at most a partial confound.

## Phase 6b/6c — window structure, Lorenz dt sweep, probes (2026-09-26)

**Window structure (model-free, `backend/characterize_window_structure.py`).** Across Lorenz
dt 0.005–0.02, return smoothness, turns per window, lobe crossings per window and `vol_20`
scale/variability all move monotonically with dt — no single metric breaks between 0.01 and
0.02. Separately: the single-period `ar1_forced` conditions I1/I2 are exactly pure sines
(sine-fit R² = 1.000000) with amplitude 11 (period 70) and 20.5 (period 140). They fail 0/5,
while amplitude-1 sines transfer at periods 15/50/200 — so the handoff's "5 phases can't be
resolved from one window" explanation for the `ar1_forced` family is doubtful. Amplitude alone
does not explain it either: I2 (amplitude 20, vol 14.7×) fails, N1 (amplitude ~20, vol 13.3×)
transfers.

**Lorenz dt sweep — transfer switches off between dt 0.01 and 0.0125.** Pretrain seed 42,
fine-tune seeds 42–44. Best checkpoints at steps 5K–14K, so not budget-limited.

| Condition | `lorenz_dt` | Fine-tune runs | Mean best val_loss | Transfer |
|---|---|---|---|---|
| N1 | 0.01 | 1581–1585, 1593–1595 | 0.615 / 0.577 | 8/8 |
| N3 | 0.0125 | 1602–1604 | 0.825 | 0/3 |
| N4 | 0.015 | 1605–1607 | 0.827 | 0/3 |
| N5 | 0.0175 | 1608–1610 | 0.824 | 0/3 |
| H1 | 0.02 | 1502–1504 | 0.826 | 0/3 |

**Frozen probes (`backend/probe_representations.py`) predict transfer.** Layer-3 Ridge R² for
USDJPY future `vol_20` on each pretrain checkpoint, before any fine-tuning:

| Checkpoint | Layer-3 vol R² | Fine-tune |
|---|---|---|
| ddm_only / xor / lfsr | 0.11–0.13 | no transfer |
| H1 Lorenz dt=0.02 | 0.130 | no transfer |
| N3 Lorenz dt=0.0125 | 0.137 | no transfer |
| M1 Sine p15 | 0.200 | 0.599 |
| N1 Lorenz dt=0.01 | 0.218 | 0.615 |
| D2 Delay | 0.386 | 0.516 |
| M2 Sine p200 | 0.405 | 0.535 |
| D1 Sine p50 | 0.483 | 0.507 |
| N2 Delay stride=2 | 0.494 | 0.324 |

There is a clean gap between 0.137 and 0.200, and the order almost matches the fine-tune
losses. Whether transfer happens is decided during pretraining — by whether the volatility
feature forms — so the open question moves to the pretrain dynamics.

## Open questions

- What H1 (dt=0.02) lacks that N1 (dt=0.01) has, for the same attractor. Candidates to measure
  per bar: return autocorrelation, curvature/smoothness at the 60-bar window scale, fraction of
  windows containing a lobe switch.
- Why the N3 pretrain doesn't form the volatility feature when N1's does: per-segment pretrain
  loss (DDM part vs Lorenz part) over training.
- Sine amplitude sweep at period 50 to test amplitude directly (would reinterpret I–L).
- Extended (40K-step) fine-tunes for N1/N2 so magnitudes are converged before comparing them.
