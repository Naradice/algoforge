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

## Phase 6d — per-segment pretraining trajectory, N1 vs N3 (2026-09-26)

`backend/segment_pretrain_trajectory.py` evaluates each pretrain's 20 intermediate checkpoints
(every 2,000 steps) on the pretrain val windows split by source segment (via
`OHLCWindowDataset.window_start_timestamps`), plus the USDJPY frozen vol probe. Raw output:
`backend/segment_pretrain_trajectory.json`.

| | N1 (dt=0.01, transfers) | N3 (dt=0.0125, fails) |
|---|---|---|
| Lorenz segment R² | 0.95–0.99 | 0.965–0.995 |
| DDM segment R² | < 0 until ~18K steps, then 0.12–0.16 | ≤ 0 at every checkpoint (min −0.61) |
| USDJPY vol probe, layer 3 | 0.11 → rises from ~16K → 0.20–0.24 | 0.11–0.15 throughout |
| DDM target variance (z-scored) | ~0.009 | ~0.0055 |

- N3 does not fit Lorenz at DDM's expense — DDM loss never gets worse, it simply never drops
  below predicting the mean. N3 fits Lorenz slightly better than N1.
- In N1 the USDJPY vol feature forms together with DDM learning (probe rises ~16K, DDM R²
  turns positive ~18–20K).
- Working hypothesis: dataset-level z-scoring of the mixture target compresses DDM's `vol_20`
  variance by the synthetic segment's scale (Lorenz vol 13× DDM in N1, 21× in N3). In N3 the
  whole DDM signal (var ≈ 0.0055) is smaller than the Lorenz residual loss (0.01–0.05), so DDM
  is never learned. Learning DDM is necessary but not sufficient — B (DDM only) learns DDM and
  does not transfer.

## Phase 6e — DDM learning across all 26 pretrains (2026-09-26)

`backend/cross_condition_segment_check.py` evaluates every condition's pretrain `best.pt` (the
checkpoint its fine-tunes warm-started from) on its own pretrain val windows, split by segment.
Raw output: `backend/cross_condition_segment_check.json`.

**Whether the pretrain learned the DDM segment separates every condition.**

| Group | DDM-segment R² at end of pretraining |
|---|---|
| Transfers: D1, D2, E1–E3, M1, M2, N1 ×2 seeds, N2 ×2 seeds | 0.050 – 0.637 (min: N1 seed 99) |
| No transfer: B, D3, D4, G1, H1, H2 s99, I1, I2, J1, K1, L1, N3–N5 | −1.231 – 0.029 (max: K1) |
| H2 seed 42 (2/3, ambiguous) | 0.018 |

The margin is thin (0.029 vs 0.050), so this is a strong pattern, not yet a threshold.

- **The Phase 6d compression hypothesis is refuted.** DDM target variance divided by the
  synthetic residual MSE overlaps between groups (N1 0.69 transfers, I2 11.3 fails).
- **Fitting the synthetic segment is not what matters.** The `ar1_forced` family fits its
  synthetic segment almost perfectly (R² 0.96–0.997) and never learns DDM; M1 barely fits its
  own (R² 0.09) and learns DDM (0.21).
- **B (DDM only) does not learn DDM either (R² 0.000).** DDM's `vol_20` is heavy-tailed (shock
  spikes), so after z-scoring most windows sit in a tiny variance band (≈ 0.014) and a DDM-only
  pretrain collapses to predicting the mean. Some synthetic segments get the pretrain past that
  collapse (sine, Delay, slow Lorenz); others do not.

## Phase 6f — B never learns DDM: a normalization artifact (2026-09-26)

B's 20 intermediate checkpoints (`segment_pretrain_trajectory.py B_ddm_only`, raw output
`backend/segment_pretrain_trajectory_B_ddm_only.json`): DDM-segment R² stays in
[−0.076, 0.001] from step 2K to 40K, and the USDJPY vol probe stays at scratch level
(0.095–0.109). The DDM-only pretrain collapses to predicting the mean immediately and never
leaves it.

**Cause: cross-gap returns inflate the target normalizer.** The DDM pretrain data is 48
independent simulation runs concatenated with 1-day gaps; each run starts at a different price,
so `pct_change` across a gap reaches 21%. `vol_20` on the 20 rows after each of the 59 gaps
reaches 4.8e-2 vs a clean maximum of 1.0e-3. `require_contiguous` drops those windows from
training, but `normalize="zscore"` computes its statistics over all rows:

| | Value |
|---|---|
| Rows whose `vol_20` spans a gap | 1,180 of 300,000 (0.4%) |
| `vol_20` std, all rows | 1.027e-3 |
| `vol_20` std, excluding gap-affected rows | 1.221e-4 |
| Inflation | 8.4× std (≈ 70× variance) |

So every real training window's target has z-variance ≈ 1/70 ≈ 0.014 — exactly the DDM variance
measured in Phase 6e (0.0142). The USDJPY fine-tune target is essentially unaffected: 4,203
weekend gaps inflate its `vol_20` std by only 1.02×.

**Implication.** Condition B — the baseline every "does the synthetic segment help" comparison
is made against — is degenerate: it never learns DDM because of this artifact. Every mixture
pretrain shares the same contamination from its DDM runs, and the synthetic segment changes the
normalizer too. Whether a correctly normalized DDM-only pretrain transfers by itself is untested.

## Phase 7a — B′: corrected target normalization is not enough (2026-09-27)

B′ = DDM-only pretrain (run 1611) with `normalize_scope="valid_windows"` (target z-variance of
real windows 0.014 → 1.000); everything else as B. Raw output:
`backend/segment_pretrain_trajectory_Bprime_ddm_only.json`.

- DDM-segment R² stays in [−0.002, 0.002] at all 20 checkpoints; the USDJPY vol probe stays at
  scratch level (0.103–0.116). **Case 3** of the planned decision tree: DDM is still not learned.
- The information is there: a single linear feature — std of the last 19 input differences —
  explains 96% of the target variance in every condition (B′ 0.957; DDM segments of D1/N1/N3/K1
  0.959–0.962).
- **The input side is the remaining bottleneck.** The input (`close`, z-scored over the whole
  dataset) is dominated by the price-level spread across the 48 concatenated simulation runs,
  so DDM's per-bar change is ~0.2% of the input scale:

| Condition / segment | Median per-bar input change (z-units) |
|---|---|
| B′ DDM | 0.0020 |
| D1 / N1 / N3 / K1 DDM | 0.0014–0.0020 |
| Synthetic segments (D1, N1, N3, K1) | 0.023–0.080 |

  Synthetic segments move 10–50× more per bar, which may be what gets a mixture pretrain past
  mean-collapse — but N1 (0.064) and N3 (0.080) are not separated by it.

## Open questions

- What N3 (dt=0.0125) lacks that N1 (dt=0.01) has, for the same attractor — the window-scale
  metrics above don't show it (Phase 6b).
- B′ fine-tunes (runs 1612–1614) — expected near baseline since the pretrain learned nothing.
- Input representation: a pretrain whose input exposes per-bar changes (e.g. z-scored returns
  instead of z-scored price levels). Changes the fine-tune input too, so it needs its own
  from-scratch USDJPY baseline.
- Sine amplitude sweep at period 50 to test amplitude directly (would reinterpret I–L).
- Extended (40K-step) fine-tunes for N1/N2 so magnitudes are converged before comparing them.
