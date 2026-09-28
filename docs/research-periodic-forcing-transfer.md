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

## Phase 7b — returns input: the pretraining question dissolves (2026-09-27)

Input switched to per-dataset z-scored log returns (`src_normalize="returns_zscore"`,
`normalize_scope="valid_windows"`) for both pretrain and fine-tune. DDM and USDJPY differ 1.6× in
raw return std but have near-identical mean_vol/σ_r (0.81 vs 0.79) and sd_vol/σ_r (0.60 vs 0.61),
so per-dataset z-scoring aligns the input→target map (tails still differ: kurtosis 41 vs 278).

| Run(s) | What | Result |
|---|---|---|
| 1615 | B′-C: DDM-only pretrain, returns input | DDM R² 0.94 at 2K steps, 0.95–0.96 thereafter; USDJPY layer-3 vol probe 0.98–0.99 |
| — | Untrained model, returns input | layer-3 probe 0.34 |
| — | Hand feature: std of last 19 input returns, linear | R² 0.994 on USDJPY |
| 1616–1618 | A″: USDJPY from scratch, returns input | best val_loss 0.0170 ± 0.0021 (R² 0.98) |
| 1619–1621 | B′-C fine-tunes | crashed (OOM, 2026-09-27 21:40; marked error), not re-run |

For comparison, the old pipeline's USDJPY fine-tunes sit at 0.84 (R² 0.16) for B and 0.32–0.60
(R² 0.4–0.68) for the best transferring conditions.

- With returns input the task is nearly trivial: `vol_20` at the next bar shares 19 of its 20
  returns with the input window, and a single hand-made feature explains 99.4% of it. A from-scratch
  model reaches R² 0.98 with no pretraining at all.
- **What Phases 1–6 measured was how much each pretrain helped the model overcome the level-input
  representation** (z-scored prices, in which per-bar changes are ~0.2% of the input scale) — a real,
  reproducible effect, but not transfer of volatility forecasting from synthetic to real data.
- B′-C fine-tunes could at most close part of the 0.017 gap to zero, so they were not re-run.
- Consequence for future work: a transfer question needs a target that is not computable from the
  window — e.g. realized volatility over a horizon that does not overlap the input.

## Phase 8a — difficulty of a future-only volatility target (2026-09-27)

Target y_t = std(r_{t+1} … r_{t+20}) of log returns after a 60-return input window (no overlap);
only windows whose whole 80-return span is gap-free. `backend/future_vol_task_difficulty.py`;
raw output `backend/future_vol_task_difficulty.json` (USDJPY + DDM, chronological/random) and
`backend/future_vol_task_difficulty_usdjpy.json` (USDJPY, adds the blocked split and |r| k-NN).

Splits: chronological = last 20% as test with an 80-window purge; random = random 80/20 windows
(like `regime_controlled`); blocked = 1-day blocks assigned randomly 80/20, purged at each
boundary (same regime mix as random, no adjacent-window sharing).

**USDJPY, 926K windows — R² of log y**

| Predictor | Chronological | Random | Blocked |
|---|---|---|---|
| mean | −0.303 | −0.098 | −0.120 |
| time of day | −0.135 | 0.131 | 0.081 |
| persistence (std of last 20 returns, no fitting) | 0.465 | 0.552 | 0.595 |
| persistence, linear fit | 0.520 | 0.572 | 0.592 |
| HAR (log std over last 5/20/60, linear) | **0.558** | **0.644** | **0.676** |
| k-NN on raw z-returns | −1.579 | −0.563 | −0.487 |
| k-NN on \|z-returns\| | 0.228 | 0.502 | 0.540 |
| MLP 2×64 on z-returns | 0.486 | 0.621 | 0.644 |

**DDM, 295K windows:** every predictor is at R² ≈ 0 (best: persistence fit 0.003 level, HAR
0.017 log; unfitted persistence −0.83 log). The DDM v3_shock pretrain data has no predictable
future volatility at this horizon.

- USDJPY future volatility is predictable (log R² 0.56–0.68) but persistence-dominated: an
  unfitted predictor gets 0.47–0.60, HAR adds ~0.08, and neither k-NN nor the MLP beats HAR.
  Headroom for "extracting state from 60 bars" beyond HAR is unmeasured but not obviously large.
- Split leakage is small for these predictors (blocked ≥ random). The random-vs-chronological gap
  is regime shift: the chronological test period (2022) has much higher volatility, so even the
  mean scores −0.30. A high-capacity model could still exploit adjacency, so blocked is the safe
  choice for training runs.
- Level-scale R² is unstable under heavy tails (MLP chronological level R² −2155 vs log 0.486);
  the target should be log volatility.
- DDM cannot teach this skill as-is: a DDM-only pretrain on this target would be fitting noise.

## Phase 8b — DDM variant screen: no DDM variant has predictable future volatility (2026-09-28)

`backend/ddm_variant_vol_screen.py`; raw output `backend/ddm_variant_vol_screen.json`. Each
variant: 12 independent runs × 5,000 candles (the pretrain data's layout), split by run 9/3;
windows never cross runs. USDJPY: last 1M rows split into gap-free segments, chronological 3/4
by segment (so its numbers differ from Phase 8a's blocked split). Target: log std of the next 20
returns after a 60-return window.

| Source | Persistence R² (log) | HAR R² (log) | HAR coef 5/20/60 | \|r\| ACF lag 1/5/20/60/240 | log-vol block ACF lag 1/3/12 | Kurtosis |
|---|---|---|---|---|---|---|
| USDJPY | 0.434 | 0.523 | 0.07 / 0.24 / 0.54 | 0.34 / 0.29 / 0.26 / 0.22 / 0.15 | 0.76 / 0.66 / 0.40 | 278 |
| v3_shock (current pretrain) | −0.950 | −0.007 | 0.01 / 0.04 / 0.10 | 0.21 / 0.00 / 0.01 / 0.01 / −0.01 | 0.07 / 0.07 / 0.04 | 37 |
| v3, no shock | −1.009 | −0.086 | 0.00 / −0.01 / 0.52 | 0.06 / 0.01 / 0.01 / 0.02 / 0.02 | 0.21 / 0.21 / 0.21 | 1 |
| decayed shock, τ=300 | −0.563 | −0.170 | 0.00 / −0.02 / 0.15 | 0.05 / 0.01 / 0.01 / 0.01 / 0.01 | 0.14 / 0.15 / 0.17 | 3 |
| spread feedback a=0.25 | −0.747 | −0.035 | 0.02 / 0.05 / 0.13 | 0.22 / 0.00 / 0.02 / 0.00 / 0.02 | 0.13 / 0.07 / 0.11 | 44 |
| loss limit 1.404 | −0.628 | 0.022 | 0.02 / 0.04 / 0.08 | 0.20 / 0.01 / 0.02 / 0.00 / 0.00 | 0.12 / 0.06 / 0.11 | 45 |

- No DDM variant has volatility clustering: the |r| ACF is ~0 from lag 5 on in every variant
  (first lag below 0.05: 2), while USDJPY's is still 0.15 at lag 240 (long memory).
- The flat log-vol ACF of "no shock" / "decayed shock" (≈0.21 / 0.15 at every lag) is a
  between-run level difference, not within-run clustering — HAR cannot use it (R² < 0).
- Consequence: a B‴ (DDM → USDJPY on the future-volatility target) with any existing DDM variant
  would pretrain on a target with no learnable signal. Testing whether "synthetic volatility
  dynamics transfer" needs a synthetic source that has them.

## Phase 8c — A‴: a scratch Transformer does not beat a strong linear baseline (2026-09-28)

A‴ = USDJPY from scratch on the future-only target (runs 1622–1624; `future-vol-scratch` mode:
returns input, `future_log_vol_20`, blocked split, 20K-step budget, early stopping after 5 checks
without improvement — stopped at 13 / 11 / 19 checks). All numbers are MSE of the z-scored log
volatility on A‴'s own blocked validation split (166,960 windows; variance 1.042).

| Model | Val MSE | R² |
|---|---|---|
| Persistence, linear fit | 0.386 | 0.629 |
| Linear on the 60 log\|r\| values (ridge) | 0.380 | 0.636 |
| HAR, std 5/20/60 (the bar used when A‴ was launched) | 0.328 | 0.685 |
| HAR, RMS 5/20/60 | 0.321 | 0.692 |
| HAR, RMS at 1/2/5/10/20/40/60 | 0.320 | 0.693 |
| RMS HAR 1–60 + 60 log\|r\| + signed return sums (ridge) | 0.3175 | 0.695 |
| A‴, best of its validation checks (selected on this set) | 0.3155 (0.312–0.318) | 0.697 |
| A‴, final checkpoint | 0.324 (0.318–0.334) | 0.689 |
| A‴, mean of its last 5 checks | 0.324 | 0.689 |

- The first reading ("A‴ beats HAR by ~5× the seed spread") came from two biases: the HAR
  bar used std instead of RMS (0.328 vs 0.321), and A‴'s number was a best-of-N selection on the
  same validation set, while check-to-check noise is about ±0.01.
- A‴ performs on par with a well-specified linear volatility-memory model. There is no evidence
  the Transformer extracts state from the 60 bars beyond what linear volatility memory gives.

## Conclusion of Phases 6–8 (2026-09-28)

1. The Phase 1–5 "transfer" was recovery from an input representation problem, not transfer of
   volatility forecasting: the price-level input hid per-bar changes (~0.2% of its range), and the
   target (`vol_20` at the next bar) shared 19 of its 20 returns with the input window.
2. The DDM pretrain data never taught DDM itself: cross-gap rows inflated the target normalizer
   8.4×, and even with that fixed the level input kept DDM unlearnable. With a returns input DDM is
   learned immediately.
3. On a future-only target (log RV of the next 20 bars) USDJPY is predictable (R² ≈ 0.69), but
   almost entirely through volatility persistence: a scratch Transformer matches a strong linear
   baseline and leaves no headroom that pretraining could improve.
4. No DDMv3 variant has volatility clustering (|r| ACF ≈ 0 from lag 5), so none can serve as a
   pretraining source for future-volatility forecasting.

Next (agreed 2026-09-28): search USDJPY for a target where nonlinear models clearly beat a strong
linear baseline (model-free screen); only then test pretraining as sample efficiency on a small
USDJPY fraction, and only then choose a synthetic source matched to that target.

## Phase 9a — screen for nonlinear headroom on USDJPY (2026-09-28)

`backend/nonlinear_headroom_screen.py`; raw output `backend/nonlinear_headroom_screen.json`. Input:
the 60 returns before t. Common window set: rows t−60 … t+240 gap-free (784,571 windows); whole
days split 80/20 with a 300-bar purge; 300K train / 90.5K test windows (139 test days). CIs: 95%,
paired bootstrap over test days (B = 500) of the absolute loss gain vs the linear baseline.
Linear baseline = ridge on RMS at 1/2/5/10/20/40/60, the 60 log|r| values and signed sums over
5/20/60 (Phase 8c's strong baseline).

**Screen (no time-of-day features), R²**

| Target | Persistence | Linear | HistGB | MLP | k-NN | HistGB gain vs linear, MSE [95% CI abs] |
|---|---|---|---|---|---|---|
| rv20 = log RMS next 20 | 0.544 | 0.654 | 0.661 | 0.645 | 0.645 | +2.0% [+0.0010, +0.0029] |
| rv60 | 0.473 | 0.651 | 0.665 | 0.640 | 0.642 | +4.0% [+0.0016, +0.0055] |
| rv240 | −0.055 | 0.443 | 0.496 | 0.447 | 0.443 | +9.6% [+0.0049, +0.0180] |
| dvol20 = log RMS next 20 − last 20 | −0.001 | 0.240 | 0.255 | 0.236 | 0.172 | +1.9% [+0.0010, +0.0028] |
| jump20 = max\|r\| next 20 > 4 × RMS last 60 (8.3%); log loss / AUC | 0.287 / 0.50 | 0.275 / 0.661 | 0.273 / 0.675 | 0.279 / 0.648 | 0.281 / 0.623 | +0.6% [+0.0001, +0.0034] |

**Follow-up: what the horizon trend was.** Adding time-of-day (4 harmonics) and weekday to the
linear model lifts rv240 from R² 0.443 to 0.674 and rv60 from 0.651 to 0.721 — the growing HistGB
gain with horizon was mostly the model reconstructing the clock (session seasonality) from the
recent volatility profile. With time-of-day given to both sides:

| Target | Linear + ToD + squares | + hour × RMS(5/20/60) | HistGB + ToD | MLP + ToD | HistGB gain vs the interaction linear [95% CI abs] |
|---|---|---|---|---|---|
| rv20 | 0.684 | 0.688 | 0.699 | 0.664 | +3.7% [+0.0020, +0.0048] |
| rv60 | 0.726 | 0.733 | 0.742 | 0.711 | +3.2% [+0.0007, +0.0037] |
| rv240 | 0.687 | 0.691 | 0.704 | 0.666 | +4.5% [+0.0005, +0.0049] |

- Beyond persistence, the dominant structure is intraday seasonality — an explicit feature (the
  clock), not something to learn from the 60 bars.
- With the clock given, a tree model keeps a small but significant 3–4.5% MSE edge over a linear
  model with explicit hour × volatility interactions. The MLP is ~6% *worse* than linear at every
  horizon, and k-NN never helps, so no smooth nonlinear model shows headroom here.
- No target clears the bar "a nonlinear model beats the strong linear baseline clearly" in a way a
  neural sequence model could plausibly exploit; the only surviving gain is small and tree-specific.

## Phase 9b — learning curves: the linear model saturates on little data (2026-09-28)

`backend/learning_curve_screen.py`; raw output `backend/learning_curve_screen.json`. Same windows,
features and test sample as Phase 9a; all models get time-of-day/weekday. Training subsets are
random whole training days (3 draws per fraction; full pool 555 days, capped at 300K windows).
R² on the fixed test sample, mean over draws:

| Training days | rv20 linear | rv20 HistGB | rv20 MLP | rv60 linear | rv60 HistGB | rv60 MLP |
|---|---|---|---|---|---|---|
| 1% (6) | 0.406 | 0.389 | −11.47 | 0.254 | 0.352 | −13.31 |
| 2% (11) | 0.597 | 0.547 | −2.09 | 0.658 | 0.542 | −1.96 |
| 5% (28) | 0.633 | 0.632 | 0.196 | 0.694 | 0.644 | 0.226 |
| 10% (56) | 0.673 | 0.668 | 0.495 | 0.719 | 0.692 | 0.510 |
| 25% (139) | 0.685 | 0.690 | 0.623 | 0.728 | 0.722 | 0.573 |
| 100% (555) | 0.688 | 0.700 | 0.667 | 0.733 | 0.742 | 0.708 |

- The strong linear baseline is within 0.015 of its full-data R² at 10% of the days and within
  0.005 at 25%; at 5% (28 days) it already reaches 0.63 / 0.69.
- The MLP is far more data-hungry: 0.2–0.5 at 5–10%, diverging (R² −2 to −13) at 1–2%, and still
  below linear with all the data.
- A small-data regime where a neural net trails the linear model badly does exist, so "does
  pretraining improve a neural net's sample efficiency" is a well-posed question. But the most
  pretraining could deliver is catching a neural net up to a linear model that needs no
  pretraining and saturates on about two months of data — no practical gain on these targets.

Decision (2026-09-28): Option 3 is not pursued for practical value on USDJPY volatility targets.
It stays open only as a scientific question (synthetic pretraining vs. neural-net sample
efficiency), to be prioritized separately.

## Open questions

- What N3 (dt=0.0125) lacks that N1 (dt=0.01) has, for the same attractor — the window-scale
  metrics above don't show it (Phase 6b).
- B′ fine-tunes (runs 1612–1614): mean best val_loss 0.824, 0/3 transfer (B: 0.837, 0/3) — as expected from a pretrain that learned nothing.
- Input representation: a pretrain whose input exposes per-bar changes (e.g. z-scored returns
  instead of z-scored price levels). Changes the fine-tune input too, so it needs its own
  from-scratch USDJPY baseline.
- Sine amplitude sweep at period 50 to test amplitude directly (would reinterpret I–L).
- Extended (40K-step) fine-tunes for N1/N2 so magnitudes are converged before comparing them.
