# T1: effect of the trade-interval clock (M2-S2) on fixed-time OHLC and real-market comparison

Study Manager: #895 (plan v0.4, approved 2026-11-17 with the conditions folded into the plan), goal #886 / tree #874.
Scripts: `backend/research/t1_clock_ohlc.py` (calibration, G1, Phase 2, Phase 4 inputs), `t1_phase3_real.py` (Phase 3).
The specification (constants, thresholds, definitions) was committed before the first result was computed.

## Question

Keep the DDMv3 (v3_shock) **price events fixed** and assign different clocks to the same events. What changes in fixed-time
OHLC statistics? The clock acts only after the fact (it does not change agent behaviour).

| id | clock |
|---|---|
| C0 | repository clock as-is (`tick_time`, i.i.d. noise per step) |
| C0s | C0 multiplied by a fixed constant so that its mean interval is 3.0 s (scale-only diagnostic) |
| C1 | M2-S2 (n = 16, lam = 0.1), independent of the dynamics, mean interval 3.0 s |
| C2 | i.i.d. exponential, mean 3.0 s |
| C3 | C1 with the post-burn-in intervals permuted (20 shuffles per price stream, averaged inside the stream) |

Design: 20 independent price-event streams (DDMv3 v3_shock, N = 500, 100 000 trades each, seeds 1-20, no failed seed);
clock seed = price seed + 1000; 60 s bars; first 500 events excluded for every condition; continuous grid (empty bars
forward-filled, zero return) is the main evaluation, `dropna` is reported for compatibility. **The independent unit of every
confidence interval is the price seed** (percentile bootstrap, 10 000 draws); the 20 shuffles of C3 are conditional
repetitions and are not counted as runs.

## Calibration (fixed before the evaluation seeds were generated)

* F-pool from Model-1 (N = 100, c = U[-0.01, 0.01], dt = 0.01, 3 x 20 000 trades), `Var(log F) = 0.984`, centred to `E[log F] = 0`.
* `kappa = E[T]/m0 = 1.886` (20 chains x 400 000 events) -> `m0 = 1.591` for a 3.0 s mean.
* C0 mean interval **0.0245 s** (about 2 450 trades per minute) -> scale constant for C0s = 122.7.
* Transient diagnostic (4 000 chains started at log m0 +-2): bias at event 500 = -0.007 / -0.015 (SE 0.011), tolerance 0.05 -> pass.

## Gate G1 (clock implementation and model properties only; no real-market fit)

5-run pilot and 20-run full set both passed: all intervals positive and times monotonic, C1 and C3 interval multisets identical
in every shuffle, per-run ACF(log T, 1) of C1 in 0.341-0.379 (limits 0.20-0.50) and CV 1.29-1.47 (limits 0.9-1.8),
quarter-wise stationarity within the pre-set tolerances in 20/20 runs, event-count candles identical across clocks in 20/20 runs.
C1 mean count per M1 bar is 20.0, i.e. the research convention of 20 trades per minute is reproduced on average.

## Phase 2: fixed-time OHLC (continuous grid, 20 price seeds, mean and 95% CI over seeds)

| statistic | C0 | C0s | C1 | C2 | C3 | C1 - C3 | C1 - C2 |
|---|---|---|---|---|---|---|---|
| ACF(log T, 1) (operational check, limit >= 0.20) | -0.035 | -0.035 | 0.360 | -0.001 | 0.000 | **+0.360** [0.355, 0.365] | +0.362 |
| M1 trade-count VMR | 1.13 | 1.32 | 12.57 | 1.00 | 1.71 | **+10.9** [10.6, 11.1] (relative +638%, [624%, 651%]) | +11.6 |
| M1 count ACF(1) | -0.036 | -0.012 | 0.704 | 0.000 | 0.030 | +0.674 [0.668, 0.679] | +0.704 |
| M1 |return| ACF(1) | 0.196 | 0.216 | 0.234 | 0.201 | 0.205 | **+0.029** [0.015, 0.043] | +0.033 [0.014, 0.053] |
| M1 |return| ACF mean of lags 1-10 | 0.082 | 0.022 | 0.038 | 0.020 | 0.022 | +0.016 [0.013, 0.020] | +0.018 |
| M1 return variance | 4.5e-6 | 4.27e-8 | 4.18e-8 | 4.20e-8 | 4.19e-8 | -0.3% | -0.5% |
| M1 return excess kurtosis | -0.23 | 43.8 | 39.2 | 38.2 | 39.2 | -0.02 [-2.5, 2.4] | +1.0 [-2.9, 4.4] |
| mean H-L range | 0.447 | 0.0232 | 0.0220 | 0.0234 | 0.0232 | -0.0012 (-5%) [-0.0013, -0.0011] | -0.0013 |
| empty-bar rate | 0 | 0 | 0.10% | 0 | 0.09% | 0.0003% | 0.10% |
| M1 bars per run | 39 | ~4 900 | ~4 900 | ~4 900 | ~4 900 | | |

* **Event-count candles (20 trades = 1 candle) are bit-identical for all clocks (20/20 runs)**, as required.
* The |return| ACF(1) difference C1 - C3 (+0.029, CI [0.015, 0.043]) is reported as an estimate; the 0.03 line is a
  reference only, not a decision threshold. Per price seed it ranges from -0.031 to +0.091 (positive in 15 of 20 runs).
* The VMR gap exceeds the 10% exploratory margin by a large factor. It is not to be read as an improvement of the market model.
* C3 has VMR 1.71 although it has no temporal dependence: the heavy-tailed marginal alone produces some over-dispersion, so
  about 6% of the C1 excess over the Poisson value (12.57 vs 1.00) is marginal, and the rest comes from the ordering.
* C0 as-is gives only ~39 bars per 100 000 trades (2 450 trades per bar). Its ACF-type numbers rest on 39 bars per run and are
  not reliable; C0 vs C1 mixes scale, marginal distribution and dependence, so only C0s and C2/C3 isolate single factors.
  The `MIN_BARS` guard was lowered from 200 to 30 for this reason after the pilot, before any OHLC result had been seen.
* `dropna` and the continuous grid differ by only ~5 dropped bars per run (0.1%) here, so the compatibility result equals
  the main result at 20 trades per bar. At 2 450 trades per bar (C0) empty bars do not occur.

## Phase 3: real quote updates (not executions)

MT5 USDJPY bid/ask updates, broker time. The tick file has **no data from 2021-11-29 to 2022-02-28** (except part of the
last day), so the planned periods were moved, before any real statistic was computed, to selection 2021-06-01..08-31 and
evaluation 2021-09-01..11-26 (finding #914). Window chosen on the selection period: broker hours 14-19 (about 6 700 updates
per hour). 63 evaluation days (one day, 2021-09-01, was inspected once while testing the code, after the period change).

| statistic | real (63 days) | C0s | C1 | C2 | C3 | C1 closer than C0s and C2 (CI excludes 0)? |
|---|---|---|---|---|---|---|
| ACF(log dt, 1) | 0.276 | -0.035 | 0.360 | -0.001 | 0.000 | yes |
| CV(dt) | 2.47 | 1.18 | 1.34 | 1.00 | 1.34 | yes |
| ACF(updates per M1 bar, 1) | 0.616 | -0.012 | 0.704 | 0.000 | 0.030 | yes |
| ACF(|M1 return|, 1) | 0.121 | 0.216 | 0.234 | 0.201 | 0.205 | **no** (C1 farther; CI includes 0) |

By the pre-registered rule (>= 3 of 4 statistics, both distance differences with CI excluding 0) the label is **"C1 improved"**.
This must be read with three limits:

1. Three of the four statistics measure interval/activity clustering, the very property C0s and C2 lack by construction;
   they are not independent evidence of a better market model. The only non-clock statistic, |return| ACF(1), is not improved
   (all model conditions are above the real 0.12, C1 the highest).
2. The real quote updates are more dispersed (CV 2.47) than any model clock and C1 over-clusters in ACF(log dt) (0.36 vs 0.28)
   and in count ACF (0.70 vs 0.62). The match is qualitative.
3. The real data are quote updates, the model has executions; the parameters of M2-S2 were chosen in T0 and not re-selected here.

## Phase 4: impact on earlier work

| work | time base | clock dependence | class |
|---|---|---|---|
| #874 Phase 1-3 (ml_model 115, dataset 48), transfer experiments (datasets 50/52, mixtures 54-84), `ddm_step4-6` CRPS/Atlas/ATR, `ddm_ensemble_backtest` | 20 trades = 1 candle | candles identical for every clock (checked on 20 streams) | **A** |
| price-level drift / non-stationarity of DDM at scale (#93-95) | price path | the clock does not enter the price dynamics | **A** (by construction) |
| Goal #57 line on `collect()` datasets 23-28, 34-47: data-volume sweeps (#58/#59), tasks A/B/C (#65), token comparison (#68) | `tick_time` M1 bars (~2 450 trades per bar under C0) | with C1-type clustering, activity concentration changes strongly (VMR, count ACF) and |return| ACF(1) by about +0.03 (CI excludes 0 versus C3, includes 0 versus C0s) | **B for activity-related statistics, C (limited) for volatility-prediction tasks (#65 tasks B/C)**; re-evaluate only these, on clustered-clock data, if the question is whether the return-to-volatility structure depends on activity clustering |
| coupled clock and price generation | - | not tested (T1-M out of scope) | D: not evaluated |

Caveat: T1 used v3_shock, N = 500; datasets 23-28 and 34-47 are plain v3, N = 300. The classification is an extrapolation
until the same comparison is run on those settings.

## Go / No-Go for F1

**Go.** The current DDMv3 can serve as the F1 baseline: count-based work is unaffected. Fixed-time work must state the time
convention (3 s per trade, 20 trades per minute) and must not use the repository clock as-is (2 450 trades per bar).
The clustering clock is an optional, separately labelled model (M2-S1/S2), not a correction.

## Limits and amendments

* M2-S2 (n = 16, lam = 0.1) is the exploratory T0 setting; no parameters were tuned on these results.
* Phase 3 period change (data gap) and `MIN_BARS` 200 -> 30 are the only deviations from plan v0.4; both were made before the
  corresponding results were computed.
* T1 measures post-hoc time assignment, not the effect of timing on agent behaviour.
* Probabilistic-forecast performance (CRPS, Atlas) was not re-run; the matrix above says where it would matter.
