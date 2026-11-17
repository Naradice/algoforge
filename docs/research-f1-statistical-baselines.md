# F1: statistical probabilistic-forecast baselines versus the frozen DDM (Study Manager #889)

Plan v0.3 = approved v0.2 + the three pre-implementation notes of the 2026-11-17 review (simultaneous confidence
bands, HAR-RV generation procedure, leak test with frozen learned state). This file is the specification of record and was
committed **before** any validation or final-period score was computed. Results are appended in a separate section.

## Question

How do the statistical baselines (unconditional and ATR-conditional empirical distributions, Gaussian, Student-t,
GARCH-family, HAR-RV) and the DDM (§11 decayed-shock ensemble) compare as predictive distributions of the cumulative
log return over h = 1, 4, 12, 24 windows (1 window = 60 M1 candles), under identical information and scoring?

The earlier DDM CRPS gains (#881) were DDM-versus-DDM. A look-ahead in the DDM bucket-to-candidate mapping was found by
code reading (finding #915): the mapping used the realised 1000-candle forward statistics of the **test** windows. F1 rebuilds
the mapping from estimation-period windows only.

## Data, windows, periods

* Series: `datasets/derived/usdjpy_1min_volatility.parquet` (USDJPY M1 close, 2 239 284 rows, 2016-08-31..2022-08-31).
* Block j = rows [60j, 60j+59]; origin row t_j = 60j+59. Information at the origin: closes up to and including row t_j.
* Target R_h = log(C[t_j+60h] / C[t_j]). Observation window for ATR: the 60 closes of block j.
* **Gap rule (amended before any result, for feasibility):** a step between consecutive rows longer than 5 minutes is a market
  gap (weekend, holiday, outage). Missing quiet minutes (<= 5 min) are not gaps, because rows are "candles" as in all earlier
  work. A strict 1-minute rule leaves only ~550 validation origins at h = 24. Origins whose observation block or target
  window contains a gap are dropped. Valid origins (validation / final): h = 1: 6 079 / 6 125; h = 4: 5 881 / 5 945;
  h = 12: 5 353 / 5 465; h = 24: 4 571 / 4 752.
* Periods (origin time and target end both inside): estimation 2016-08-31..2020-08-31, validation 2020-09-01..2021-08-31,
  final 2021-09-01..2022-08-31 (**partially known** holdout, see the audit below; used once, never for selection).
  Supplementary 2022-09-01..2022-12-30 (HistData) only if the two sources align (checked and reported).

## Information available at origin t

| model | uses at t | learned state (monthly, from data strictly before the month start) |
|---|---|---|
| B0 empirical | nothing | empirical distribution of completed targets (target end before month start) |
| B1 Gaussian, B2 Student-t | nothing | variance (and nu) of completed targets |
| B3-B6 GARCH(1,1)-N, -t, GJR-t, EGARCH-t | block returns up to t | MLE parameters on block returns before the month start |
| B4s GARCH-t with hour-of-day scaling (sensitivity) | block returns up to t, hour of day | hour-of-day factors and parameters from data before the month start |
| B7 HAR-RV (N and t innovations) | block RV up to t | OLS coefficients, residual pool, innovation law from data before the month start |
| B8 ATR-bucket empirical (control) | ATR of the 60 closes of block j | bucket edges (20 quantiles) from estimation origins; bucket distribution of completed targets |
| DDM | ATR of block j -> bucket | edges and bucket->candidate mapping from estimation windows only; fixed ensembles |

B0/B1/B2/B8/B3-B7 are re-estimated monthly with an expanding window; the DDM mapping is frozen on the estimation period (a
stated asymmetry). Location is zero for all models.

## Statistical model definitions

* Block return r_j = log(C[t_j]/C[t_{j-1}]) when both blocks are gap-free; the GARCH recursion runs over the valid block
  returns in time order (state is carried across gaps, not reset). Fit with `arch` (pinned 8.0.0), zero mean, returns scaled by
  100: GARCH(1,1), GJR-GARCH(1,1,1), EGARCH(1,1,1); normal or standardised Student-t (variance 1, nu >= 2.1).
* Multi-step distribution: for each origin a fresh simulation of 1 000 independent paths of 24 steps from the filtered
  variance at t, updating the conditional variance with each simulated shock; cumulative sums at h = 1, 4, 12, 24.
* **HAR-RV (note 2).** RV_j = sum of the 60 squared M1 returns of block j. Features: log RV_j, log of the mean RV over the
  last 24 blocks, log of the mean RV over the last 120 blocks (available blocks only). Regression of log RV_{j+1} on them
  (OLS). Two **different** distributions are used and are not to be confused:
  (a) the log-RV residual law: the empirical (bootstrap) pool of the in-sample OLS residuals, mean zero, drives the simulated
  log-RV path (the recursion uses simulated RV in the 24/120 averages); (b) the return innovation law: zeta_j = z_j/std(z), with
  z_j = r_j / sqrt(RV_j) observed in the fit sample, standard normal (HAR-N) or standardised Student-t (HAR-t, nu by MLE).
  Simulated return r_s = std(z) * sqrt(RV_s) * zeta_s with (a) and (b) independent. The simulated RV is a realised variance,
  not a conditional variance, so no exp(sigma^2/2) correction is applied: the bootstrap handles it.
* B1: N(0, v_h), v_h = mean of R_h^2 over completed targets. B2: Student-t scaled to variance v_h, nu_h by MLE
  (standardised, nu >= 2.1). B0: bootstrap draws of completed R_h. B8: bootstrap draws of the completed R_h in the origin's
  ATR bucket. All random draws use a seed derived from (model, origin).
* Hour-of-day sensitivity B4s: returns divided by the hour-of-day standard deviation (hour of the block end) estimated on
  data before the month start; simulated step s is multiplied by the factor of hour (h_origin + s) mod 24.
* Not run (declared): M1-based GARCH versions (B3m, B4m); they are a separate additional comparison.

## DDM

* Frozen: atlas `ddm_regime_atlas.csv`, `select_ensemble` (std-first shortlist, TOP_K = 8), 20 quantiles, §11 decayed shock
  (P0 = 0.002, S = 0.3, tau = 300, floor = 0), "20 trades = 1 candle", simulate_long_path (N = 300).
* Rebuilt: bucket edges = 19 interior quantiles of the block ATR over estimation origins; for each bucket up to 300 random
  estimation windows (fixed seed) whose forward 1 000-candle statistics (same `full_characterize` of closes[60j : 60j+1001])
  end inside the estimation period; mean statistics -> `select_ensemble` -> 8 candidates (out-of-range flag recorded, candidate
  still used as before).
* Ensemble: 1 000 successful independent long paths per bucket (round-robin over the 8 candidates; failed paths skipped),
  cumulative log return of the first 60h candles at h = 1, 4, 12, 24. Three independent ensemble seeds; the DDM CRPS is the
  mean over the three seeds and the seed SD is reported (flag if SD > 0.5% of CRPS, one quarter of the 2% margin).
* The old (look-ahead) mapping is rebuilt only to measure its effect and reproduce earlier numbers; it is not used for any
  comparison with the statistical models.

## Scoring (all models identical)

* Predictive distribution = M = 1 000 draws. CRPS: fair estimator mean|x - y| - sum_{i<j}|x_i - x_j| / (M(M-1)); the standard
  (biased) estimator is reported in an appendix.
* PIT: randomised for ties (fixed seed), 20 bins, KS distance; Monte Carlo spread from 100 randomisations.
  Coverage of central 50/80/90/95% intervals; tail scores = pinball loss at 1%, 5%, 95%, 99%.
* Per-origin scores are stored; comparisons are paired on identical origins.

## Model selection, primary comparison, decision rule

* Per horizon h, the statistical model with the lowest **validation** mean fair CRPS is chosen from {B0, B1, B2, B3, B4, B5, B6,
  B7-N, B7-t} (B8 and B4s excluded; ties within 0.5% resolved by the order B1 < B2 < B3 < B4 < B7-N < B5 < B6 < B7-t < B0).
* Final period: four pre-fixed comparisons, one per horizon, **Delta_h = (CRPS_DDM,h - CRPS_Stat*(h),h) / CRPS_Stat*(h),h; a
  negative value favours the DDM.** B8 versus DDM and all other models are exploratory.
* **Simultaneous inference (note 1):** week blocks (ISO week of the origin; the same blocks for all horizons) are resampled
  10 000 times; Delta*_h is recomputed on each resample; se_h = SD of Delta*_h; the critical value c is the 95th percentile of
  max_h |Delta*_h - Delta_h| / se_h; the simultaneous 95% band is Delta_h +- c se_h. This replaces the separate Holm correction
  for the decision; Diebold-Mariano p-values (Newey-West lag h + 23) with Holm adjustment are reported as secondary.
* Provisional practical margin +-2% (not justified by the earlier DDM-vs-DDM gains). Using the simultaneous band: upper < -2% ->
  DDM better; lower > +2% -> statistical model better; band inside [-2%, +2%] -> equivalent; otherwise **inconclusive**
  (including a band that excludes 0 but straddles a margin). Because the final period is only a partially known holdout,
  these labels are **pre-specified classifications for this evaluation period only**, not confirmatory claims for unseen data.

## Leak tests (note 3)

For 150 random origins in each of the validation and final periods and for every model path (statistical models, B8, DDM), with
the learned state **fixed** (monthly parameters, bucket edges, mapping, ensembles), all data strictly after the origin row is
replaced by random numbers, every derived array (block returns, RV, ATR) is rebuilt from the perturbed series, and the
predictive samples must be bit-identical to the unperturbed ones for the same seed. A failure stops the run.

## Final-period independence audit (what earlier work looked at 2021-09..2022-08)

* DDM CRPS, atlas and shock selection (`ddm_*.py`): rows 300 000-600 000 (2017-06-22..2018-04-12) only.
* `delta_forward_validation`, `delta_timeseries_usdjpy`: rows 300 000-620 000.
* `real_data_transfer_test`: random windows over the whole series (concurrent correlation); `phase10a_target_screen`
  (datasets 29 and 149): target screening over the whole period; `ddm_variant_vol_screen` (last 1M rows, last quarter from
  2021-12-31 as test) and `future_vol_task_difficulty` (chronological last 20%, 2022): volatility predictability screening.
* None of these selected a probabilistic model, a distribution family or a DDM candidate, but the volatility persistence that
  GARCH/HAR exploit was studied on this period. The final period is therefore a **partially known** holdout, 2022 is a
  high-volatility regime, and no generalisation beyond it is claimed. No F1 modelling choice uses the final period.

## Environment

`arch==8.0.0` (added to `backend/requirements.txt`; installing it changed no existing package); numpy 2.3.3, scipy 1.17.1,
pandas 3.0.1, statsmodels 0.14.6, Python 3.13.17.

## Implementation amendments (made before any validation/final score; no result had been computed)

1. Common origin set: an origin is evaluated only if the ATR block, the block return at the origin and the target window are
   all gap-free, so that every model (including the GARCH family and HAR-RV) can use the block ending at the origin. The
   counts quoted above are upper bounds; the exact counts are reported with the results.
2. DDM mapping windows additionally require the 1 001 forward closes to be gap-free under the same 5-minute rule (the earlier
   code did not check contiguity; forward statistics that include a weekend jump would be inconsistent with gap-free targets).
3. Simulation size: statistical models draw 1 000 independent paths per origin directly (M = 1 000), instead of 10 000 paths
   followed by sub-sampling.

## Interpretation limits (to be stated with every F1 result; consistent with the approved plan v0.2/v0.3)

These are not implementation defects. Each one is a property of the approved design that limits what the result may be taken to
mean; the final report must repeat them next to the classification.

1. **Weekly block bootstrap.** Whole calendar weeks are resampled jointly for all four horizons, so dependence inside a week
   (about 120 origins, longer than the 24-origin target overlap at h = 24) and across horizons is preserved. Dependence
   **between adjacent weeks** (volatility clustering that spans a week boundary) is not preserved, so the standard errors and the
   simultaneous band may be somewhat too narrow in persistent-volatility periods. The Newey-West Diebold-Mariano p-values
   (lag h + 23) are reported as a cross-check, not as a correction.
2. **HAR-RV independence assumption.** The log-RV residual and the return innovation are drawn independently, so any leverage
   or sign-volatility dependence is absent from B7 (the GJR/EGARCH models carry asymmetry). If B7 loses to an asymmetric model
   this cannot be attributed to the HAR mean structure alone.
3. **DDM-only frozen mapping (declared asymmetry).** The DDM bucket-to-candidate mapping and its ensembles are frozen on the
   estimation period (to 2020-08-31), while B0-B8 are re-estimated every month with an expanding window. A statistical model
   can therefore follow the 2021-2022 regime change, the DDM cannot. A statistical advantage in the final period may reflect
   this updating and not a better structure; a DDM advantage, if it appears, is not helped by it. The comparison measures
   "frozen DDM versus continuously updated baselines" and must not be described as a comparison of model structures alone. A
   sensitivity analysis with the baselines also frozen at 2020-08-31 would separate the two effects; it is not part of the
   approved plan and is recorded as an optional follow-up.
4. **Partially known final period and regime.** 2021-09..2022-08 is a high-volatility, regime-changing year that earlier
   work has looked at (volatility screening); labels are classifications under the pre-specified rule for this period only.
5. **Margin and equivalence.** The +-2% margin is provisional and was not derived from statistical-versus-DDM differences;
   a band that straddles a margin is reported as inconclusive.
6. **ATR definition.** The block ATR uses 59 absolute differences (leak-free) and is not identical to the 60-difference
   definition of earlier work; DDM and B8 share it, so their comparison is internally consistent.
7. **Atlas versus real windows.** The atlas statistics are computed with the same function on 1 000 simulated candles (999
   returns); a real forward window has 1 001 closes (1 000 returns). The numerical consistency of definition, units and
   order is checked in `test_f1_atlas_consistency.py`; the one-return difference is immaterial and not corrected.

## Execution procedure (run guard)

Each stage script first runs `f1_guard.guard_main`: with `--preflight` it prints HEAD, dirty files and the `state_hash`
(HEAD + hash of every loaded repository module + input hashes) and exits. The caller passes exactly these values to the Study
Manager `check_execution_gate` and saves the response as `gate_approval_<stage>.json`. Without an approval that belongs to this
node, this HEAD and this state hash, and a passing response (Stage C approved, audited commit matches, clean workspace, no
unresolved critical), the stage refuses to run; modified or untracked loaded repository files always refuse. There is no
environment variable or flag that bypasses it. `end_run` re-measures after the run and marks the run invalid if anything
differs. The approval file is evidence written by the caller; the Study Manager record of the gate call is the authority in the
post-run audit.
