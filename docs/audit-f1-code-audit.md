# F1 (#889) code audit: Stage A (coding-agent pre-audit) and Stage B (material for the independent reviewer)

**This is the coding agent's own audit. It is not a completed code audit.** Stage C (independent review of the pinned
commit) is still required before the experiment is run.

## 1. Identification

| item | value |
|---|---|
| Repository | https://github.com/Naradice/algoforge |
| Pinned commit (code under review) | `5ecf3b262924d42f098b8422c336872b1b188a66` |
| Plan of record | `docs/research-f1-statistical-baselines.md` (spec committed at `4c88e81` / `ac1d444` before any score) and Study Manager #889 v0.3 |
| Entry points | `backend/research/f1_ddm.py` (`map`, `ens`), `f1_run.py` (`score`, `leak`), `f1_analyze.py` (`select`, `final`) |
| Commands | `python research/f1_ddm.py map W` ; `python research/f1_ddm.py ens W {1,2,3}` ; `python research/f1_run.py score W validation` ; `python research/f1_analyze.py select W` ; `python research/f1_run.py leak W` ; `python research/f1_run.py score W final` ; `python research/f1_analyze.py final W` (W = fresh work directory) |
| Dependencies | Python 3.13.17, arch 8.0.0, numpy 2.3.3, scipy 1.17.1, pandas 3.0.1, statsmodels 0.14.6 (`backend/requirements.txt` pins arch) |
| Main parameters | `f1_common.py`: periods, `GAP_TOL_SECONDS=300`, `M_SAMPLES=1000`; `f1_ddm.py`: shock P0=0.002 S=0.3 tau=300, `N_PATHS=1000`, `MAX_PER_BUCKET=300`; `f1_analyze.py`: margin 0.02, `N_BOOT=10000`, tie tolerance 0.005 |
| Input data | `artifacts/datasets/derived/usdjpy_1min_volatility.parquet`, 55 060 647 bytes, sha256 `c25eb5720a5552b0230fb6ce2354de1cee020e6b7ca95da46034b75f2a53c828`; `backend/ddm_regime_atlas.csv` sha256 `9dcdb2ffa555f0f1dd4fec3bc3b597e5874efeb75ad00c3c22a9d86a3ac4a5f1`; mapping `ddm_mapping.json` sha256 `15312749b7c1648cf63b79550e5b11fda8ad63d32c4ee5d9dddd237f4273d436` |
| Seeds | statistical samplers `default_rng([model_id, block_j])`; DDM simulation seed `8_000_000 + seed_id*1_000_000 + bucket*20_000 + attempt`; mapping sample `20261117`; PIT randomisation `default_rng([99, model_id, h, month, year])`; bootstrap `20261117` |

Permalinks below use the pinned commit: `https://github.com/Naradice/algoforge/blob/5ecf3b262924d42f098b8422c336872b1b188a66/<path>#L<line>`
(abbreviated as `[file:line]`).

## 2. Plan requirement -> implementation -> verification

| id | requirement (plan v0.3) | implementation | verification |
|---|---|---|---|
| R-01 | estimation / validation / final periods; origin and target end inside the period; purge | `f1_common.py` PERIODS L21, `period_mask` L103 | `test_period_mask_purges_targets_crossing_the_period_end` |
| R-02 | 60-candle blocks, origin = last row, gap rule (> 5 min), target `log C[t+60h]/C[t]` | `Series.contiguous` L50, `target` L90, `atr_valid` L99, `GAP_TOL_SECONDS` L26 | `test_gap_rule_and_targets`, `test_missing_quiet_minutes_do_not_break_blocks` |
| R-03 | information at origin only rows <= t (ATR, block return, RV, HAR features) | `atr_window` L59, `block_returns` L68, `block_rv` L76 (`f1_common.py`); `har_features` L140 (`f1_models.py`) | `test_derived_arrays_depend_only_on_rows_up_to_origin` |
| R-04 | learned state (monthly) built only from data before the month start | `f1_models.py` `build_state` L85 (`done` L93, `past` L100, HAR rows L125-127) | `test_state_depends_only_on_data_before_month_start` (perturbs all later data) |
| R-05 | models B0-B8, B4s: definitions, per-path sequential simulation, standardised t, HAR with separate residual and innovation laws | `sample_B0/B1/B2/B8` L160-175, `sample_arch` L187, `sample_har` L215, `std_t` L59, `fit_std_t_nu` L64 | `test_samplers_are_deterministic_and_ignore_the_future` (all samplers) |
| R-06 | fair CRPS M=1000, plain CRPS, randomised PIT, coverage, pinball loss | `f1_common.py` `fair_crps` L120, `plain_crps` L132, `score_samples` L147 | `test_fair_crps_matches_bruteforce`, `test_plain_crps_matches_bruteforce`, `test_fair_crps_is_unbiased_for_small_ensembles`, `test_crps_quantile_grid_matches_analytic`, `test_pit_uniform_for_calibrated_samples_and_ties_randomised`, `test_gaussian_baseline_crps_close_to_analytic` |
| R-07 | per-horizon validation selection among B0-B7 with 0.5% tie rule and complexity order | `f1_analyze.py` `stage_select` L45, `SELECT_POOL` L21, `TIE_ORDER` L22 | `test_stage_select_and_final_on_synthetic_scores` |
| R-08 | Delta_h sign, simultaneous max-t band over 4 horizons with week blocks, label rules, DM (Newey-West h+23), Holm | `stage_final` L83 (bootstrap L110-118, labels L133-140), `newey_west_dm` L70 | `test_newey_west_matches_naive_for_iid_and_holm_is_monotone`, synthetic end-to-end test (known ratios 0.90 / 1.00 / 1.10 give the three labels) |
| R-09 | DDM mapping from estimation windows only (forward 1001 closes gap-free and inside the estimation period), frozen atlas / `select_ensemble` / shock, 1000 paths per bucket, 3 seeds, horizon of exactly 60h candle steps | `f1_ddm.py` `stage_map` L82, `simulate_long_path` L39, `cum_returns` L65, `bucket_job` L69, `stage_ens` L123 | `test_ddm_cumulative_return_covers_exactly_60h_candle_steps`; mapping built from real data in an earlier run (log reviewed by hand, no automated test) |
| R-10 | leak test with learned state fixed | `f1_run.py` `leak_job` L117, `stage_leak` L150 | executed only after the ensembles exist (not yet run); unit tests above cover the same property |
| R-11 | reproducible seeds | `rng_for` (`f1_models.py` L156), seeds in table above | `test_samplers_are_deterministic_and_ignore_the_future` |
| R-12 | same origins for every compared model; results persisted per origin | `assert_same_origins` (`f1_analyze.py` L34), `stage_score` (`f1_run.py` L84) | manual one-month run with dummy ensembles (56 model/horizon cells, identical origins) |
| R-13 | environment pinned | `backend/requirements.txt` (`arch==8.0.0`) | `pip install --dry-run` showed no change to other packages |

## 3. Important code changes found by this audit (diff since the pre-audit commit)

Diff to read: https://github.com/Naradice/algoforge/compare/704d6f6...5ecf3b262924d42f098b8422c336872b1b188a66
(restrict to `backend/research/f1_*.py` and `backend/tests/unit/test_f1_pipeline.py`).

* D1 DDM horizon off by one candle step (inherited from the earlier `path_metrics`): the DDM cumulative return used
  `closes[60h-1]/closes[0]` (59 steps at h = 1, about 0.8% too small a standard deviation) while the real target spans 60h
  steps. Fixed in `f1_ddm.py` (simulate `24*60 + 1` candles, use `closes[60h]/closes[0]`). Found by test, not by running the
  experiment; no DDM ensemble from the old code is to be used.
* D2 Origins with realised variance exactly 0 gave `log RV = -inf` in HAR-RV (4 validation origins, 0 final). Excluded from the
  common origin set (`f1_models.py` L51).
* D3 The leak test did not cover the learned state; the state-invariance test was added.
* D4 No guard that compared models were scored on identical origins; assertions added in `f1_analyze.py`.
* D5 PIT Monte Carlo spread (plan: 100 randomisations) was not implemented; replaced by recording the share of origins
  with ties (`tie` in `score_samples`). In a one-month trial ties occur at about 1% of origins for B0/B8 only (discrete price
  grid) and never for the other models. **This is a deviation from the plan** to be confirmed by the reviewer.

## 4. Automated tests

`python -m pytest tests/unit/test_f1_pipeline.py` from `backend/`: **19 passed** (16 unit/integration tests + 3 parametrised
end-to-end cases of selection and final analysis on synthetic scores). The state/sampler/Gaussian-CRPS tests use the real
USDJPY artifact and are skipped if it is missing.

## 5. Potential defects, leaks and statistical issues not yet resolved

1. `f1_run.stage_leak` has not been executed against real ensembles.
2. `f1_run.month_job` / `stage_score` orchestration is covered only by a manual one-month run with dummy ensembles (no
   automated test); `f1_ddm.stage_map` has no automated test.
3. Planned but not implemented: effect of the old (look-ahead) mapping versus the rebuilt mapping (a deliverable of the plan);
   supplementary period 2022-09..12 (data-source alignment unchecked); M1-based GARCH (declared not run).
4. The DDM ensemble is shared by all origins of an ATR bucket and is frozen on the estimation period, while all other models are
   re-estimated monthly (declared asymmetry).
5. ATR uses 59 absolute differences of the 60 closes in the block; the earlier code used 60 differences including one close after
   the window. The new definition is leak-free but not identical to the earlier one.
6. The simultaneous band assumes calendar-week blocks are exchangeable. A week holds about 120 origins, longer than the longest
   target overlap (24 origins at h = 24), but dependence across adjacent weeks (volatility clustering) is not modelled.
7. HAR-RV assumes independence of the log-RV residual and the return innovation (stated in the plan).
8. Process incident: while implementing, `git add backend/research` also committed 15 unrelated untracked research scripts
   (f99223b). They were untracked again in `d0b7a03` and are still kept locally; they remain in the history of `f99223b` on the
   remote. Nothing in them is used by F1.
9. Process incident: a background ensemble job (old code) was stopped through the task tool, but its worker processes could not be
   terminated (the kill command was denied) and may still be running; outputs of that run in the old work directory must not be
   used. The audited run uses a new work directory.

## 6. Files and functions that need independent review first

1. `backend/research/f1_models.py`: `build_state`, `sample_arch`, `sample_har`, `Context.__init__`.
2. `backend/research/f1_ddm.py`: `stage_map`, `simulate_long_path`, `cum_returns`.
3. `backend/research/f1_analyze.py`: `stage_final` (bootstrap, band, labels), `stage_select`.
4. `backend/research/f1_common.py`: `Series.contiguous`, `target`, `period_mask`, `fair_crps`, `score_samples`.
5. `backend/research/f1_run.py`: `month_job`, `leak_job`.

## 7. After the run (post-run audit checklist)

Executed commit equals the pinned commit; run settings equal the plan; `final_result.json` main statistics recomputed from
`scores_final.npz` by an independent script; list of deviations and failures (`failures_*.json`, `leak_test.json`).
