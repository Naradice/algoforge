# F1 (#889) code audit: Stage A (coding-agent pre-audit) and Stage B (material for the independent reviewer)

**This is the coding agent's own audit. It is not a completed code audit.** Stage C (independent review of the pinned
commit) is still required before the experiment is run. Revision 2 answered the Stage C interim comments (A)-(D); revision 3 (this file) answers the second interim review and pins a new code commit.

## 1. Identification

| item | value |
|---|---|
| Repository | https://github.com/Naradice/algoforge |
| **Code commit under review (pinned)** | `bf1ebc66cc29c54d13aae6faedab1722b981cbaa` (revision 2 was `d5cedc77dd445fae403840afa13c961a6c8f59f7`, revision 1 `5ecf3b262924d42f098b8422c336872b1b188a66`; both are superseded) |
| **Audit document commit** | the commit that contains this file (the code commit above does not contain it; the document commit only adds documentation) |
| Plan of record | `docs/research-f1-statistical-baselines.md` (spec committed at `4c88e81` / `ac1d444` before any score) and Study Manager #889 v0.3 |
| Entry points | `backend/research/f1_ddm.py` (`map`, `ens`), `f1_run.py` (`score`, `leak`), `f1_analyze.py` (`select`, `final`) |
| Commands | `python research/f1_ddm.py map W` ; `python research/f1_ddm.py ens W {1,2,3}` ; `python research/f1_run.py score W validation` ; `python research/f1_analyze.py select W` ; `python research/f1_run.py leak W` ; `python research/f1_run.py score W final` ; `python research/f1_analyze.py final W` (W = fresh work directory) |
| Dependencies | Python 3.13.17, arch 8.0.0, numpy 2.3.3, scipy 1.17.1, pandas 3.0.1, statsmodels 0.14.6 (`backend/requirements.txt` pins arch) |
| Main parameters | `f1_common.py`: periods, `GAP_TOL_SECONDS=300`, `M_SAMPLES=1000`; `f1_ddm.py`: shock P0=0.002 S=0.3 tau=300, `N_PATHS=1000`, `MAX_PER_BUCKET=300`; `f1_analyze.py`: margin 0.02, `N_BOOT=10000`, `PIT_REPEATS=100`, tie tolerance 0.005 |
| Input data | `artifacts/datasets/derived/usdjpy_1min_volatility.parquet`, 55 060 647 bytes, sha256 `c25eb5720a5552b0230fb6ce2354de1cee020e6b7ca95da46034b75f2a53c828`; `backend/ddm_regime_atlas.csv` sha256 `9dcdb2ffa555f0f1dd4fec3bc3b597e5874efeb75ad00c3c22a9d86a3ac4a5f1`; `ddm_mapping.json` of the earlier mapping run sha256 `15312749b7c1648cf63b79550e5b11fda8ad63d32c4ee5d9dddd237f4273d436` (the mapping logic is unchanged apart from refactoring; it is rebuilt in the audited run) |
| Seeds | statistical samplers `default_rng([model_id, block_j])`; DDM simulation seed `8_000_000 + seed_id*1_000_000 + bucket*20_000 + attempt`; mapping sample `20261117`; PIT randomisation `default_rng([99, model_id, h, month, year])` and the 100 repeats `default_rng(0)`; bootstrap `20261117` |

Permalinks use the code commit: `https://github.com/Naradice/algoforge/blob/d5cedc77dd445fae403840afa13c961a6c8f59f7/<path>#L<line>`.

## 2. Plan requirement -> implementation -> verification

| id | requirement (plan v0.3) | implementation | verification |
|---|---|---|---|
| R-01 | periods; origin and target end inside the period; purge | `f1_common.py` PERIODS L21, `period_mask` L103 | `test_period_mask_purges_targets_crossing_the_period_end` |
| R-02 | 60-candle blocks, origin = last row, gap rule (> 5 min), target `log C[t+60h]/C[t]` | `Series.contiguous` L50, `target` L90, `atr_valid` L99, `GAP_TOL_SECONDS` L26 | `test_gap_rule_and_targets`, `test_missing_quiet_minutes_do_not_break_blocks` |
| R-03 | information at origin only rows <= t | `atr_window` L59, `block_returns` L68, `block_rv` L76; `har_features` (`f1_models.py` L140) | `test_derived_arrays_depend_only_on_rows_up_to_origin` |
| R-04 | learned state built only from data before the month start | `f1_models.py` `build_state` L85 | `test_state_depends_only_on_data_before_month_start` |
| R-05 | models B0-B8, B4s; per-path sequential simulation; standardised t; HAR with separate residual and innovation laws | `sample_B*` L160-175, `sample_arch` L187, `sample_har` L215 (`f1_models.py`) | `test_samplers_are_deterministic_and_ignore_the_future` |
| R-06 | fair CRPS (M=1000), plain CRPS, randomised PIT, coverage, pinball | `f1_common.py` `fair_crps` L120, `plain_crps` L132, `score_samples` L147 | six CRPS/PIT tests in `test_f1_pipeline.py` (brute force, unbiasedness, analytic value, quantile grid, PIT uniformity and ties, Gaussian baseline) |
| R-07 | per-horizon validation selection with tie rule | `f1_analyze.py` `stage_select` L61 | `test_stage_select_and_final_on_synthetic_scores`, end-to-end test |
| R-08 | Delta_h sign, simultaneous max-t band, label rules, DM (lag h+23), Holm | `stage_final` L99, `newey_west_dm` L86 | `test_newey_west_matches_naive_for_iid_and_holm_is_monotone`, synthetic end-to-end (ratios 0.90 / 1.00 / 1.10 give the three labels) |
| R-09 | **DDM mapping from estimation windows only; bucket edges; forward 1001 closes** | `f1_ddm.py` `build_mapping` L82 (pure, injectable), `stage_map` L118, `simulate_long_path` L39, `cum_returns` L65, `bucket_job` L69 | **`test_stage_map_windows_edges_and_forward_range`, `test_stage_map_ignores_data_after_the_estimation_period_and_is_reproducible`, `test_stage_map_excludes_windows_with_gaps_and_windows_running_past_the_period_end`**, `test_ddm_cumulative_return_covers_exactly_60h_candle_steps` |
| R-10 | leak test with learned state fixed | `f1_run.py` `leak_job` L133, `stage_leak` L166 | not yet run against real ensembles; unit tests above cover the same property |
| R-11 | reproducible seeds | `rng_for` (`f1_models.py` L156) | determinism asserted for every sampler and for the whole month |
| R-12 | **score generation -> result saving -> analysis; identical origins for all models** | `f1_run.py` `score_month` L50, `aggregate_scores` L89, `write_scores` L100, `stage_score` L108; `assert_same_origins` (`f1_analyze.py` L50) | **`test_score_month_to_saved_results_end_to_end`** |
| R-13 | environment pinned | `backend/requirements.txt` (`arch==8.0.0`) | dry run showed no change to other packages |
| R-14 | **PIT uncertainty (100 re-randomisations)** | `score_samples` stores `pit_lo`, `pit_w` (`f1_common.py` L147); `pit_randomisation_spread` (`f1_analyze.py` L37, `PIT_REPEATS` L34) | `test_pit_randomisation_spread_is_zero_without_ties_and_positive_with_ties` |

### What the new tests check

* **stage_map** (artificial 60-day series with stochastic volatility, artificial estimation period): bucket edges equal the
  quantiles of the estimation-period block ATR; every used window starts in the estimation period, falls in its bucket, spans
  exactly rows `[60j, 60j+1000]` (1001 closes, the slice handed to the characteriser equals the slice of the series in order),
  is gap-free and ends inside the estimation period; the cap per bucket and the seed behave as specified; replacing all data
  after the estimation period leaves edges, used blocks and statistics unchanged; windows containing a gap are excluded, the
  window starting right after the gap is accepted, and the last ~16 windows are purged.
* **score_month -> results -> analysis** (artificial 120-day series, one week scored, fake ensembles): all 14 models x 4
  horizons are present; no failures; every model has the same origin set, equal to an independent recomputation; all scores are
  finite and in range; two scores are recomputed by hand from the sampler / ensemble and match to 1e-12; the whole month is
  bit-identical on a second run; `write_scores` produces a file with sorted origins and an empty failure file; `stage_select`
  and `stage_final` then run on that file and produce the expected structure including the PIT repeat spread.

## 3. Code changes found by this audit (diff since the pre-audit commit)

Diff to read: https://github.com/Naradice/algoforge/compare/704d6f6...d5cedc77dd445fae403840afa13c961a6c8f59f7
(restrict to `backend/research/f1_*.py` and `backend/tests/unit/test_f1_*.py`).

* D1 DDM horizon off by one candle step (inherited from the earlier `path_metrics`: `closes[60h-1]/closes[0]`, 59 steps at
  h = 1, about 0.8% too small a standard deviation). Fixed in `f1_ddm.py`.
* D2 Origins with realised variance exactly 0 gave `log RV = -inf` in HAR-RV (4 validation origins, 0 final). Excluded.
* D3 State-invariance test added. D4 Same-origin assertions added.
* D5 (interim comment D) PIT: the plan asked for the Monte Carlo spread of the randomised PIT from 100 re-randomisations. It
  is now implemented: `pit_lo`/`pit_w` are stored per origin and the final analysis reports, per model and horizon, the mean and
  SD of the KS distance and the largest SD of a histogram bin over 100 repeats. With no ties the spread is exactly zero.

## 4. Automated tests

From `backend/`: `python -m pytest tests/unit/test_f1_pipeline.py tests/unit/test_f1_stage_map_and_run.py`: **24 passed**
(19 + 5) in about 60 s on the code commit (the working tree files equal the commit). Tests that need the real USDJPY artifact
are skipped when it is missing.

## 5. Interim review comment (C): accidentally pushed files

Commit `f99223b` also contained 15 unrelated, previously untracked research scripts
(`analyze_holding_time_and_timestop.py`, `backtest_dynamic_sltp.py`, `backtest_ml_filter.py`, `build_rebound_dataset.py`,
`check_data.py`, `create_strategy_8.py`, `discover_extrema_with_ml.py`, `patch_live_runner.py`, `rigorous_rebound_backtest.py`,
`strategy7_ml_screen.py`, `strategy7_robustness.py`, `sweep_strategy7_execution.py`, `train_range_quantile_model.py`,
`train_rebound_model.py`, `verify_standalone_pivot.py`; all under `backend/research/`). A pattern search over the diff of
those 15 files (api key, secret, password, token, bearer, authorization, credential, private key, well-known key prefixes,
URLs with embedded credentials, login, account, broker) found **no matches**; the only URL is a local API address
(`http://127.0.0.1:8000/api/v1`). The pattern search does not prove absence of every kind of sensitive content: the files are
trading-strategy research code (strategy logic is not a credential but may be proprietary), and the visibility of the remote
repository could not be checked from this environment (the GitHub CLI is not installed). They were untracked in `d0b7a03`; the
history of `f99223b` still contains them, and removing them from history would require a history rewrite and force push, which
was **not** done. Decision for the owner: check the repository visibility and decide whether a history rewrite is wanted.

## 6. Unresolved items and potential issues

1. `f1_run.stage_leak` has not been executed against real ensembles (they do not exist yet; the unit tests cover the property).
2. Old (look-ahead) versus rebuilt mapping effect: planned deliverable, not implemented. Supplementary period 2022-09..12: not
   implemented (data-source alignment unchecked). M1-based GARCH: declared not run.
3. The DDM ensemble is shared by all origins of an ATR bucket and frozen on the estimation period; the other models are
   re-estimated monthly (declared asymmetry).
4. ATR uses 59 absolute differences of the 60 closes of the block; the earlier code used 60 differences including one close
   after the window (leak-free now, not identical to the earlier definition).
5. The simultaneous band treats calendar-week blocks as exchangeable; weeks hold about 120 origins (longer than the longest
   target overlap of 24 origins) but dependence across adjacent weeks is not modelled.
6. HAR-RV assumes independence of the log-RV residual and the return innovation (stated in the plan).
7. `stage_map` with the real atlas and the real `select_ensemble` is covered only by the earlier real run (log reviewed by hand)
   and by tests with injected fakes; the real `full_characterize` is not tested here.
8. Process incident: a background ensemble job (old code) was stopped through the task tool, but its worker processes could not
   be terminated (the kill command was denied) and may still be running; outputs of that run in the old work directory must not
   be used. The audited run uses a fresh work directory.

## 7. Files and functions for the independent review

1. `backend/research/f1_ddm.py`: `build_mapping`, `simulate_long_path`, `cum_returns`.
2. `backend/research/f1_models.py`: `build_state`, `sample_arch`, `sample_har`, `Context.__init__`.
3. `backend/research/f1_run.py`: `score_month`, `leak_job`, `write_scores`.
4. `backend/research/f1_analyze.py`: `stage_final`, `stage_select`, `pit_randomisation_spread`.
5. `backend/research/f1_common.py`: `Series.contiguous`, `target`, `period_mask`, `fair_crps`, `score_samples`.
6. `backend/tests/unit/test_f1_pipeline.py`, `test_f1_stage_map_and_run.py`: are the assertions strong enough?

## 8. After the run (post-run audit checklist)

Executed commit equals the pinned commit; run settings equal the plan; `final_result.json` main statistics recomputed from
`scores_final.npz` by an independent script; list of deviations and failures (`failures_*.json`, `leak_test.json`).

## 9. Revision 3: second interim review

Machine-readable requirement list for the Study Manager: `docs/audit/f1_requirements.json` (same commit as this file).
The code commit is `bf1ebc66cc29c54d13aae6faedab1722b981cbaa`; this document and the JSON file are added by a later commit.

* **Test target equals production path.** `test_stage_map_calls_build_mapping_with_the_production_functions` runs the real
  `stage_map` with `build_mapping` replaced by a recorder and checks that it receives the production `full_characterize`,
  `select_ensemble`, the atlas file content, the series from `load_series` and the documented defaults, and that the JSON is
  written. `test_month_job_passes_the_loaded_objects_to_score_month` and
  `test_stage_score_writes_what_write_scores_aggregates` do the same for `month_job` / `stage_score` (month order, arguments,
  concatenation sorted by origin, files written).
* **Expectations not generated by the implementation.** `test_origin_sets_equal_an_independent_brute_force_construction`
  rebuilds the origin sets with plain Python loops over raw timestamps and closes (gap rule, previous-block return, RV > 0,
  period and target-end purge); `test_scores_recomputed_by_brute_force` recomputes a DDM CRPS by broadcasting pairwise
  distances, the 90% hit from `np.percentile` and the PIT as a plain fraction.
* **Leak test: implementation, perturbation, pass condition, power.** `leak_job` perturbs everything after the origin,
  rebuilds all derived arrays, keeps the learned state fixed and compares all statistical models and DDM seeds bit by bit.
  `test_leak_job_detects_a_sampler_that_reads_the_future` injects a sampler that adds the realised next-block return and
  requires `leak_job` to flag exactly that model at every tested origin. Running it on the real ensembles is **R-10b, an
  execution-time gate** (after the ensembles exist, before any score computation; a failure stops the study).
* **Executed commit check.** `f1_guard.start_run` is called first by every stage of `f1_ddm.py`, `f1_run.py` and
  `f1_analyze.py`: it writes `run_manifest_<stage>_<time>.json` (HEAD, dirty files, command line, input sha256, package
  versions), refuses a HEAD different from `F1_EXPECTED_COMMIT` and modified or untracked F1 files (unless `F1_ALLOW_DIRTY=1`,
  recorded). Tests: `test_guard_*`. The Study Manager execution gate is a second, independent check.
* Tests: 33 passed in about 74 s (`test_f1_pipeline.py`, `test_f1_stage_map_and_run.py`, `test_f1_wiring_and_guard.py`).
* **Study Manager registration status.** `validate_audit_requirements` accepts the schema (validated with the fields
  `repository`, `commit`, `requirements[id,title,description,severity,status,implementation[path,function,lines],
  verification{tests}]`), but with `auto_register=true` the server returned `name 'timezone' is not defined`, so the audit state of
  #889 is still `none` and `check_execution_gate` correctly reports that Stage C is not completed. The execution gate has
  therefore not been passed; the experiment has not been run.
