# T0-A: Threshold-dealer-model (Yamada–Takayasu–Takayasu) Fig.3 reproducibility audit

Study Manager: goal #886 → experiment #894 (parent #874). Code: `backend/research/tdm_fig3_audit.py`
(the production simulator `backend/data/collectors/ddm_simulator.py` is **not modified**; frozen at
commit `e5232c0`, file clean in the working tree).

Primary source: K. Yamada, H. Takayasu, M. Takayasu, *Characterization of foreign exchange market using
the threshold-dealer-model*, Physica A 382 (2007) 340; arXiv:physics/0608099v2 (read from the PDF text).

## 1. What the paper defines

| Item | Paper |
|---|---|
| Model-1 | `db_i/dt' = σ_i c_i` (eq.1); σ=+1 buyer / −1 seller; ask `a_i = b_i + L`; trade iff `max{b_i} − min{a_j} ≥ 0` (eq.2); buyer/seller swap after a trade; price quoted between bid and ask, minimum unit |
| Model-2 | `db_i/dt = σ_i c_i /(G_k<T_k>_τ)`, `dt = G_k<T_k>_τ dt'` (eq.3); `G_k = T'^β_k / T'`, `β = α−1`, `α≈1.25`; `T'` = mean of `T'^α_k`; `<T_k>_τ` = moving average of the (modulated) interval over the last τ seconds. Two time axes: `t'` (model-1 step time) and `t` (real time) |
| Model-3 | adds `ε_i d_i <dP>_η`, ε=±1 follower/contrarian (mixed), `<dP>_η = Σ w_j dP_t / W` |
| Model-4 | both; `w_j = exp(−0.3 j)`, η=150 s, d_i∈[5,7] |
| Fig.3 | intervals vs time (left) and cumulative distribution of intervals normalised by the mean (right), model-1 vs model-2; **N=500, L=1.0, dt=0.01, c_i=[−0.01,0.01], τ=150** |
| Not specified | initial b_i, initial <T>_τ, window-empty rule, how c_i<0 is handled, number of steps |

Derived consistency check: since G_k is constant inside interval k, eq.(3) leaves the price trajectory in
`t'` unchanged and only rescales the clock: `T_k = G_k<T>_τ T'_k = F_k <T>_τ`, `F_k = T'^α_k / mean(T'^α)`.
So **C1 (Model-2 internal dynamics) ≡ C0 (Model-1) by construction**; only C2 (the transformed time) differs.

## 2. Difference table (paper vs current DDM)

| # | Item | Current DDMv1/v3 (`ddm_simulator.py`) | Class |
|---|---|---|---|
| 1 | Matching rule (eq.2), position swap, bid/ask=b / b+L | identical (`_contruct`); price = midpoint floored to trade_unit | match |
| 2 | **Model-2 self-modulation (G_k, <T>_τ)** | **absent.** Time advances `tick_time += tick_time_unit × noise` on *every* step, noise i.i.d. from weights 1/(i+1) over 1..100 (inherited from the original stocknet simulator). Nothing depends on past intervals | **implementation mismatch (major)** |
| 3 | Time axes | one clock, `tick_time` (i.i.d. noise per step); paper has `t'` and `t` | implementation mismatch |
| 4 | c_i | `U[0.01,0.02]` positive only; paper `[−c,c]` with c=0.015 (Fig.1), 0.01 (Fig.3), 0.02 (Fig.5) | parameter / **unresolved** (sign handling) |
| 5 | N, dt | default N=300, price step per step = full `tend` (≡dt=1, the Fig.5 setting); Fig.3 uses N=500, dt=0.01 | parameter |
| 6 | Trend term | `dealer_sensitive∈[−3.5,−1.5]` (all agents, **same negative sign**) × WMA of last 5 *trades* with U(0,1) weights; paper: ε_i=±1 mixed, d_i∈[5,7], exp(−0.3j) weights over η=150 *seconds* | implementation mismatch (Model-3/4, out of scope for the Fig.3 step); sign convention needs a separate check |
| 7 | Fig.3 statistic | not computed anywhere in the repo | evaluation gap |
| 8 | Real-data time (Fig.2): tick = quote/trade? | unknown; repo uses tick_time as physical seconds when aggregating to OHLC | observation/time-mapping, **unresolved** (T1) |

## 3. Minimal reproduction (Model-1 vs Model-2, Fig.3 conditions)

Setup: N=500, L=1, dt=0.01, τ=150, α=1.25, 20 000 trades, seeds 1–3, both readings of c_i
("signed" = literal `U[−0.01,0.01]`; "abs" = `U[0,0.01]`), initial <T>_τ = mean(T'). Also the current DDMv1
code as-is (N=100, 500; seed 1). Raw numbers: `research/out` not kept; rerun
`python research/tdm_fig3_audit.py 20000 out.json` (~5 min).

| Run | mean T | CV | max/median | ACF(log T) lag1 | level q1→q4 |
|---|---|---|---|---|---|
| C0 Model-1 (all seeds, both c modes) | 0.8–1.7 (scale set by c) | 0.69 | 6.7–9.4 | \|·\|<0.02 | flat |
| C2 Model-2, signed s1/s2/s3 | 2.85 / 1.99 / 1.04 | 0.87 | 11.2–13.2 | 0.02 / −0.01 / 0.00 | 2.5→3.3 / 2.2→2.0 / 1.0→1.1 |
| C2 Model-2, abs s1/s2/s3 | **0.024** / 0.27 / 1.23 | 0.87 | 12.4–15.6 | ≈0 | flat |
| CODE DDMv1 N=100 / 500 | 0.045 / 0.024 | 0.94 / 1.20 | 9.1 / 17.3 | −0.03 / −0.04 | – |

Findings (observations, not yet explanations beyond what is stated):
1. C1≡C0 holds exactly (eq.3 only rescales time).
2. Literal eq.(3) gives C2 = (heavier tail via T'^1.25: CV 0.69→0.87, max/median ≈7→12) **but no clustering**
   (ACF of log T ≈ 0 at lags 1–50). With τ=150 s the window holds ~50–100 intervals, so <T>_τ behaves like a
   near-constant; the output is `F_k × const`.
3. That constant is the arbitrary start-up value: the mean interval of C2 differs by up to ~50× across seeds
   (0.024 … 2.85) and is then frozen. The "over-dense / extremely long intervals" impression from earlier
   attempts is therefore at least partly a level/initialisation artifact of an under-specified eq.(3), not a
   dynamical property (earlier logs were not re-checked, so this stays a hypothesis for that past run).
4. The current code (i.i.d. noise per step) also shows no clustering (ACF(log T) ≈ −0.03…0), as expected from #2 above.
5. The paper's clustered intervals (Fig.3 left) are **not reproduced under this reading**. Candidate missing
   pieces (one factor at a time, not yet tested): window τ counted in ticks rather than seconds; <T>_τ
   computed over the previous *t'*-intervals; G_k acting causally inside an interval; a different normalisation
   of `T'` in G_k. These remain **unresolved**.

## 4. Classification and impact

* Implementation mismatch: Model-2 self-modulation absent; Model-3/4 trend term differs.
* Parameter/observation mismatch: c_i sign/range, dt, N, tick = quote vs trade (T1).
* Unresolved: the exact eq.(3) reading that yields interval clustering.

Impact on existing results (provisional until T1; classes A no rerun / B partial / C re-verify / D hold):

| Existing work | Depends on interval clock? | Class |
|---|---|---|
| Per-trade / trade-count-indexed experiments (moment matching on tick returns, Atlas on per-trade stats) | no | A |
| OHLC (M1…) based: delta transfer, CRPS/Coverage, shock-decay DDM vs ATR | yes (candles are cut by `tick_time`; i.i.d. clock ⇒ no interval clustering) | C |
| Claims that DDM "reproduces the paper's interval law" | – | D (none supported) |
| Model-3/4 / Fig.5 comparisons | trend-term differences | D |

## 5. F1 gate

Per goal #886: full reproduction is not a gate. F1 may proceed **with current DDM labelled "DDMv3 (repo
variant; no Model-2 time modulation, same-sign WMA trend term)" and not as the paper's Model-4**; F3/F1 write-ups
must not call it a reproduction of the paper. T1 (time-scale comparison) needs its own plan approval
(node #895, still `proposed`).

## 6. Provenance of the deviations (stocknet_study/FeatureEngineering/AgentSimulation.ipynb)

Added after reviewing the original experiment notebook. The original `simulator.py` (stocknet / stocknet_study,
byte-identical apart from line endings) has no Model-2 either, so the Fig.3 gap is inherited, not a porting error.

* Time: in the notebook V1 advances `tick_time` by a fixed unit; trade intervals were injected by sampling spans from the
  empirical USDJPY span CDF, independent of the trade process. V2 only post-processed intervals (`T^alpha / mean`,
  cumsum; no moving-average feedback; the `tick_sma` argument called in the notebook is not defined). The notebook records
  "[1] says transaction span obeys exp(-kx); my data doesn't reproduce that" and "I couldn't reproduce a result of
  reference [1]". [1] is the Takayasu et al. book, not Yamada et al. [4]. The simulator's i.i.d. `time_noise_method` is a
  later in-simulator substitute for that empirical-span injection.
* Trend term: a fixed follower value (`dealer_sensitive=0.5`, cell 53) drifts monotonically to the initial price; the
  default contrarian range gives a bounded path and a lag-1 return autocorrelation of about -0.37 (market about -0.27,
  cell 64). The notebook does not state the reason for choosing the contrarian default; this reading is inferred.
* Port initial positions: the original uses i.i.d. `random.choice([-1, 1])`; the algoforge port uses a balanced 50/50
  start (its code comment claiming this matches the original `bull_ratio=0.5` is inaccurate for this file).
* Training provenance: ml_model 115 (run 1417) was trained on dataset 48 `ddm_regime_delta_regression`, generated by
  `generate_regime_datasets.py` with the algoforge `DDMv3` via `simulate(n_trades=...)` (trade-count candles, no
  `tick_time`). The run-to-dataset link rests on the dataset name only. Such trade-count-indexed training data is
  insensitive to the missing Model-2 time modulation (class A); `collect()`/`tick_time`-resampled OHLC remains class C.

So the "no Model-2" and "all-contrarian" items in section 2 are deliberate practical substitutes made when the
book's interval law could not be reproduced, not accidental implementation errors; their classification stays
"implementation mismatch".

## 7. T0-B: one-factor readings of eq.(3) (Study Manager #903)

Code `backend/research/tdm_fig3_t0b.py`; Model-1 interval series T' (N=500, L=1, dt=0.01, c=[-0.01,0.01] "signed" and
[0,0.01] "abs", seeds 1-3, 20 000 trades) are reused; only the clock modulation changes, one factor at a time from the
T0-A baseline B0 (tau=150 s, <T> averages past modulated T, init <T>=mean(T'), alpha=1.25). Medians over the 6 series;
`vr20` = variance of 20-event block means relative to the shuffled series (>1 = clustering), `drift` = log(last/first
quarter mean T), `collapsed` = share of intervals below 1e-3 x mean(T').

| variant | vr20 | acf1 | drift | max/mean | collapsed |
|---|---|---|---|---|---|
| Model-1 clock (reference) | 0.88 | 0.00 | 0.00 | 6.5 | 0 |
| B0 literal (tau=150 s) | 0.97 | 0.00 | 0.04 | 9.5 | 0 |
| tau 500 s / 50 s | 0.90 / 1.59 | 0.00 / 0.03 | 0.01 / -0.03 | 9.6 / 9.7 | 0 |
| tau 15 s / 5 s | 5.1 / 6.9 | 0.22 / 0.35 | -0.38 / -0.61 | 21 / 53 | 0 / 0.001 |
| tau 150 ticks / 20 ticks | 2.1 / 12.6 | 0.06 / 0.59 | -0.58 / -48 | 11 / 230 | 0 / 0.90 |
| initial <T> x0.1 / x10 | 0.89 / 4.8 | 0.00 / 0.22 | 0.00 / -0.38 | 9.6 / 21 | 0.001 / 0 |
| empty-window rule (mean T') | 0.99 | 0.00 | 0.04 | 9.5 | 0 |
| <T> averaged over T' instead of T | 0.93 | 0.00 | 0.00 | 9.7 | 0 |
| alpha 1.0 / 1.5 | 0.91 / 1.05 | 0.00 / 0.01 | 0.00 / 0.03 | 6.5 / 13.9 | 0 |

Findings:
1. Clustering under literal eq.(3) is controlled by the number of events in the window, tau / mean(T): it appears for
   about <=30 events per window (tau 15 s, 5 s, 150 ticks) and grows as the window shrinks. "init x10" reproduces the
   tau=15 s numbers, i.e. the unspecified initial <T> (the time scale) acts as tau.
2. The same short-window regime is unstable: the level of T drifts down (E[F]=1 implies E[log F]<0, so the
   multiplicative clock contracts), reaching total collapse (90% of intervals below 1e-3 of the Model-1 scale at 20
   ticks) together with isolated very long intervals (max/mean 21-230). This is the "dense periods plus extremely long
   intervals" pattern recalled from the earlier failed attempt (that attempt's logs were not re-checked, so this is
   consistent with, not proof of, the earlier cause).
3. Averaging over the modulated T is necessary (averaging T' gives no clustering); alpha only changes the tail
   (1.0 no fat tail, 1.5 heavier), not the clustering; the empty-window rule is irrelevant for tau >= 5 s.
4. Not established: whether the paper's tau=150 s sits in the clustered regime. That needs the paper's time scale of T
   (mean tick interval in seconds); Fig.3 is only available as a plot and its mean interval is not stated. A
   stabilising ingredient (e.g. mean-reversion of <T> or a geometric-mean normalisation) is not in the paper and is not
   tested here.
5. Numerical note: a running-sum moving average goes negative once T collapses by ~14 orders of magnitude; the script
   re-sums short windows exactly.
