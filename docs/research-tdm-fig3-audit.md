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

## 8. T0-C/D/E: stationarity of the self-modulation clock, Fig.3 time scale, stabilised extension

Scripts: `backend/research/tdm_fig3_pdf_extract.py` (Fig.3 read-out), `tdm_fig3_t0c.py` (stationarity),
`tdm_fig3_t0c_excerpt.py` (excerpt plausibility). Model-1 intervals are rescaled to the paper's Fig.3 time scale and
bootstrapped (they are ~white) into iid `F_k = T'^1.25 / E[T'^1.25]`; `E[log F] = -0.389`, `Var(log F) = 0.957`, `CV(F) = 0.87`.

**Correction to section 3 / T0-A finding 3.** The "level frozen at an arbitrary start-up value" reading was wrong:
the 0.024x-1.7x level spread of T0-A is an early collapse (within the first quarter) combined with different Model-1
time scales per (c mode, seed), not a frozen initial value. Section 8 supersedes it.

### D. Which time scale is tau = 150 s? (Fig.3 read from the PDF drawing)

Fig.3 left draws each transaction as a vertical tick on a 0-500 s axis; the tick positions were read from the vector
drawing (`tdm_fig3_pdf_extract.py`, pixel read-out, no data table in the paper):

| | n | mean interval | CV | max/mean | ACF1 |
|---|---|---|---|---|---|
| paper model-1 excerpt | 69 | 7.3 s | 0.75 | 4.6 | -0.15 |
| paper model-2 excerpt | 50 | 9.2 s | 1.43 | 7.6 | +0.31 |

So tau = 150 s holds about **16 events**, inside the clustering regime of T0-B (<= ~30 events). My Model-1 intervals
(mean 0.8-1.7 in the unit dt = 0.01) were 4-9 times faster than the paper's 7.3 s, which put tau = 150 s at ~90 events
and explains the null result of T0-A as a **time-scale mismatch**, not a different equation. The unit conversion of the
paper's dt = 0.01 / c_i to seconds is not stated and is not resolved here. Caveats: one 450-500 s excerpt, 50-69 events,
pixel read-out; the 49-interval excerpt has an ACF1 standard error of ~0.14.

### C. Does the clock have a stationary interval level?

Decomposition: `log T_k = log F_k + log A_k`. `A` is the window mean of past T, and because `E F = 1` it is a
conditional martingale: it has no restoring force, so its fluctuations are permanent and its log drifts down (Jensen).
Measured on the paper model with a fixed n-event window (60 000 events, median of 6 runs; E1 in the JSON):

| n | ACF1 | log level change Q1 to Q4 |
|---|---|---|
| 8 / 16 / 40 | 0.62 / 0.60 / 0.53 | collapse to underflow in every run |
| 80 | 0.38 | -10.4 |
| 150 | 0.21 | -3.0 |
| 300 | 0.09 | -0.8 |

For n >= 80 the collapse scales as ~n^-1.9 while the clustering (ACF1) scales as ~n^-1, i.e. the per-event log drift is
about **-1.6e-3 x ACF1^2** (n = 80/150/300 give 1.6/1.5/2.2e-3). **In the paper-faithful model, clustering strength and
level collapse are not independent knobs; both are set by the window size.** A constant re-normalisation does not
separate them: for n = 150 the level falls at c = 1.00 (1/6 runs rise) and rises in 6/6 runs at c = 1.01; for n = 16 even
c = 1.03 still collapses (-74). The break-even c depends on the window size, which in the seconds-window model moves with
the level, so a constant c only creates an unstable equilibrium.

Paper model with a 150 s window (12 runs, geometric-mean interval, paper time scale, 100 000 events):

| events | 0-500 | 3000-6000 | 12000-24000 | 90000-100000 |
|---|---|---|---|---|
| geometric mean T (init 1x) | 2.7 | 4.2 | 0.96 | 0.27 |
| events per window | 38 | 38 | 158 | 398 |
| ACF1 | 0.23 | 0.72 | 0.38 | 0.05 |
| CV, max/mean | 0.96, 8.6 | 5.5, 143 | 1.4, 25 | 0.9, 12 |

The level falls ~10x over 10^5 events, the window fills up (38 to ~400 events) and the clustering fades; in between
there are bursts with CV 4-6 and max/mean 100-300. Initial levels 0.3x / 1x / 3x are forgotten by ~12 000 events (same
trajectory afterwards, with common random numbers), so the long-run level is a trend, not a fixed point. In short the
literal model produces transient clustering that decays with the level collapse; it is not a stationary interval process.

### E. Stabilised extensions (separate models, not a reproduction of the paper)

* **M2-S1** = paper clock + mean reversion in log (`A = exp((1-lam) log mean + lam log m0)`). lam = 0.01 holds the level
  (geometric mean 2.8-3.1 from event 12 000 to 100 000) with ACF1 0.7, but strongly heavy-tailed (CV 6-7, max/mean ~300);
  lam = 0.002 still decays (2.7 to 0.8); lam = 0.05 is stable and milder (ACF1 0.66, CV 2.7, max/mean 100).
* **M2-S2** = log-domain AR(n): `log T_k = log F_k + (1-lam) mean_n log T + lam log m0`, with `E log F = 0` (exactly
  linear, stationary iff lam > 0; 400 000 events x 8 runs):

| n, lam | ACF1(T) | CV | max/mean | Var(log T) | drift Q1 to Q4 | init offset remaining |
|---|---|---|---|---|---|---|
| 16, 0 | 0.56 | 13.8 | 1125 | 314 | -5.8 | 4.00 (permanent) |
| 16, 0.01 | 0.57 | 6.0 | 561 | 6.2 | 0.18 | 0 |
| 16, 0.1 | 0.35 | 1.32 | 36 | 1.4 | 0.02 | 0 |
| 50, 0.03 | 0.37 | 1.41 | 38 | 1.5 | 0.05 | 0 |

With lam > 0 the level is stationary and forgets its start while the clustering is kept (n = 16, lam = 0.1: ACF1 0.35,
CV 1.3 against the paper excerpt's 0.31 / 1.43). **Clustering and level stability can therefore be separated, but only
by ingredients that are not in the paper (a restoring force and a log-normalisation).** Conceptually, the empirical
statement that interval / moving-average interval is white noise concerns a real-market activity level that is driven
externally (time of day, news); inside the model the same level is endogenous and has a unit root (a conjecture, not
tested).

Plausibility of the paper's excerpt (`tdm_fig3_t0c_excerpt.py`; 49-interval excerpts; percentile rank of the observed
CV 1.43 / ACF1 0.31 / max-mean 7.6, and the share of excerpts within +-0.15 ACF1 and CV 0.8-1.25x of the observation):

| model | CV pct | ACF1 pct | max/mean pct | joint share |
|---|---|---|---|---|
| paper clock, run start, init 1x / 3x / 5x | 97 / 90 / 85 | 91 / 73 / 62 | 98 / 90 / 87 | 0.7% / 3.6% / 4.7% |
| paper clock, after level collapse | 100 | 99 | 99 | 0.2% |
| M2-S2 n=16, lam 0.03 / 0.1 | 100 | 95-96 | 98 | 1.5% / 1.3% |

The paper's excerpt is a rare draw under every model (it starts with intervals of 47, 42 and 70 s). It is most compatible
with the start-up transient of the literal clock with a high initial level and least with the stationary extensions,
but a single excerpt cannot discriminate.

### Conclusions and open questions

1. Eq.(3) alone does not generate a stationary interval process; clustering and collapse are coupled (collapse
   rate ~ ACF1^2).
2. tau = 150 s is a clustering-regime window (~16 events) at the paper's time scale.
3. Stabilised variants (restoring force + log normalisation) separate the two but must be reported as models M2-S1/S2.

Open: the paper's dt / c_i to seconds conversion; whether the Model-4 results (volatility clustering, Fig.5) depend on
the interval clustering; whether the initial level of the paper's run was above its long-run level; the effect of
replacing the repository's i.i.d. clock by M2-S2 on OHLC-based results (class C, T1).

## 9. T0-F: seconds conversion of dt and c_i (Study Manager #886)

Script `backend/research/tdm_fig3_time_scale.py` (PDF read-out; `--sweep` for the Model-1 parameter sweep).

**What the paper's own time axes say** (pixel read-out of the vector drawings, no data tables):

| source | mean interval | CV | ACF1 |
|---|---|---|---|
| Fig.2 real USD/JPY, 62 ticks in a 460 s excerpt | 7.5 s | 1.18 | +0.23 |
| Fig.5F real data (area under the complementary CDF) | ~8.0 s | - | - |
| Fig.3 model-1 excerpt (t' axis) | 7.3 s | 0.75 | -0.15 |
| Fig.3 model-2 excerpt | 9.2 s | 1.43 | +0.31 |
| Fig.5F model-4 | ~6.7 s | - | - |

All of them sit at the real-data scale of about 7-9 s.

**Literal reading** (dt = seconds per step, c_i per second, `Delta b = sigma c dt`, c_i signed), Model-1 of
`tdm_fig3_audit.model1_intervals`, mean of 3 seeds x 6000 trades:

| N | c_max | dt | mean T' | CV | ACF1 |
|---|---|---|---|---|---|
| 100 | 0.010 | 0.01 | **8.3 s** | 0.67 | -0.03 |
| 100 | 0.015 | 0.01 | 5.5 s | 0.66 | -0.04 |
| 150 | 0.010 | 0.01 | 5.6 s | 0.68 | -0.03 |
| 200 | 0.010 | 0.01 | 4.0 s | 0.68 | -0.02 |
| **500** | **0.010** | 0.01 | **1.65 s** (Fig.3 caption) | 0.69 | -0.01 |
| 100 | 0.02 | 1.0 | 4.3 s (2 seeds, 4000 trades, earlier run) | - | - |

Findings:
1. The mean interval scales as `T' ~ 8.3 s / (N c_max)` at dt = 0.01 (and as `steps x dt` in general; the number of
   steps per trade scales as `1/(N c dt)`: 8.2 vs 8.6 for the Fig.3 and Fig.5 parameter sets). So **the seconds scale
   is fixed by N x c_max, and dt only converts steps to seconds**.
2. With the Fig.3 caption (N = 500) the literal reading gives 1.65 s, 4.4x faster than the 7.3 s drawn in the figure
   (9x if c_i is read as |c|). The literal reading reproduces the drawn 7.3 s and the real-data scale for
   **N x c_max ~ 1.1, e.g. N = 100, c = 0.01 (8.3 s)** - the N of Fig.1 and Fig.5. Either the Fig.3 caption's N = 500
   is wrong (N = 100 would fit) or the figure's time axis was calibrated to the real data (mean ~7.5 s) rather than
   given by the stated dt. The text cannot distinguish the two; both give the same practical rule below.
3. CV (0.66-0.69) and ACF1 (about -0.03) of Model-1 do not depend on N or c_max (the drawn excerpt: 0.75 / -0.15, within
   the sampling error of 69 events), so the time-scale choice does not change the statistics of T', only their unit.
4. Model-2/4 intervals cannot be used for this calibration: the level of the modulated clock is not fixed by the
   dynamics (section 8-C), so the 9.2 s / 6.7 s of model-2/4 only show that the authors' level is near the real data.

**Practical rule used from here on:** keep dt = 0.01 and c_i signed U[-0.01, 0.01], but take N = 100 (mean T' ~ 8 s, the
paper's real-data scale) as the Fig.3 setting for time-scale-sensitive work, or equivalently rescale a Model-1 series to a
mean of 7.3 s (what sections 7-8 did; their conclusions do not depend on N because CV and ACF1 of T' are N-independent).
A literal N = 500 run must not be combined with tau = 150 s without this rescaling (it puts ~90 events in the window and
removes the clustering, which is the T0-A null result).

Not resolved: whether the caption N = 500 is a typo; the paper does not give the initial b_i distribution or the number
of steps.
