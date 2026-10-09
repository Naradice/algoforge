# FX strategy search (2026-10-09)

Harness: `backend/research/fx_search/` (vectorised, H1/D1, 9 HistData pairs 2000–2022, net of assumed spreads,
10%-vol-targeted per instrument, equal-weight across pairs). Run order: `prep.py` → `screen.py` → `wf.py`,
`ml.py` + `evalml.py`, `carry.py`, `more.py`, `combo.py`. Cached data lives in `artifacts/research/fx/` (not tracked).

Assumptions: spreads 0.6–2.5 pips/unit turnover (`core.SPREAD`, deliberately conservative); policy rates in `carry.py`
are **approximate annual averages entered from memory** (rank-order is reliable, levels are not); HistData is bid-only,
fixed EST; bad ticks removed in `core.load`.

## Results (net of cost, Sharpe)

| Idea | Result |
|---|---|
| H1 Donchian / MA-cross / TS-momentum / RSI / Bollinger-reversion, annual walk-forward param selection | −0.5 … +0.15 OOS (2004–22). No edge after costs. |
| Hour-of-day drift (trailing mean, t-stat gated) | Strongly negative net (turnover); not tradable. |
| Pooled HistGB on 15 vol-normalised features → 24h forward return, annual expanding walk-forward | IC ≈ +0.008 avg (range −0.018…+0.045); net SR −0.13…+0.03. No edge. |
| Carry (sign of policy-rate differential, monthly) | SR 0.42 full sample, **0.18 from 2009**: edge came from 2000–06 rate differentials. |
| Currency-strength momentum, weekly rebalance (L = 5/60/120/250d) | SR 0.15–0.27, positive in every L; reversal variants are negative. |
| **Momentum basket + carry, 50/50** | **SR 0.48 (0.29 from 2009), maxDD −10% of 10%-vol-per-leg units**, corr(mom,carry)=0.08 |

## Interpretation
- Price-only short-horizon prediction (technical rules and GBM) does not survive FX costs on this data.
- The only positive, low-turnover sources are slow ones: currency-level momentum and carry; they diversify each other,
  but the combined post-2009 Sharpe (~0.3) is modest and not far above what the multiple-testing noise allows.
- Not yet done: true out-of-sample beyond 2022 (MT5 / Dukascopy data exist), vol-timing of carry, real spread data
  from the OANDA tick files, replay of the sleeves through the AlgoForge backtest engine (engine has no cross-sectional
  or carry condition handler — would need a new handler).

## Round 2 — daily framework on MT5 D1 (2000–2025-05), true holdout 2023+
Code: `backend/research/fx_search/{d1,carry2,final}.py`. Rates extended to 2025 (approximate, from memory).
Design rules were fixed on ≤2022 data; 2023-01…2025-05 is a genuine holdout.

| Sleeve (net of spread, monthly/weekly rebal.) | SR all | SR 2009+ | SR 2023+ |
|---|---|---|---|
| carry, sign | 0.45 | 0.32 | 0.89 |
| carry, weight ∝ rate diff (clip ±1.5, thr 0.5%) | 0.52 | 0.40 | 0.87 |
| + vol guard (scale by 1y-median / 20d vol of book, cap 1) | **0.56** | 0.30 | 0.81 |
| currency momentum 60d+250d (weekly) | 0.37 | 0.29 | 0.42 |
| **0.65 carry* + 0.35 momentum (risk-parity)** | **0.62** | **0.39** | **0.81** |

Maxdrawdown −13% (10%-vol-per-leg units) vs −25% for plain carry. Robust to ±0.5% random noise on every
currency-year rate (20 draws: mean SR 0.59, worst 0.52 all-sample, 0.15 post-2009). Mom5/mom120 variants were
unstable (sign flips between neighbours) → only 60/250 used. Caveat: most of the all-sample carry P&L is 2001–06;
2008 carry crash (−14%) is the dominant risk; rates are approximate.

## Round 3 — real rates (FRED) replace approximate ones
`d1.py` now uses FRED 3-month interbank rates (IR3TIB01{US,EZ,GB,JP,AU,CH}M156N; JPY pre-2002 from IRSTCB01JPM156N),
1-month lag, daily forward-fill. Download: `https://fred.stlouisfed.org/graph/fredgraph.csv?id=<ID>` into
`artifacts/research/fx/rates/` (not tracked). Results barely moved, i.e. the conclusions do not rely on my rate guesses:

| | SR all | 2009+ | 2023+ | maxDD |
|---|---|---|---|---|
| carry, sign | 0.42 | 0.26 | 0.79 | −24% |
| carry*, diff-weighted + vol guard | 0.52 | 0.32 | 0.80 | −23% |
| **0.65 carry* + 0.35 mom60/250** | **0.58** | **0.41** | **0.80** | **−14%** |

Negative / rejected (`ext.py`): rate-change momentum (SR 0.08–0.14), cross-sectional carry top-2/bottom-2 (0.29 < sign carry),
daily GBM on carry/momentum/vol features (IC −0.075 pooled, net SR −0.5; no usable edge, sign-flipping it would be snooping).
Data end 2025-05 (MT5 D1); no later FX price data located locally.

## Round 4 — widening the search (all negative or not tradable)
Code: `backend/research/fx_search/{d2,scan,sm,m7,micro,final2,final3}.py`. d2 = 10 currencies / 30 MT5 D1 pairs (2007-05…2025-05),
real FRED rates incl. CAD/NZD/NOK/SEK, spread = 2× recent MT5 median.

- **Carry is regime-dependent, not a steady premium.** On the 2007+ window plain carry SR is ≈ −0.1 (2008 crash −36…−44% at 10% vol);
  2009+ SR 0.2–0.4, and that is concentrated in 2023–24 (+12%, +35% at 10% vol); 2010–2022 ≈ flat. The 0.5–0.6 all-sample SR of
  rounds 2–3 came largely from 2000–06. Wider universe (30 pairs) is worse than the 9 majors (SR 2009+ 0.21 vs 0.34; CHF crosses lose).
- VIX-jump guard (VIX/VIX₋₂₀ > 1.3 → size ×0.3) improves 9-pair SR 2009+ 0.34→0.40, 2023+ 0.80→1.02, DD −30%→−20%; faster vol scaling
  (max of 20/60/250d) trims 2008 only from −41% to −36%. Tail risk of carry is not removable with these tools.
- Currency momentum is **not robust**: positive on the 9 pairs (SR~0.3), negative on the 30-pair universe (60d: −0.3, 250d: −0.2 post-2009).
  Drop it from the recommended blend (or ≤ 20% weight).
- Rejected (train ≤2015 / test ≥2016): cross-asset lead-lag (64 instruments × 30 pairs × 1d/5d: 23 train-significant, 0 survive in test),
  calendar effects (dow/day-of-month/month: 2 train-significant, 1 weak survivor, AUDNZD Fri), pair-spread mean reversion
  (125 spreads, mean SR ≈ 0; walk-forward selection −0.07), value/long-term reversal (3y/5y: SR ≈ 0), momentum on carry-adjusted returns,
  monthly and daily GBM with carry/VIX/momentum features (IC consistently negative −0.07…−0.08 OOS → relations learned in-sample reverse).
- Tick microstructure (USDJPY Dukascopy M1 with OFI/imbalance, GBM walk-forward 2019–22): IC +0.02…0.03 real, but top-decile gross edge
  0.04–0.11 bp/trade vs ~0.86 bp spread → not tradable by taking liquidity.

## Conclusion of the search
Out of everything tested, the only repeatable positive-expectancy source is low-turnover carry on the major pairs, with a VIX/vol
guard; it is dominated by rare crash risk and its payoff depends on rate differentials being wide. There is no evidence of a
cost-surviving directional price-prediction edge in FX (technical rules, momentum, reversal, ML at M1/H1/D1/monthly, cross-asset, calendar).
