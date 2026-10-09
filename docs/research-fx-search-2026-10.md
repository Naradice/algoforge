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
