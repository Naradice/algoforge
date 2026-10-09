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
