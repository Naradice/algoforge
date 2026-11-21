# Strategy Layer

## Concepts

A **Strategy** defines *what* to trade and *how* to decide. It is stored as a JSON definition and executed by the strategy engine against historical data (backtest), simulated live data (paper), or a real broker (live).

```
Strategy
├── symbol          e.g. "AAPL", "EURUSD=X"
├── indicators[]    technical indicators to compute each bar
├── entry           entry condition block (direction + conditions + logic)
├── exit            exit condition block (conditions + logic)
└── risk            position sizing and stop-loss / take-profit
```

---

## Strategy Definition Schema

```json
{
  "symbol": "AAPL",
  "indicators": [
    { "id": "macd", "type": "macd", "params": { "fast": 12, "slow": 26, "signal_period": 9 } },
    { "id": "rsi",  "type": "rsi",  "params": { "period": 14 } },
    { "id": "atr",  "type": "atr",  "params": { "period": 14 } }
  ],
  "entry": {
    "direction": "buy",
    "conditions": [
      { "left": "macd_line", "op": ">", "right": "macd_signal" },
      { "left": "rsi",       "op": "<", "right": 70 }
    ],
    "logic": "and"
  },
  "exit": {
    "conditions": [
      { "left": "macd_line", "op": "<", "right": "macd_signal" }
    ],
    "logic": "or"
  },
  "groups": {
    "trend_up": {
      "name": "Trend Up",
      "conditions": [
        { "left": "macd_line", "op": ">", "right": "macd_signal" },
        { "left": "renko_direction", "op": "==", "right": 1 }
      ],
      "logic": "and"
    }
  },
  "risk": {
    "sl_pct": 0.02,
    "tp_pct": 0.04,
    "position_size": 1.0,
    "trailing_stop": true,
    "trailing_atr_multiplier": 3.0,
    "trailing_clip_with_price": false,
    "trailing_only_in_profit": true
  }
}
```

---

## Indicators

Each indicator spec: `{ "id": "<column_name>", "type": "<type>", "params": {...} }`

The `id` becomes the column name referenced in conditions.

| Type | Output columns | Key params |
|------|---------------|------------|
| `macd` | `<id>_line`, `<id>_signal`, `<id>_hist` | `fast`, `slow`, `signal_period` |
| `rsi` | `<id>` | `period` |
| `atr` | `<id>` | `period` |
| `ema` | `<id>` | `period` |
| `sma` | `<id>` | `period` |
| `bb` | `<id>_upper`, `<id>_middle`, `<id>_lower` | `period`, `std_dev` |
| `slope` | `<id>` | `period`, `column` |
| `renko` | `<id>_direction`, `<id>_flip`, `<id>_momentum` | `atr_window`, `brick_size` |
| `event` | `<id>_near`, `<id>_mins_to_next`, `<id>_mins_since_last`, `<id>_surprise` | `calendar` (event_calendar artifact path), `currencies`, `min_impact`=high, `pre_minutes`=60, `post_minutes`=60, `title_contains`, `surprise_window`=24. Point-in-time: only events released ≤ bar time feed `surprise`. |
| `carry` | `<id>` (annual base-quote 3m rate differential), `<id>_w` (diff/`scale` clipped to ±`cap`, 0 inside ±`threshold`) | `base`, `quote`, `threshold`=0.005, `scale`=0.03, `cap`=1.5, `rates_dir` (FRED CSVs in `$ARTIFACT_STORE_PATH/research/fx/rates`). Add `risk.swap_col: "<id>"` to accrue it as swap P&L. |

All indicators also have access to the base OHLCV columns: `open`, `high`, `low`, `close`, `volume`.

---

## Condition Types

### Standard comparison
```json
{ "left": "rsi", "op": "<", "right": 30 }
{ "left": "macd_line", "op": ">", "right": "macd_signal" }
```
`left` and `right` can be column names (strings) or numeric literals. `op`: `<`, `>`, `<=`, `>=`, `==`, `!=`.

### Condition Groups

Groups let you name a reusable set of conditions and reference them by name inside any entry or exit block.

**Define** groups at the top level under `"groups"` (keyed by group id):
```json
"groups": {
  "trend_up": {
    "name": "Trend Up",
    "conditions": [
      { "left": "macd_line", "op": ">", "right": "macd_signal" },
      { "left": "renko_direction", "op": "==", "right": 1 }
    ],
    "logic": "and"
  }
}
```

**Reference** a group inside any condition array:
```json
{ "type": "group_ref", "group_id": "trend_up" }
```

A `group_ref` evaluates to the result of the entire group (True/False). Example — exit on group OR a range signal:
```json
"exit": {
  "conditions": [
    { "type": "group_ref", "group_id": "trend_up" },
    { "left": "rt_is_range", "op": "==", "right": 1 }
  ],
  "logic": "or"
}
```

Groups cannot nest other `group_ref` conditions.

### Streak
```json
{
  "type": "streak",
  "left": "macd_line",
  "op": ">",
  "right": "macd_signal",
  "min_streak": 3
}
```
True when the sub-condition (`left op right`) has been continuously True for **at least `min_streak` consecutive bars** up to and including the current bar. The count resets to 0 the moment the sub-condition is False.

Example timeline (MACD line > MACD signal, `min_streak: 3`):
```
bar 1:  True  → streak 1, condition False
bar 2:  True  → streak 2, condition False
bar 3:  True  → streak 3, condition True  ✓
bar 4:  False → streak 0, condition False
bar 5:  True  → streak 1, condition False
```

`left` and `right` follow the same rules as standard comparison (column name or numeric literal).

### ML signal
```json
{
  "type": "ml_signal",
  "model_id": 3,
  "direction": "buy",
  "step": 1,
  "min_confidence": 0.0
}
```
Calls `POST /models/{model_id}/predict` with the last N bars. Returns true when the predicted direction matches and confidence ≥ threshold.

### LLM signal
```json
{
  "type": "llm_signal",
  "direction": "buy",
  "lookback": 10,
  "model": "claude-sonnet-4-6",
  "columns": ["close", "volume", "macd_line", "rsi"],
  "cache": true
}
```
Sends recent bar data to an LLM and parses a buy/sell/hold response. `cache: true` avoids duplicate API calls for the same bar.

---

## Entry / Exit Blocks

Both entry and exit use the same structure:

```json
{
  "direction": "buy",
  "conditions": [ ...condition objects... ],
  "logic": "and"
}
```

- `direction`: `"buy"` or `"sell"` (entry only)
- `logic`: `"and"` (all must be true) | `"or"` (any must be true)
- `conditions`: array of condition objects — any mix of comparison, ml_signal, llm_signal

---

## Risk

```json
{
  "sl_pct": 0.02,
  "tp_pct": 0.04,
  "position_size": 1.0
}
```

| Field | Description |
|-------|-------------|
| `sl_pct` | Stop-loss as fraction of entry price (0.02 = 2%) |
| `tp_pct` | Take-profit as fraction of entry price |
| `position_size` | Fraction of equity or fixed lot size |
| `swap_col` | Optional indicator column holding the annual long-side carry (e.g. `carry`). Each trade accrues `side × volume × mean(rate) × days_held / 365` into its profit and the equity curve (buy = +rate, sell = −rate). Omitted = price-only P&L. |

---

## Execution Modes

### Backtest
- Replays historical OHLC data bar by bar from a Dataset
- Requires `dataset_id` pointing to a ready Dataset
- Produces: trades, equity curve, run metrics (win rate, Sharpe, max drawdown, etc.)

### Paper
- Subscribes to live yfinance price feed
- Executes trade logic but uses simulated order fills (no real money)
- Useful for forward-testing strategies in real-time

### Live *(not yet implemented)*
- Connects to a real broker (MT5, Coincheck)
- Requires broker credentials configured in Settings → Brokers
- Same condition logic as paper mode

---

## Run Lifecycle

```
POST /strategies/{id}/runs  →  status: pending
        ↓ arq enqueues execute_strategy_run
status: running  (publishes events via SSE)
        ↓ backtest completes / paper stopped / error
status: completed | stopped | error
```

After completion:
- `RunMetric` rows written (win_rate, sharpe_ratio, max_drawdown, profit_factor, total_trades, total_pnl)
- `Trade` rows written per trade
- `equity_curve` stored in `strategy_runs.equity_curve`

---

## Version History

Every time `definition` changes via `PATCH /strategies/{id}`, a `StrategyVersion` snapshot is created automatically. Use `GET /strategies/{id}/versions` to retrieve history. This allows comparing definition changes against run performance.

---

## Chat Interface

Each strategy has a WebSocket chat endpoint:

```
WS /api/v1/ws/strategies/{strategy_id}/runs/{run_id}/chat
```

Client sends: `{ "message": "Why is the win rate low?" }`
Server streams back: `{ "role": "agent", "content": "...", "is_final": true }`

The AI agent has access to: current run metrics, recent trades, and the strategy definition as context.


---

## Portfolio engine (multi-symbol, cross-sectional)

`strategy/portfolio/engine.py` — `run_portfolio_backtest(definition)` (also the MCP tool `run_portfolio_backtest`).
The single-symbol engine cannot express baskets; this one takes a universe of daily bars plus sleeves/overlays:

```json
{
  "universe": [{"symbol": "USDJPY", "artifact_path": "research/fx/d1_merged/USDJPY.parquet",
                "base": "USD", "quote": "JPY", "spread_pct": 0.00004}],
  "rates":    {"dir": "research/fx/rates", "series": {"USD": "IR3TIB01USM156N", "JPY": "IR3TIB01JPM156N"}},
  "sleeves":  [{"type": "carry", "weight": 0.65, "threshold": 0.005, "scale": 0.03, "cap": 1.5},
               {"type": "momentum", "weight": 0.35, "lookbacks": [60, 250]}],
  "overlays": [{"type": "vol_guard", "cap": 1.0},
               {"type": "series_jump", "path": "research/fx/rates/VIXCLS.csv", "column": "VIXCLS",
                "ratio": 1.3, "lookback": 20, "cut": 0.7}],
  "risk":     {"target_vol": 0.10, "rebalance": "M", "max_leverage": 5.0, "vol_window": 60}
}
```
Positions apply from t+1; carry accrues daily at (rate differential)/252 on the held position; costs are charged on
turnover. Returns are per-instrument fractions at `target_vol`, averaged over instruments available that day.
Reference run (9 majors, carry + VIX-jump guard, 2000→2026-10): Sharpe 0.65, 2009+ 0.50, max DD −20%
(`backend/research/fx_search/portfolio_check.py`). Not yet persisted as Strategy/Run rows or exposed over HTTP.

### Sleeves, overlays and risk keys (portfolio engine)

| Sleeve `type` | Meaning | Keys |
|---|---|---|
| `carry` | sign/size ∝ (base − quote) short-rate differential | `threshold`, `scale`, `cap`, `rebalance`, `net_of_markup` (trade only the part above `risk.swap_markup`) |
| `momentum` | currency-strength momentum from pair returns | `lookbacks`, `rebalance` (default W) |
| `scores` | external per-symbol scores (e.g. ML predictions) from parquet | `path`, `column`, `mode` (`sign`\|`rank`), `threshold`, `k`, `ffill` |

| Overlay `type` | Meaning |
|---|---|
| `vol_guard` | shrink when 20d book vol ≫ its 1y norm |
| `series_jump` | cut when a series (VIX…) rose more than `ratio` over `lookback` days |
| `series_below_ma` | cut while a series (S&P 500…) is below its `window`-day MA (risk-asset trend filter) |
| `agent_risk` | LLM risk officer (multiplier ∈ {0, 0.5, 1}); only for dates ≥ `min_date` |

`risk` keys: `target_vol`, `rebalance`, `max_leverage`, `vol_window`, `swap_markup` (annual haircut on gross exposure),
`max_gross`, `currency_cap` (per-currency net exposure, equity units), `dd_stop`/`dd_cut` (drawdown circuit breaker).
Result includes `stress` (book P&L in 2008, 2011, 2015 SNB, 2016 Brexit, 2020, 2024-08 windows) and `targets`
(latest signed weights). A strategy whose definition contains `universe` runs through this engine from the normal
`POST /strategies/{id}/runs` (modes `backtest` and `paper`; `paper` re-computes the current target book).

---

## Backtesting & Execution Anti-Patterns (Guidelines & Lessons Learned)

To prevent severe performance discrepancies between backtest and live execution, all strategy implementations must strictly adhere to the following quantitative engineering principles.

### 1. Prohibition of Intra-Bar Lookahead Bias (足内先読み・タイムトラベルの厳禁)
- **The Core Rule:** When an entry/exit signal is triggered on bar $i$ (close/evaluation), the execution **must occur at bar $i+1$ Open or later**.
- **No Same-Bar High/Low Settlement:** Never evaluate SL (Stop Loss) or TP (Take Profit) against the High/Low of the very same bar $i$ where the condition was detected. Doing so creates an artificial "wick/pinbar illusion," falsely assuming the trade caught the extreme wick of the candle and immediately exited with profit, inflating Win Rate and Profit Factor (e.g. producing false PF 2.80+ when real next-bar execution is PF < 1.0).
- **No Ad-Hoc Execution Loops:** Do not create separate "toy backtesters" (such as `analyst_strategy.py`). All strategies must integrate into the standard engine (`strategy/engine/backtest.py`), which natively enforces the `pending_entry -> next bar open fill -> subsequent bar SL/TP checks` execution lifecycle.

### 2. Strict Separation of Out-of-Sample Evaluation (学習内評価の禁止とパージング)
- **No Full-Retrained In-Sample Scoring:** Never evaluate or publish backtest results by scoring the training dataset with a model that was retrained on that entire dataset.
- **Walk-Forward with Purging:** ML-filtered strategies must be evaluated using strictly sequential Walk-Forward splits. Any training events whose label holding window overlaps with the start of the validation window must be purged to prevent label leakage.
- **Distinguish Filtering from Re-Simulation:** Filtering existing historical trades by score is an offline analytical study, not an execution backtest. An actual backtest must simulate daily quota preservation, subsequent intraday candidates upon rejection, and next-bar order executions.

### 3. Mandatory Fail-Closed Architecture for Live Runners (ライブ執行のフェイルクローズ徹底)
- **Whitelist-Only Order Approval:** When an ML filter or safety guard is enabled, orders may proceed **only if** inference succeeds and returns a strictly finite numeric probability that meets or exceeds the threshold (`math.isfinite(prob) and prob >= threshold`).
- **Default to Rejection (Fail-Closed):** Any anomalous state—including empty candles, missing data, API timeout, unhandled exceptions, `NaN`, or $\pm\infty$—must unconditionally cancel the order candidate (`dir_to_enter = None`). Never permit a "fail-open" path where an unverified signal defaults to execution.

