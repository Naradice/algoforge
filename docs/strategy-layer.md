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
  "risk": {
    "sl_pct": 0.02,
    "tp_pct": 0.04,
    "position_size": 1.0
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

All indicators also have access to the base OHLCV columns: `open`, `high`, `low`, `close`, `volume`.

---

## Condition Types

### Standard comparison
```json
{ "left": "rsi", "op": "<", "right": 30 }
{ "left": "macd_line", "op": ">", "right": "macd_signal" }
```
`left` and `right` can be column names (strings) or numeric literals. `op`: `<`, `>`, `<=`, `>=`, `==`, `!=`.

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
