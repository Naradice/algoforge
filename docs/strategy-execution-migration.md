# Strategy Execution Migration Plan

## Goal

Retire `trade_strategy` as a runtime dependency and keep **AlgoForge** as the single strategy platform for backtest, paper trading, and live trading.

The migration target is **not** strict parity with legacy `trade_strategy` backtests. The target is a single execution model that is:

1. close to real trading behavior,
2. deterministic and explainable,
3. shared across backtest, paper, and live modes,
4. simple enough to maintain in one codebase.

## Why this change is needed

The current migration is incomplete because the same named strategy can still produce materially different results between:

- legacy `trade_strategy`,
- AlgoForge backtest,
- future paper/live execution in AlgoForge.

The biggest causes are:

- different indicator evaluation timing,
- different order fill timing,
- different slippage / commission assumptions,
- different stop-loss / take-profit defaults,
- different handling of trailing stops and range filters.

If AlgoForge becomes the only maintained platform, these differences must be removed inside AlgoForge rather than preserved from the legacy engine.

## Canonical execution model

AlgoForge should adopt one canonical model and reuse it everywhere.

### 1. Signal timing

- A strategy evaluates signals only from data available up to the current completed bar.
- No backtest logic may depend on future bars or indicator values calculated from future history.
- Indicator state must be equivalent between backtest replay and paper/live stepping.

### 2. Entry timing

- Signal detected on bar **N**
- Market entry filled on the first executable price of bar **N+1**

This is the default because it is safer and more realistic than same-bar close fills from backtests built on OHLC data.

### 3. Exit timing

- Stop-loss / take-profit orders are evaluated against the next bar's intrabar range.
- Signal-based exits are executed as market orders using the configured fill model.
- Trailing-stop updates occur in a deterministic order and must match paper/live behavior.

### 4. Fill model

One reusable execution component must own:

- spread / slippage,
- commissions / fees,
- stop-loss / take-profit fills,
- trailing stops,
- max position limits,
- cooldown and daily loss controls,
- end-of-data liquidation behavior.

Backtest and paper trading should both use this component. Live trading should keep the same decision flow and replace only the broker adapter.

### 5. Data fidelity modes

AlgoForge should support two realism tiers:

- **Bar mode**: conservative OHLC-based execution rules
- **Intrabar mode**: higher-fidelity fills when tick or lower-timeframe data exists

Bar mode remains the default because many datasets only contain OHLC candles.

## Migration principles

### Keep one strategy kernel

Signal logic must be shared across:

- backtest,
- paper,
- live.

There should not be one path that precomputes indicators for the full dataset and another path that updates indicators incrementally.

### Separate strategy from execution

Strategy code should answer:

- should we enter long?
- should we exit short?

Execution code should answer:

- when is the order filled?
- at what price?
- did SL or TP trigger first?
- how do fees and slippage change PnL?

### Remove hidden defaults

Risk and execution parameters must be explicit. In particular:

- `sl_pct`
- `tp_pct`
- `slippage_pct`
- `commission_pct`
- trailing-stop settings

Default UI values may exist, but engine behavior should remain transparent in stored strategy definitions and run metadata.

## Recommended rollout

### Phase 1: lock the target semantics

- document canonical entry / exit / stop / fee behavior,
- document bar-mode assumptions and limitations,
- use those rules as the reference contract for all later work.

### Phase 2: remove look-ahead style indicator behavior

- backtest must evaluate indicators only from `df[:i+1]`,
- condition context must use bar-local indicator state,
- this is the first priority because it directly affects signal correctness.

### Phase 3: introduce a shared execution simulator

- extract fills, stop logic, and PnL mechanics into one broker-simulator component,
- reuse it in backtest and paper mode,
- keep live mode compatible with the same order lifecycle.

### Phase 4: migrate legacy strategies through adapters

Provide translators for legacy concepts such as:

- `MACDRenko`,
- slope filters,
- range detection,
- trailing stops.

The goal is one-time migration of definitions, not permanent dual-engine support.

### Phase 5: add regression and parity fixtures

Create golden datasets and expected trades for a small set of reference strategies:

- `macdrenko`,
- one crossover trend strategy,
- one mean-reversion strategy.

Validate:

- backtest vs paper-sim parity,
- bar-mode determinism,
- known deltas from legacy `trade_strategy`.

## First implementation target

The first change should be:

**make AlgoForge backtests evaluate indicators incrementally instead of precomputing them across the full dataset before replay.**

This removes the largest source of non-realistic signal behavior and aligns backtest logic more closely with paper/live stepping.

## Definition of done for the migration

AlgoForge is the sole maintained engine when:

1. strategy signals are produced by one shared kernel,
2. backtest and paper trading share one execution simulator,
3. live trading uses the same order lifecycle contract,
4. core migrated strategies have golden regression coverage,
5. `trade_strategy` is no longer required for strategy execution.
