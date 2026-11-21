"""Fixed-grid Strategy 7 research replay on cached FX168 levels and M30 quotes."""
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load():
    bars = pd.read_parquet(ROOT / "artifacts/datasets/USDJPY_M30_latest.parquet")
    bars.index = pd.to_datetime(bars.index, utc=True)
    bars = bars.sort_index()
    levels = pd.read_csv(ROOT.parent.parent / "algoforge-live-research/strategy7-fx168-levels.csv")
    levels = levels.rename(columns={"report_date": "date"})
    levels = levels[levels.horizon.eq("Daily")].dropna(subset=["expected_low", "expected_high"])
    if levels.date.duplicated().any() or (levels.expected_low >= levels.expected_high).any():
        raise ValueError("Ambiguous or invalid FX168 levels")
    levels = levels.drop_duplicates("date", keep="last").set_index("date")
    bars["day"] = bars.index.strftime("%Y-%m-%d")
    bars = bars.join(levels[["expected_low", "expected_high"]], on="day")
    drift = bars.close.diff()
    bars["efficiency"] = bars.close.diff(24).abs() / drift.abs().rolling(24).sum()
    bars["vol_ratio"] = drift.abs().rolling(12).mean() / drift.abs().rolling(96).mean()
    bars["minute"] = bars.index.hour * 60 + bars.index.minute
    return bars[(bars.day >= levels.index.min()) & (bars.day <= levels.index.max())]


def replay(frame, mode, hold, regime, offset, sl, tp, cost, *, details=False, standard=False):
    position = pending = None
    used = set()
    results = []
    ledger = []
    retest = None
    for i, row in enumerate(frame.itertuples()):
        o, h, l, c = row.open, row.high, row.low, row.close
        if pending is not None and position is None:
            signal_i, day, side = pending
            if day == row.day and row.minute < 1410 and (
                row.Index - frame.index[signal_i] == pd.Timedelta(minutes=30)
            ):
                position = (i, side, o, row.Index)
                used.add((day, side))
            pending = None

        valid = 30 <= row.minute < 1410 and np.isfinite(row.expected_low + row.expected_high)
        ranged = regime == "all" or (
            previous is not None and (
                previous.efficiency <= .30 if regime == "low_efficiency" else previous.vol_ratio <= .85
            )
        ) if i else regime == "all"

        # Resting orders are placed before the bar, with the published bounds and
        # prior-bar regime. Skip bars that touch both sides without a known order.
        if retest and (row.day != retest[0] or i > retest[3]):
            retest = None
        if position is None and valid and ranged and (mode == "limit" or retest):
            buy = row.expected_low + offset / 100
            sell = row.expected_high - offset / 100
            buy_hit = l < buy and (row.day, 1) not in used
            sell_hit = h > sell and (row.day, -1) not in used
            if retest:
                buy_hit &= retest[1] == 1
                sell_hit &= retest[1] == -1
            if buy < sell and buy_hit != sell_hit:
                side, price = (1, buy) if buy_hit else (-1, sell)
                position = (i, side, price, row.Index)
                used.add((row.day, side))
                retest = None

        if position is not None:
            entry_i, side, entry, entry_time = position
            stop = entry - side * sl / 100
            target = entry + side * tp / 100
            session_end = row.minute >= 1410 or row.day != entry_time.strftime("%Y-%m-%d")
            day_close = hold == 0 and session_end
            if isinstance(hold, str) and (row.minute == 1410 or (i and previous.day != row.day and previous.minute < 1410)):
                inside = row.expected_low < o < row.expected_high
                favorable = side * (o - entry) > 0
                room = (row.expected_high - o if side == 1 else o - row.expected_low) * 100
                day_close = not inside or (hold == "profit" and not favorable) or (hold == "room" and room < tp / 2)
            elapsed = hold != 0 and row.Index - entry_time >= pd.Timedelta(days=2 if isinstance(hold, str) else hold)
            stop_hit = l <= stop if side == 1 else h >= stop
            tp_hit = h >= target if side == 1 else l <= target
            exit_price = None
            if day_close or elapsed:
                exit_price = o
            elif standard and tp_hit:
                exit_price = target
            elif stop_hit:
                # On a limit fill bar the open may precede the fill.
                exit_price = stop if i == entry_i and mode in ("limit", "retest") else (min(o, stop) if side == 1 else max(o, stop))
            elif tp_hit and (mode not in ("limit", "retest") or i != entry_i):
                exit_price = target
            elif i == len(frame) - 1:
                exit_price = c
            if exit_price is not None:
                net = (exit_price - entry) * side * 100 - cost
                results.append(net)
                if details:
                    ledger.append((entry_time, row.Index, side, entry, exit_price, net))
                position = None

        if position is None and retest is None and valid and ranged and mode in ("market", "reclaim", "retest"):
            if (mode == "market" or c > row.expected_low >= l) and l <= row.expected_low + offset / 100 and (row.day, 1) not in used:
                if mode == "retest":
                    retest = (row.day, 1, i, i + 4)
                else:
                    pending = (i, row.day, 1)
            elif (mode == "market" or c < row.expected_high <= h) and h >= row.expected_high - offset / 100 and (row.day, -1) not in used:
                if mode == "retest":
                    retest = (row.day, -1, i, i + 4)
                else:
                    pending = (i, row.day, -1)
        previous = row
    if details:
        return pd.DataFrame(ledger, columns=["entry_time", "exit_time", "side", "entry", "exit", "pnl"])
    pnl = np.asarray(results)
    equity = np.r_[0., pnl.cumsum()]
    return dict(trades=len(pnl), net=round(float(pnl.sum()), 3),
                avg=round(float(pnl.mean()), 4) if pnl.size else 0,
                dd=round(float((np.maximum.accumulate(equity) - equity).max()), 3))


def main():
    bars = load()
    days = bars.loc[bars.expected_low.notna(), "day"].unique()
    if len(days) < 30:
        raise ValueError(f"Only {len(days)} usable days")
    cutoff = days[int(len(days) * .7)]
    train, test = bars[bars.day < cutoff], bars[bars.day >= cutoff]
    print(f"Usable FX168 days: {len(days)}", flush=True)
    grid = list(product(("market", "limit", "reclaim"), (0, 1, 2, 3), ("all", "low_efficiency", "quiet"),
                        (0, 5, 10), ((15, 40), (20, 20), (30, 40))))
    rows = []
    for mode, hold, regime, offset, (sl, tp) in grid:
        config = (mode, hold, regime, offset, sl, tp)
        fit = replay(train, *config, 1.2)
        rows.append(dict(zip(("mode", "hold", "regime", "offset", "sl", "tp"), config), **fit))
    results = pd.DataFrame(rows)
    path = ROOT.parent.parent / "algoforge-live-research"
    results.to_csv(path / "strategy7-feed-exploration.csv", index=False)
    viable = results[results.trades >= 20].sort_values(["avg", "net"], ascending=False)
    columns = ["mode", "hold", "regime", "offset", "sl", "tp"]
    baseline = ("market", 0, "all", 10, 15, 40)
    print(f"days={len(days)}, first={days[0]}, holdout_start={cutoff}, last={days[-1]}, grid={len(grid)}")
    print(viable.head(8).to_string(index=False))
    print("baseline", replay(train, *baseline, 1.2), replay(test, *baseline, 1.2))
    if not viable.empty and viable.iloc[0]["avg"] > 0:
        best = tuple(viable.iloc[0][columns])
        print("selected", best)
        for cost in (0.6, 1.2, 2.4):
            print("holdout", cost, replay(test, *best, cost))


if __name__ == "__main__":
    main()
