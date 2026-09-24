"""
Phase 5c-period-structure (user-requested): pure analysis, no new training runs. H2 (transfers,
2/3 seeds) and K1 (clean negative, 0/5 seeds) use the same generator (DDM + AR(1)-forced), the
same period COUNT (5), a similar range, and both sets happen to be all-prime -- yet H2 transfers
and K1 doesn't. This computes every mechanically-derivable feature of each period set (values,
frequencies, adjacent/pairwise differences, ratios, beat periods, distance to k*50 -- D1/D2's own
recurrence_lag) side by side, to find what actually differs before designing another training run.

Usage: python analyze_period_structure.py
"""

from __future__ import annotations

import itertools

import numpy as np

H2_PERIODS = [47.0, 71.0, 97.0, 127.0, 157.0]   # transfers, 2/3 seeds
K1_PERIODS = [53.0, 79.0, 103.0, 131.0, 149.0]  # clean negative, 0/5 seeds
RECURRENCE_SCALE = 50.0  # D1(Sine period=50)/D2(Delay, measured recurrence_lag=50)


def _adjacent_diffs(periods: list[float]) -> list[float]:
    return [b - a for a, b in zip(periods, periods[1:])]


def _pairwise(periods: list[float]):
    """All C(5,2)=10 unordered pairs, smaller period first."""
    return list(itertools.combinations(sorted(periods), 2))


def _distance_to_k_times_scale(period: float, scale: float, k_max: int = 4) -> tuple[int, float]:
    """Nearest k in [1, k_max] minimizing |period - k*scale|; returns (k, signed distance)."""
    ks = np.arange(1, k_max + 1)
    dists = period - ks * scale
    i = int(np.argmin(np.abs(dists)))
    return int(ks[i]), float(dists[i])


def analyze_set(name: str, periods: list[float]) -> None:
    periods = sorted(periods)
    freqs = [1.0 / p for p in periods]

    print(f"\n{'=' * 100}\n{name}: periods={periods}\n{'=' * 100}")

    print(f"\n-- period / frequency --")
    print(f"{'period':>10}{'freq (1/period)':>20}")
    for p, f in zip(periods, freqs):
        print(f"{p:>10.1f}{f:>20.6f}")

    print(f"\n-- adjacent differences (consecutive gaps) --")
    adj = _adjacent_diffs(periods)
    print(" ".join(f"{d:.1f}" for d in adj))

    print(f"\n-- distance to k x {RECURRENCE_SCALE:.0f} (D1/D2's own recurrence scale), k in 1..4 --")
    print(f"{'period':>10}{'nearest k':>12}{'k*scale':>10}{'signed dist':>14}")
    for p in periods:
        k, dist = _distance_to_k_times_scale(p, RECURRENCE_SCALE)
        print(f"{p:>10.1f}{k:>12}{k * RECURRENCE_SCALE:>10.0f}{dist:>14.1f}")

    print(f"\n-- all {len(_pairwise(periods))} pairwise features --")
    header = (f"{'pair':>14}{'period diff':>13}{'period ratio':>14}"
              f"{'freq diff':>12}{'freq ratio':>12}{'beat period':>13}")
    print(header)
    for p1, p2 in _pairwise(periods):
        f1, f2 = 1.0 / p1, 1.0 / p2
        period_diff = p2 - p1
        period_ratio = p2 / p1
        freq_diff = f1 - f2  # positive since p1 < p2 -> f1 > f2
        freq_ratio = f1 / f2
        beat_period = 1.0 / abs(f1 - f2) if freq_diff != 0 else float("inf")
        print(f"{f'{p1:.0f}-{p2:.0f}':>14}{period_diff:>13.1f}{period_ratio:>14.4f}"
              f"{freq_diff:>12.6f}{freq_ratio:>12.4f}{beat_period:>13.1f}")

    all_pairs = _pairwise(periods)
    period_diffs = [p2 - p1 for p1, p2 in all_pairs]
    beat_periods = [1.0 / abs(1.0 / p1 - 1.0 / p2) for p1, p2 in all_pairs]
    print(f"\n-- summary stats across all 10 pairs --")
    print(f"period diffs: min={min(period_diffs):.1f} max={max(period_diffs):.1f} "
          f"mean={np.mean(period_diffs):.1f} std={np.std(period_diffs):.1f} "
          f"n_distinct={len(set(round(d) for d in period_diffs))}/10")
    print(f"beat periods: min={min(beat_periods):.1f} max={max(beat_periods):.1f} "
          f"mean={np.mean(beat_periods):.1f} median={np.median(beat_periods):.1f}")


def compare() -> None:
    analyze_set("H2 (TRANSFERS, 2/3 seeds)", H2_PERIODS)
    analyze_set("K1 (CLEAN NEGATIVE, 0/5 seeds)", K1_PERIODS)

    print(f"\n{'#' * 100}\nDIRECT COMPARISON\n{'#' * 100}")

    h2_adj, k1_adj = _adjacent_diffs(sorted(H2_PERIODS)), _adjacent_diffs(sorted(K1_PERIODS))
    print(f"\nAdjacent diffs -- H2: {h2_adj} (repeats: 30,30)   K1: {k1_adj}")
    print(f"H2 adjacent-diff repeat check: {len(set(round(d) for d in h2_adj))} distinct values "
          f"out of {len(h2_adj)} -- {'HAS a repeated gap' if len(set(round(d) for d in h2_adj)) < len(h2_adj) else 'no repeat'}")
    print(f"K1 adjacent-diff repeat check: {len(set(round(d) for d in k1_adj))} distinct values "
          f"out of {len(k1_adj)} -- {'HAS a repeated gap' if len(set(round(d) for d in k1_adj)) < len(k1_adj) else 'no repeat'}")

    for name, periods in (("H2", H2_PERIODS), ("K1", K1_PERIODS)):
        pairs = _pairwise(sorted(periods))
        beat_periods = [1.0 / abs(1.0 / p1 - 1.0 / p2) for p1, p2 in pairs]
        n_long_beats = sum(1 for b in beat_periods if b > 1000)
        print(f"\n{name}: {n_long_beats}/10 pairs have beat period > 1000 bars "
              f"(near-commensurate / slow envelope) -- beat periods: "
              f"{sorted(round(b) for b in beat_periods)}")

    print(f"\nDistance-to-k*50 pattern (k in 1..3, i.e. 50/100/150):")
    for name, periods in (("H2", H2_PERIODS), ("K1", K1_PERIODS)):
        dists = [_distance_to_k_times_scale(p, RECURRENCE_SCALE, k_max=3)[1] for p in sorted(periods)]
        print(f"  {name}: signed distances = {[round(d, 1) for d in dists]}, "
              f"mean|dist|={np.mean(np.abs(dists)):.1f}")


if __name__ == "__main__":
    compare()
