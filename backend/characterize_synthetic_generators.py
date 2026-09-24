"""
Phase 5d-recheck (user-requested): re-run the synthetic-generator characterization now that the
FULL investigation has concluded -- sine (D1) and delay (D2) are the ONLY generators that
transfer (consistently, 3/3 seeds, across pretrain-seed replications); ar1 (G1), lorenz (H1),
ar1_forced/H2 (initially looked unstable-positive at 2/3, but a pretrain-seed replication came
back a clean 0/3 -- confirmed a fluke, not a reproducible effect), xor (D3), and lfsr (D4) all do
NOT transfer. Every "structural proxy" tried for ar1_forced's periods (count, specific values,
gap pattern) also failed to reproduce transfer, so ar1_forced is now treated as a NEGATIVE
control alongside ar1/lorenz/xor/lfsr, not a partial positive.

This is a return to basics per the user's explicit request: rather than inventing new forcing
signals, lay out every measurable property of the generators we ALREADY have data for and look
for whichever one (if any) is shared by EXACTLY {sine, delay} and absent from every negative
control {ar1, lorenz, ar1_forced, xor, lfsr}.

Goal: find which measurable quantity is shared by EXACTLY the generators that transfer and
absent from the ones that don't, turning the "bounded + deterministic + fixed characteristic
timescale" hypothesis into something measured rather than an impression.

Axes (per the user's list):
  - autocorrelation decay time: first lag where |ACF(lag)| drops below 1/e
  - dominant frequency / spectral peak: periodogram peak frequency (as a period, in bars) and
    spectral concentration (top-peak power / total power) -- "peakiness" of the spectrum
  - characteristic recurrence time: lag of the first significant secondary ACF peak
    (scipy.signal.find_peaks on the ACF beyond lag 0) -- None if no significant peak exists,
    which is itself the discriminating signal for "no fixed recurrence timescale"
  - boundedness: max|x-mean|/std (bounded range in std units) and a variance-stationarity ratio
    (first-half var / second-half var) -- close to 1 and O(1) range both indicate a stationary,
    bounded process (true of all these generators by construction, so this axis may NOT
    discriminate -- worth measuring precisely because a non-discriminating axis rules itself out)
  - Lyapunov exponent: largest exponent via Rosenstein's method (time-delay embedding + nearest-
    neighbor divergence tracking) -- positive means sensitive dependence / chaos, ~0 or negative
    means non-chaotic (periodic, or a contracting linear recurrence)
  - entropy / predictability: normalized permutation entropy (Bandt-Pompe, order 5) in [0, 1],
    1 = maximally unpredictable ordinal complexity, 0 = fully ordered

Usage: python characterize_synthetic_generators.py
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, periodogram
from statsmodels.tsa.stattools import acf

from data.collectors.synthetic_function import (
    _ar1, _ar1_forced, _lfsr, _lorenz, _mackey_glass, _temporal_xor,
)

LENGTH = 60_000  # matches ABLATION_COMPONENT_ROWS -- the actual per-component row count used
                  # in every DDM+X mixture dataset in this investigation

GENERATORS = {
    # name -> (array, transfer_status) -- transfer_status from the FULL, now-complete
    # investigation (D1-D4, G1, H1-H2, and the H2 pretrain-seed replication that confirmed H2's
    # apparent 2/3 transfer was a checkpoint-specific fluke, not reproducible).
    "sine": (lambda: np.sin(2 * np.pi * np.arange(LENGTH) / 50.0), "TRANSFERS (D1: 0.507, 3/3)"),
    "delay": (lambda: _mackey_glass(LENGTH, tau=17), "TRANSFERS (D2: 0.516, 3/3)"),
    "xor": (lambda: 2 * _temporal_xor(LENGTH, seed=2003) - 1, "no effect (D3: 0/3)"),
    "lfsr": (lambda: 2 * _lfsr(LENGTH, bits=8, seed=2004) - 1, "no effect (D4: 0/3)"),
    "ar1": (lambda: _ar1(LENGTH, phi=0.98, sigma=1.0, seed=2005), "no effect (G1: 0.825, 0/3)"),
    "lorenz": (lambda: _lorenz(LENGTH), "no effect (H1: 0.826, 0/3)"),
    "ar1_forced": (
        lambda: _ar1_forced(LENGTH, phi=0.98, amplitude=1.0),
        "no effect (H2: FLUKE -- 2/3 on pretrain_seed=42 did not replicate, 0/3 on seed=99)",
    ),
}


def acf_decay_time(x: np.ndarray, max_lag: int = 2000) -> int | None:
    a = acf(x, nlags=max_lag, fft=True)
    below = np.where(np.abs(a) < 1 / np.e)[0]
    return int(below[0]) if len(below) else None


def spectral_peak(x: np.ndarray) -> tuple[float, float]:
    """Returns (dominant_period_in_bars, spectral_concentration). Concentration = power of the
    dominant bin (+/-1 neighbor) / total power, excluding the DC (freq=0) bin."""
    freqs, power = periodogram(x - x.mean(), fs=1.0)
    power = power[1:]
    freqs = freqs[1:]  # drop DC
    peak_idx = int(np.argmax(power))
    dominant_period = 1.0 / freqs[peak_idx] if freqs[peak_idx] > 0 else float("inf")
    lo, hi = max(0, peak_idx - 1), min(len(power), peak_idx + 2)
    concentration = power[lo:hi].sum() / power.sum()
    return dominant_period, concentration


def recurrence_time(x: np.ndarray, max_lag: int = 2000) -> tuple[int | None, float | None]:
    """First significant secondary ACF peak beyond lag 0. 'Significant' = prominence >= 0.1
    (ACF units) and height >= 0.15 -- thresholds chosen to reject noise ripple but catch a real
    periodic/quasi-periodic recurrence. Returns (lag, peak_height); (None, None) if none found."""
    a = acf(x, nlags=max_lag, fft=True)
    peaks, props = find_peaks(a[1:], prominence=0.1, height=0.15)
    if len(peaks) == 0:
        return None, None
    first = int(np.argmin(peaks))
    return int(peaks[first]) + 1, float(props["peak_heights"][first])


def boundedness_stats(x: np.ndarray) -> tuple[float, float]:
    mean, std = x.mean(), x.std()
    range_in_std = (x.max() - x.min()) / std if std > 0 else float("nan")
    half = len(x) // 2
    var1, var2 = x[:half].var(), x[half:].var()
    stationarity_ratio = min(var1, var2) / max(var1, var2) if max(var1, var2) > 0 else float("nan")
    return range_in_std, stationarity_ratio


def lyapunov_rosenstein(
    x: np.ndarray, m: int = 5, emb_tau: int | None = None,
    theiler: int = 50, max_k: int = 100, n_ref: int = 2000, seed: int = 0,
) -> float | None:
    """Largest Lyapunov exponent via Rosenstein et al. (1993). Time-delay embed into R^m with
    delay emb_tau (default: the series' own ACF 1/e decay time, floored at 1), find each
    reference point's nearest neighbor outside a Theiler window (to avoid temporally-correlated
    "neighbors"), track mean log divergence d_i(k) = ||X_{i+k} - X_{j(i)+k}|| for k=0..max_k,
    and take the slope of the linear region of <ln d(k)> vs k as the per-step exponent. Returns
    None if too few valid neighbor pairs survive to fit a slope."""
    if emb_tau is None:
        emb_tau = max(1, acf_decay_time(x, max_lag=200) or 1)
    n = len(x) - (m - 1) * emb_tau
    if n < 2 * theiler + max_k + 10:
        return None
    embedded = np.empty((n, m))
    for i in range(m):
        embedded[:, i] = x[i * emb_tau : i * emb_tau + n]

    rng = np.random.default_rng(seed)
    ref_idx = rng.choice(np.arange(n - max_k), size=min(n_ref, n - max_k), replace=False)

    log_divs = np.zeros(max_k + 1)
    counts = np.zeros(max_k + 1)
    for i in ref_idx:
        dists = np.linalg.norm(embedded - embedded[i], axis=1)
        dists[max(0, i - theiler) : i + theiler + 1] = np.inf
        j = int(np.argmin(dists))
        if not np.isfinite(dists[j]) or j >= n - max_k or i >= n - max_k:
            continue
        for k in range(max_k + 1):
            d = np.linalg.norm(embedded[i + k] - embedded[j + k])
            if d > 0:
                log_divs[k] += np.log(d)
                counts[k] += 1

    valid = counts > n_ref * 0.3
    if valid.sum() < 10:
        return None
    mean_log_div = np.full(max_k + 1, np.nan)
    mean_log_div[valid] = log_divs[valid] / counts[valid]
    ks = np.arange(max_k + 1)[valid]
    ys = mean_log_div[valid]
    # linear region: fit over the first half of the valid range (early-time divergence, before
    # saturation from the system's finite attractor size dominates)
    cut = max(5, len(ks) // 2)
    slope, _ = np.polyfit(ks[:cut], ys[:cut], 1)
    return float(slope)


def permutation_entropy(x: np.ndarray, m: int = 5, delay: int = 1) -> float:
    """Bandt-Pompe permutation entropy, normalized to [0, 1] by log(m!)."""
    n = len(x) - (m - 1) * delay
    patterns: dict[tuple, int] = {}
    for i in range(n):
        window = x[i : i + m * delay : delay]
        pattern = tuple(np.argsort(window))
        patterns[pattern] = patterns.get(pattern, 0) + 1
    counts = np.array(list(patterns.values()), dtype=np.float64)
    probs = counts / counts.sum()
    h = -np.sum(probs * np.log(probs))
    from math import factorial, log
    return float(h / log(factorial(m)))


def main() -> None:
    rows = []
    for name, (gen_fn, status) in GENERATORS.items():
        print(f"Generating + analyzing '{name}'...")
        x = gen_fn()
        decay = acf_decay_time(x)
        dom_period, concentration = spectral_peak(x)
        rec_lag, rec_height = recurrence_time(x)
        range_std, stationarity = boundedness_stats(x)
        lyap = lyapunov_rosenstein(x)
        pent = permutation_entropy(x)
        rows.append({
            "name": name, "status": status,
            "acf_decay_lag": decay,
            "dominant_period": round(dom_period, 1) if np.isfinite(dom_period) else None,
            "spectral_concentration": round(concentration, 4),
            "recurrence_lag": rec_lag,
            "recurrence_height": round(rec_height, 3) if rec_height is not None else None,
            "range_in_std": round(range_std, 2),
            "var_stationarity_ratio": round(stationarity, 3),
            "lyapunov_per_step": round(lyap, 5) if lyap is not None else None,
            "perm_entropy": round(pent, 4),
        })

    print("\n" + "=" * 140)
    header = (f"{'name':<12}{'status':<24}{'acf_decay':>10}{'dom_period':>12}{'spec_conc':>11}"
              f"{'rec_lag':>9}{'rec_height':>11}{'range/std':>11}{'stationarity':>13}"
              f"{'lyapunov':>11}{'perm_entropy':>13}")
    print(header)
    print("-" * 140)
    for r in rows:
        print(f"{r['name']:<12}{r['status']:<24}{str(r['acf_decay_lag']):>10}"
              f"{str(r['dominant_period']):>12}{r['spectral_concentration']:>11}"
              f"{str(r['recurrence_lag']):>9}{str(r['recurrence_height']):>11}"
              f"{r['range_in_std']:>11}{r['var_stationarity_ratio']:>13}"
              f"{str(r['lyapunov_per_step']):>11}{r['perm_entropy']:>13}")
    print("=" * 140)

    import json
    with open("synthetic_generator_characteristics.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("\nSaved synthetic_generator_characteristics.json")

    # -- discrimination check: does each axis cleanly separate {sine, delay} from every
    # negative control? --
    positives = {"sine", "delay"}
    by_name = {r["name"]: r for r in rows}
    negatives = [n for n in by_name if n not in positives]

    print("\n" + "=" * 100)
    print("DISCRIMINATION CHECK -- does this axis separate {sine, delay} from ALL negatives?")
    print("=" * 100)

    def _report(axis: str, present_fn) -> None:
        pos_vals = {n: present_fn(by_name[n]) for n in positives}
        neg_vals = {n: present_fn(by_name[n]) for n in negatives}
        pos_all_true = all(pos_vals.values())
        neg_all_false = not any(neg_vals.values())
        verdict = "CLEAN DISCRIMINATOR" if (pos_all_true and neg_all_false) else "does not discriminate"
        print(f"\n{axis}: {verdict}")
        print(f"  positives: {pos_vals}")
        print(f"  negatives: {neg_vals}")

    _report("has a recurrence_lag (ACF secondary peak found)", lambda r: r["recurrence_lag"] is not None)
    _report("acf_decay_lag <= 15 (fast e-folding decay)", lambda r: (r["acf_decay_lag"] or 999) <= 15)
    _report("spectral_concentration >= 0.3 (peaky spectrum)", lambda r: r["spectral_concentration"] >= 0.3)
    _report("perm_entropy <= 0.4 (locally ordered)", lambda r: r["perm_entropy"] <= 0.4)
    _report("lyapunov <= 0.01 (non-chaotic-ish)", lambda r: (r["lyapunov_per_step"] or 999) <= 0.01)
    _report("range_in_std <= 5 (tightly bounded)", lambda r: r["range_in_std"] <= 5)


if __name__ == "__main__":
    main()
