"""
Synthetic function collector — generates simple, deterministic (or pseudo-random-but-simple)
time series from closed-form formulas/recurrences, for the "dataset axis" of the mechanism-hunt
methodology (see docs/model-layer.md's "Comparing training runs" section): the same tokenization
and training harness applied across generative rules of deliberately different character, to
check whether a finding (an effective representation, a scaling effect) generalizes beyond one
signal's particular structure or was an artifact of it.

Datasource config shape (stored in datasources.config):
    {
        "function": "sine" | "sine_sum" | "delay" | "xor" | "lfsr" | "ar1" | "lorenz" | "ar1_forced",
        "period": 50,        # T -- base period, in bars (sine / sine_sum only)
        "amplitude": 1.0,    # A -- wave amplitude (sine/sine_sum/xor/lfsr; ignored by delay/ar1)
        "freq_ratio": 5,     # sine_sum only -- 2nd wave oscillates this many times faster than the base
        "tau": 17,           # delay only -- Mackey-Glass delay parameter (see below)
        "stride": 1,         # delay only -- keep every stride-th recurrence step (time-rescaling:
                              # multiplies the per-bar Lyapunov exponent by ~stride)
        "lorenz_dt": 0.02,   # lorenz only -- RK4 step per bar (time-rescaling: per-bar Lyapunov
                              # exponent ~= 0.905 * lorenz_dt, attractor geometry unchanged)
        "lfsr_bits": 8,      # lfsr only -- shift-register width; supported: 4, 5, 8, 16
        "ar_phi": 0.98,      # ar1 / ar1_forced -- AR(1) persistence coefficient (see below)
        "ar_sigma": 1.0,     # ar1 only -- AR(1) innovation std dev
        "forced_periods": None,  # ar1_forced only -- list of forcing periods, overrides the
                              # default 5-period mixture (e.g. [140.0] for a single clean
                              # characteristic timescale)
        "base_price": 100.0, # vertical offset so the series looks like a price series
        "noise": 0.0,        # gaussian noise std dev added on top; 0 = pure deterministic
        "length": 2000,      # number of bars to generate
        "timeframe": "M5",   # bar spacing
        "seed": 42,          # RNG seed -- noise (all functions), bit generation (xor), initial
                              # register state (lfsr), innovations (ar1). Unused (irrelevant) for
                              # sine/sine_sum/delay, which are fully deterministic from their
                              # formula alone.
        "start_ts": "2024-01-01",  # first bar timestamp
    }

Formulas (t = bar index, 0..length-1):
    "sine":     x_t = base_price + amplitude * sin(2*pi * t / period)
    "sine_sum": x_t = base_price + sin(2*pi * t / period) + amplitude * sin(2*pi * freq_ratio * t / period)
    "delay":    x_t = base_price + (Mackey-Glass delay-differential equation, discrete-time form)
                dx/dt = 0.2 * x(t-tau) / (1 + x(t-tau)^10) - 0.1 * x(t)
                The canonical chaotic-time-series benchmark in reservoir-computing/nonlinear
                dynamics literature (tau=17 is the standard "mildly chaotic" setting). Deterministic
                given tau and the fixed initial history, but long-range unpredictable in practice
                (sensitive dependence on initial conditions) -- the "complex, deterministic delay
                recurrence" point on the dataset axis, contrasting with sine's simple periodicity.
    "xor":      a_t ~ iid Bernoulli(0.5); x_t = base_price + amplitude * (2*(a_{t-1} XOR a_{t-2}) - 1)
                Classic "temporal XOR" — the next value depends nonlinearly (non-additively) on two
                specific past bits. Not linearly separable from either bit alone, a standard
                benchmark for whether a sequence model can learn nonlinear temporal combination
                rather than just correlation/periodicity.
    "lfsr":     x_t = base_price + amplitude * (2*bit_t - 1), where bit_t is a Fibonacci linear
                feedback shift register's output bit (see _LFSR_TAPS for the primitive-polynomial
                tap sets used per register width). Deterministic and low-complexity to *generate*
                (one XOR of a few register bits per step, period exactly 2^bits - 1), but its
                statistical profile (near-uniform bit frequency, near-zero autocorrelation except
                exactly at the period) looks close to random -- a direct test of whether a model
                (or the token-characteristics framework: entropy, LZ compression) can tell "looks
                complex" apart from "is complex to generate".
    "ar1":      x_t = base_price + z_t, z_t = ar_phi * z_{t-1} + eps_t, eps_t ~ N(0, ar_sigma).
                Textbook AR(1) process -- smooth and strongly autocorrelated (ar_phi close to 1
                gives long persistence, a "smooth random walk"-like trajectory) but STOCHASTIC and
                non-periodic, unlike sine (deterministic, periodic) or delay (deterministic,
                chaotic). Added specifically to test whether transfer to DDM+X requires smoothness
                per se, or specifically a deterministic/reproducible generator -- see the
                DDM/USDJPY transfer investigation's Phase 5 representation-probing follow-up.
    "lorenz":   x_t = base_price + (Lorenz system's x-coordinate, RK4-integrated, dt=0.02,
                sigma=10, rho=28, beta=8/3 -- the canonical chaotic-attractor parameters).
                Deterministic, continuous-valued, non-periodic, chaotic (sensitive dependence on
                initial conditions) -- a SEPARATE deterministic-chaotic generator from delay's
                Mackey-Glass delay-differential-equation form, added (Phase 5b) to test whether
                "deterministic + continuous + chaotic dynamics" transfers as a general category,
                or was specific to Mackey-Glass's particular recurrence structure. Fixed initial
                condition [1,1,1] -- seed is unused/irrelevant, same convention as sine/delay.
    "ar1_forced": Same linear recurrence as ar1 (z_t = ar_phi * z_{t-1} + f_t) but f_t is a
                DETERMINISTIC sum of incommensurate-period sinusoids (default: 5 periods -- 47,
                71, 97, 127, 157 bars -- pairwise-unrelated so the sum doesn't exactly repeat
                within any practical dataset length) instead of iid Gaussian innovations.
                Isolates whether it's specifically the injected per-step RANDOMNESS that blocks
                ar1's transfer effect, holding the recurrence/smoothing structure (same ar_phi,
                comparable forcing amplitude via `amplitude`) fixed -- a direct determinism-only
                control against plain "ar1". Fully deterministic; seed unused.
                `forced_periods` (list of floats, ar1_forced only) overrides the default 5-period
                mixture -- pass a single-element list (e.g. [140.0]) for a clean, directly-
                controlled characteristic recurrence timescale instead of one that emerges
                indirectly from 5 superposed periods (Phase 5c-timescale, user-requested: does
                the transfer effect's strength depend on the specific timescale value, at lag 70
                vs 140 vs 280, once "a detectable recurrence exists at all" is held fixed).

Both sine/sine_sum are a practical reading of "x_t periodic with period T" and "sin(t) + A*sin(T*t)",
reparameterized around a bar-count period so the result is a usable series at any timeframe --
raw sin(t) with integer t oscillates every ~6.3 bars, too fast to be a useful comparison signal.

Returns: CollectResult(artifact_path, row_count, from_ts, to_ts)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ARTIFACT_STORE = Path(os.getenv("ARTIFACT_STORE_PATH", "artifacts"))

_PANDAS_FREQ = {
    "M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min",
    "H1": "1h", "H4": "4h", "D1": "1D", "W1": "1W", "MN": "1MS",
}


@dataclass
class CollectResult:
    artifact_path: str   # relative to ARTIFACT_STORE
    row_count: int
    from_ts: datetime
    to_ts: datetime


# Fibonacci LFSR tap positions (1-indexed from the LSB) for known maximal-length (period =
# 2^bits - 1) primitive polynomials. Widths chosen to span "short enough to see the period
# within a normal dataset length" (4, 5) through "long enough to look genuinely random over a
# typical window" (16).
_LFSR_TAPS: dict[int, list[int]] = {
    4: [4, 3],
    5: [5, 3],
    8: [8, 6, 5, 4],
    16: [16, 15, 13, 4],
}


def _mackey_glass(length: int, tau: float, burn_in: int = 1000, stride: int = 1) -> np.ndarray:
    """Discrete-time Mackey-Glass delay recurrence (beta=0.2, gamma=0.1, n=10 -- the standard
    parameters used throughout the reservoir-computing/chaotic-time-series literature). tau=17 is
    the canonical "mildly chaotic" setting; below ~tau=4.5 the system settles to a fixed point
    instead. burn_in discards the initial transient before the trajectory settles onto its
    attractor, so the returned series doesn't depend on the arbitrary constant initial history.

    stride > 1 keeps every stride-th recurrence step -- the same trajectory viewed at a coarser
    time resolution, so the per-bar Lyapunov exponent scales by ~stride with the attractor
    unchanged (the Lyapunov-dial experiment's time-rescaling control)."""
    stride = max(1, int(stride))
    tau_steps = max(1, int(round(tau)))
    total = length * stride + burn_in + tau_steps + 1
    x = np.empty(total, dtype=np.float64)
    x[: tau_steps + 1] = 1.2  # standard constant initial history
    beta, gamma, n = 0.2, 0.1, 10
    for t in range(tau_steps, total - 1):
        lagged = x[t - tau_steps]
        x[t + 1] = x[t] + beta * lagged / (1 + lagged ** n) - gamma * x[t]
    start = burn_in + tau_steps
    return x[start : start + length * stride : stride]


def _temporal_xor(length: int, seed: int) -> np.ndarray:
    """x_t = a_{t-1} XOR a_{t-2} for iid Bernoulli(0.5) bits a -- returns values in {0, 1}."""
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=length + 2)
    return (bits[:length] ^ bits[1 : length + 1]).astype(np.float64)


def _lfsr(length: int, bits: int, seed: int) -> np.ndarray:
    """Fibonacci LFSR output bit stream -- returns values in {0, 1}, period exactly 2^bits - 1."""
    n = int(bits)
    taps = _LFSR_TAPS.get(n)
    if taps is None:
        raise ValueError(f"Unsupported lfsr_bits={n} (supported: {sorted(_LFSR_TAPS)})")
    rng = np.random.default_rng(seed)
    state = int(rng.integers(1, 2 ** n))  # nonzero seed state (all-zero state never changes)
    mask = (1 << n) - 1
    out = np.empty(length, dtype=np.float64)
    for t in range(length):
        out[t] = state & 1
        feedback = 0
        for tap in taps:
            feedback ^= (state >> (tap - 1)) & 1
        state = ((state << 1) | feedback) & mask
    return out


def _ar1(length: int, phi: float, sigma: float, seed: int) -> np.ndarray:
    """AR(1): z_t = phi*z_{t-1} + eps_t, eps_t ~ N(0, sigma). phi close to 1 gives a smooth,
    strongly-autocorrelated ("smooth random walk"-like) but stationary, purely stochastic
    trajectory -- no periodicity, no deterministic recurrence."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, sigma, length)
    z = np.empty(length, dtype=np.float64)
    z[0] = eps[0]
    for t in range(1, length):
        z[t] = phi * z[t - 1] + eps[t]
    return z


def _lorenz(length: int, dt: float = 0.02, sigma_param: float = 10.0, rho: float = 28.0,
            beta: float = 8.0 / 3.0, burn_in: int = 1000) -> np.ndarray:
    """Lorenz system (canonical chaotic attractor), RK4-integrated, x-coordinate returned --
    deterministic, continuous-valued, non-periodic, chaotic, and a completely separate generator
    from Mackey-Glass (a 3-variable coupled ODE flow, not a scalar delay-differential equation).
    Fixed initial condition [1,1,1] (off the unstable origin) -- no RNG, no seed dependence,
    same convention as sine/delay.

    dt is the integration step per output bar: the per-bar largest Lyapunov exponent is
    ~0.905 * dt (the canonical system's exponent is ~0.905 per unit time), so varying dt dials
    per-bar chaos while leaving the attractor itself unchanged. burn_in is in time units of the
    default dt=0.02 (20 time units) so a smaller dt still discards the same transient."""
    burn_in = int(round(burn_in * 0.02 / dt))
    def deriv(state: np.ndarray) -> np.ndarray:
        x, y, z = state
        return np.array([sigma_param * (y - x), x * (rho - z) - y, x * y - beta * z])

    total = length + burn_in
    state = np.array([1.0, 1.0, 1.0])
    xs = np.empty(total, dtype=np.float64)
    for i in range(total):
        k1 = deriv(state)
        k2 = deriv(state + 0.5 * dt * k1)
        k3 = deriv(state + 0.5 * dt * k2)
        k4 = deriv(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        xs[i] = state[0]
    return xs[burn_in:]


_AR1_FORCED_PERIODS = (47.0, 71.0, 97.0, 127.0, 157.0)


def _ar1_forced(
    length: int, phi: float, amplitude: float,
    periods: tuple[float, ...] = _AR1_FORCED_PERIODS,
) -> np.ndarray:
    """Same linear recurrence as _ar1 (z_t = phi*z_{t-1} + f_t), but f_t is a DETERMINISTIC sum
    of incommensurate-period sinusoids instead of iid random innovations -- isolates whether it's
    specifically the injected per-step randomness (not the recurrence/smoothing structure) that
    makes plain ar1 fail to transfer. No RNG, no seed dependence.

    `periods` defaults to the 5-sinusoid mixture above, but callers pass a single-element tuple
    (Phase 5c-timescale, user-requested) to get a clean, directly-controlled characteristic
    recurrence timescale instead of one that emerges indirectly from 5 superposed periods --
    isolates "does the transfer effect's strength depend on the recurrence timescale's specific
    value" from "does a detectable recurrence exist at all"."""
    t = np.arange(length, dtype=np.float64)
    forcing = sum(np.sin(2 * np.pi * t / p) for p in periods) * (amplitude / len(periods))
    z = np.empty(length, dtype=np.float64)
    z[0] = forcing[0]
    for i in range(1, length):
        z[i] = phi * z[i - 1] + forcing[i]
    return z


def _generate_series(
    function: str,
    length: int,
    period: float,
    amplitude: float,
    freq_ratio: float,
    tau: float = 17.0,
    lfsr_bits: int = 8,
    ar_phi: float = 0.98,
    ar_sigma: float = 1.0,
    seed: int = 42,
    forced_periods: tuple[float, ...] | None = None,
    stride: int = 1,
    lorenz_dt: float = 0.02,
) -> np.ndarray:
    t = np.arange(length, dtype=np.float64)
    if function == "sine":
        return amplitude * np.sin(2 * np.pi * t / period)
    if function == "sine_sum":
        return np.sin(2 * np.pi * t / period) + amplitude * np.sin(2 * np.pi * freq_ratio * t / period)
    if function == "delay":
        return _mackey_glass(length, tau, stride=stride)
    if function == "xor":
        return amplitude * (2 * _temporal_xor(length, seed) - 1)
    if function == "lfsr":
        return amplitude * (2 * _lfsr(length, lfsr_bits, seed) - 1)
    if function == "ar1":
        return _ar1(length, ar_phi, ar_sigma, seed)
    if function == "lorenz":
        return _lorenz(length, dt=lorenz_dt)
    if function == "ar1_forced":
        periods = tuple(forced_periods) if forced_periods else _AR1_FORCED_PERIODS
        return _ar1_forced(length, ar_phi, amplitude, periods=periods)
    raise ValueError(
        f"Unknown synthetic function: {function!r} "
        f"(expected 'sine', 'sine_sum', 'delay', 'xor', 'lfsr', 'ar1', 'lorenz', or 'ar1_forced')"
    )


def collect(datasource_id: int, config: dict) -> CollectResult:
    function = config.get("function", "sine")
    length = int(config.get("length", 2000))
    period = float(config.get("period", 50))
    amplitude = float(config.get("amplitude", 1.0))
    freq_ratio = float(config.get("freq_ratio", 5))
    tau = float(config.get("tau", 17))
    lfsr_bits = int(config.get("lfsr_bits", 8))
    stride = int(config.get("stride", 1))
    lorenz_dt = float(config.get("lorenz_dt", 0.02))
    ar_phi = float(config.get("ar_phi", 0.98))
    ar_sigma = float(config.get("ar_sigma", 1.0))
    base_price = float(config.get("base_price", 100.0))
    noise = float(config.get("noise", 0.0))
    seed = int(config.get("seed", 42))
    timeframe = config.get("timeframe", "M5")
    start_ts = config.get("start_ts") or "2024-01-01"
    forced_periods_cfg = config.get("forced_periods")
    forced_periods = tuple(float(p) for p in forced_periods_cfg) if forced_periods_cfg else None

    if length < 2:
        raise ValueError("length must be at least 2")
    if period <= 0:
        raise ValueError("period must be positive")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if lorenz_dt <= 0:
        raise ValueError("lorenz_dt must be positive")

    values = base_price + _generate_series(
        function, length, period, amplitude, freq_ratio, tau=tau, lfsr_bits=lfsr_bits,
        ar_phi=ar_phi, ar_sigma=ar_sigma, seed=seed, forced_periods=forced_periods,
        stride=stride, lorenz_dt=lorenz_dt,
    )
    if noise > 0:
        rng = np.random.default_rng(seed)
        values = values + rng.normal(0, noise, length)

    freq = _PANDAS_FREQ.get(timeframe, "5min")
    index = pd.date_range(start=pd.Timestamp(start_ts, tz="UTC"), periods=length, freq=freq)

    df = pd.DataFrame({
        "open": values, "high": values, "low": values, "close": values,
        "volume": np.ones(length),
    }, index=index)
    df.index.name = "datetime"

    out_dir = ARTIFACT_STORE / "datasets" / f"src_{datasource_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_rel = f"datasets/src_{datasource_id}/{function}_{timeframe}.parquet"
    df.to_parquet(ARTIFACT_STORE / artifact_rel)

    from data.artifact_store import upload as _upload
    _upload(ARTIFACT_STORE / artifact_rel)

    return CollectResult(
        artifact_path=artifact_rel,
        row_count=len(df),
        from_ts=df.index[0].to_pydatetime().replace(tzinfo=timezone.utc),
        to_ts=df.index[-1].to_pydatetime().replace(tzinfo=timezone.utc),
    )
