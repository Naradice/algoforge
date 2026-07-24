"""
Sliding-window dataset for OHLC parquet files.

Compatible with the stocknet trainer interface:
    ds[start : start + batch_size]  →  (src_tensor, tgt_tensor)
where tensors are [batch, seq_len, features].

Normalisation modes (apply to the prediction target, tgt, always):
    "returns"  — log returns of each feature column (stationary, recommended)
    "diff"     — raw first differences (stationary, no log — use when the series isn't
                 price-like/multiplicative, or when comparing representations that need the
                 same additive scale as the raw series; see "diff" vs "returns" note below)
    "minmax"   — min-max scale each column to [0, 1]
    "zscore"   — standardize to zero mean and unit variance
    "robust"   — scale by median and IQR (robust to outliers)
    "none"     — raw values

token_level (opt-in, defaults to None): decouples how the model's *input* history (src) is
represented from `normalize` above, which continues to control the *target* (tgt) exclusively.
Exists to compare input tokenization schemes (raw continuous / differenced / discretized-symbol)
without changing what's being predicted or how prediction error is scored — every scheme still
regresses the same continuous target with the same loss, so results are directly comparable via
val_loss alone. See docs/model-layer.md, "Comparing training runs" / token_level.
    None             — src uses the exact same array as tgt (current/default behaviour)
    "diff"           — src = raw first differences of the underlying series (continuous)
    "quantize_diff"  — src = raw first differences, discretized into `n_bins` integer token ids
                       via quantile (equal-frequency) bin edges fit on the training split only.
                       Quantile, not equal-width, bins matter here: a differenced periodic signal's
                       histogram is typically U-shaped (density concentrated at the extremes, not
                       at zero), so equal-width bins produce wildly unbalanced token frequencies.
    "cluster"        — src = pattern/shape tokens: k-means (k=`n_clusters`) over sliding, per-window
                       z-scored shapes of `cluster_window` consecutive differences, fit on the
                       training split only. Groups short movement patterns (e.g. "uptrend",
                       "range", "sharp drop") into a token per shape rather than per raw value.
Only single-feature (`feature_cols` of length 1) datasets support token_level -- multi-feature
tokenized input isn't implemented.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch


class OHLCWindowDataset:
    # Max tick batch files to load from a DDM directory (100 × 10 000 = 1 M ticks)
    _MAX_TICK_FILES = 100
    # Max OHLC rows to keep after loading; keeps window arrays from exceeding ~1 GB RAM
    _MAX_OHLC_ROWS = 50_000

    def __init__(
        self,
        artifact_path: str,
        obs_len: int,
        pred_len: int,
        feature_cols: list[str] | None = None,
        normalize: str = "returns",
        val_split: float = 0.2,
        device: str = "cpu",
        preprocessing: dict | None = None,
        max_rows: int | None = None,
        token_level: str | None = None,
        n_bins: int = 7,
        cluster_window: int = 20,
        n_clusters: int = 20,
    ) -> None:
        df, feature_cols = self._load_preprocessed_df(artifact_path, feature_cols, preprocessing, max_rows)
        raw = df[feature_cols].values.astype(np.float32)

        tgt_data = self._apply_normalize(raw, normalize)

        # Store normalisation params for inverse transform at inference
        self._normalize = normalize
        self._norm_min = tgt_data.min(axis=0) if normalize == "minmax" else None
        self._norm_max = tgt_data.max(axis=0) if normalize == "minmax" else None

        self.token_level = token_level
        self.vocab_size: int | None = None
        self.token_bin_edges: np.ndarray | None = None
        self.token_stream: np.ndarray | None = None  # flat, pre-windowing -- see compute_token_characteristics
        self.cluster_centroids: np.ndarray | None = None  # only set for token_level="cluster"

        if token_level is None:
            src_data = tgt_data
        else:
            if len(feature_cols) != 1:
                raise ValueError("token_level requires exactly one feature_col (multi-feature tokenized input isn't implemented)")
            diff_full = np.diff(raw, axis=0)  # length n-1
            # tgt must stay index-aligned with src regardless of which token_level is used, so
            # always trim tgt to the same n-1 length as a differenced series -- tgt_data[i]
            # remains "the value at time i+1", paired with src derived from the transition
            # ending at time i+1, both knowable as of time i+1 (causally consistent).
            if len(tgt_data) == len(raw):
                tgt_data = tgt_data[1:]
            if token_level == "diff":
                src_data = diff_full
            elif token_level == "quantize_diff":
                # Bin edges fit on an approximate train-fraction prefix only, to avoid leaking
                # validation-region statistics into the vocabulary -- val_split applied directly
                # to row count here (not the window-count split below) since edges must exist
                # before windowing can happen; the resulting boundary differs from the window
                # split by at most one window's worth of rows, immaterial for quantile estimation.
                train_boundary = max(1, int(len(diff_full) * (1 - val_split)))
                qs = np.linspace(0, 1, n_bins + 1)[1:-1]
                edges = np.quantile(diff_full[:train_boundary, 0], qs)
                self.token_bin_edges = edges
                self.vocab_size = n_bins
                src_data = np.digitize(diff_full[:, 0], edges).astype(np.int64).reshape(-1, 1)
                self.token_stream = src_data.reshape(-1)
            elif token_level == "cluster":
                # Pattern tokens: k-means over sliding shape windows of the differenced series
                # (a "shape" is cluster_window consecutive deltas -- movement over that span, not
                # a price level, so it stays meaningful regardless of where in a trend/regime the
                # window sits). Each shape is z-scored *individually* (removes that window's own
                # scale) before clustering, the standard normalization for shape/shapelet
                # clustering -- otherwise windows just cluster by volatility, not by pattern
                # shape. On a periodic signal this should recover something close to a phase
                # partition of the cycle: sorting centroids by their dominant frequency content
                # tends to lay them out in cycle order, a useful sanity check that clustering is
                # doing something structural rather than arbitrary.
                from sklearn.cluster import KMeans

                w = cluster_window
                n_diff = len(diff_full)
                if n_diff < w:
                    raise ValueError(f"Series too short ({n_diff} diffs) for cluster_window={w}")
                n_shapes = n_diff - w + 1
                shape_idx = np.arange(w)[None, :] + np.arange(n_shapes)[:, None]
                shapes = diff_full[shape_idx, 0]  # [n_shapes, w]
                mu = shapes.mean(axis=1, keepdims=True)
                sigma = shapes.std(axis=1, keepdims=True)
                sigma = np.where(sigma == 0, 1.0, sigma)
                shapes_norm = (shapes - mu) / sigma

                # Fit on an approximate train-fraction prefix only -- same leakage rationale as
                # quantize_diff's bin edges above.
                train_boundary = max(n_clusters, int(n_shapes * (1 - val_split)))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                kmeans.fit(shapes_norm[:train_boundary])
                cluster_ids = kmeans.predict(shapes_norm).astype(np.int64)

                self.vocab_size = n_clusters
                self.cluster_centroids = kmeans.cluster_centers_
                src_data = cluster_ids.reshape(-1, 1)
                self.token_stream = cluster_ids
                # Shape window i covers diff_full[i:i+w], i.e. underlying time range
                # [i+1, i+w] -- its token represents "the pattern ending at time i+w", which
                # pairs with tgt_data[i+w-1] ("value at time i+w" after tgt_data's existing
                # 1-row trim above). Trim the front of tgt_data to match.
                tgt_data = tgt_data[w - 1:]
            else:
                raise ValueError(f"Unknown token_level: {token_level!r}")

        # Build windows
        n = len(tgt_data)
        total_len = obs_len + pred_len + 1   # +1 for the teacher-forced tgt shift
        n_windows = n - total_len + 1

        split_idx = int(n_windows * (1 - val_split))
        self._train_src, self._train_tgt = self._make_windows(
            src_data[:split_idx + total_len - 1], tgt_data[:split_idx + total_len - 1], obs_len, pred_len + 1
        )
        val_src_data = src_data[split_idx:]
        val_tgt_data = tgt_data[split_idx:]
        if len(val_tgt_data) >= total_len:
            self._val_src, self._val_tgt = self._make_windows(val_src_data, val_tgt_data, obs_len, pred_len + 1)
        else:
            self._val_src, self._val_tgt = self._train_src, self._train_tgt

        self._is_train = True
        self.device = device
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.n_features = len(feature_cols)

    @staticmethod
    def _apply_normalize(data: np.ndarray, normalize: str) -> np.ndarray:
        if normalize == "returns":
            data = np.log(data + 1e-8)
            return np.diff(data, axis=0)
        elif normalize == "diff":
            return np.diff(data, axis=0)
        elif normalize == "minmax":
            mn, mx = data.min(axis=0), data.max(axis=0)
            rng = np.where(mx - mn == 0, 1.0, mx - mn)
            return (data - mn) / rng
        elif normalize == "zscore":
            mu = data.mean(axis=0)
            sigma = data.std(axis=0)
            sigma = np.where(sigma == 0, 1.0, sigma)
            return (data - mu) / sigma
        elif normalize == "robust":
            median = np.median(data, axis=0)
            q25 = np.percentile(data, 25, axis=0)
            q75 = np.percentile(data, 75, axis=0)
            iqr = np.where(q75 - q25 == 0, 1.0, q75 - q25)
            return (data - median) / iqr
        return data

    @classmethod
    def _load_preprocessed_df(
        cls,
        artifact_path: str,
        feature_cols: list[str] | None,
        preprocessing: dict | None,
        max_rows: int | None = None,
    ) -> tuple[pd.DataFrame, list[str]]:
        """Load + preprocess a dataset artifact, up to (but not including) normalization.

        Returns the DataFrame with its real DatetimeIndex intact — this is "what will be fed
        to the model" after indicators/clustering and the row cap, before feature_cols are
        pulled out as a plain ndarray and normalized. Shared by __init__ and
        compute_effective_characteristics below.

        `max_rows` overrides the default `_MAX_OHLC_ROWS` cap (opt-in per training run via the
        `max_rows` hyperparam) — the default exists to keep window arrays from exceeding ~1 GB
        RAM on typical datasets, not as a hard ceiling; a caller that wants a genuine data-size
        comparison needs to be able to raise it.
        """
        store = Path(os.getenv("ARTIFACT_STORE_PATH", "artifacts"))
        full_path = store / artifact_path

        if full_path.is_dir():
            # DDM tick directory: load a capped sample and resample to M1 OHLC
            from data.parquet_reader import load_ddm_ticks
            tick_df = load_ddm_ticks(full_path, max_files=cls._MAX_TICK_FILES)
            ohlc = tick_df["price"].resample("1min").ohlc()
            ohlc.columns = ["open", "high", "low", "close"]
            ohlc["volume"] = tick_df["price"].resample("1min").count()
            df = ohlc.dropna()
        else:
            df = pd.read_parquet(full_path)

        df.columns = [c.lower() for c in df.columns]
        df = df.sort_index().dropna()

        # Apply indicators and clustering before row cap so indicators have full history
        if preprocessing:
            from model.trainers.preprocessing import apply_preprocessing
            df = apply_preprocessing(df, preprocessing)

        effective_cap = max_rows if max_rows is not None else cls._MAX_OHLC_ROWS
        if len(df) > effective_cap:
            df = df.iloc[-effective_cap:]

        if feature_cols is None:
            feature_cols = ["close"]
        feature_cols = [c for c in feature_cols if c in df.columns]
        if not feature_cols:
            feature_cols = [df.columns[-1]]

        return df, feature_cols

    @staticmethod
    def _make_windows(src_data: np.ndarray, tgt_data: np.ndarray, obs_len: int, tgt_len: int):
        """src_data and tgt_data must be the same length and index-aligned (tgt_data[i] is the
        value "at" the same point in time src_data[i] is derived from) -- see token_level above
        for how src can differ in representation (continuous/differenced/tokenized) from tgt."""
        n = len(src_data)
        total = obs_len + tgt_len
        srcs, tgts = [], []
        for i in range(n - total + 1):
            srcs.append(src_data[i : i + obs_len])
            tgts.append(tgt_data[i + obs_len - 1 : i + obs_len - 1 + tgt_len])  # overlapping by 1 for teacher forcing
        return np.array(srcs, dtype=src_data.dtype), np.array(tgts, dtype=np.float32)

    def train(self) -> None:
        self._is_train = True

    def eval(self) -> None:
        self._is_train = False

    def __len__(self) -> int:
        return len(self._train_src) if self._is_train else len(self._val_src)

    def __getitem__(self, key):
        src_arr = self._train_src if self._is_train else self._val_src
        tgt_arr = self._train_tgt if self._is_train else self._val_tgt

        if isinstance(key, slice):
            src = torch.tensor(src_arr[key], device=self.device)
            tgt = torch.tensor(tgt_arr[key], device=self.device)
        else:
            src = torch.tensor(src_arr[key], device=self.device)
            tgt = torch.tensor(tgt_arr[key], device=self.device)

        return src, tgt


# Structure/complexity analyses from data/characteristics.py's registry — see that module's
# docstring for what each one measures (long-range dependence, periodicity, multiscale
# wavelet structure, entropy/nonlinearity, regime changes).
_EFFECTIVE_CHARACTERISTIC_KEYS = [
    "long_range_dependence",
    "spectral_periodicity",
    "multiscale_wavelet",
    "complexity_nonlinearity",
    "regime_changes",
]


def compute_effective_characteristics(
    artifact_path: str,
    feature_cols: list[str] | None,
    preprocessing: dict | None,
    max_rows: int | None = None,
) -> dict:
    """Structure characteristics of the data a training run will actually consume: after
    preprocessing (indicators/clustering) and the row cap, on the primary feature column,
    before normalization (whose output breaks the log-return math these analyses rely on).

    Best-effort per analysis — an error in one doesn't blank the rest. Callers should also
    wrap the call itself, since a completely unreadable/degenerate dataset can still raise
    before any per-analysis try/except is reached (e.g. during loading).
    """
    from data.characteristics import CHARACTERISTIC_REGISTRY

    df, resolved_feature_cols = OHLCWindowDataset._load_preprocessed_df(artifact_path, feature_cols, preprocessing, max_rows)
    series_df = pd.DataFrame({"close": df[resolved_feature_cols[0]]})

    results: dict = {}
    for name in _EFFECTIVE_CHARACTERISTIC_KEYS:
        try:
            results[name] = CHARACTERISTIC_REGISTRY[name](series_df)
        except Exception as e:
            results[name] = {"error": str(e)}
    return results


def compute_token_characteristics(tokens: np.ndarray, vocab_size: int) -> dict:
    """Information-theoretic structure of a discretized token stream (OHLCWindowDataset's
    token_stream, produced by token_level="quantize_diff" or future discrete token_levels).

    Lets Validation Loss be plotted against how complex/predictable a *representation* actually
    is (effective vocabulary, entropy rate, compressibility, ...) instead of against training row
    count alone — two token_levels or vocab sizes that use the same number of rows can still
    differ hugely in how much structure the model actually has to learn.

    Best-effort per metric, same pattern as compute_effective_characteristics — one metric
    failing (e.g. too few tokens for a stable n-gram estimate) never blanks the rest.
    """
    tokens = np.asarray(tokens).reshape(-1)
    results: dict = {}

    try:
        results["token_entropy"] = _token_entropy(tokens, vocab_size)
    except Exception as e:
        results["token_entropy"] = {"error": str(e)}

    try:
        h = results["token_entropy"]["bits"] if isinstance(results["token_entropy"], dict) else None
        results["effective_vocab_size"] = float(2 ** h) if h is not None else None
    except Exception as e:
        results["effective_vocab_size"] = {"error": str(e)}

    try:
        results["token_zipf"] = _token_zipf_fit(tokens)
    except Exception as e:
        results["token_zipf"] = {"error": str(e)}

    try:
        results["token_mutual_information"] = _adjacent_mutual_information(tokens, vocab_size)
    except Exception as e:
        results["token_mutual_information"] = {"error": str(e)}

    try:
        results["ngram_entropy"] = _ngram_entropy_rates(tokens, vocab_size, max_n=3)
    except Exception as e:
        results["ngram_entropy"] = {"error": str(e)}

    try:
        results["lz_compression_ratio"] = _lz_compression_ratio(tokens)
    except Exception as e:
        results["lz_compression_ratio"] = {"error": str(e)}

    return results


def _token_entropy(tokens: np.ndarray, vocab_size: int) -> dict:
    """Shannon entropy of the unigram token distribution, in bits, plus the same value
    normalized by the maximum possible entropy (log2(vocab_size), a uniform distribution) so
    entropy is comparable across runs with different vocab sizes."""
    counts = np.bincount(tokens, minlength=vocab_size)
    p = counts[counts > 0] / counts.sum()
    bits = float(-np.sum(p * np.log2(p)))
    max_bits = float(np.log2(vocab_size)) if vocab_size > 1 else 1.0
    return {"bits": bits, "normalized": bits / max_bits if max_bits > 0 else 0.0}


def _token_zipf_fit(tokens: np.ndarray) -> dict:
    """Fits the token rank-frequency curve to a power law (Zipf's law: frequency ∝ rank^-alpha)
    via a log-log linear regression. alpha near 1 and a high r2 mean the token stream's usage
    pattern looks "language-like"; a low r2 means frequency doesn't follow a clean power law at
    all (e.g. a near-uniform or bimodal distribution)."""
    counts = np.bincount(tokens)
    freqs = np.sort(counts[counts > 0])[::-1].astype(np.float64)
    if len(freqs) < 3:
        return {"alpha": None, "r2": None, "note": "too few distinct tokens for a stable fit"}
    ranks = np.arange(1, len(freqs) + 1, dtype=np.float64)
    log_r, log_f = np.log(ranks), np.log(freqs)
    slope, intercept = np.polyfit(log_r, log_f, 1)
    pred = slope * log_r + intercept
    ss_res = np.sum((log_f - pred) ** 2)
    ss_tot = np.sum((log_f - log_f.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"alpha": float(-slope), "r2": float(r2)}


def _adjacent_mutual_information(tokens: np.ndarray, vocab_size: int) -> float:
    """I(X_t; X_{t+1}) in bits, from the empirical joint distribution of consecutive tokens --
    how much knowing the current token reduces uncertainty about the next one. 0 means adjacent
    tokens are statistically independent (pure noise, from the model's point of view); higher
    values mean there's short-range structure a model could in principle exploit."""
    x, y = tokens[:-1], tokens[1:]
    joint = np.zeros((vocab_size, vocab_size))
    np.add.at(joint, (x, y), 1)
    joint /= joint.sum()
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        outer = px * py
        ratio = np.where((joint > 0) & (outer > 0), joint / outer, 1.0)
        terms = np.where(joint > 0, joint * np.log2(ratio), 0.0)
    return float(terms.sum())


def _ngram_entropy_rates(tokens: np.ndarray, vocab_size: int, max_n: int = 3) -> dict:
    """Block entropy H(n) (Shannon entropy of the n-gram distribution) for n=1..max_n, plus the
    conditional entropy rate estimates h(n) = H(n) - H(n-1) -- "bits of surprise in the next
    token, given n-1 tokens of context". A falling h(n) as n grows means longer context keeps
    reducing uncertainty (learnable long-range structure); a flat h(n) means n-1 tokens of
    context already captures everything predictable (a low-order Markov process, or noise)."""
    block_entropy: dict[int, float] = {}
    for n in range(1, max_n + 1):
        if len(tokens) < n or vocab_size ** n > 10_000_000:
            break
        ids = np.zeros(len(tokens) - n + 1, dtype=np.int64)
        for i in range(n):
            ids += tokens[i: len(tokens) - n + 1 + i].astype(np.int64) * (vocab_size ** (n - 1 - i))
        counts = np.bincount(ids)
        p = counts[counts > 0] / counts.sum()
        block_entropy[n] = float(-np.sum(p * np.log2(p)))

    conditional_rates = {
        n: block_entropy[n] - block_entropy[n - 1]
        for n in block_entropy if n > 1 and (n - 1) in block_entropy
    }
    return {"block_entropy": block_entropy, "conditional_rates": conditional_rates}


def _lz_compression_ratio(tokens: np.ndarray) -> float:
    """Generic LZ-family (zlib/DEFLATE) compressed size over raw size of the token stream's byte
    encoding -- a standard, well-tested compressibility proxy (not the classic LZ76 production
    count specifically, but the same family of algorithmic-complexity idea: a highly repetitive
    or low-entropy token stream compresses well; a high-entropy or near-random one doesn't).

    Encodes as uint8 (vocab sizes in practice are tiny, e.g. n_bins=7) rather than a wider dtype
    -- int32 would pad every token with 3 constant zero bytes, which zlib then "compresses away"
    regardless of the actual token sequence, making the ratio mostly measure dtype padding
    instead of token structure.
    """
    import zlib

    if tokens.max(initial=0) >= 256:
        raise ValueError("_lz_compression_ratio assumes vocab_size < 256 (uint8 encoding)")
    raw_bytes = tokens.astype(np.uint8).tobytes()
    compressed = zlib.compress(raw_bytes, level=9)
    return len(compressed) / len(raw_bytes)
