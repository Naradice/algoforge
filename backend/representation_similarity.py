"""Phase 5 step 3 (user-requested): is DDM+Sine's representation a genuinely NEW feature, or just
a simple combination of DDM-only's and Sine-only's own representations?

Reuses the per-layer, per-checkpoint hidden-state arrays saved by probe_representations.py
extract() (same USDJPY validation windows, same sample order across every checkpoint, so
population-level representation-similarity metrics like CKA are directly comparable). Computes
linear CKA (Kornblith et al. 2019) between every checkpoint pair, at every layer.

If ddm_sine's layer-3 representation has CKA close to 1.0 against EITHER ddm_only or sine_only
alone, that's evidence DDM+Sine mostly just inherits one component's representation. If CKA is
low against both (while still being the checkpoint whose probe R^2 jumps dramatically, per
probe_representations.py's results), that supports the interaction hypothesis: DDM+Sine forms a
representation neither ingredient has on its own.

Usage: python representation_similarity.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent / "probe_artifacts"

# Focus comparison set -- the three conditions the "is it additive or an interaction" question is
# actually about, plus ddm_delay/ddm_xor as generalization/negative-control checks, plus (Phase
# 5e-period-representation, user-requested) ddm_sine_p15/ddm_sine_p200/ddm_lfsr -- do the three
# TRANSFERRING Sine periods (15/50/200) form the SAME representation as each other (high CKA), and
# does ddm_lfsr (exactly periodic, but NOT transferring) look like ddm_only/scratch instead of
# ddm_sine (low CKA against ddm_sine, high against ddm_only)?
LABELS = [
    "scratch", "ddm_only", "sine_only", "ddm_sine", "ddm_delay", "ddm_xor",
    "ddm_lfsr", "ddm_sine_p15", "ddm_sine_p200",
]


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA between two [n_samples, d] representation matrices of the SAME samples (d can
    differ between x and y; here it doesn't, d_model=64 throughout). Kornblith et al. 2019's
    feature-space formulation -- equivalent to the full n x n HSIC-based version for a linear
    kernel, but O(d^2) instead of O(n^2), cheap here since d_model=64 << n_samples."""
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    xty = x.T @ y
    xtx = x.T @ x
    yty = y.T @ y
    num = np.linalg.norm(xty, ord="fro") ** 2
    denom = np.linalg.norm(xtx, ord="fro") * np.linalg.norm(yty, ord="fro")
    return float(num / denom) if denom > 0 else float("nan")


def main() -> None:
    data = {label: np.load(OUTPUT_DIR / f"{label}.npz") for label in LABELS}
    n_layers = sum(1 for k in data[LABELS[0]].files if k.startswith("layer_"))

    pairs = [
        ("ddm_only", "sine_only"),
        ("ddm_only", "ddm_sine"),
        ("sine_only", "ddm_sine"),
        ("ddm_only", "ddm_delay"),
        ("ddm_only", "ddm_xor"),
        ("scratch", "ddm_sine"),
        # Phase 5e-period-representation: do the three transferring periods share a representation?
        ("ddm_sine", "ddm_sine_p15"),
        ("ddm_sine", "ddm_sine_p200"),
        ("ddm_sine_p15", "ddm_sine_p200"),
        # ddm_lfsr (periodic, does NOT transfer) -- should look like ddm_only, not ddm_sine, if the
        # representation is specific to smooth/single-tone periodicity rather than periodicity-as-
        # a-label.
        ("ddm_only", "ddm_lfsr"),
        ("ddm_sine", "ddm_lfsr"),
        ("ddm_only", "ddm_sine_p15"),
        ("ddm_only", "ddm_sine_p200"),
    ]

    for layer_idx in range(n_layers):
        print(f"\n=== layer {layer_idx} linear CKA ===")
        for a, b in pairs:
            cka = linear_cka(data[a][f"layer_{layer_idx}"], data[b][f"layer_{layer_idx}"])
            print(f"  CKA({a:>10}, {b:<10}) = {cka:.4f}")


if __name__ == "__main__":
    main()
