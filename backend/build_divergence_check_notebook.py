"""Throwaway: build a small, self-contained Colab notebook that checks which (delta, wma)
candidates diverge under spread feedback (a=1.0, window=150) -- an experiment in its own right
(does colab_runner.py's generic exec_notebook/download interface work for a plain diagnostic
script, not just a full TrainingRun?), run in parallel with the slow local phase1 run rather than
waiting on it.

Deliberately embeds DDMv3/_DDMv1's source directly (inspect.getsource) instead of following
model_core's "pip install a git-pinned package" convention: unlike model_core, backend/data/ has
no standalone installable package, and building one is out of scope for a one-off pipeline test.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import nbformat as nbf

from data.collectors.ddm_simulator import DDMv3, _DDMv1

CANDIDATES = [
    (-1.5, 15), (-1.4, 15), (-1.3, 15), (-1.2, 15), (-1.1, 15),
    (-1.0, 10), (-0.9, 10), (-0.8, 7), (-0.7, 7), (-0.6, 7),
    (-0.5, 7), (-0.4, 5), (-0.3, 5), (-0.2, 10), (-0.1, 7), (0.0, 15),
    (-0.8, 5), (-0.5, 10), (0.0, 20), (-1.0, 20),
]
SPREAD_A = 1.0
SPREAD_WINDOW = 150
N_TRADES = 24 * 60 * 20  # MAX_HORIZON(24) * CANDLES_PER_WINDOW(60) * TRADES_PER_CANDLE(20)


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell(
        "# DDM spread-feedback divergence pre-check\n"
        "Throwaway experiment: does `colab_runner.py`'s generic exec_notebook/download path work "
        "for a plain diagnostic script (not a TrainingRun)? Checks which (delta, wma) candidates "
        f"diverge under spread_feedback_a={SPREAD_A}, window={SPREAD_WINDOW}."
    ))

    cells.append(nbf.v4.new_code_cell("import random\nimport numpy as np\nimport json\nimport time"))

    ddm_source = inspect.getsource(_DDMv1) + "\n\n" + inspect.getsource(DDMv3)
    cells.append(nbf.v4.new_code_cell(
        "# --- Embedded from backend/data/collectors/ddm_simulator.py (inspect.getsource) ---\n"
        "from collections.abc import Iterable\n"
        "import pandas as pd\n\n" + ddm_source
    ))

    check_code = f"""
CANDIDATES = {CANDIDATES!r}
SPREAD_A = {SPREAD_A!r}
SPREAD_WINDOW = {SPREAD_WINDOW!r}
N_TRADES = {N_TRADES!r}

results = []
t0 = time.time()
for delta, wma in CANDIDATES:
    random.seed(0); np.random.seed(0)
    model = DDMv3(num_agent=300, max_volatility=0.02, min_volatility=0.01, wma=wma,
                  dealer_sensitive_min=-3.5+delta, dealer_sensitive_max=-1.5+delta,
                  spread_feedback_a=SPREAD_A, spread_feedback_window=SPREAD_WINDOW)
    status = "ok"
    amps = []
    try:
        for price, tt in model.simulate_stream(n_trades=N_TRADES):
            amps.append(model.spread / model._spread_z0)
    except RuntimeError:
        status = "diverged"
    except (ValueError, OverflowError):
        status = "error"
    entry = {{"delta": delta, "wma": wma, "status": status}}
    if amps:
        arr = np.array(amps)
        if np.all(np.isfinite(arr)):
            entry.update(mean_amp=float(arr.mean()), p99_amp=float(np.percentile(arr, 99)), max_amp=float(arr.max()))
        else:
            entry["status"] = "non_finite"
    results.append(entry)
    print(entry)

print(f"done in {{time.time()-t0:.1f}}s")
with open("/content/divergence_check.json", "w") as f:
    json.dump(results, f, indent=2)
print("wrote /content/divergence_check.json")
"""
    cells.append(nbf.v4.new_code_cell(check_code))

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    return nb


if __name__ == "__main__":
    nb = build_notebook()
    out_dir = Path(__file__).resolve().parent.parent / "notebooks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ddm_divergence_check.ipynb"
    nbf.write(nb, out_path)
    print(f"Wrote {out_path}")
