"""
Generate a self-contained Google Colab notebook that reproduces an algoforge training run on a
dataset snapshot (data/snapshot_service.py) — runnable on a bare Colab CPU runtime, no algoforge
backend connection required.

Rather than hand-copying model/dataset logic into the notebook (which would silently drift from
algoforge's own implementation over time), this embeds the *actual current source* of:
  - the requested architecture's model class (model/architectures/{arch}.py)
  - OHLCWindowDataset's core numeric methods, `_apply_normalize` and `_make_windows`
    (model/trainers/dataset.py) — the two methods that actually determine a run's numbers
  - the training/eval loop (model/trainers/supervised.py)
via `inspect.getsource()`, so the notebook's numerics match algoforge's pipeline exactly as of
the commit it was generated from (recorded in the notebook's first cell) — not a
reimplementation that could quietly diverge.

Scope: only architecture="lstm" on the default (non-tokenized, no preprocessing recipe, no
clustering, chronological split) OHLCWindowDataset path — see SimpleWindowDataset in the
generated notebook. This covers the common case for a tiny from-scratch model check. Extend
_SUPPORTED_ARCHITECTURES only after checking the new architecture's forward()/training-loop
usage actually matches what this generator wires up — e.g. Seq2SeqTransformer's teacher-forcing
call shape needs verifying before being added here, it isn't automatically compatible just
because train_epoch/eval_epoch are shared by both.
"""
from __future__ import annotations

import inspect
import json
import subprocess
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import nbformat as nbf

from model.architectures.lstm import LSTMModel
from model.trainers.dataset import OHLCWindowDataset
from model.trainers.supervised import _split_tgt, eval_epoch, train_epoch

_SUPPORTED_ARCHITECTURES = {"lstm"}

DEFAULT_HYPERPARAMS = {
    "obs_len": 60,
    "pred_len": 10,
    "epochs": 20,
    "batch_size": 32,
    "lr": 0.001,
    "hidden_dim": 32,
    "num_layers": 2,
    "dropout": 0.1,
    "val_split": 0.2,
    "normalize": "returns",
    "feature_cols": ["close"],
    "seed": 42,
}


def _dedent_source(obj) -> str:
    """inspect.getsource() on a @staticmethod includes the decorator line and the class's
    indentation; strip both so the result stands alone as a top-level def."""
    src = textwrap.dedent(inspect.getsource(obj))
    lines = [ln for ln in src.splitlines() if ln.strip() != "@staticmethod"]
    return "\n".join(lines)


def _git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def build_notebook(
    architecture: str,
    model_name: str,
    dataset_id: int,
    snapshot_id: int,
    snapshot_url: str,
    snapshot_sha256: str,
    hyperparams: dict,
) -> nbf.NotebookNode:
    if architecture not in _SUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"Unsupported architecture for notebook export: {architecture!r} "
            f"(supported: {sorted(_SUPPORTED_ARCHITECTURES)})"
        )

    hp = {**DEFAULT_HYPERPARAMS, **hyperparams}
    # backend/model/notebook_export.py -> backend/model -> backend -> algoforge/
    repo_root = Path(__file__).resolve().parent.parent.parent
    commit = _git_commit(repo_root)
    generated_at = datetime.now(timezone.utc).isoformat()

    cells = []

    cells.append(nbf.v4.new_markdown_cell(f"""# {model_name} — algoforge reproduction notebook

Self-contained: runs on a bare Colab CPU runtime, no algoforge backend connection required.

- **Architecture**: `{architecture}`
- **Dataset**: algoforge dataset id `{dataset_id}`, snapshot id `{snapshot_id}`
- **Snapshot sha256**: `{snapshot_sha256}`
- **Generated**: {generated_at} from algoforge commit `{commit}`

The model/dataset/training-loop code below is copied verbatim (via Python's `inspect.getsource`,
at notebook-generation time) from the algoforge backend at the commit above — not a
reimplementation — so results should match what algoforge itself produces training the same
hyperparams on the same snapshot.

After training, download `best.pt` and `metrics.json` (the last cell does this automatically
inside Colab) and register them with algoforge — see the last cell of this notebook.
"""))

    cells.append(nbf.v4.new_code_cell(
        "!pip install -q pyarrow  # torch/numpy/pandas already present on Colab runtimes"
    ))

    cells.append(nbf.v4.new_code_cell(f"""import hashlib
import urllib.request

DATASET_ID = {dataset_id}
GIT_COMMIT = {commit!r}
SNAPSHOT_URL = {snapshot_url!r}
SNAPSHOT_SHA256 = {snapshot_sha256!r}
LOCAL_PATH = "dataset_snapshot.parquet"

urllib.request.urlretrieve(SNAPSHOT_URL, LOCAL_PATH)

_digest = hashlib.sha256()
with open(LOCAL_PATH, "rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        _digest.update(chunk)
_actual = _digest.hexdigest()
assert _actual == SNAPSHOT_SHA256, (
    f"snapshot hash mismatch: expected {{SNAPSHOT_SHA256}}, got {{_actual}} -- "
    "the file may have changed since this notebook was generated, or downloaded incorrectly"
)
print("snapshot OK:", _actual)
"""))

    hp_json = json.dumps(hp, indent=4)
    cells.append(nbf.v4.new_code_cell(f"""import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HYPERPARAMS = {hp_json}

_seed = HYPERPARAMS.get("seed")
if _seed is not None:
    torch.manual_seed(_seed)
    np.random.seed(_seed)
    random.seed(_seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
"""))

    model_src = _dedent_source(LSTMModel)
    cells.append(nbf.v4.new_code_cell(f"# --- model/architectures/lstm.py (verbatim, commit {commit}) ---\n{model_src}"))

    normalize_src = _dedent_source(OHLCWindowDataset._apply_normalize)
    make_windows_src = _dedent_source(OHLCWindowDataset._make_windows)
    cells.append(nbf.v4.new_code_cell(f"""# --- model/trainers/dataset.py: OHLCWindowDataset's core numeric methods (verbatim) ---
# Only the default path is reproduced here: no token_level, no preprocessing recipe, no
# clustering, chronological split only — see notebook_export.py's module docstring.
{normalize_src}


{make_windows_src}


class SimpleWindowDataset:
    \"\"\"Minimal re-creation of OHLCWindowDataset's default path: load a single Parquet
    snapshot, normalize, window, chronological train/val split. _apply_normalize and
    _make_windows above are algoforge's own code, copied verbatim — not reimplemented.\"\"\"

    def __init__(self, parquet_path, obs_len, pred_len, feature_cols, normalize, val_split, device):
        df = pd.read_parquet(parquet_path)
        raw = df[feature_cols].values.astype(np.float32)
        data = _apply_normalize(raw, normalize)

        all_src, all_tgt = _make_windows(data, data, obs_len, pred_len + 1)
        n_windows = len(all_src)
        split_idx = int(n_windows * (1 - val_split))
        self._train_src, self._train_tgt = all_src[:split_idx], all_tgt[:split_idx]
        if n_windows - split_idx > 0:
            self._val_src, self._val_tgt = all_src[split_idx:], all_tgt[split_idx:]
        else:
            self._val_src, self._val_tgt = self._train_src, self._train_tgt

        self._is_train = True
        self.device = device
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.n_features = len(feature_cols)

    def train(self):
        self._is_train = True

    def eval(self):
        self._is_train = False

    def __len__(self):
        return len(self._train_src) if self._is_train else len(self._val_src)

    def __getitem__(self, key):
        src_arr = self._train_src if self._is_train else self._val_src
        tgt_arr = self._train_tgt if self._is_train else self._val_tgt
        return torch.tensor(src_arr[key], device=self.device), torch.tensor(tgt_arr[key], device=self.device)


# train_epoch/eval_epoch below are copied from a module that imports the real OHLCWindowDataset
# only for a type hint (`ds: OHLCWindowDataset`) -- that module also has
# `from __future__ import annotations`, deferring the hint to a string there, but that pragma
# doesn't carry over into this notebook's cells. Alias it so the hint still resolves here too.
OHLCWindowDataset = SimpleWindowDataset
"""))

    split_tgt_src = _dedent_source(_split_tgt)
    train_epoch_src = _dedent_source(train_epoch)
    eval_epoch_src = _dedent_source(eval_epoch)
    cells.append(nbf.v4.new_code_cell(
        f"# --- model/trainers/supervised.py (verbatim) ---\n{split_tgt_src}\n\n{train_epoch_src}\n\n{eval_epoch_src}"
    ))

    cells.append(nbf.v4.new_code_cell("""dataset = SimpleWindowDataset(
    LOCAL_PATH,
    obs_len=HYPERPARAMS["obs_len"],
    pred_len=HYPERPARAMS["pred_len"],
    feature_cols=HYPERPARAMS["feature_cols"],
    normalize=HYPERPARAMS["normalize"],
    val_split=HYPERPARAMS["val_split"],
    device=device,
)

model = LSTMModel(
    input_dim=len(HYPERPARAMS["feature_cols"]),
    hidden_dim=HYPERPARAMS["hidden_dim"],
    output_dim=len(HYPERPARAMS["feature_cols"]),
    pred_len=HYPERPARAMS["pred_len"],
    num_layers=HYPERPARAMS["num_layers"],
    dropout=HYPERPARAMS["dropout"],
    device=device,
)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("num_params:", num_params)

optimizer = torch.optim.Adam(model.parameters(), lr=HYPERPARAMS["lr"])
criterion = nn.MSELoss()
"""))

    cells.append(nbf.v4.new_code_cell("""import json

epoch_metrics = []
best_val_loss = float("inf")
best_epoch = 0

for epoch in range(1, HYPERPARAMS["epochs"] + 1):
    train_loss = train_epoch(model, dataset, optimizer, criterion, HYPERPARAMS["batch_size"])
    val_loss = eval_epoch(model, dataset, criterion, HYPERPARAMS["batch_size"])
    epoch_metrics.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_loss": val_loss}, "best.pt")
    print(f"epoch {epoch}/{HYPERPARAMS['epochs']}: train={train_loss:.6f} val={val_loss:.6f}")

metadata = {
    "dataset_id": DATASET_ID,
    "hyperparams": HYPERPARAMS,
    "epoch_metrics": epoch_metrics,
    "best_epoch": best_epoch,
    "val_loss": best_val_loss,
    "num_params": num_params,
    "external_ref": {
        "platform": "colab",
        "git_commit": GIT_COMMIT,
        "dataset_snapshot_sha256": SNAPSHOT_SHA256,
    },
}
with open("metrics.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("best_epoch:", best_epoch, "best_val_loss:", best_val_loss)
print("Wrote best.pt and metrics.json")
"""))

    cells.append(nbf.v4.new_code_cell("""try:
    from google.colab import files
    files.download("best.pt")
    files.download("metrics.json")
except ImportError:
    print("Not running in Colab -- best.pt and metrics.json are in the working directory.")
"""))

    cells.append(nbf.v4.new_markdown_cell(f"""## Registering this run with algoforge

From `algoforge/backend`, with the downloaded `best.pt` and `metrics.json` in the current
directory:

```
python -m scripts.import_training_run --model-id <MODEL_ID> --dataset-id {dataset_id} \\
    --checkpoint best.pt --metadata metrics.json \\
    --notebook-url <this notebook's Colab/GitHub URL>
```

This registers the run as a `TrainingRun(source="external")` — it then shows up in
`/model/compare` and the model detail page alongside runs algoforge trained itself.
"""))

    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"name": model_name, "provenance": []},
    }
    return nb
