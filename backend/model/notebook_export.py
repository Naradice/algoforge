"""
Generate a self-contained Google Colab notebook that reproduces an algoforge training run on a
dataset snapshot (data/snapshot_service.py) — runnable on a bare Colab CPU runtime, no algoforge
backend connection required.

Installs and imports the ACTUAL model_core package (backend/model_core/ — see that package's
own __init__.py for why it has no backend dependencies) at execution time, from the exact git
commit this notebook was generated from:

    pip install "git+https://github.com/<org>/algoforge.git@<commit>#subdirectory=backend/model_core"

Both this backend's own celery worker (`from model_core.architectures import build_model`, see
celery_worker.py) and a Colab notebook generated here import the SAME package — not a
hand-copied snapshot of its source — so there is no possibility of drift between what a local
run and a Colab run actually execute. (An earlier version of this generator used
inspect.getsource() to embed source text directly; that only guaranteed textual identity at
generation time, not that both paths ran the same code — see docs/colab-workflow.md's history
for why this changed.)

Scope: architecture="lstm" is the only one verified against this generator's training-loop
wiring so far — see _SUPPORTED_ARCHITECTURES. Extending it only requires confirming the new
architecture's forward()/train_epoch call shape matches (build_model/OHLCWindowDataset/
train_epoch/eval_epoch themselves need no changes, since they're imported from model_core, not
regenerated here). token_level / a preprocessing recipe / split_mode are technically reachable
now that OHLCWindowDataset itself runs unmodified (see the dataset cell below) but are NOT
exposed by model/colab_trainer.py's check_colab_supported yet — that restriction hasn't been
lifted because it hasn't been verified end-to-end, not because of a generator limitation.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import nbformat as nbf

_SUPPORTED_ARCHITECTURES = {"lstm"}

DEFAULT_HYPERPARAMS = {
    "obs_len": 60,
    "pred_len": 10,
    "epochs": 20,
    "batch_size": 32,
    "lr": 0.001,
    "val_split": 0.2,
    "normalize": "returns",
    "feature_cols": ["close"],
    "seed": 42,
}


def _git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _git_remote_url(repo_root: Path) -> str:
    """HTTPS URL (ending in .git) of this repo's remote — prefers "origin", falls back to
    whatever remote exists (this repo's is named "remote", not "origin"). Used to build the
    `pip install git+...` URL model_core is installed from; a notebook that can't specify where
    to install model_core from can't run, so this raises rather than silently generating a
    broken notebook."""
    try:
        remotes = subprocess.check_output(
            ["git", "remote"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).split()
    except Exception as exc:
        raise RuntimeError("could not list git remotes -- is this running inside a git repo?") from exc
    if not remotes:
        raise RuntimeError("no git remote configured -- can't build a pip install URL for model_core")
    remote_name = "origin" if "origin" in remotes else remotes[0]
    url = subprocess.check_output(
        ["git", "remote", "get-url", remote_name], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
    ).strip()
    if not url.endswith(".git"):
        url += ".git"
    return url


def build_notebook(
    architecture: str,
    model_name: str,
    model_config: dict,
    dataset_id: int,
    snapshot_id: int,
    snapshot_url: str,
    snapshot_sha256: str,
    hyperparams: dict,
) -> nbf.NotebookNode:
    """*model_config* is the MLModel's own config (architecture-shape params like hidden_dim/
    num_layers — see model_core/architectures/__init__.py's ARCHITECTURE_DEFAULTS) — kept
    separate from *hyperparams* (obs_len/epochs/lr/etc.) the same way celery_worker.py's
    _train_model keeps model_config and hp separate, so a Colab run builds the model with the
    same shape as the MLModel record actually specifies, not whatever a hyperparams default
    happens to be."""
    if architecture not in _SUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"Unsupported architecture for notebook export: {architecture!r} "
            f"(supported: {sorted(_SUPPORTED_ARCHITECTURES)})"
        )

    hp = {**DEFAULT_HYPERPARAMS, **hyperparams}
    # backend/model/notebook_export.py -> backend/model -> backend -> algoforge/
    repo_root = Path(__file__).resolve().parent.parent.parent
    commit = _git_commit(repo_root)
    remote_url = _git_remote_url(repo_root)
    install_url = f"git+{remote_url}@{commit}#subdirectory=backend/model_core"
    generated_at = datetime.now(timezone.utc).isoformat()

    cells = []

    cells.append(nbf.v4.new_markdown_cell(f"""# {model_name} — algoforge reproduction notebook

Self-contained: runs on a bare Colab CPU runtime, no algoforge backend connection required.

- **Architecture**: `{architecture}`
- **Dataset**: algoforge dataset id `{dataset_id}`, snapshot id `{snapshot_id}`
- **Snapshot sha256**: `{snapshot_sha256}`
- **Generated**: {generated_at} from algoforge commit `{commit}`

This notebook installs and imports algoforge's actual `model_core` package
(`{install_url}`) rather than embedding a copy of its source — so it runs the exact same
model/dataset/training-loop code this backend's own training worker would, not a
reimplementation that could quietly drift from it.

After training, download `best.pt` and `metrics.json` (the last cell does this automatically
inside Colab) and register them with algoforge — see the last cell of this notebook.
"""))

    cells.append(nbf.v4.new_code_cell(f'!pip install -q "{install_url}"'))

    cells.append(nbf.v4.new_code_cell(f"""import hashlib
import os
import urllib.request

DATASET_ID = {dataset_id}
GIT_COMMIT = {commit!r}
SNAPSHOT_URL = {snapshot_url!r}
SNAPSHOT_SHA256 = {snapshot_sha256!r}

# OHLCWindowDataset resolves artifact_path as Path(ARTIFACT_STORE_PATH) / artifact_path (same
# convention the backend uses) -- point it at the current directory so the downloaded snapshot's
# plain filename resolves directly.
os.environ["ARTIFACT_STORE_PATH"] = "."
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
    model_config_json = json.dumps(model_config, indent=4)
    cells.append(nbf.v4.new_code_cell(f"""import random

import numpy as np
import torch
import torch.nn as nn

from model_core.architectures import build_model
from model_core.trainers.dataset import OHLCWindowDataset
from model_core.trainers.supervised import train_epoch, eval_epoch

ARCHITECTURE = {architecture!r}
HYPERPARAMS = {hp_json}
MODEL_CONFIG = {model_config_json}

_seed = HYPERPARAMS.get("seed")
if _seed is not None:
    torch.manual_seed(_seed)
    np.random.seed(_seed)
    random.seed(_seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
"""))

    cells.append(nbf.v4.new_code_cell("""dataset = OHLCWindowDataset(
    LOCAL_PATH,
    obs_len=HYPERPARAMS["obs_len"],
    pred_len=HYPERPARAMS["pred_len"],
    feature_cols=HYPERPARAMS["feature_cols"],
    normalize=HYPERPARAMS["normalize"],
    val_split=HYPERPARAMS["val_split"],
    device=device,
)

model = build_model(
    ARCHITECTURE,
    {
        **MODEL_CONFIG,
        "input_dim": dataset.n_features,
        "output_dim": dataset.n_features,
        "pred_len": HYPERPARAMS["pred_len"],
    },
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
    "model_config": MODEL_CONFIG,
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
