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

Scope: any architecture build_model() accepts works here — the generated notebook calls
model_core.trainers.get_trainer_fns(architecture) / get_default_criterion(architecture) the
same way celery_worker.py's _train_model does, rather than hardcoding a whitelist of
individually-verified architectures, so a new supervised/GAN/VAE architecture added to
model_core needs no change here to also work on Colab. Excluded: NON_GRADIENT_ARCHITECTURES
("rl_agent", "ar", "ma", "arma") — not torch.nn.Module-based, build_model() itself rejects
them regardless of config. split_mode ("chronological"/"regime_controlled") is passed through
to OHLCWindowDataset and verified end-to-end (see model/colab_trainer.py's
check_colab_supported). token_level and a preprocessing recipe are technically reachable now
that OHLCWindowDataset itself runs unmodified (see the dataset cell below) but are NOT exposed
by check_colab_supported yet — that restriction hasn't been lifted because it hasn't been
verified end-to-end (token_level needs vocab_size/embedding_dim wired into the model-build
cell; a preprocessing recipe needs the separate `finance_client` package installed too), not
because of a generator limitation.
"""
from __future__ import annotations

import pprint
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import nbformat as nbf

from model_core.architectures import NON_GRADIENT_ARCHITECTURES

DEFAULT_HYPERPARAMS = {
    "obs_len": 60,
    "pred_len": 10,
    "epochs": 20,
    "batch_size": 32,
    "lr": 0.001,
    "val_split": 0.2,
    "normalize": "returns",
    "feature_cols": ["close"],
    "split_mode": "chronological",
    "token_level": None,
    "n_bins": 7,
    "cluster_window": 20,
    "n_clusters": 20,
    "n_digits": 3,
    "sax_paa_size": 5,
    "embedding_dim": None,
    "preprocessing": None,
    "optimizer": "adam",
    "beta1": 0.9,
    "beta2": 0.999,
    "momentum": 0.0,
    "weight_decay": None,
    "disable_lr_scheduler": False,
    "shuffle": False,
    "lr_warmup_epochs": 0,
    "early_stop_patience": None,
    "divergence_factor": None,
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
    if architecture in NON_GRADIENT_ARCHITECTURES:
        raise ValueError(
            f"Unsupported architecture for notebook export: {architecture!r} is not "
            f"torch.nn.Module-based (see model_core.architectures.NON_GRADIENT_ARCHITECTURES)"
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
    if hp.get("token_level") == "cluster":
        # model_core's own [cluster] extra exists for this (see its pyproject.toml), but
        # expressing "package[extra] @ git+URL" correctly alongside #subdirectory in one pip
        # invocation is fiddly to get right -- a separate plain install is simpler and no less
        # correct, since scikit-learn has no version coupling to model_core itself.
        cells.append(nbf.v4.new_code_cell(
            '!pip install -q scikit-learn  # only needed for token_level="cluster" (KMeans)'
        ))
    if hp.get("preprocessing"):
        # A preprocessing recipe (indicators/clustering) is applied via
        # model_core.trainers.preprocessing.apply_preprocessing, which itself imports the
        # separate finance_client package (its indicator implementations) -- not one of
        # model_core's own dependencies, so it needs its own install here. --no-deps: skips
        # finance_client's own declared deps (a couple aren't needed just to compute
        # indicators, e.g. Windows-only MetaTrader5, which fails outright on Colab's Linux
        # runtime); the packages actually needed to import it are listed explicitly instead,
        # mirroring algoforge/backend/requirements.txt's own comment on why.
        cells.append(nbf.v4.new_code_cell(
            "# preprocessing recipe support needs finance_client (its indicator implementations)\n"
            "!pip install -q scipy python-dotenv PyYAML websocket-client google-auth-oauthlib "
            "pandas_datareader statsmodels yfinance matplotlib\n"
            '!pip install -q --no-deps "git+https://github.com/Naradice/finance_client.git"'
        ))

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

    # repr(), not json.dumps() -- json.dumps renders None/True/False as null/true/false, which
    # is JSON syntax, not Python. That's invisible until a value is actually None (no default
    # here was until token_level/embedding_dim were added), at which point the generated cell
    # fails with NameError: name 'null' is not defined. pformat gives the same readability with
    # correct Python literals.
    hp_json = pprint.pformat(hp, indent=4, sort_dicts=False)
    model_config_json = pprint.pformat(model_config, indent=4, sort_dicts=False)
    cells.append(nbf.v4.new_code_cell(f"""import random

import numpy as np
import torch
import torch.nn as nn

from model_core.architectures import build_model
from model_core.trainers import OHLCWindowDataset, get_default_criterion, get_trainer_fns

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
    split_mode=HYPERPARAMS["split_mode"],
    preprocessing=HYPERPARAMS["preprocessing"],
    token_level=HYPERPARAMS["token_level"],
    n_bins=HYPERPARAMS["n_bins"],
    cluster_window=HYPERPARAMS["cluster_window"],
    n_clusters=HYPERPARAMS["n_clusters"],
    n_digits=HYPERPARAMS["n_digits"],
    sax_paa_size=HYPERPARAMS["sax_paa_size"],
    device=device,
)

_model_kwargs = {
    **MODEL_CONFIG,
    "input_dim": dataset.n_features,
    "output_dim": dataset.n_features,
    "pred_len": HYPERPARAMS["pred_len"],
    # See OHLCWindowDataset.effective_seq_len's docstring (model_core/trainers/dataset.py)
    # for why this is needed (decoder_only) and harmless for every other architecture.
    # Same call celery_worker.py's _train_model makes -- not a separate calculation.
    "seq_len": dataset.effective_seq_len,
}
# token_level (see OHLCWindowDataset): when set, src is a stream of integer token ids rather
# than continuous features -- pass the fitted vocab size through so the model builds an
# embedding front-end instead of its usual continuous input path. Same condition
# celery_worker.py's _train_model checks.
if dataset.vocab_size is not None:
    _model_kwargs["vocab_size"] = dataset.vocab_size
    _model_kwargs["embedding_dim"] = HYPERPARAMS["embedding_dim"]

model = build_model(ARCHITECTURE, _model_kwargs, device=device)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("num_params:", num_params)

# optimizer/beta1/beta2/momentum/weight_decay: same construction celery_worker.py's
# _train_model uses -- see its comments for why weight_decay defaults to None (each optimizer's
# own torch default) rather than 0, and why beta1/beta2 are exposed at all (isolating whether an
# optimizer's own per-parameter moment estimates, not just an epoch-denominated LR schedule,
# explain a result).
_optimizer_name = str(HYPERPARAMS["optimizer"]).lower()
_weight_decay = HYPERPARAMS["weight_decay"]
if _optimizer_name == "sgd":
    _opt_kwargs = {"momentum": HYPERPARAMS["momentum"]}
    if _weight_decay is not None:
        _opt_kwargs["weight_decay"] = float(_weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), lr=HYPERPARAMS["lr"], **_opt_kwargs)
elif _optimizer_name == "adamw":
    _opt_kwargs = {"betas": (HYPERPARAMS["beta1"], HYPERPARAMS["beta2"])}
    if _weight_decay is not None:
        _opt_kwargs["weight_decay"] = float(_weight_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=HYPERPARAMS["lr"], **_opt_kwargs)
else:
    _opt_kwargs = {"betas": (HYPERPARAMS["beta1"], HYPERPARAMS["beta2"])}
    if _weight_decay is not None:
        _opt_kwargs["weight_decay"] = float(_weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=HYPERPARAMS["lr"], **_opt_kwargs)

# Dispatches to the same supervised/GAN/VAE loop celery_worker.py's _train_model uses for this
# architecture -- see model_core.trainers's module docstring for which architectures use which.
train_fn, eval_fn = get_trainer_fns(ARCHITECTURE)
criterion = get_default_criterion(ARCHITECTURE)

# disable_lr_scheduler: see celery_worker.py's _train_model -- an opt-in escape hatch for
# step-count-controlled comparisons (ReduceLROnPlateau's patience is epoch-denominated, same
# confound as early_stop_patience). None for gan/vae (criterion is None there too), matching
# _train_model's own `if criterion and not disable_lr_scheduler` condition exactly.
scheduler = (
    torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    if criterion and not HYPERPARAMS["disable_lr_scheduler"] else None
)
"""))

    cells.append(nbf.v4.new_code_cell("""import csv
import json
import os

_target_lr = HYPERPARAMS["lr"]
_lr_warmup_epochs = HYPERPARAMS["lr_warmup_epochs"]
_early_stop_patience = HYPERPARAMS["early_stop_patience"]
_divergence_factor = HYPERPARAMS["divergence_factor"]
epochs_since_improvement = 0
epoch_metrics = []
best_val_loss = float("inf")
best_epoch = 0

for epoch in range(1, HYPERPARAMS["epochs"] + 1):
    # lr_warmup_epochs: opt-in linear ramp from lr/N to the full target lr over the first N
    # epochs -- same as celery_worker.py's _train_model. The scheduler.step() call below is
    # held off during warmup so ReduceLROnPlateau doesn't fight the ramp with its own reductions.
    if _lr_warmup_epochs > 0 and epoch <= _lr_warmup_epochs:
        _warmup_lr = _target_lr * epoch / _lr_warmup_epochs
        for pg in optimizer.param_groups:
            pg["lr"] = _warmup_lr

    train_loss = train_fn(model, dataset, optimizer, criterion, HYPERPARAMS["batch_size"], HYPERPARAMS["shuffle"])
    val_loss = eval_fn(model, dataset, criterion, HYPERPARAMS["batch_size"])

    if scheduler and epoch >= _lr_warmup_epochs:
        scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        epochs_since_improvement = 0
        torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_loss": val_loss}, "best.pt")
    else:
        epochs_since_improvement += 1
    # best_epoch/best_val_loss recorded per-row (as of this epoch) so a consumer reading only
    # the last row -- see below -- gets the same "current status" progress.json used to convey,
    # without a second file. epoch_metrics itself (this full list) is still exactly what
    # metrics.json's "epoch_metrics" field gets at the end of the run.
    epoch_metrics.append({
        "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
        "best_epoch": best_epoch, "best_val_loss": best_val_loss,
    })
    # Rewritten in full every epoch (not appended -- the whole history is tiny even for
    # thousands of epochs) so an orchestrator polling this session (model/colab_trainer.py,
    # while colab exec is otherwise blocked until the whole run finishes) can show live progress
    # and decide when to stop early -- see that module's _poll_and_maybe_stop. Also doubles as
    # this run's resilience backup: if the final metrics.json fetch after colab exec returns
    # ever fails (e.g. a degraded colab-cli connection -- see colab_runner.download_with_retry's
    # docstring), colab_trainer.py falls back to whatever this file held as of the last
    # successful poll, so at most the last _POLL_INTERVAL_SECONDS-or-so of epochs are at risk of
    # being lost instead of the entire run's history. Atomic write (tmp file + rename) so a
    # concurrent `colab download` of metrics_log.csv never reads a half-written file.
    with open("metrics_log.csv.tmp", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "best_epoch", "best_val_loss"])
        writer.writeheader()
        writer.writerows(epoch_metrics)
    os.replace("metrics_log.csv.tmp", "metrics_log.csv")
    print(f"epoch {epoch}/{HYPERPARAMS['epochs']}: train={train_loss:.6f} val={val_loss:.6f}")

    # early_stop_patience: stops once this many consecutive epochs pass with no new best
    # val_loss (a plateau). divergence_factor: stops the moment val_loss exceeds
    # divergence_factor x best_val_loss (a blow-up), independent of patience -- see
    # celery_worker.py's _train_model for the full rationale for keeping these distinct.
    if _early_stop_patience is not None and epochs_since_improvement >= _early_stop_patience:
        print(f"early-stopped at epoch {epoch} (no improvement for {_early_stop_patience} epochs, best={best_val_loss:.6f} @ epoch {best_epoch})")
        break
    if _divergence_factor is not None and val_loss > best_val_loss * _divergence_factor:
        print(f"diverged at epoch {epoch}: val={val_loss:.6f} exceeds {_divergence_factor}x best ({best_val_loss:.6f}) -- stopping")
        break

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
