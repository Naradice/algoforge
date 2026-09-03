"""Generate a self-contained Colab notebook that reproduces an algoforge training run.

Usage (from algoforge/backend):

    # Using an existing MLModel (architecture + config come from the DB) and an uploaded
    # snapshot (see scripts/export_dataset_snapshot.py --upload-gdrive)
    python -m scripts.generate_colab_notebook --model-id 11 --model-name tiny-lstm-v1 \\
        --snapshot-id 2 --hyperparams-json '{"obs_len":60,"epochs":20}' \\
        --out ../notebooks/tiny-lstm-v1.ipynb

    # Or without a DB model record, specifying architecture/config directly, and pointing
    # directly at a URL/hash (e.g. to generate before uploading anywhere)
    python -m scripts.generate_colab_notebook --architecture lstm --model-config-json '{"hidden_dim":32}' \\
        --model-name tiny-lstm-v1 --dataset-id 3 --snapshot-url https://... --snapshot-sha256 <hex> \\
        --out ../notebooks/tiny-lstm-v1.ipynb

See model/notebook_export.py for what the generated notebook contains and its scope
(architecture="lstm" only, default OHLCWindowDataset path only).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass


async def _resolve_snapshot(snapshot_id: int) -> tuple[int, str, str]:
    from sqlalchemy import select

    from data.models import DatasetSnapshot
    from database import async_session_factory

    async with async_session_factory() as db:
        snapshot = (
            await db.execute(select(DatasetSnapshot).where(DatasetSnapshot.id == snapshot_id))
        ).scalar_one_or_none()
        if snapshot is None:
            raise SystemExit(f"DatasetSnapshot {snapshot_id} not found")
        if snapshot.status != "uploaded" or not snapshot.export_ref or "url" not in snapshot.export_ref:
            raise SystemExit(
                f"DatasetSnapshot {snapshot_id} has not been uploaded to an external provider yet "
                f"(status={snapshot.status!r}). Run scripts/export_dataset_snapshot.py --upload-gdrive "
                "first, or pass --dataset-id/--snapshot-url/--snapshot-sha256 directly."
            )
        return snapshot.dataset_id, snapshot.export_ref["url"], snapshot.sha256


async def _resolve_model(model_id: int) -> tuple[str, dict]:
    from sqlalchemy import select

    from database import async_session_factory
    from model.models import MLModel

    async with async_session_factory() as db:
        model = (await db.execute(select(MLModel).where(MLModel.id == model_id))).scalar_one_or_none()
        if model is None:
            raise SystemExit(f"MLModel {model_id} not found")
        return model.architecture, dict(model.config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", type=int, help="Load architecture + config from this MLModel")
    parser.add_argument("--architecture", help="Required if --model-id is not given")
    parser.add_argument("--model-config-json", default="{}", help="Used only if --model-id is not given")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--snapshot-id", type=int, help="An uploaded DatasetSnapshot id")
    parser.add_argument("--dataset-id", type=int, help="Use with --snapshot-url/--snapshot-sha256 instead of --snapshot-id")
    parser.add_argument("--snapshot-url", help="Direct download URL, if not using --snapshot-id")
    parser.add_argument("--snapshot-sha256", help="sha256 of the file at --snapshot-url")
    parser.add_argument("--hyperparams-json", default="{}")
    parser.add_argument("--out", required=True, help="Output .ipynb path")
    args = parser.parse_args()

    if args.model_id is None and not args.architecture:
        raise SystemExit("Provide either --model-id or --architecture")
    if args.snapshot_id is None and not (args.dataset_id is not None and args.snapshot_url and args.snapshot_sha256):
        raise SystemExit("Provide either --snapshot-id, or all of --dataset-id/--snapshot-url/--snapshot-sha256")

    async def _resolve_all():
        # Both DB lookups run inside one asyncio.run() -- calling asyncio.run() twice in a row
        # here hit a Windows/asyncpg issue where the second run's connection-pool teardown
        # touches an already-closed proactor event loop.
        if args.model_id is not None:
            architecture, model_config = await _resolve_model(args.model_id)
        else:
            architecture, model_config = args.architecture, json.loads(args.model_config_json)

        if args.snapshot_id is not None:
            dataset_id, snapshot_url, snapshot_sha256 = await _resolve_snapshot(args.snapshot_id)
            snapshot_id = args.snapshot_id
        else:
            dataset_id, snapshot_url, snapshot_sha256 = args.dataset_id, args.snapshot_url, args.snapshot_sha256
            snapshot_id = 0
        return architecture, model_config, dataset_id, snapshot_url, snapshot_sha256, snapshot_id

    architecture, model_config, dataset_id, snapshot_url, snapshot_sha256, snapshot_id = asyncio.run(_resolve_all())

    import nbformat as nbf

    from model.notebook_export import build_notebook

    nb = build_notebook(
        architecture=architecture,
        model_name=args.model_name,
        model_config=model_config,
        dataset_id=dataset_id,
        snapshot_id=snapshot_id,
        snapshot_url=snapshot_url,
        snapshot_sha256=snapshot_sha256,
        hyperparams=json.loads(args.hyperparams_json),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(out_path))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
