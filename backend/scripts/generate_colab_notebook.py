"""Generate a self-contained Colab notebook that reproduces an algoforge training run.

Usage (from algoforge/backend):

    # Using an already-uploaded snapshot (see scripts/export_dataset_snapshot.py --upload-gdrive)
    python -m scripts.generate_colab_notebook --architecture lstm --model-name tiny-lstm-v1 \\
        --snapshot-id 2 --hyperparams-json '{"obs_len":60,"epochs":20}' \\
        --out ../notebooks/tiny-lstm-v1.ipynb

    # Or pointing directly at a URL/hash, e.g. to generate before uploading anywhere
    python -m scripts.generate_colab_notebook --architecture lstm --model-name tiny-lstm-v1 \\
        --dataset-id 3 --snapshot-url https://... --snapshot-sha256 <hex> \\
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--snapshot-id", type=int, help="An uploaded DatasetSnapshot id")
    parser.add_argument("--dataset-id", type=int, help="Use with --snapshot-url/--snapshot-sha256 instead of --snapshot-id")
    parser.add_argument("--snapshot-url", help="Direct download URL, if not using --snapshot-id")
    parser.add_argument("--snapshot-sha256", help="sha256 of the file at --snapshot-url")
    parser.add_argument("--hyperparams-json", default="{}")
    parser.add_argument("--out", required=True, help="Output .ipynb path")
    args = parser.parse_args()

    if args.snapshot_id is not None:
        dataset_id, snapshot_url, snapshot_sha256 = asyncio.run(_resolve_snapshot(args.snapshot_id))
        snapshot_id = args.snapshot_id
    elif args.dataset_id is not None and args.snapshot_url and args.snapshot_sha256:
        dataset_id, snapshot_url, snapshot_sha256 = args.dataset_id, args.snapshot_url, args.snapshot_sha256
        snapshot_id = 0
    else:
        raise SystemExit("Provide either --snapshot-id, or all of --dataset-id/--snapshot-url/--snapshot-sha256")

    import nbformat as nbf

    from model.notebook_export import build_notebook

    nb = build_notebook(
        architecture=args.architecture,
        model_name=args.model_name,
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
