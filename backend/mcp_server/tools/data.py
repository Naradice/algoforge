"""MCP tools — dataset and datasource inspection."""

from __future__ import annotations

import sqlalchemy as sa

from mcp_server import mcp


@mcp.tool()
async def list_datasets() -> list[dict]:
    """
    List all datasets with their symbol, timeframe, row count, and status.
    Datasets are the OHLC data artifacts used for backtesting and model training.
    """
    from database import async_session_factory
    from data.models import Dataset

    async with async_session_factory() as db:
        rows = (
            await db.execute(sa.select(Dataset).order_by(Dataset.created_at.desc()))
        ).scalars().all()

    return [
        {
            "id": d.id,
            "name": d.name,
            "datasource_id": d.datasource_id,
            "symbol": d.symbol,
            "timeframe": d.timeframe,
            "from_ts": d.from_ts.isoformat() if d.from_ts else None,
            "to_ts": d.to_ts.isoformat() if d.to_ts else None,
            "row_count": d.row_count,
            "status": d.status,
            "artifact_path": d.artifact_path,
            "created_at": d.created_at.isoformat(),
        }
        for d in rows
    ]


@mcp.tool()
async def get_dataset_characteristics(dataset_id: int) -> dict:
    """
    Get statistical characteristics of a dataset: ACF, Hurst exponent, fat tails,
    diffusion, seasonality, and other market microstructure metrics.

    Args:
        dataset_id: ID of the dataset.
    """
    from database import async_session_factory
    from data.models import Dataset, DataCharacteristics

    async with async_session_factory() as db:
        ds = (
            await db.execute(sa.select(Dataset).where(Dataset.id == dataset_id))
        ).scalar_one_or_none()

        if ds is None:
            return {"error": f"Dataset {dataset_id} not found"}

        char = (
            await db.execute(
                sa.select(DataCharacteristics)
                .where(DataCharacteristics.dataset_id == dataset_id)
                .order_by(DataCharacteristics.computed_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()

    result = {
        "dataset_id": dataset_id,
        "name": ds.name,
        "symbol": ds.symbol,
        "timeframe": ds.timeframe,
        "row_count": ds.row_count,
        "from_ts": ds.from_ts.isoformat() if ds.from_ts else None,
        "to_ts": ds.to_ts.isoformat() if ds.to_ts else None,
    }

    if char is None:
        result["characteristics"] = None
        result["note"] = "No characteristics computed yet. Run POST /datasets/{id}/characteristics/compute."
    else:
        result["characteristics"] = char.metrics
        result["computed_at"] = char.computed_at.isoformat()
        result["interpretation"] = _interpret_characteristics(char.metrics)

    return result


@mcp.tool()
async def list_datasources() -> list[dict]:
    """
    List all configured datasources (OHLC download, DDM simulation, web report).
    Shows which collection jobs are associated with each source.
    """
    from database import async_session_factory
    from data.models import Datasource, CollectionJob

    async with async_session_factory() as db:
        sources = (
            await db.execute(sa.select(Datasource).order_by(Datasource.created_at.desc()))
        ).scalars().all()

        result = []
        for s in sources:
            jobs = (
                await db.execute(
                    sa.select(CollectionJob).where(CollectionJob.datasource_id == s.id)
                )
            ).scalars().all()

            result.append({
                "id": s.id,
                "name": s.name,
                "type": s.type,
                "config": s.config,
                "created_at": s.created_at.isoformat(),
                "collection_jobs": [
                    {
                        "id": j.id,
                        "schedule": j.schedule,
                        "status": j.status,
                        "last_run_at": j.last_run_at.isoformat() if j.last_run_at else None,
                        "last_error": j.last_error,
                    }
                    for j in jobs
                ],
            })

    return result


@mcp.tool()
async def get_dataset_preview(dataset_id: int, rows: int = 5) -> dict:
    """
    Get the first and last N rows of a dataset for a quick data sanity check.

    Args:
        dataset_id: ID of the dataset.
        rows:       Number of rows from head and tail to return (max 20 each).
    """
    import os
    from pathlib import Path

    import pandas as pd

    from database import async_session_factory
    from data.models import Dataset

    rows = min(rows, 20)

    async with async_session_factory() as db:
        ds = (
            await db.execute(sa.select(Dataset).where(Dataset.id == dataset_id))
        ).scalar_one_or_none()

    if ds is None:
        return {"error": f"Dataset {dataset_id} not found"}
    if ds.artifact_path is None:
        return {"error": "Dataset has no artifact (collection not yet run)"}

    store = Path(os.getenv("ARTIFACT_STORE_PATH", "artifacts"))
    path = store / ds.artifact_path

    if not path.exists():
        return {"error": f"Artifact file not found at {path}"}

    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]

    head = df.head(rows).reset_index()
    tail = df.tail(rows).reset_index()

    return {
        "dataset_id": dataset_id,
        "name": ds.name,
        "total_rows": len(df),
        "columns": list(df.columns),
        "head": head.to_dict(orient="records"),
        "tail": tail.to_dict(orient="records"),
    }


def _interpret_characteristics(metrics: dict) -> str:
    if not metrics:
        return "No metrics to interpret."
    parts = []
    if "hurst_exponent" in metrics:
        h = metrics["hurst_exponent"]
        regime = "trending" if h > 0.55 else "mean-reverting" if h < 0.45 else "random walk"
        parts.append(f"Hurst {h:.3f} → {regime}")
    if "autocorrelation_lag1" in metrics:
        ac = metrics["autocorrelation_lag1"]
        parts.append(f"Lag-1 autocorrelation {ac:.3f}")
    if "kurtosis" in metrics:
        k = metrics["kurtosis"]
        parts.append(f"Kurtosis {k:.2f} ({'fat tails' if k > 3 else 'normal tails'})")
    if "volatility_annualised" in metrics:
        parts.append(f"Annualised vol {metrics['volatility_annualised']:.1%}")
    return "; ".join(parts) if parts else "Characteristics computed — see raw metrics."


@mcp.tool()
async def create_datasource(name: str, type: str, config: dict) -> dict:
    """
    Create a new data source.

    Args:
        name:   Datasource name.
        type:   Type: "ohlc_download", "web_report", "manual_upload", "economic_calendar".
        config: Type-specific configuration (symbol, timeframe, provider, etc.).
    """
    from database import async_session_factory
    from data.service import data_service
    from data.models import DatasourceCreate

    async with async_session_factory() as db:
        body = DatasourceCreate(name=name, type=type, config=config)
        ds = await data_service.create_datasource(db, body)
        return {"id": ds.id, "name": ds.name, "type": ds.type}


@mcp.tool()
async def collect_data(datasource_id: int, from_ts: str | None = None, to_ts: str | None = None) -> dict:
    """
    Trigger data collection for a datasource.
    Returns the collection job ID to track progress.

    Args:
        datasource_id: ID of the datasource to collect from.
        from_ts:       ISO datetime for start of collection window (optional).
        to_ts:         ISO datetime for end of collection window (optional).
    """
    from database import async_session_factory
    from data.service import data_service

    async with async_session_factory() as db:
        job = await data_service.trigger_datasource_collection(db, datasource_id)
        return {"job_id": job.id, "status": job.status}


@mcp.tool()
async def analyze_dataset(dataset_id: int, analyses: list[str] | None = None) -> dict:
    """
    Trigger characteristic analysis for a dataset.
    Analysis includes: return distribution, ACF, CCDF, Hurst exponent, ADF test.

    Args:
        dataset_id: ID of the dataset to analyze.
        analyses:   Optional list of specific analysis names. None = run all.
    """
    from database import async_session_factory
    from data.service import data_service

    async with async_session_factory() as db:
        dataset = await data_service.trigger_analysis(db, dataset_id)
        return {"dataset_id": dataset.id, "status": dataset.status}


@mcp.tool()
async def get_dataset_info(dataset_id: int) -> dict:
    """
    Get detailed information about a dataset including row count, time range, and status.

    Args:
        dataset_id: ID of the dataset.
    """
    from database import async_session_factory
    from data.service import data_service

    async with async_session_factory() as db:
        ds = await data_service.get_dataset(db, dataset_id)
        chars = await data_service.get_characteristics(db, dataset_id)
    result = {
        "id": ds.id,
        "name": ds.name,
        "symbol": ds.symbol,
        "timeframe": ds.timeframe,
        "status": ds.status,
        "row_count": ds.row_count,
        "from_ts": ds.from_ts.isoformat() if ds.from_ts else None,
        "to_ts": ds.to_ts.isoformat() if ds.to_ts else None,
        "artifact_path": ds.artifact_path,
    }
    if chars:
        result["characteristics"] = chars.metrics
    return result
