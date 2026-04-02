import io
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ── Datasources ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_datasource(client):
    r = await client.post("/api/v1/datasources", json={
        "name": "OHLC Feed",
        "type": "ohlc_download",
        "config": {"symbol": "USDJPY"},
    })
    assert r.status_code == 201
    body = r.json()
    assert body["data"]["name"] == "OHLC Feed"
    assert body["data"]["type"] == "ohlc_download"


@pytest.mark.asyncio
async def test_list_datasources(client):
    await client.post("/api/v1/datasources", json={"name": "DS List", "type": "manual_upload"})
    r = await client.get("/api/v1/datasources")
    assert r.status_code == 200
    assert isinstance(r.json()["data"], list)
    assert r.json()["meta"]["total"] >= 1


@pytest.mark.asyncio
async def test_get_datasource(client):
    cr = await client.post("/api/v1/datasources", json={"name": "DS Get", "type": "web_report"})
    ds_id = cr.json()["data"]["id"]

    r = await client.get(f"/api/v1/datasources/{ds_id}")
    assert r.status_code == 200
    assert r.json()["data"]["id"] == ds_id


@pytest.mark.asyncio
async def test_get_datasource_not_found(client):
    r = await client.get("/api/v1/datasources/999999")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_update_datasource(client):
    cr = await client.post("/api/v1/datasources", json={"name": "DS Update", "type": "manual_upload"})
    ds_id = cr.json()["data"]["id"]

    r = await client.patch(f"/api/v1/datasources/{ds_id}", json={"name": "DS Updated"})
    assert r.status_code == 200
    assert r.json()["data"]["name"] == "DS Updated"


@pytest.mark.asyncio
async def test_delete_datasource(client):
    cr = await client.post("/api/v1/datasources", json={"name": "DS Delete", "type": "manual_upload"})
    ds_id = cr.json()["data"]["id"]

    r = await client.delete(f"/api/v1/datasources/{ds_id}")
    assert r.status_code == 204

    r2 = await client.get(f"/api/v1/datasources/{ds_id}")
    assert r2.status_code == 404


# ── Collection Jobs ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_collection_job(client):
    cr = await client.post("/api/v1/datasources", json={"name": "Job DS", "type": "ohlc_download"})
    ds_id = cr.json()["data"]["id"]

    r = await client.post("/api/v1/collection-jobs", json={"datasource_id": ds_id})
    assert r.status_code == 201
    body = r.json()
    assert body["data"]["datasource_id"] == ds_id
    assert body["data"]["status"] == "idle"


@pytest.mark.asyncio
async def test_list_collection_jobs(client):
    cr = await client.post("/api/v1/datasources", json={"name": "Job DS List", "type": "ohlc_download"})
    ds_id = cr.json()["data"]["id"]
    await client.post("/api/v1/collection-jobs", json={"datasource_id": ds_id})

    r = await client.get("/api/v1/collection-jobs")
    assert r.status_code == 200
    assert len(r.json()["data"]) >= 1


@pytest.mark.asyncio
async def test_patch_collection_job(client):
    cr = await client.post("/api/v1/datasources", json={"name": "Job DS Patch", "type": "ohlc_download"})
    ds_id = cr.json()["data"]["id"]
    jr = await client.post("/api/v1/collection-jobs", json={"datasource_id": ds_id, "schedule_cron": "0 * * * *"})
    job_id = jr.json()["data"]["id"]

    r = await client.patch(f"/api/v1/collection-jobs/{job_id}", json={"enabled": False})
    assert r.status_code == 200
    assert r.json()["data"]["id"] == job_id


@pytest.mark.asyncio
async def test_delete_collection_job(client):
    cr = await client.post("/api/v1/datasources", json={"name": "Job DS Del", "type": "ohlc_download"})
    ds_id = cr.json()["data"]["id"]
    jr = await client.post("/api/v1/collection-jobs", json={"datasource_id": ds_id})
    job_id = jr.json()["data"]["id"]

    r = await client.delete(f"/api/v1/collection-jobs/{job_id}")
    assert r.status_code == 204


# ── Datasets ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_datasets(client):
    r = await client.get("/api/v1/datasets")
    assert r.status_code == 200
    assert isinstance(r.json()["data"], list)


def _write_test_parquet(tmpdir: str) -> str:
    """Write a minimal OHLC parquet file and return its relative path."""
    store = Path(tmpdir)
    artifact_path = "test_ds.parquet"
    idx = pd.date_range("2000-01-03", periods=20, freq="1min", tz="UTC")
    df = pd.DataFrame({
        "open": np.ones(20) * 100.0,
        "high": np.ones(20) * 101.0,
        "low": np.ones(20) * 99.0,
        "close": np.ones(20) * 100.0,
        "volume": np.ones(20, dtype=int) * 30,
    }, index=idx)
    df.index.name = "datetime"
    df.to_parquet(store / artifact_path)
    return artifact_path


@pytest.mark.asyncio
async def test_rename_dataset(client):
    """PATCH /datasets/{id} should update the dataset name."""
    cr = await client.post("/api/v1/datasources", json={"name": "DS Rename", "type": "manual_upload"})
    ds_id = cr.json()["data"]["id"]

    # Create a dataset by upload (simplest way to get a dataset record via API)
    csv_content = "datetime,open,high,low,close,volume\n2000-01-03 00:01:00,100,101,99,100,30\n"
    files = {"file": ("data.csv", csv_content.encode(), "text/csv")}
    data = {"datasource_id": str(ds_id)}
    upload_r = await client.post("/api/v1/datasets/upload", files=files, data=data)
    assert upload_r.status_code == 202, upload_r.text
    dataset_id = upload_r.json()["data"]["dataset_id"]

    r = await client.patch(f"/api/v1/datasets/{dataset_id}", json={"name": "Renamed Dataset"})
    assert r.status_code == 200
    assert r.json()["data"]["name"] == "Renamed Dataset"


@pytest.mark.asyncio
async def test_delete_dataset(client):
    """DELETE /datasets/{id} should remove the record."""
    cr = await client.post("/api/v1/datasources", json={"name": "DS Del", "type": "manual_upload"})
    ds_id = cr.json()["data"]["id"]

    csv_content = "datetime,open,high,low,close,volume\n2000-01-03 00:01:00,100,101,99,100,30\n"
    files = {"file": ("data.csv", csv_content.encode(), "text/csv")}
    upload_r = await client.post("/api/v1/datasets/upload", files=files, data={"datasource_id": str(ds_id)})
    assert upload_r.status_code == 202
    dataset_id = upload_r.json()["data"]["dataset_id"]

    del_r = await client.delete(f"/api/v1/datasets/{dataset_id}")
    assert del_r.status_code == 204

    get_r = await client.get(f"/api/v1/datasets/{dataset_id}")
    assert get_r.status_code == 404


@pytest.mark.asyncio
async def test_dataset_preview_ready(client):
    """GET /datasets/{id}/preview should return rows for a ready dataset."""
    artifact_store = os.environ.get("ARTIFACT_STORE_PATH", "/tmp/algoforge_test_artifacts")
    artifact_path = _write_test_parquet(artifact_store)

    # Insert dataset record directly via upload (which creates a ready record)
    csv_content = "datetime,open,high,low,close,volume\n" + "\n".join(
        f"2000-01-03 00:0{i}:00,100,101,99,100,30" for i in range(1, 6)
    ) + "\n"
    files = {"file": ("data.csv", csv_content.encode(), "text/csv")}
    upload_r = await client.post("/api/v1/datasets/upload", files=files)
    assert upload_r.status_code == 202
    dataset_id = upload_r.json()["data"]["dataset_id"]

    r = await client.get(f"/api/v1/datasets/{dataset_id}/preview?rows=3")
    assert r.status_code == 200
    rows = r.json()["data"]
    assert isinstance(rows, list)
    assert len(rows) <= 3


@pytest.mark.asyncio
async def test_dataset_preview_not_found(client):
    r = await client.get("/api/v1/datasets/999999/preview")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_dataset_get_not_found(client):
    r = await client.get("/api/v1/datasets/999999")
    assert r.status_code == 404


# ── Config endpoints ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_dataset_config_analyses(client):
    r = await client.get("/api/v1/dataset-config/analyses")
    assert r.status_code == 200
    assert "data" in r.json()
