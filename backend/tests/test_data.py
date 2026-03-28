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


# ── Config endpoints ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_dataset_config_analyses(client):
    r = await client.get("/api/v1/dataset-config/analyses")
    assert r.status_code == 200
    assert "data" in r.json()
