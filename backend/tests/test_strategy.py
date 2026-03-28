import pytest


# ── Strategy CRUD ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_strategy(client):
    r = await client.post("/api/v1/strategies", json={"name": "My Strat", "description": "desc", "definition": {}})
    assert r.status_code == 201
    body = r.json()
    assert body["data"]["name"] == "My Strat"
    assert body["data"]["status"] == "inactive"


@pytest.mark.asyncio
async def test_list_strategies(client):
    # Create one first
    await client.post("/api/v1/strategies", json={"name": "List Test", "description": ""})
    r = await client.get("/api/v1/strategies")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["data"], list)
    assert len(body["data"]) >= 1
    assert body["meta"]["total"] >= 1


@pytest.mark.asyncio
async def test_get_strategy(client):
    cr = await client.post("/api/v1/strategies", json={"name": "Get Test", "description": ""})
    strat_id = cr.json()["data"]["id"]

    r = await client.get(f"/api/v1/strategies/{strat_id}")
    assert r.status_code == 200
    assert r.json()["data"]["id"] == strat_id


@pytest.mark.asyncio
async def test_get_strategy_not_found(client):
    r = await client.get("/api/v1/strategies/999999")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_update_strategy(client):
    cr = await client.post("/api/v1/strategies", json={"name": "Update Test", "description": ""})
    strat_id = cr.json()["data"]["id"]

    r = await client.patch(f"/api/v1/strategies/{strat_id}", json={"name": "Updated Name"})
    assert r.status_code == 200
    assert r.json()["data"]["name"] == "Updated Name"


@pytest.mark.asyncio
async def test_delete_strategy(client):
    cr = await client.post("/api/v1/strategies", json={"name": "Delete Test", "description": ""})
    strat_id = cr.json()["data"]["id"]

    r = await client.delete(f"/api/v1/strategies/{strat_id}")
    assert r.status_code == 204

    r2 = await client.get(f"/api/v1/strategies/{strat_id}")
    assert r2.status_code == 404


@pytest.mark.asyncio
async def test_strategy_status_filter(client):
    await client.post("/api/v1/strategies", json={"name": "Active Strat", "description": ""})
    r = await client.get("/api/v1/strategies?status=inactive")
    assert r.status_code == 200
    for item in r.json()["data"]:
        assert item["status"] == "inactive"


# ── Strategy Runs ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_run(client):
    cr = await client.post("/api/v1/strategies", json={"name": "Run Test Strat", "description": ""})
    strat_id = cr.json()["data"]["id"]

    r = await client.post(f"/api/v1/strategies/{strat_id}/runs", json={"mode": "backtest"})
    assert r.status_code == 202
    body = r.json()
    assert body["data"]["mode"] == "backtest"
    assert body["data"]["status"] == "pending"


@pytest.mark.asyncio
async def test_list_runs(client):
    cr = await client.post("/api/v1/strategies", json={"name": "List Runs Strat", "description": ""})
    strat_id = cr.json()["data"]["id"]
    await client.post(f"/api/v1/strategies/{strat_id}/runs", json={"mode": "backtest"})

    r = await client.get(f"/api/v1/strategies/{strat_id}/runs")
    assert r.status_code == 200
    assert len(r.json()["data"]) == 1


@pytest.mark.asyncio
async def test_get_run(client):
    cr = await client.post("/api/v1/strategies", json={"name": "Get Run Strat", "description": ""})
    strat_id = cr.json()["data"]["id"]
    rr = await client.post(f"/api/v1/strategies/{strat_id}/runs", json={"mode": "backtest"})
    run_id = rr.json()["data"]["id"]

    r = await client.get(f"/api/v1/strategies/{strat_id}/runs/{run_id}")
    assert r.status_code == 200
    assert r.json()["data"]["id"] == run_id


# ── Version history ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_version_created_on_definition_change(client):
    cr = await client.post("/api/v1/strategies", json={"name": "Version Strat", "description": "", "definition": {"v": 1}})
    strat_id = cr.json()["data"]["id"]

    # Patch with a new definition — should create a version snapshot
    await client.patch(f"/api/v1/strategies/{strat_id}", json={"definition": {"v": 2}})

    r = await client.get(f"/api/v1/strategies/{strat_id}/versions")
    assert r.status_code == 200
    versions = r.json()["data"]
    assert len(versions) >= 1


@pytest.mark.asyncio
async def test_no_version_on_name_only_change(client):
    cr = await client.post("/api/v1/strategies", json={"name": "NoVer Strat", "description": "", "definition": {"v": 1}})
    strat_id = cr.json()["data"]["id"]

    await client.patch(f"/api/v1/strategies/{strat_id}", json={"name": "Renamed"})

    r = await client.get(f"/api/v1/strategies/{strat_id}/versions")
    assert r.status_code == 200
    assert len(r.json()["data"]) == 0
