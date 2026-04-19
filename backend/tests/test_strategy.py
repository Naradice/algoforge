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


# ── Trades endpoint ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_run_trades_empty(client):
    cr = await client.post("/api/v1/strategies", json={"name": "Trades Strat", "description": ""})
    strat_id = cr.json()["data"]["id"]
    rr = await client.post(f"/api/v1/strategies/{strat_id}/runs", json={"mode": "backtest"})
    run_id = rr.json()["data"]["id"]

    r = await client.get(f"/api/v1/strategies/{strat_id}/runs/{run_id}/trades")
    assert r.status_code == 200
    body = r.json()
    assert body["data"] == []
    assert body["meta"]["total"] == 0


@pytest.mark.asyncio
async def test_get_run_trades_returns_trade_fields(client, db_session):
    from datetime import datetime, timezone
    from strategy.models import Trade

    cr = await client.post("/api/v1/strategies", json={"name": "Trade Fields Strat", "description": ""})
    strat_id = cr.json()["data"]["id"]
    rr = await client.post(f"/api/v1/strategies/{strat_id}/runs", json={"mode": "backtest"})
    run_id = rr.json()["data"]["id"]

    async with db_session() as db:
        trade = Trade(
            id=9001,  # explicit id — SQLite BigInteger doesn't autoincrement
            run_id=run_id, symbol="USDJPY", direction="buy",
            entry_price=150.0, exit_price=151.0, volume=0.1,
            profit=1000.0,
            opened_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            closed_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
            exit_reason="tp", phase="is", mae=0.5, mfe=1.5,
        )
        db.add(trade)
        await db.commit()

    r = await client.get(f"/api/v1/strategies/{strat_id}/runs/{run_id}/trades")
    assert r.status_code == 200
    body = r.json()
    assert body["meta"]["total"] == 1
    t = body["data"][0]
    assert t["symbol"] == "USDJPY"
    assert t["direction"] == "buy"
    assert t["entry_price"] == 150.0
    assert t["exit_price"] == 151.0
    assert t["profit"] == 1000.0
    assert t["exit_reason"] == "tp"
    assert t["phase"] == "is"
    assert t["mae"] == 0.5
    assert t["mfe"] == 1.5


@pytest.mark.asyncio
async def test_get_run_trades_wrong_strategy_returns_404(client, db_session):
    cr1 = await client.post("/api/v1/strategies", json={"name": "Owner Strat", "description": ""})
    strat1_id = cr1.json()["data"]["id"]
    rr = await client.post(f"/api/v1/strategies/{strat1_id}/runs", json={"mode": "backtest"})
    run_id = rr.json()["data"]["id"]

    cr2 = await client.post("/api/v1/strategies", json={"name": "Other Strat", "description": ""})
    strat2_id = cr2.json()["data"]["id"]

    r = await client.get(f"/api/v1/strategies/{strat2_id}/runs/{run_id}/trades")
    assert r.status_code == 404
