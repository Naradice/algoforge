import pytest


@pytest.mark.asyncio
async def test_list_webhooks_empty(client):
    r = await client.get("/api/v1/webhooks")
    assert r.status_code == 200
    assert isinstance(r.json()["data"], list)


@pytest.mark.asyncio
async def test_create_webhook(client):
    r = await client.post("/api/v1/webhooks", json={
        "url": "https://example.com/hook",
        "events": ["run.completed", "run.error"],
        "secret": "supersecret",
    })
    assert r.status_code == 201
    body = r.json()
    assert body["data"]["url"] == "https://example.com/hook"
    assert body["data"]["active"] is True
    assert "run.completed" in body["data"]["events"]


@pytest.mark.asyncio
async def test_list_webhooks_after_create(client):
    await client.post("/api/v1/webhooks", json={
        "url": "https://example.com/hook2",
        "events": ["run.completed"],
        "secret": "s3cr3t",
    })
    r = await client.get("/api/v1/webhooks")
    assert r.status_code == 200
    assert len(r.json()["data"]) >= 1


@pytest.mark.asyncio
async def test_delete_webhook(client):
    cr = await client.post("/api/v1/webhooks", json={
        "url": "https://example.com/hook-del",
        "events": [],
        "secret": "s",
    })
    wh_id = cr.json()["data"]["id"]

    r = await client.delete(f"/api/v1/webhooks/{wh_id}")
    assert r.status_code == 204


@pytest.mark.asyncio
async def test_delete_webhook_not_found(client):
    r = await client.delete("/api/v1/webhooks/999999")
    assert r.status_code == 404
