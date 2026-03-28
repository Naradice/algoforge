import pytest


# ── Model CRUD ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_model(client):
    r = await client.post("/api/v1/models", json={
        "name": "LSTM v1",
        "architecture": "lstm",
        "config": {"hidden_size": 128},
    })
    assert r.status_code == 201
    body = r.json()
    assert body["data"]["name"] == "LSTM v1"
    assert body["data"]["architecture"] == "lstm"
    assert body["data"]["status"] == "created"


@pytest.mark.asyncio
async def test_list_models(client):
    await client.post("/api/v1/models", json={"name": "Model List", "architecture": "lstm"})
    r = await client.get("/api/v1/models")
    assert r.status_code == 200
    assert isinstance(r.json()["data"], list)
    assert r.json()["meta"]["total"] >= 1


@pytest.mark.asyncio
async def test_list_models_filter_architecture(client):
    await client.post("/api/v1/models", json={"name": "Transformer v1", "architecture": "seq2seq_transformer"})
    r = await client.get("/api/v1/models?architecture=seq2seq_transformer")
    assert r.status_code == 200
    for item in r.json()["data"]:
        assert item["architecture"] == "seq2seq_transformer"


@pytest.mark.asyncio
async def test_get_model(client):
    cr = await client.post("/api/v1/models", json={"name": "Model Get", "architecture": "lstm"})
    model_id = cr.json()["data"]["id"]

    r = await client.get(f"/api/v1/models/{model_id}")
    assert r.status_code == 200
    assert r.json()["data"]["id"] == model_id


@pytest.mark.asyncio
async def test_get_model_not_found(client):
    r = await client.get("/api/v1/models/999999")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_update_model(client):
    cr = await client.post("/api/v1/models", json={"name": "Model Update", "architecture": "lstm"})
    model_id = cr.json()["data"]["id"]

    r = await client.patch(f"/api/v1/models/{model_id}", json={"name": "Model Updated"})
    assert r.status_code == 200
    assert r.json()["data"]["name"] == "Model Updated"


@pytest.mark.asyncio
async def test_delete_model(client):
    cr = await client.post("/api/v1/models", json={"name": "Model Delete", "architecture": "lstm"})
    model_id = cr.json()["data"]["id"]

    r = await client.delete(f"/api/v1/models/{model_id}")
    assert r.status_code == 204

    r2 = await client.get(f"/api/v1/models/{model_id}")
    assert r2.status_code == 404


# ── Training Runs ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_training_run(client):
    cr = await client.post("/api/v1/models", json={"name": "Model TR", "architecture": "lstm"})
    model_id = cr.json()["data"]["id"]

    r = await client.post(f"/api/v1/models/{model_id}/training-runs", json={
        "dataset_id": 1,
        "hyperparams": {"lr": 0.001, "epochs": 10},
    })
    assert r.status_code == 202
    body = r.json()
    assert body["data"]["model_id"] == model_id
    assert body["data"]["status"] == "pending"


@pytest.mark.asyncio
async def test_list_training_runs(client):
    cr = await client.post("/api/v1/models", json={"name": "Model TR List", "architecture": "lstm"})
    model_id = cr.json()["data"]["id"]
    await client.post(f"/api/v1/models/{model_id}/training-runs", json={"dataset_id": 1})

    r = await client.get(f"/api/v1/models/{model_id}/training-runs")
    assert r.status_code == 200
    assert len(r.json()["data"]) == 1


@pytest.mark.asyncio
async def test_stop_training_run(client):
    cr = await client.post("/api/v1/models", json={"name": "Model Stop", "architecture": "lstm"})
    model_id = cr.json()["data"]["id"]
    tr = await client.post(f"/api/v1/models/{model_id}/training-runs", json={"dataset_id": 1})
    run_id = tr.json()["data"]["id"]

    r = await client.post(f"/api/v1/training-runs/{run_id}/stop")
    assert r.status_code == 200
    # stop_requested should be set
    status_r = await client.get(f"/api/v1/training-runs/{run_id}/status")
    assert status_r.status_code == 200


@pytest.mark.asyncio
async def test_get_epoch_metrics_empty(client):
    cr = await client.post("/api/v1/models", json={"name": "Model Metrics", "architecture": "lstm"})
    model_id = cr.json()["data"]["id"]
    tr = await client.post(f"/api/v1/models/{model_id}/training-runs", json={"dataset_id": 1})
    run_id = tr.json()["data"]["id"]

    r = await client.get(f"/api/v1/training-runs/{run_id}/metrics")
    assert r.status_code == 200
    assert r.json()["data"] == []


# ── Config endpoints ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_architectures(client):
    r = await client.get("/api/v1/model-config/architectures")
    assert r.status_code == 200
    data = r.json()["data"]
    assert isinstance(data, list)
    assert len(data) > 0
