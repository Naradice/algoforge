"""Target assessments: POST creates the row and enqueues assess_target; GET returns the verdict.

conftest stubs celery_app (enqueue is a no-op), so each test runs the task body the worker would
run -- celery_worker._assess_target -- directly after the POST."""
import numpy as np
import pandas as pd


async def _dataset(db_session, tmp_path, monkeypatch, n_days=30, per_day=300, name="ds", seed=0):
    from data.models import Dataset

    monkeypatch.setenv("ARTIFACT_STORE_PATH", str(tmp_path))
    rng = np.random.default_rng(seed)
    idx, vals, price = [], [], 100.0
    for d in range(n_days):
        sig = 1e-3 * 3.0 ** (d % 4)            # volatility constant within a day, varying by day
        p = price * np.exp(np.cumsum(rng.normal(0, sig, per_day)))
        price = p[-1]
        idx.append(pd.date_range(pd.Timestamp("2024-01-01") + pd.Timedelta(days=d), periods=per_day, freq="1min"))
        vals.append(p)
    close = np.concatenate(vals)
    df = pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1.0},
                      index=idx[0].append(idx[1:]))
    df.to_parquet(tmp_path / f"{name}.parquet")
    async with db_session() as db:
        ds = Dataset(name="assessment test", symbol=name, artifact_path=f"{name}.parquet", status="ready", row_count=len(df))
        db.add(ds)
        await db.commit()
        return ds.id


async def _run_worker(assessment_id):
    import celery_worker
    return await celery_worker._assess_target(assessment_id)


async def test_assessment_runs_and_reports_a_verdict(client, db_session, tmp_path, monkeypatch):
    dataset_id = await _dataset(db_session, tmp_path, monkeypatch)
    r = await client.post("/api/v1/target-assessments", json={
        "dataset_id": dataset_id, "target": "future_log_rv", "horizon": 60, "obs": 60,
        "with_time": False, "models": ["hgb"]})
    assert r.status_code == 202, r.text
    aid = r.json()["data"]["id"]
    assert r.json()["data"]["status"] == "pending"
    await _run_worker(aid)

    r = await client.get(f"/api/v1/target-assessments/{aid}")
    body = r.json()["data"]
    assert body["status"] == "completed", body
    assert body["verdict"] == "trivial"       # daily-constant volatility over a 60-return horizon
    assert body["reason"] and body["result"]["metrics"]["linear"]["r2"] > 0.95


async def test_invalid_parameters_are_rejected(client, db_session, tmp_path, monkeypatch):
    dataset_id = await _dataset(db_session, tmp_path, monkeypatch, n_days=2)
    r = await client.post("/api/v1/target-assessments", json={
        "dataset_id": dataset_id, "target": "sharpe", "horizon": 999, "models": ["xgboost"]})
    assert r.status_code == 422
    assert "target must be one of" in r.text and "horizon" in r.text and "models" in r.text


async def test_unknown_dataset_and_assessment_404(client):
    r = await client.post("/api/v1/target-assessments", json={"dataset_id": 987654})
    assert r.status_code == 404
    r = await client.get("/api/v1/target-assessments/987654")
    assert r.status_code == 404


async def test_failure_is_recorded_not_left_running(client, db_session, tmp_path, monkeypatch):
    dataset_id = await _dataset(db_session, tmp_path, monkeypatch, n_days=1, per_day=50)  # too few windows
    r = await client.post("/api/v1/target-assessments", json={"dataset_id": dataset_id, "with_time": False,
                                                              "models": ["hgb"]})
    aid = r.json()["data"]["id"]
    await _run_worker(aid)
    body = (await client.get(f"/api/v1/target-assessments/{aid}")).json()["data"]
    assert body["status"] == "error" and "not enough" in body["error_message"]


async def test_assessment_with_other_instruments(client, db_session, tmp_path, monkeypatch):
    main_id = await _dataset(db_session, tmp_path, monkeypatch, name="MAIN")
    other_id = await _dataset(db_session, tmp_path, monkeypatch, name="OTHER", seed=1)
    r = await client.post("/api/v1/target-assessments", json={
        "dataset_id": main_id, "target": "future_return", "horizon": 5, "with_time": False,
        "models": ["hgb"], "exog_dataset_ids": [other_id]})
    assert r.status_code == 202, r.text
    aid = r.json()["data"]["id"]
    await _run_worker(aid)
    body = (await client.get(f"/api/v1/target-assessments/{aid}")).json()["data"]
    assert body["status"] == "completed", body
    assert body["result"]["exog"] == ["OTHER"] and "exog_ci95" in body["result"]["metrics"]["linear"]

    r = await client.post("/api/v1/target-assessments", json={"dataset_id": main_id, "exog_dataset_ids": [main_id]})
    assert r.status_code == 422 and "exog_dataset_ids" in r.text
    r = await client.post("/api/v1/target-assessments", json={"dataset_id": main_id, "exog_dataset_ids": [987654]})
    assert r.status_code == 404
