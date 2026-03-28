"""Unit tests for Pydantic schema validation."""
import pytest
from pydantic import ValidationError

from strategy.models import StrategyCreate, StrategyUpdate, StrategyRunCreate
from model.models import MLModelCreate, MLModelUpdate, TrainingRunCreate, HyperparamSearchCreate
from data.models import DatasourceCreate, DatasourceUpdate, CollectionJobCreate, CollectionJobUpdate
from schemas import DataResponse, Meta


# ── Strategy schemas ───────────────────────────────────────────────────────────

class TestStrategyCreate:
    def test_requires_name(self):
        with pytest.raises(ValidationError):
            StrategyCreate()  # type: ignore

    def test_name_only_is_valid(self):
        s = StrategyCreate(name="My Strategy")
        assert s.name == "My Strategy"

    def test_description_defaults_empty(self):
        assert StrategyCreate(name="X").description == ""

    def test_definition_defaults_empty_dict(self):
        assert StrategyCreate(name="X").definition == {}

    def test_accepts_nested_definition(self):
        s = StrategyCreate(name="X", definition={"entry": {"conditions": []}})
        assert s.definition["entry"]["conditions"] == []


class TestStrategyUpdate:
    def test_all_fields_optional(self):
        u = StrategyUpdate()
        assert u.name is None
        assert u.description is None
        assert u.definition is None
        assert u.status is None

    def test_partial_update_is_valid(self):
        u = StrategyUpdate(name="New Name")
        assert u.name == "New Name"
        assert u.definition is None


class TestStrategyRunCreate:
    def test_requires_mode(self):
        with pytest.raises(ValidationError):
            StrategyRunCreate()  # type: ignore

    def test_optional_fields_default_none(self):
        r = StrategyRunCreate(mode="backtest")
        assert r.dataset_id is None
        assert r.broker_client is None
        assert r.from_ts is None
        assert r.to_ts is None

    def test_accepts_all_fields(self):
        from datetime import datetime, timezone
        r = StrategyRunCreate(mode="paper", broker_client="mt5", dataset_id=5,
                              from_ts=datetime(2024, 1, 1, tzinfo=timezone.utc))
        assert r.mode == "paper"
        assert r.broker_client == "mt5"


# ── ML Model schemas ────────────────────────────────────────────────────────────

class TestMLModelCreate:
    def test_requires_name(self):
        with pytest.raises(ValidationError):
            MLModelCreate(architecture="lstm")  # type: ignore

    def test_requires_architecture(self):
        with pytest.raises(ValidationError):
            MLModelCreate(name="M")  # type: ignore

    def test_config_defaults_empty(self):
        m = MLModelCreate(name="M", architecture="lstm")
        assert m.config == {}

    def test_valid_creation(self):
        m = MLModelCreate(name="M", architecture="seq2seq_transformer", config={"d_model": 64})
        assert m.config["d_model"] == 64


class TestMLModelUpdate:
    def test_all_fields_optional(self):
        u = MLModelUpdate()
        assert u.name is None
        assert u.config is None
        assert u.status is None

    def test_partial_update(self):
        u = MLModelUpdate(status="deployed")
        assert u.status == "deployed"
        assert u.name is None


class TestTrainingRunCreate:
    def test_requires_dataset_id(self):
        with pytest.raises(ValidationError):
            TrainingRunCreate()  # type: ignore

    def test_hyperparams_default_empty(self):
        t = TrainingRunCreate(dataset_id=1)
        assert t.hyperparams == {}

    def test_accepts_hyperparams(self):
        t = TrainingRunCreate(dataset_id=2, hyperparams={"lr": 0.001, "epochs": 50})
        assert t.hyperparams["lr"] == 0.001


class TestHyperparamSearchCreate:
    def test_requires_all_fields(self):
        with pytest.raises(ValidationError):
            HyperparamSearchCreate(model_id=1)  # type: ignore

    def test_valid_search_grid(self):
        h = HyperparamSearchCreate(model_id=1, dataset_id=2,
                                   search_grid={"lr": [0.001, 0.0001], "batch_size": [32, 64]})
        assert len(h.search_grid["lr"]) == 2


# ── Data schemas ────────────────────────────────────────────────────────────────

class TestDatasourceCreate:
    def test_requires_name_and_type(self):
        with pytest.raises(ValidationError):
            DatasourceCreate(name="DS")  # type: ignore
        with pytest.raises(ValidationError):
            DatasourceCreate(type="ohlc_download")  # type: ignore

    def test_config_defaults_empty(self):
        d = DatasourceCreate(name="DS", type="manual_upload")
        assert d.config == {}


class TestDatasourceUpdate:
    def test_all_optional(self):
        u = DatasourceUpdate()
        assert u.name is None
        assert u.config is None


class TestCollectionJobCreate:
    def test_requires_datasource_id(self):
        with pytest.raises(ValidationError):
            CollectionJobCreate()  # type: ignore

    def test_schedule_cron_optional(self):
        j = CollectionJobCreate(datasource_id=1)
        assert j.schedule_cron is None

    def test_with_cron(self):
        j = CollectionJobCreate(datasource_id=1, schedule_cron="0 * * * *")
        assert j.schedule_cron == "0 * * * *"


class TestCollectionJobUpdate:
    def test_all_optional(self):
        u = CollectionJobUpdate()
        assert u.schedule_cron is None
        assert u.enabled is None

    def test_disable_job(self):
        u = CollectionJobUpdate(enabled=False)
        assert u.enabled is False


# ── Response envelope ──────────────────────────────────────────────────────────

class TestDataResponse:
    def test_wraps_dict(self):
        r = DataResponse(data={"id": 1, "name": "test"})
        assert r.data == {"id": 1, "name": "test"}

    def test_wraps_list(self):
        r = DataResponse(data=[1, 2, 3])
        assert r.data == [1, 2, 3]

    def test_wraps_none(self):
        r = DataResponse(data=None)
        assert r.data is None

    def test_default_meta_is_empty(self):
        r = DataResponse(data=42)
        assert r.meta.total is None
        assert r.meta.page is None
        assert r.meta.page_size is None

    def test_meta_with_values(self):
        r = DataResponse(data=[], meta=Meta(total=100, page=2, page_size=20))
        assert r.meta.total == 100
        assert r.meta.page == 2
        assert r.meta.page_size == 20
