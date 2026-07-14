"""Unit tests for Pydantic schema validation."""
import pytest
from pydantic import ValidationError

from strategy.models import StrategyCreate, StrategyUpdate, StrategyRunCreate, TradeRead
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
    def test_dataset_id_and_preprocessed_dataset_id_both_optional_at_schema_level(self):
        # Requiring "at least one of dataset_id / preprocessed_dataset_id" is a service-layer
        # business rule (ModelService.create_training_run raises 422), not a schema constraint —
        # a bare preprocessed_dataset_id-only payload must validate here.
        t = TrainingRunCreate()  # type: ignore
        assert t.dataset_id is None
        assert t.preprocessed_dataset_id is None

    def test_hyperparams_default_empty(self):
        t = TrainingRunCreate(dataset_id=1)
        assert t.hyperparams == {}

    def test_accepts_hyperparams(self):
        t = TrainingRunCreate(dataset_id=2, hyperparams={"lr": 0.001, "epochs": 50})
        assert t.hyperparams["lr"] == 0.001

    def test_accepts_preprocessed_dataset_id(self):
        t = TrainingRunCreate(preprocessed_dataset_id=5, hyperparams={"epochs": 10})
        assert t.preprocessed_dataset_id == 5
        assert t.dataset_id is None


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


# ── TradeRead schema ───────────────────────────────────────────────────────────

class TestTradeRead:
    def _base(self):
        from datetime import datetime, timezone
        return dict(
            id=1, run_id=2, symbol="USDJPY", direction="buy",
            entry_price=150.0, volume=0.1,
            opened_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

    def test_required_fields_valid(self):
        t = TradeRead(**self._base())
        assert t.symbol == "USDJPY"
        assert t.direction == "buy"
        assert t.entry_price == 150.0

    def test_optional_fields_default_none(self):
        t = TradeRead(**self._base())
        assert t.exit_price is None
        assert t.profit is None
        assert t.closed_at is None
        assert t.exit_reason is None
        assert t.phase is None
        assert t.mae is None
        assert t.mfe is None

    def test_full_trade_valid(self):
        from datetime import datetime, timezone
        t = TradeRead(
            id=1, run_id=2, symbol="EURUSD", direction="sell",
            entry_price=1.1, exit_price=1.09, volume=1.0,
            sl_price=1.11, tp_price=1.08, profit=100.0,
            opened_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            closed_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
            exit_reason="tp", phase="oos", mae=0.003, mfe=0.012,
        )
        assert t.exit_reason == "tp"
        assert t.phase == "oos"
        assert t.mfe == 0.012

    def test_from_attributes_builds_from_orm_like_object(self):
        from datetime import datetime, timezone

        class FakeTrade:
            id = 5
            run_id = 3
            symbol = "BTCUSD"
            direction = "buy"
            entry_price = 50000.0
            exit_price = None
            volume = 0.01
            sl_price = None
            tp_price = None
            profit = None
            opened_at = datetime(2024, 6, 1, tzinfo=timezone.utc)
            closed_at = None
            exit_reason = None
            phase = None
            mae = None
            mfe = None

        t = TradeRead.model_validate(FakeTrade())
        assert t.id == 5
        assert t.symbol == "BTCUSD"
        assert t.exit_price is None

    def test_missing_required_field_raises(self):
        from pydantic import ValidationError
        from datetime import datetime, timezone
        with pytest.raises(ValidationError):
            TradeRead(id=1, run_id=2, direction="buy", entry_price=1.0, volume=0.1,
                      opened_at=datetime(2024, 1, 1, tzinfo=timezone.utc))  # missing symbol


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
