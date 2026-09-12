"""Unit tests for model/notebook_export.py's build_notebook -- in particular the additions made
for the DDM->USDJPY transfer experiment: passing tgt_feature_cols/require_contiguous/max_rows
through to the generated OHLCWindowDataset call, the step-denominated (max_steps) training loop
branch, and warm_start_checkpoint support."""
from __future__ import annotations

import ast

import pytest

BASE_HP = {
    "obs_len": 60, "pred_len": 1, "feature_cols": ["close"],
    "normalize": "zscore", "seed": 42,
}
MODEL_CONFIG = {"d_model": 16, "nhead": 2, "num_layers": 1, "dim_feedforward": 32}


def _build(hyperparams, **kwargs):
    from model.notebook_export import build_notebook

    return build_notebook(
        architecture="decoder_only", model_name="test", model_config=MODEL_CONFIG,
        dataset_id=1, snapshot_id=1, snapshot_url="https://example.com/x.parquet",
        snapshot_sha256="abc", hyperparams=hyperparams, **kwargs,
    )


def _code_cells(nb):
    return [c["source"] for c in nb["cells"] if c["cell_type"] == "code"]


def _assert_all_cells_parse(nb):
    for i, src in enumerate(_code_cells(nb)):
        lines = [l for l in src.split("\n") if not l.strip().startswith("!")]
        try:
            ast.parse("\n".join(lines))
        except SyntaxError as e:
            pytest.fail(f"cell {i} has invalid Python syntax: {e}\n---\n{src}")


class TestEpochModeUnchanged:
    def test_default_hyperparams_still_produce_epoch_loop(self):
        nb = _build(BASE_HP)
        _assert_all_cells_parse(nb)
        joined = "\n".join(_code_cells(nb))
        assert 'for epoch in range(1, HYPERPARAMS["epochs"] + 1):' in joined
        assert "while steps_done < _max_steps:" not in joined


class TestStepModeTrainingLoop:
    def test_max_steps_set_produces_step_loop_not_epoch_loop(self):
        hp = {**BASE_HP, "max_steps": 1000, "val_every_steps": 200}
        nb = _build(hp)
        _assert_all_cells_parse(nb)
        joined = "\n".join(_code_cells(nb))
        assert "while steps_done < _max_steps:" in joined
        assert 'for epoch in range(1, HYPERPARAMS["epochs"] + 1):' not in joined
        assert "get_step_trainer_fn" in joined


class TestDatasetParamPassthrough:
    def test_tgt_feature_cols_and_require_contiguous_reach_the_dataset_cell(self):
        hp = {**BASE_HP, "tgt_feature_cols": ["vol_20"], "require_contiguous": True, "max_rows": 12345}
        nb = _build(hp)
        _assert_all_cells_parse(nb)
        joined = "\n".join(_code_cells(nb))
        assert 'tgt_feature_cols=HYPERPARAMS["tgt_feature_cols"]' in joined
        assert 'require_contiguous=HYPERPARAMS["require_contiguous"]' in joined
        assert "'tgt_feature_cols': ['vol_20']" in joined.replace('"', "'")


class TestWarmStartCheckpoint:
    def test_no_warm_start_url_omits_download_and_load(self):
        nb = _build(BASE_HP)
        _assert_all_cells_parse(nb)
        joined = "\n".join(_code_cells(nb))
        assert "WARM_START_URL" not in joined
        assert "warm-started from" not in joined

    def test_warm_start_url_adds_download_and_load_before_optimizer(self):
        nb = _build(
            BASE_HP,
            warm_start_checkpoint_url="https://example.com/best.pt",
            warm_start_checkpoint_sha256="deadbeef",
        )
        _assert_all_cells_parse(nb)
        cells = _code_cells(nb)
        joined = "\n".join(cells)
        assert "WARM_START_URL = 'https://example.com/best.pt'" in joined
        assert "WARM_START_SHA256 = 'deadbeef'" in joined
        assert "model.load_state_dict(_ckpt[\"model_state\"])" in joined

        # The load must happen after build_model(...) and before the optimizer is constructed,
        # in the SAME cell -- same placement/rationale as celery_worker.py's
        # _apply_warm_start_checkpoint (a clean optimizer state for the fine-tuning run).
        build_cell = next(c for c in cells if "model = build_model(" in c)
        assert "model.load_state_dict" in build_cell
        assert build_cell.index("model = build_model(") < build_cell.index("model.load_state_dict")
        assert build_cell.index("model.load_state_dict") < build_cell.index("optimizer = torch.optim")
