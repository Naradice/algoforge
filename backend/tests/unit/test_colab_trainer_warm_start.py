"""Unit tests for model/colab_trainer.py's check_colab_supported -- specifically the
warm_start_checkpoint existence check added for the DDM->USDJPY transfer experiment (fail fast
at TrainingRun creation time, before a Colab runtime is provisioned, same philosophy as its
existing architecture/token_level/split_mode checks)."""
from __future__ import annotations

import os

import pytest
from fastapi import HTTPException


@pytest.fixture
def artifact_store(tmp_path):
    orig = os.environ.get("ARTIFACT_STORE_PATH")
    os.environ["ARTIFACT_STORE_PATH"] = str(tmp_path)
    yield tmp_path
    if orig is None:
        os.environ.pop("ARTIFACT_STORE_PATH", None)
    else:
        os.environ["ARTIFACT_STORE_PATH"] = orig


class TestWarmStartCheckpointCheck:
    def test_no_warm_start_checkpoint_passes(self, artifact_store):
        from model.colab_trainer import check_colab_supported

        check_colab_supported("decoder_only", None, {})  # must not raise

    def test_existing_checkpoint_passes(self, artifact_store):
        from model.colab_trainer import check_colab_supported

        ckpt_dir = artifact_store / "models" / "1" / "training_2"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "best.pt").write_bytes(b"fake checkpoint")

        check_colab_supported("decoder_only", None, {"warm_start_checkpoint": "models/1/training_2/best.pt"})

    def test_missing_checkpoint_raises_422(self, artifact_store):
        from model.colab_trainer import check_colab_supported

        with pytest.raises(HTTPException) as exc_info:
            check_colab_supported("decoder_only", None, {"warm_start_checkpoint": "models/1/training_2/best.pt"})
        assert exc_info.value.status_code == 422
        assert exc_info.value.detail["code"] == "COLAB_WARM_START_CHECKPOINT_NOT_FOUND"
