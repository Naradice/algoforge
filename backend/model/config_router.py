"""Model configuration / schema endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from schemas import DataResponse

model_config_router = APIRouter(prefix="/model-config", tags=["model-config"])

ARCHITECTURE_SCHEMAS = {
    "seq2seq_transformer": {
        "type": "object",
        "properties": {
            "d_model": {"type": "integer", "default": 64},
            "nhead": {"type": "integer", "default": 4},
            "num_encoder_layers": {"type": "integer", "default": 2},
            "num_decoder_layers": {"type": "integer", "default": 2},
            "dim_feedforward": {"type": "integer", "default": 256},
            "dropout": {"type": "number", "default": 0.1},
            "input_size": {"type": "integer"},
            "output_size": {"type": "integer"},
        },
    },
    "lstm": {
        "type": "object",
        "properties": {
            "hidden_size": {"type": "integer", "default": 128},
            "num_layers": {"type": "integer", "default": 2},
            "dropout": {"type": "number", "default": 0.1},
            "input_size": {"type": "integer"},
            "output_size": {"type": "integer"},
            "bidirectional": {"type": "boolean", "default": False},
        },
    },
    "timegan": {
        "type": "object",
        "properties": {
            "hidden_dim": {"type": "integer", "default": 24},
            "num_layer": {"type": "integer", "default": 3},
            "seq_len": {"type": "integer", "default": 24},
            "batch_size": {"type": "integer", "default": 128},
        },
    },
    "rl_agent": {
        "type": "object",
        "properties": {
            "algorithm": {"type": "string", "enum": ["ppo", "dqn", "a2c"], "default": "ppo"},
            "policy": {"type": "string", "default": "MlpPolicy"},
            "learning_rate": {"type": "number", "default": 0.0003},
        },
    },
}


@model_config_router.get("/architectures")
async def list_architectures():
    return DataResponse(data=list(ARCHITECTURE_SCHEMAS.keys()))


@model_config_router.get("/architectures/{architecture}")
async def get_architecture_schema(architecture: str):
    schema = ARCHITECTURE_SCHEMAS.get(architecture.lower())
    if not schema:
        raise HTTPException(status_code=404, detail={"code": "ARCHITECTURE_NOT_FOUND", "message": f"Architecture {architecture!r} not found"})
    return DataResponse(data=schema)
