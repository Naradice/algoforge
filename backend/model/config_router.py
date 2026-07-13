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
        },
        "_note": "input_dim and output_dim are set automatically from the selected feature columns at training time.",
    },
    "lstm": {
        "type": "object",
        "properties": {
            "hidden_dim": {"type": "integer", "default": 128},
            "num_layers": {"type": "integer", "default": 2},
            "dropout": {"type": "number", "default": 0.1},
        },
        "_note": "input_dim and output_dim are set automatically from the selected feature columns at training time.",
    },
    "timegan": {
        "type": "object",
        "properties": {
            "hidden_dim": {"type": "integer", "default": 24},
            "num_layer": {"type": "integer", "default": 3},
            "seq_len": {"type": "integer", "default": 24},
            "batch_size": {"type": "integer", "default": 128},
        },
        "_note": "input_dim and output_dim are set automatically from the selected feature columns at training time.",
    },
    "rl_agent": {
        "type": "object",
        "properties": {
            "algorithm": {"type": "string", "enum": ["ppo", "dqn", "a2c"], "default": "ppo"},
            "policy": {"type": "string", "default": "MlpPolicy"},
            "learning_rate": {"type": "number", "default": 0.0003},
        },
    },
    "cnn_lstm": {
        "type": "object",
        "properties": {
            "cnn_filters":  {"type": "integer", "default": 64},
            "kernel_size":  {"type": "integer", "default": 3},
            "cnn_layers":   {"type": "integer", "default": 2},
            "lstm_hidden":  {"type": "integer", "default": 128},
            "lstm_layers":  {"type": "integer", "default": 1},
            "dropout":      {"type": "number",  "default": 0.2},
        },
        "_note": "input_dim and output_dim are set automatically from the selected feature columns at training time.",
    },
    "tcn": {
        "type": "object",
        "properties": {
            "num_channels": {"type": "integer", "default": 64},
            "num_levels":   {"type": "integer", "default": 4},
            "kernel_size":  {"type": "integer", "default": 3},
            "dropout":      {"type": "number",  "default": 0.2},
        },
        "_note": "input_dim and output_dim are set automatically from the selected feature columns at training time.",
    },
    "vae": {
        "type": "object",
        "properties": {
            "latent_dim":      {"type": "integer", "default": 32},
            "encoder_hidden":  {"type": "integer", "default": 128},
            "decoder_hidden":  {"type": "integer", "default": 128},
            "encoder_layers":  {"type": "integer", "default": 2},
            "dropout":         {"type": "number",  "default": 0.1},
        },
        "_note": "input_dim and output_dim are set automatically from the selected feature columns at training time.",
    },
    "nbeats": {
        "type": "object",
        "properties": {
            "hidden_units": {"type": "integer", "default": 256},
            "nb_blocks":    {"type": "integer", "default": 3},
            "theta_dim":    {"type": "integer", "default": 64},
        },
        "_note": "input_dim and output_dim are set automatically from the selected feature columns at training time.",
    },
}


PREPROCESSING_SCHEMA = {
    "indicators": {
        "sma": {
            "params": [{"name": "period", "type": "integer", "default": 20}],
            "outputs": ["sma_{period}"],
        },
        "ema": {
            "params": [{"name": "period", "type": "integer", "default": 20}],
            "outputs": ["ema_{period}"],
        },
        "rsi": {
            "params": [{"name": "period", "type": "integer", "default": 14}],
            "outputs": ["rsi_{period}"],
        },
        "macd": {
            "params": [
                {"name": "fast",   "type": "integer", "default": 12},
                {"name": "slow",   "type": "integer", "default": 26},
                {"name": "signal", "type": "integer", "default": 9},
            ],
            "outputs": ["macd", "macd_signal", "macd_hist"],
        },
        "bbands": {
            "params": [
                {"name": "period", "type": "integer", "default": 20},
                {"name": "std",    "type": "number",  "default": 2.0},
            ],
            "outputs": ["bb_upper_{period}", "bb_mid_{period}", "bb_lower_{period}", "bb_width_{period}"],
        },
        "atr": {
            "params": [{"name": "period", "type": "integer", "default": 14}],
            "outputs": ["atr_{period}"],
            "requires": ["high", "low", "close"],
        },
        "returns": {
            "params": [{"name": "period", "type": "integer", "default": 1}],
            "outputs": ["returns_{period}"],
        },
        "volatility": {
            "params": [{"name": "period", "type": "integer", "default": 20}],
            "outputs": ["vol_{period}"],
        },
    },
    "clustering": {
        "methods": ["kmeans"],
        "params": [
            {"name": "enabled",    "type": "boolean", "default": False},
            {"name": "method",     "type": "string",  "default": "kmeans"},
            {"name": "n_clusters", "type": "integer", "default": 5},
            {"name": "on_cols",    "type": "array",   "default": ["close"]},
        ],
        "output_col": "cluster_{n_clusters}",
    },
    "normalize_modes": ["returns", "minmax", "zscore", "robust", "none"],
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


@model_config_router.get("/preprocessing")
async def get_preprocessing_schema():
    return DataResponse(data=PREPROCESSING_SCHEMA)
