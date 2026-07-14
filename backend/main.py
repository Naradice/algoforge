"""
AlgoForge — FastAPI application entry point.

Start with:
    uvicorn main:app --reload --port 8000
"""

import asyncio
import logging
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from database import engine, Base
from log_writer import start_log_writer
from strategy.router import router as strategy_router
from strategy.config_router import config_router as strategy_config_router
from model.router import router as model_router
from model.training_runs_router import tr_router
from model.preprocessed_datasets_router import pd_router
from model.config_router import model_config_router
from data.router import router as data_router
from logs_router import router as logs_router
from ws_router import ws_router
from mcp_server import mcp
from webhooks.router import webhook_router

# Configure Python logging (console output in development)
logging.basicConfig(
    level=logging.DEBUG if os.getenv("ALGOFORGE_DEBUG") else logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s — %(message)s",
)

app = FastAPI(
    title="AlgoForge API",
    version="0.1.0",
    description="Unified algorithmic trading platform — Strategy · Model · Data",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
API_PREFIX = "/api/v1"
app.include_router(strategy_router, prefix=API_PREFIX)
app.include_router(model_router, prefix=API_PREFIX)
app.include_router(data_router, prefix=API_PREFIX)
app.include_router(logs_router, prefix=API_PREFIX)
app.include_router(ws_router, prefix=API_PREFIX)
app.include_router(webhook_router, prefix=API_PREFIX)
app.include_router(strategy_config_router, prefix=API_PREFIX)
app.include_router(tr_router, prefix=API_PREFIX)
app.include_router(pd_router, prefix=API_PREFIX)
app.include_router(model_config_router, prefix=API_PREFIX)

# MCP server — accessible at /mcp (SSE transport for Claude Desktop)
# Try known API names across fastmcp versions; skip gracefully if unavailable.
_mcp_mounted = False
for _attr in ("get_asgi_app", "http_app", "sse_app", "asgi_app"):
    _fn = getattr(mcp, _attr, None)
    if _fn is not None:
        try:
            app.mount("/mcp", _fn() if callable(_fn) else _fn)
            _mcp_mounted = True
            break
        except Exception:
            pass
if not _mcp_mounted:
    logging.getLogger("main").warning("MCP server could not be mounted — upgrade fastmcp or run it standalone")


_HTTP_CODE_MAP: dict[int, str] = {
    400: "BAD_REQUEST",
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    409: "CONFLICT",
    422: "UNPROCESSABLE",
}


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    if isinstance(exc.detail, dict) and "code" in exc.detail:
        code = exc.detail["code"]
        message = exc.detail.get("message", code)
        detail = exc.detail.get("detail", {})
    else:
        code = _HTTP_CODE_MAP.get(exc.status_code, "UNKNOWN_ERROR")
        message = str(exc.detail)
        detail = {}
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": code, "message": message, "detail": detail}},
    )


@app.on_event("startup")
async def startup() -> None:
    asyncio.create_task(start_log_writer())


@app.get("/api/v1/health", tags=["health"])
async def health():
    return {"status": "ok", "version": app.version}
