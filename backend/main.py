"""
AlgoForge — FastAPI application entry point.

Start with:
    uvicorn main:app --reload --port 8000
"""

import asyncio
import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database import engine, Base
from log_writer import start_log_writer
from strategy.router import router as strategy_router
from model.router import router as model_router
from data.router import router as data_router
from logs_router import router as logs_router

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


@app.on_event("startup")
async def startup() -> None:
    # Start the async log writer background task
    asyncio.create_task(start_log_writer())


@app.get("/api/v1/health", tags=["health"])
async def health():
    return {"status": "ok", "version": app.version}
