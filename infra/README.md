# AlgoForge Infra

Docker Compose configuration for running the full AlgoForge stack.

## Services

| Service | Image / Build | Port | Description |
|---------|--------------|------|-------------|
| `postgres` | `postgres:16-alpine` | 5432 | Primary database |
| `redis` | `redis:7-alpine` | 6379 | Job queue + event bus |
| `backend` | `../backend` | 8000 | FastAPI API + arq worker |
| `web` | `../web` | 3000 | Next.js dashboard |
| `ml_worker` | `../ml_worker` | — | Python 3.8 RL training worker |

## Quick start

```bash
cd infra
cp .env.example .env
# Edit .env — at minimum set GOOGLE_API_KEY if you want LLM features
docker compose up --build
```

Open http://localhost:3000.

## Environment file

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Key variables:

| Variable | Description |
|----------|-------------|
| `GOOGLE_API_KEY` | Gemini API key — required for LLM conditions and AI chat |
| `MT5_LOGIN` / `MT5_PASSWORD` / `MT5_SERVER` | MetaTrader 5 credentials for live trading |
| `SECRET_KEY` | JWT signing secret — **change in production** |
| `ALGOFORGE_NO_REDIS` | Set to `1` to use in-process event bus (testing only) |

## Individual service commands

```bash
# Start only infrastructure (postgres + redis)
docker compose up postgres redis -d

# Start a single service
docker compose up backend -d

# View logs
docker compose logs -f backend

# Rebuild a single service
docker compose up --build backend

# Stop everything
docker compose down

# Stop and remove volumes (WARNING: deletes all data)
docker compose down -v
```

## Volumes

| Volume | Used by | Contents |
|--------|---------|----------|
| `postgres_data` | postgres | Database files |
| `redis_data` | redis | Redis persistence |
| `artifacts` | backend, ml_worker | Parquet datasets + model checkpoints |

The `artifacts` volume is shared between `backend` and `ml_worker` so both can read/write model files.

## Startup sequence

The backend container runs `alembic upgrade head` before starting uvicorn, ensuring the database schema is always up to date:

```yaml
command: >
  sh -c "alembic upgrade head && uvicorn main:app --host 0.0.0.0 --port 8000"
```

In `docker-compose.dev.yml`, the backend still uses `uvicorn --reload`, but generated `artifacts/` paths are excluded from file watching so data collection and backtests do not trigger reload crashes on Docker Desktop / Windows.

Service dependencies are handled via health checks:
- `backend` and `ml_worker` wait for postgres and redis to be healthy before starting
- `web` waits for `backend` to start

## Production notes

For production deployment:

1. **Secrets** — Use Docker secrets or a secrets manager instead of `.env` files
2. **Reverse proxy** — Place nginx or Traefik in front of `backend` (port 8000) and `web` (port 3000)
3. **TLS** — Terminate TLS at the reverse proxy; update `CORS_ORIGINS` accordingly
4. **Postgres** — Use a managed database service (RDS, Cloud SQL) with proper backups
5. **Artifact storage** — Replace the `artifacts` volume with S3 or equivalent; update `ARTIFACT_STORE_PATH`
6. `SECRET_KEY` — Generate a random 32+ byte hex string
