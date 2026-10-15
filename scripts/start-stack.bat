@echo off
REM Start the local dev stack after a reboot: Docker infra, API, Celery workers (solo pool) and Beat.
REM Usage: scripts\start-stack.bat   (idempotent for containers; start workers only once)
setlocal
set ROOT=%~dp0..
set ARTIFACT_STORE_PATH=%ROOT%\artifacts
set DATABASE_URL=postgresql+asyncpg://algoforge:algoforge@127.0.0.1:5432/algoforge

docker info >nul 2>&1 || (echo Starting Docker Desktop... & start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe" & timeout /t 45 /nobreak >nul)
docker start infra-postgres-1 infra-redis-1
timeout /t 10 /nobreak >nul

cd /d %ROOT%\backend
start "api"      /min cmd /c "uvicorn main:app --port 8000 > uvicorn_api.log 2>&1"
start "collect"  /min cmd /c "celery -A celery_worker worker -Q collection --pool=solo -n collect@%%h --loglevel=info > celery_collect.log 2>&1"
start "chars"    /min cmd /c "celery -A celery_worker worker -Q characteristics --pool=solo -n chars@%%h --loglevel=info > celery_chars.log 2>&1"
start "train"    /min cmd /c "celery -A celery_worker worker -Q training --pool=solo -n train@%%h --loglevel=info > celery_train.log 2>&1"
start "backtest" /min cmd /c "celery -A celery_worker worker -Q backtest,portfolio --pool=solo -n backtest@%%h --loglevel=info > celery_backtest.log 2>&1"
start "beat"     /min cmd /c "celery -A celery_worker beat --loglevel=info > celery_beat.log 2>&1"
echo Stack starting. Check: curl http://localhost:8000/api/v1/ops/queues
endlocal
