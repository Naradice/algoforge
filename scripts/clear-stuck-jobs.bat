@echo off
REM Clear stuck Celery dedup lock keys from Redis.
REM Run this if "Run now" does nothing after a worker was killed (Ctrl+C).

cd /d "%~dp0..\backend"
python clear_stuck_jobs.py
pause
