@echo off
REM Clear stuck arq in-progress / retry keys from Redis.
REM Run this if the arq worker shows "already running elsewhere" for every job,
REM which happens on Windows when the worker is killed mid-job (Ctrl+C) and
REM the cleanup signal handler never fires.
REM
REM Usage: run from any directory while Redis is running.

cd /d "%~dp0..\backend"
python -c "
import redis, os
from dotenv import load_dotenv
load_dotenv()
r = redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379/0'))

all_arq = r.keys('arq:*')
if not all_arq:
    print('No arq keys found.')
else:
    for k in all_arq:
        r.delete(k)
        print(f'Deleted: {k.decode()}')
    print(f'Done — cleared {len(all_arq)} arq keys.')
    print('Restart the arq worker, then re-queue any jobs from the UI.')
"
pause
