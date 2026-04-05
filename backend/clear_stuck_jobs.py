"""Clear orphaned Celery dedup lock keys from Redis.

Run via:  scripts/clear-stuck-jobs.bat
Or directly:  python backend/clear_stuck_jobs.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

import redis

url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
try:
    r = redis.from_url(url)
    r.ping()
except Exception as e:
    print(f"Cannot connect to Redis at {url}: {e}")
    sys.exit(1)

locks = r.keys(b"algoforge:enqueued:*")
arq   = r.keys(b"arq:*")
all_keys = locks + arq

if not all_keys:
    print("No stuck keys found - Redis is clean.")
else:
    for k in all_keys:
        r.delete(k)
        print(f"Deleted: {k.decode()}")
    print(f"\nDone - cleared {len(all_keys)} key(s).")
    print("You can now click Run now from the UI.")
