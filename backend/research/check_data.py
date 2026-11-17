import asyncio
import os
import sys

backend_dir = r"c:\Users\ksato\workspace\finance\algoforge\backend"
sys.path.insert(0, backend_dir)

from database import async_session_factory
from sqlalchemy import text
import pandas as pd

async def main():
    async with async_session_factory() as session:
        result = await session.execute(text("SELECT * FROM report_forecasts LIMIT 1"))
        cols = list(result.keys())
        print("Columns in report_forecasts:", cols)

asyncio.run(main())
