"""Issue an AlgoForge API key for an external MCP consumer (requirements.md R-3).

Usage (from backend/):
    python -m scripts.create_api_key --name "research-agent-service"

Prints the raw key once. Only the hash is stored in the DB — save it now.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


async def _create(name: str, scopes: list[str]) -> None:
    from auth import APIKey, generate_api_key
    from database import async_session_factory

    raw, key_hash = generate_api_key()
    async with async_session_factory() as db:
        db.add(APIKey(name=name, key_hash=key_hash, scopes=scopes))
        await db.commit()

    print(f"Created API key {name!r}. Store this now — it will not be shown again:\n")
    print(f"  {raw}\n")
    print("Use it as: Authorization: Bearer <key>")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="Human-readable label, e.g. the consumer service name")
    parser.add_argument("--scope", action="append", default=[], dest="scopes", help="Optional scope tag (repeatable)")
    args = parser.parse_args()
    asyncio.run(_create(args.name, args.scopes))


if __name__ == "__main__":
    main()
