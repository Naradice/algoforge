"""ASGI-level auth gate for the /mcp mount (requirements.md R-3).

FastMCP is mounted as a raw ASGI sub-app in main.py, not a FastAPI router, so the
usual `Depends(require_api_key)` chain doesn't apply to it. This wraps the mounted
app directly and checks the same `api_keys` table auth.py already defines.

Bypassed when ALGOFORGE_NO_AUTH=1, matching auth.optional_auth's dev-only escape hatch.
"""
from __future__ import annotations

import os

_NO_AUTH = os.getenv("ALGOFORGE_NO_AUTH", "").lower() in ("1", "true")

_UNAUTHORIZED_BODY = b'{"error":{"code":"UNAUTHORIZED","message":"Valid API key required for /mcp. See auth.py / scripts/create_api_key.py."}}'


class MCPAuthMiddleware:
    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or _NO_AUTH:
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        auth_header = headers.get(b"authorization", b"").decode("latin-1")
        token = auth_header[len("Bearer "):] if auth_header.startswith("Bearer ") else None

        if not token or not await _is_valid_key(token):
            await send({
                "type": "http.response.start",
                "status": 401,
                "headers": [(b"content-type", b"application/json")],
            })
            await send({"type": "http.response.body", "body": _UNAUTHORIZED_BODY})
            return

        await self.app(scope, receive, send)


async def _is_valid_key(token: str) -> bool:
    import sqlalchemy as sa
    from datetime import datetime, timezone
    from auth import APIKey, _hash_key
    from database import async_session_factory

    key_hash = _hash_key(token)
    async with async_session_factory() as db:
        result = await db.execute(
            sa.select(APIKey).where(APIKey.key_hash == key_hash, APIKey.active == True)
        )
        api_key = result.scalar_one_or_none()
        if api_key is None:
            return False
        await db.execute(
            sa.update(APIKey).where(APIKey.id == api_key.id).values(last_used_at=datetime.now(timezone.utc))
        )
        await db.commit()
    return True
