import hashlib
import os
import secrets
from datetime import datetime, timezone
from typing import Annotated

from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from .database import Base, get_db

import sqlalchemy as sa


class APIKey(Base):
    __tablename__ = "api_keys"

    id = sa.Column(sa.Integer, primary_key=True)
    name = sa.Column(sa.Text, nullable=False)
    key_hash = sa.Column(sa.Text, nullable=False, unique=True, index=True)
    scopes = sa.Column(sa.ARRAY(sa.Text), nullable=False, server_default="{}")
    active = sa.Column(sa.Boolean, nullable=False, server_default="true")
    created_at = sa.Column(sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())
    last_used_at = sa.Column(sa.DateTime(timezone=True), nullable=True)


_security = HTTPBearer(auto_error=False)


def _hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def generate_api_key() -> tuple[str, str]:
    """Return (raw_key, key_hash). Store only the hash."""
    raw = "af_" + secrets.token_urlsafe(32)
    return raw, _hash_key(raw)


async def require_api_key(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(_security)] = None,
    db: AsyncSession = Depends(get_db),
) -> APIKey:
    if credentials is None:
        raise HTTPException(status_code=401, detail="API key required")

    key_hash = _hash_key(credentials.credentials)
    result = await db.execute(select(APIKey).where(APIKey.key_hash == key_hash, APIKey.active == True))
    api_key = result.scalar_one_or_none()
    if api_key is None:
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")

    await db.execute(
        update(APIKey).where(APIKey.id == api_key.id).values(last_used_at=datetime.now(timezone.utc))
    )
    return api_key


# Convenience: skip auth when ALGOFORGE_NO_AUTH=1 (dev only)
_NO_AUTH = os.getenv("ALGOFORGE_NO_AUTH", "").lower() in ("1", "true")


async def optional_auth(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(_security)] = None,
    db: AsyncSession = Depends(get_db),
) -> APIKey | None:
    if _NO_AUTH:
        return None
    if credentials is None:
        return None
    return await require_api_key(credentials, db)
