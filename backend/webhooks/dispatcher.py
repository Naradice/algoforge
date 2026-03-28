"""Webhook HMAC signing and async delivery."""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger("webhooks.dispatcher")


async def dispatch(db, event_type: str, payload: dict) -> None:
    """Fire all matching active webhooks for event_type."""
    import sqlalchemy as sa
    from webhooks.models import WebhookRegistration

    try:
        result = await db.execute(
            sa.select(WebhookRegistration).where(
                WebhookRegistration.active == True,
            )
        )
        registrations = [r for r in result.scalars().all() if event_type in r.events]
    except Exception:
        logger.exception("Failed to load webhook registrations")
        return

    if not registrations:
        return

    body = {
        "event": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": payload,
    }
    body_bytes = json.dumps(body, default=str).encode()

    import httpx
    async with httpx.AsyncClient(timeout=10) as client:
        for reg in registrations:
            sig = hmac.new(reg.secret.encode(), body_bytes, hashlib.sha256).hexdigest()
            try:
                resp = await client.post(
                    reg.url,
                    content=body_bytes,
                    headers={
                        "Content-Type": "application/json",
                        "X-AlgoForge-Signature": sig,
                    },
                )
                await _update_status(db, reg.id, resp.status_code)
            except Exception:
                logger.exception(f"Webhook delivery failed for {reg.url}")
                await _update_status(db, reg.id, 0)


async def _update_status(db, webhook_id: int, status_code: int) -> None:
    import sqlalchemy as sa
    from webhooks.models import WebhookRegistration
    from datetime import datetime, timezone
    try:
        await db.execute(
            sa.update(WebhookRegistration).where(WebhookRegistration.id == webhook_id).values(
                last_fired_at=datetime.now(timezone.utc),
                last_status=status_code,
            )
        )
    except Exception:
        pass  # Don't fail the main operation if status update fails
