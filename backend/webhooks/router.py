"""Webhook management endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from schemas import DataResponse
from webhooks.models import WebhookRegistration, WebhookRegistrationCreate, WebhookRegistrationRead

webhook_router = APIRouter(prefix="/webhooks", tags=["webhooks"])


@webhook_router.get("", response_model=DataResponse[list[WebhookRegistrationRead]])
async def list_webhooks(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(WebhookRegistration).order_by(WebhookRegistration.created_at.desc()))
    items = list(result.scalars().all())
    return DataResponse(data=items)


@webhook_router.post("", response_model=DataResponse[WebhookRegistrationRead], status_code=201)
async def register_webhook(body: WebhookRegistrationCreate, db: AsyncSession = Depends(get_db)):
    obj = WebhookRegistration(url=body.url, events=body.events, secret=body.secret)
    db.add(obj)
    await db.flush()
    await db.refresh(obj)
    return DataResponse(data=obj)


@webhook_router.delete("/{webhook_id}", status_code=204)
async def delete_webhook(webhook_id: int, db: AsyncSession = Depends(get_db)):
    from fastapi import HTTPException
    result = await db.execute(select(WebhookRegistration).where(WebhookRegistration.id == webhook_id))
    obj = result.scalar_one_or_none()
    if obj is None:
        raise HTTPException(status_code=404, detail={"code": "WEBHOOK_NOT_FOUND", "message": "Webhook not found"})
    await db.delete(obj)


@webhook_router.post("/{webhook_id}/test")
async def test_webhook(webhook_id: int, db: AsyncSession = Depends(get_db)):
    from fastapi import HTTPException
    result = await db.execute(select(WebhookRegistration).where(WebhookRegistration.id == webhook_id))
    obj = result.scalar_one_or_none()
    if obj is None:
        raise HTTPException(status_code=404, detail={"code": "WEBHOOK_NOT_FOUND", "message": "Webhook not found"})

    import json
    import hmac
    import hashlib
    from datetime import datetime, timezone
    import httpx

    test_body = {
        "event": "test",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {"message": "AlgoForge webhook test"},
    }
    body_bytes = json.dumps(test_body).encode()
    sig = hmac.new(obj.secret.encode(), body_bytes, hashlib.sha256).hexdigest()

    status_code = 0
    response_body = ""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                obj.url,
                content=body_bytes,
                headers={"Content-Type": "application/json", "X-AlgoForge-Signature": sig},
            )
            status_code = resp.status_code
            response_body = resp.text[:500]
    except Exception as e:
        response_body = str(e)

    return DataResponse(data={"status_code": status_code, "response_body": response_body})
