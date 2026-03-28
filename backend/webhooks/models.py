"""Webhook registration ORM + Pydantic schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Any

import sqlalchemy as sa
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Mapped, mapped_column

from database import Base


class WebhookRegistration(Base):
    __tablename__ = "webhook_registrations"

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
    url: Mapped[str] = mapped_column(sa.Text, nullable=False)
    events: Mapped[list] = mapped_column(sa.JSON, nullable=False, server_default="[]")
    secret: Mapped[str] = mapped_column(sa.Text, nullable=False)  # stored for HMAC signing
    active: Mapped[bool] = mapped_column(sa.Boolean, nullable=False, server_default="true")
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())
    last_fired_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)
    last_status: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)


class WebhookRegistrationCreate(BaseModel):
    url: str
    events: list[str]
    secret: str


class WebhookRegistrationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    url: str
    events: list[str]
    active: bool
    created_at: datetime
    last_fired_at: datetime | None
    last_status: int | None
