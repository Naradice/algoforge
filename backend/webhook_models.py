"""ORM model for webhook registrations."""

from __future__ import annotations

from datetime import datetime

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column

try:
    from .database import Base
except ImportError:
    from database import Base


class WebhookRegistration(Base):
    __tablename__ = "webhook_registrations"

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
    url: Mapped[str] = mapped_column(sa.Text, nullable=False)
    events: Mapped[list[str]] = mapped_column(ARRAY(sa.Text), nullable=False, server_default="{}")
    secret_hash: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    active: Mapped[bool] = mapped_column(sa.Boolean, nullable=False, server_default="true")
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())
