import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool

# Load all ORM models so Alembic can see them via Base.metadata
from database import Base  # noqa: F401
import auth  # noqa: F401 — registers APIKey
import log_models  # noqa: F401 — registers Log
import webhooks.models  # noqa: F401 — registers WebhookRegistration
import strategy.models  # noqa: F401
import model.models  # noqa: F401
import data.models  # noqa: F401

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# Convert asyncpg URL to psycopg2 for synchronous migrations
_url = os.getenv("DATABASE_URL", config.get_main_option("sqlalchemy.url"))
SYNC_DATABASE_URL = _url.replace("postgresql+asyncpg://", "postgresql+psycopg2://")


def run_migrations_offline() -> None:
    context.configure(
        url=SYNC_DATABASE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = create_engine(SYNC_DATABASE_URL, poolclass=NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
