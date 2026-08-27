"""Alembic environment.

The database URL is read from `EDITGPT_DATABASE_URL` rather than `alembic.ini`, because
a URL contains a password and `alembic.ini` is committed. `target_metadata` points at the
real models so `alembic revision --autogenerate` can see drift.
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from editgpt_store.models import Base
from sqlalchemy import engine_from_config, pool

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

DEFAULT_URL = "postgresql+psycopg://editgpt:editgpt@localhost:5432/editgpt"
config.set_main_option("sqlalchemy.url", os.environ.get("EDITGPT_DATABASE_URL", DEFAULT_URL))

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
