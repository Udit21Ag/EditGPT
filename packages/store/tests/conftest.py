"""Fixtures shared by the store tests.

SQLite in memory rather than a Postgres container: `make check` must not need Docker, and
the behaviour under test here is the repository's, not the dialect's. The one thing that
would be dialect-specific — the unique constraint on `(user_id, idempotency_key)` — is
enforced by SQLite too, so it is still genuinely covered.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import sqlalchemy as sa
from editgpt_core import AssetRef, EditOp, EditSpec, Job, MaskSource
from editgpt_store import bootstrap, make_engine, make_session_factory
from sqlalchemy.orm import Session, sessionmaker


@pytest.fixture
def engine() -> Iterator[sa.Engine]:
    made = make_engine("sqlite+pysqlite:///:memory:")
    bootstrap(made)
    yield made
    made.dispose()


@pytest.fixture
def session_factory(engine: sa.Engine) -> sessionmaker[Session]:
    return make_session_factory(engine)


@pytest.fixture
def spec() -> EditSpec:
    return EditSpec(
        op=EditOp.REMOVE,
        image_ref=AssetRef(bucket="local", sha256="a" * 64, width=640, height=480),
        mask_source=MaskSource.TEXT,
        target="the car",
    )


@pytest.fixture
def job(spec: EditSpec) -> Job:
    return Job(spec=spec)
