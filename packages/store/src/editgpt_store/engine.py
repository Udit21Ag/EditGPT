"""Building an engine and getting the schema in place.

`bootstrap` is for tests and a first local run; Alembic in `migrations/` is what a
deployment uses. They are kept separate on purpose — `create_all` cannot express a
column rename or a backfill, and a project that leans on it discovers this at the worst
possible moment.
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker

from editgpt_store.models import ANONYMOUS_USER_ID, Base, User


def make_engine(url: str, *, echo: bool = False) -> sa.Engine:
    """An engine with a pool sized for one worker and one gateway on one small host."""
    if url.startswith("sqlite"):
        # StaticPool keeps an in-memory database alive across sessions; without it every
        # session gets its own empty database and tests pass against nothing.
        return sa.create_engine(
            url, echo=echo, poolclass=sa.pool.StaticPool, connect_args={"check_same_thread": False}
        )
    return sa.create_engine(url, echo=echo, pool_size=5, max_overflow=5, pool_pre_ping=True)


def make_session_factory(engine: sa.Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


def bootstrap(engine: sa.Engine) -> None:
    """Create every table and seed the anonymous user. Idempotent."""
    Base.metadata.create_all(engine)
    ensure_anonymous_user(engine)


def ensure_anonymous_user(engine: sa.Engine) -> None:
    """The sentinel owner every row points at until authentication exists."""
    with Session(engine) as session:
        if session.get(User, ANONYMOUS_USER_ID) is None:
            session.add(User(id=ANONYMOUS_USER_ID, external_id=None))
            session.commit()
