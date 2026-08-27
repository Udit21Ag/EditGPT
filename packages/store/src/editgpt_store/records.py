"""Writes to the tables that are records rather than state machines.

Images, artifacts and ledger entries are inserted and read; nothing transitions and
nothing needs two implementations, so these are plain functions over a session factory
rather than a protocol. When one of them grows a second backend, that is the moment to
give it one — not now.
"""

from __future__ import annotations

from uuid import UUID

import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker

from editgpt_store.models import ANONYMOUS_USER_ID, Artifact, CostEntry, Image


def record_image(
    session_factory: sessionmaker[Session],
    *,
    sha256: str,
    width: int,
    height: int,
    content_type: str,
    byte_size: int,
    user_id: UUID = ANONYMOUS_USER_ID,
) -> None:
    """Note that an image exists. Uploading the same bytes twice is not an error.

    The digest is the primary key, so a re-upload is a no-op rather than a duplicate row
    or a conflict the caller has to handle.
    """
    with session_factory() as session:
        if session.get(Image, sha256) is not None:
            return
        session.add(
            Image(
                sha256=sha256,
                user_id=user_id,
                width=width,
                height=height,
                content_type=content_type,
                byte_size=byte_size,
            )
        )
        session.commit()


def record_artifact(
    session_factory: sessionmaker[Session], *, job_id: UUID, sha256: str, kind: str
) -> None:
    with session_factory() as session:
        session.add(Artifact(job_id=job_id, sha256=sha256, kind=kind))
        session.commit()


def artifacts_for(session_factory: sessionmaker[Session], job_id: UUID) -> list[Artifact]:
    with session_factory() as session:
        return list(
            session.scalars(
                sa.select(Artifact)
                .where(Artifact.job_id == job_id)
                .order_by(Artifact.created_at, Artifact.id)
            )
        )


def record_cost(
    session_factory: sessionmaker[Session],
    *,
    provider: str,
    operation: str,
    units: int = 1,
    cents: float = 0.0,
    job_id: UUID | None = None,
    user_id: UUID = ANONYMOUS_USER_ID,
) -> None:
    """Record a provider call, free or not.

    Zero-cost entries are the point rather than an edge case: the free tiers this project
    runs on are capped by *call count*, so a ledger that only recorded money would show
    nothing until the moment everything started failing.
    """
    with session_factory() as session:
        session.add(
            CostEntry(
                user_id=user_id,
                job_id=job_id,
                provider=provider,
                operation=operation,
                units=units,
                cents=cents,
            )
        )
        session.commit()


def spend_since(
    session_factory: sessionmaker[Session],
    *,
    since: sa.DateTime | None = None,
    user_id: UUID = ANONYMOUS_USER_ID,
) -> tuple[int, float]:
    """(calls, cents) charged to a user, optionally since a moment."""
    with session_factory() as session:
        query = sa.select(
            sa.func.coalesce(sa.func.sum(CostEntry.units), 0),
            sa.func.coalesce(sa.func.sum(CostEntry.cents), 0.0),
        ).where(CostEntry.user_id == user_id)
        if since is not None:
            query = query.where(CostEntry.at >= since)
        calls, cents = session.execute(query).one()
        return int(calls), float(cents)
