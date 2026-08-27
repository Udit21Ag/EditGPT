"""Images, artifacts and the cost ledger."""

from __future__ import annotations

from uuid import uuid4

import pytest
from editgpt_core import Job
from editgpt_store import (
    Image,
    SqlJobStore,
    artifacts_for,
    record_artifact,
    record_cost,
    record_image,
    spend_since,
)
from sqlalchemy.orm import Session, sessionmaker


def test_an_image_is_recorded_with_its_dimensions(session_factory: sessionmaker[Session]) -> None:
    record_image(
        session_factory,
        sha256="a" * 64,
        width=640,
        height=480,
        content_type="image/png",
        byte_size=12_345,
    )
    with session_factory() as session:
        row = session.get(Image, "a" * 64)
    assert row is not None
    assert (row.width, row.height, row.byte_size) == (640, 480, 12_345)


def test_recording_the_same_image_twice_is_a_no_op(session_factory: sessionmaker[Session]) -> None:
    """The digest is the primary key, so a re-upload must not be a conflict to handle."""
    for _ in range(2):
        record_image(
            session_factory,
            sha256="b" * 64,
            width=10,
            height=10,
            content_type="image/png",
            byte_size=1,
        )
    with session_factory() as session:
        assert session.query(Image).count() == 1


def test_artifacts_come_back_in_the_order_they_were_produced(
    session_factory: sessionmaker[Session], job: Job
) -> None:
    SqlJobStore(session_factory=session_factory).create(job)
    for kind in ("mask", "intermediate", "result"):
        record_artifact(session_factory, job_id=job.id, sha256="c" * 64, kind=kind)

    assert [a.kind for a in artifacts_for(session_factory, job.id)] == [
        "mask",
        "intermediate",
        "result",
    ]


def test_a_job_with_no_artifacts_returns_an_empty_list(
    session_factory: sessionmaker[Session],
) -> None:
    assert artifacts_for(session_factory, uuid4()) == []


def test_a_free_call_is_still_recorded(session_factory: sessionmaker[Session]) -> None:
    """The free tiers here are capped by *call count*, so a ledger of money alone would
    show nothing until everything started failing."""
    record_cost(session_factory, provider="cloudflare", operation="add", units=1, cents=0.0)
    calls, cents = spend_since(session_factory)
    assert calls == 1
    assert cents == pytest.approx(0.0)


def test_spend_sums_across_jobs(session_factory: sessionmaker[Session]) -> None:
    """A budget is consumed per account, which is why the ledger is not a column on jobs."""
    record_cost(session_factory, provider="p", operation="add", units=2, cents=1.5)
    record_cost(session_factory, provider="p", operation="replace", units=3, cents=2.25)
    calls, cents = spend_since(session_factory)
    assert calls == 5
    assert cents == pytest.approx(3.75)


def test_an_empty_ledger_sums_to_zero_rather_than_none(
    session_factory: sessionmaker[Session],
) -> None:
    """SQL `SUM` over no rows is NULL; a quota check must not have to handle that."""
    assert spend_since(session_factory) == (0, 0.0)


def test_spend_is_scoped_to_a_user(session_factory: sessionmaker[Session]) -> None:
    record_cost(session_factory, provider="p", operation="add", units=4, cents=1.0)
    assert spend_since(session_factory, user_id=uuid4()) == (0, 0.0)
