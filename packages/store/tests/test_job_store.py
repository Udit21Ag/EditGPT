"""Both job stores, held to the same contract.

The two implementations are parametrised over the same tests on purpose: the in-memory
one is what the gateway falls back to and what tests run against, so if it behaves
differently from the SQL one the tests are measuring a system nobody deploys.
"""

from __future__ import annotations

from collections.abc import Callable
from uuid import UUID, uuid4

import pytest
from editgpt_core import EditSpec, Job, JobState
from editgpt_store import ANONYMOUS_USER_ID, InMemoryJobStore, JobStore, SqlJobStore
from sqlalchemy.orm import Session, sessionmaker

OTHER_USER = UUID("00000000-0000-0000-0000-0000000000ff")


@pytest.fixture(params=["memory", "sql"])
def store(request: pytest.FixtureRequest, session_factory: sessionmaker[Session]) -> JobStore:
    if request.param == "memory":
        return InMemoryJobStore()
    return SqlJobStore(session_factory=session_factory)


def test_a_created_job_can_be_read_back(store: JobStore, job: Job) -> None:
    store.create(job)
    found = store.get(job.id)
    assert found is not None
    assert found.id == job.id
    assert found.state is JobState.QUEUED
    assert found.spec.target == "the car"


def test_an_unknown_job_is_none_rather_than_an_error(store: JobStore) -> None:
    assert store.get(uuid4()) is None


def test_steps_survive_the_round_trip(store: JobStore, job: Job) -> None:
    store.create(job)
    running = job.advance(JobState.PLANNING, detail="planning", progress=0.1).advance(
        JobState.RUNNING, detail="erasing", progress=0.5
    )
    store.save(running)

    found = store.get(job.id)
    assert found is not None
    assert [s.state for s in found.steps] == [JobState.PLANNING, JobState.RUNNING]
    assert [s.detail for s in found.steps] == ["planning", "erasing"]
    assert found.progress == pytest.approx(0.5)


def test_the_same_idempotency_key_returns_the_first_job(store: JobStore, spec: EditSpec) -> None:
    """A retried request must not create a second job doing identical work."""
    first = store.create(Job(spec=spec, idempotency_key="k-1"))
    second = store.create(Job(spec=spec, idempotency_key="k-1"))
    assert second.id == first.id


def test_different_keys_create_different_jobs(store: JobStore, spec: EditSpec) -> None:
    first = store.create(Job(spec=spec, idempotency_key="k-1"))
    second = store.create(Job(spec=spec, idempotency_key="k-2"))
    assert second.id != first.id


def test_a_job_is_not_visible_to_another_user(store: JobStore, job: Job) -> None:
    store.create(job, user_id=ANONYMOUS_USER_ID)
    assert store.get(job.id, user_id=OTHER_USER) is None


def test_a_terminal_job_carries_its_outcome(store: JobStore, job: Job) -> None:
    store.create(job)
    done = (
        job.advance(JobState.PLANNING, progress=0.1)
        .advance(JobState.RUNNING, progress=0.5)
        .advance(JobState.REVIEW, progress=0.9)
        .advance(JobState.DONE, progress=1.0, result_sha256="b" * 64)
    )
    store.save(done)

    found = store.get(job.id)
    assert found is not None
    assert found.state is JobState.DONE
    assert found.result_sha256 == "b" * 64
    assert found.is_terminal


def test_saving_a_job_that_was_never_created_inserts_it(store: JobStore, job: Job) -> None:
    """The worker may be the first to persist a job if the gateway died mid-request."""
    store.save(job)
    assert store.get(job.id) is not None


@pytest.mark.parametrize("factory", [lambda: InMemoryJobStore()])
def test_saving_another_users_job_is_refused(factory: Callable[[], JobStore], job: Job) -> None:
    store = factory()
    store.create(job, user_id=ANONYMOUS_USER_ID)
    with pytest.raises(PermissionError):
        store.save(job, user_id=OTHER_USER)
