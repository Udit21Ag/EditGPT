"""Persisting jobs, without letting the database have its own opinion about them.

`editgpt_core.jobs.Job` owns what a job is and which transitions are legal. This module
only moves one between memory and a table, and it does that by round-tripping through the
same Pydantic model both the gateway and the worker import. There is deliberately no
second validation layer here: two places that both decide what a legal job is will
eventually disagree, and the one in the database is the harder to change.

Two implementations. `InMemoryJobStore` is what tests and the eager Celery mode use, so
neither needs a container. `SqlJobStore` is what runs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol
from uuid import UUID

import sqlalchemy as sa
from editgpt_core.jobs import Job, JobState, JobStep
from sqlalchemy.orm import Session, selectinload

from editgpt_store.models import ANONYMOUS_USER_ID, JobRow, JobStepRow

log = logging.getLogger(__name__)


class JobStore(Protocol):
    """Where jobs live between the request that created one and the worker that runs it."""

    def create(self, job: Job, *, user_id: UUID = ANONYMOUS_USER_ID) -> Job: ...

    def get(self, job_id: UUID, *, user_id: UUID = ANONYMOUS_USER_ID) -> Job | None: ...

    def by_idempotency_key(self, key: str, *, user_id: UUID = ANONYMOUS_USER_ID) -> Job | None: ...

    def save(self, job: Job, *, user_id: UUID = ANONYMOUS_USER_ID) -> Job: ...


def to_row(job: Job, user_id: UUID) -> JobRow:
    return JobRow(
        id=job.id,
        user_id=user_id,
        state=job.state.value,
        spec=job.spec.model_dump(mode="json"),
        idempotency_key=job.idempotency_key,
        result_sha256=job.result_sha256,
        error=job.error,
        created_at=job.created_at,
        updated_at=job.updated_at,
        steps=[
            JobStepRow(state=s.state.value, detail=s.detail, progress=s.progress, at=s.at)
            for s in job.steps
        ],
    )


def from_row(row: JobRow) -> Job:
    from editgpt_core.spec import EditSpec

    return Job(
        id=row.id,
        spec=EditSpec.model_validate(row.spec),
        state=JobState(row.state),
        idempotency_key=row.idempotency_key,
        created_at=row.created_at,
        updated_at=row.updated_at,
        steps=tuple(
            JobStep(state=JobState(s.state), at=s.at, detail=s.detail, progress=s.progress)
            for s in row.steps
        ),
        error=row.error,
        result_sha256=row.result_sha256,
    )


@dataclass
class InMemoryJobStore:
    """A dict. Enough for tests, the eager worker, and a local run with no Postgres."""

    _jobs: dict[UUID, Job] = field(default_factory=dict)
    _owners: dict[UUID, UUID] = field(default_factory=dict)

    def create(self, job: Job, *, user_id: UUID = ANONYMOUS_USER_ID) -> Job:
        if job.idempotency_key is not None:
            existing = self.by_idempotency_key(job.idempotency_key, user_id=user_id)
            if existing is not None:
                return existing
        self._jobs[job.id] = job
        self._owners[job.id] = user_id
        return job

    def get(self, job_id: UUID, *, user_id: UUID = ANONYMOUS_USER_ID) -> Job | None:
        if self._owners.get(job_id) != user_id:
            return None
        return self._jobs.get(job_id)

    def by_idempotency_key(self, key: str, *, user_id: UUID = ANONYMOUS_USER_ID) -> Job | None:
        return next(
            (
                job
                for job in self._jobs.values()
                if job.idempotency_key == key and self._owners.get(job.id) == user_id
            ),
            None,
        )

    def save(self, job: Job, *, user_id: UUID = ANONYMOUS_USER_ID) -> Job:
        if self._owners.get(job.id) not in (None, user_id):
            raise PermissionError(f"job {job.id} belongs to another user")
        self._jobs[job.id] = job
        self._owners[job.id] = user_id
        return job


@dataclass
class SqlJobStore:
    """Jobs in Postgres. One session per call, committed before returning.

    Steps are replaced wholesale on save rather than diffed. They are append-only and
    there are a handful per job, so the simple thing is also the correct thing — and a
    diff would be one more place for the in-memory job and the row to disagree.
    """

    session_factory: sa.orm.sessionmaker[Session]

    def create(self, job: Job, *, user_id: UUID = ANONYMOUS_USER_ID) -> Job:
        if job.idempotency_key is not None:
            existing = self.by_idempotency_key(job.idempotency_key, user_id=user_id)
            if existing is not None:
                log.info("job.idempotent_hit", extra={"job_id": str(existing.id)})
                return existing
        with self.session_factory() as session:
            session.add(to_row(job, user_id))
            session.commit()
        return job

    def get(self, job_id: UUID, *, user_id: UUID = ANONYMOUS_USER_ID) -> Job | None:
        with self.session_factory() as session:
            row = session.scalars(
                sa.select(JobRow)
                .options(selectinload(JobRow.steps))
                .where(JobRow.id == job_id, JobRow.user_id == user_id)
            ).one_or_none()
            return from_row(row) if row is not None else None

    def by_idempotency_key(self, key: str, *, user_id: UUID = ANONYMOUS_USER_ID) -> Job | None:
        with self.session_factory() as session:
            row = session.scalars(
                sa.select(JobRow)
                .options(selectinload(JobRow.steps))
                .where(JobRow.idempotency_key == key, JobRow.user_id == user_id)
            ).one_or_none()
            return from_row(row) if row is not None else None

    def save(self, job: Job, *, user_id: UUID = ANONYMOUS_USER_ID) -> Job:
        with self.session_factory() as session:
            row = session.get(JobRow, job.id, options=[selectinload(JobRow.steps)])
            if row is None:
                session.add(to_row(job, user_id))
                session.commit()
                return job
            if row.user_id != user_id:
                raise PermissionError(f"job {job.id} belongs to another user")

            row.state = job.state.value
            row.error = job.error
            row.result_sha256 = job.result_sha256
            row.updated_at = job.updated_at
            row.steps = [
                JobStepRow(state=s.state.value, detail=s.detail, progress=s.progress, at=s.at)
                for s in job.steps
            ]
            session.commit()
        return job
