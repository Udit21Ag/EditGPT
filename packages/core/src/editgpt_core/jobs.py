"""The job lifecycle, as a contract.

Pure Python and Pydantic — no database, no queue. Persistence lives in `editgpt_store`,
and both the gateway and the worker import *this*, so the two cannot disagree about what
a legal transition is.

The transition table is data rather than a chain of `if` statements because it is the
thing most likely to be got wrong: a worker that moves a cancelled job back to `running`
loses a user's cancellation, and no test catches that if the rule lives in three places.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Self
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from editgpt_core.errors import IllegalTransitionError
from editgpt_core.spec import EditSpec


class JobState(StrEnum):
    QUEUED = "queued"
    PLANNING = "planning"
    RUNNING = "running"
    REVIEW = "review"
    """The critic is scoring the result and may send it back to `running` for a retry."""
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL: frozenset[JobState] = frozenset({JobState.DONE, JobState.FAILED, JobState.CANCELLED})

TRANSITIONS: dict[JobState, frozenset[JobState]] = {
    JobState.QUEUED: frozenset({JobState.PLANNING, JobState.CANCELLED, JobState.FAILED}),
    JobState.PLANNING: frozenset({JobState.RUNNING, JobState.CANCELLED, JobState.FAILED}),
    # RUNNING -> RUNNING is not a self-loop for free: a retry goes through REVIEW, so the
    # decision to retry is always recorded rather than implied.
    JobState.RUNNING: frozenset({JobState.REVIEW, JobState.CANCELLED, JobState.FAILED}),
    JobState.REVIEW: frozenset(
        {JobState.RUNNING, JobState.DONE, JobState.CANCELLED, JobState.FAILED}
    ),
    JobState.DONE: frozenset(),
    JobState.FAILED: frozenset(),
    JobState.CANCELLED: frozenset(),
}


def can_transition(current: JobState, requested: JobState) -> bool:
    return requested in TRANSITIONS[current]


def check_transition(current: JobState, requested: JobState) -> None:
    """Raise unless the move is legal. Call this before persisting a state change."""
    if not can_transition(current, requested):
        allowed = [state.value for state in sorted(TRANSITIONS[current])]
        raise IllegalTransitionError(
            f"cannot move a job from {current} to {requested}; legal next states are "
            f"{allowed or 'none - this state is terminal'}"
        )


class JobStep(BaseModel):
    """One recorded stage of a job's execution, for progress and for audit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    state: JobState
    at: datetime
    detail: str = ""
    progress: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0


class Job(BaseModel):
    """A unit of work. Immutable — a transition returns a new instance.

    Immutability is deliberate: a job is handled by more than one process, and an
    in-place mutation that fails halfway leaves a record nobody can interpret.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    spec: EditSpec
    state: JobState = JobState.QUEUED
    idempotency_key: str | None = None
    """Supplied by the client. Two requests carrying the same key are the same job."""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    steps: tuple[JobStep, ...] = ()
    error: str | None = None
    result_sha256: str | None = None

    @model_validator(mode="after")
    def _terminal_states_carry_their_outcome(self) -> Self:
        if self.state is JobState.FAILED and not self.error:
            raise ValueError("a failed job must record why it failed")
        if self.state is JobState.DONE and not self.result_sha256:
            raise ValueError("a completed job must reference its result")
        return self

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL

    @property
    def progress(self) -> float:
        return self.steps[-1].progress if self.steps else 0.0

    def advance(
        self,
        to: JobState,
        *,
        detail: str = "",
        progress: float | None = None,
        error: str | None = None,
        result_sha256: str | None = None,
    ) -> Job:
        """Return a new job in state `to`, or raise if the move is not permitted."""
        check_transition(self.state, to)
        now = datetime.now(UTC)
        step = JobStep(
            state=to,
            at=now,
            detail=detail,
            progress=self.progress if progress is None else progress,
        )
        # Constructed rather than `model_copy(update=...)`: that method skips validation
        # by design, so the invariants below — a failed job records why, a done job
        # references its result — would never run on a transition, which is the only time
        # they can actually be violated.
        return Job(
            id=self.id,
            spec=self.spec,
            state=to,
            idempotency_key=self.idempotency_key,
            created_at=self.created_at,
            updated_at=now,
            steps=(*self.steps, step),
            error=error if error is not None else self.error,
            result_sha256=(result_sha256 if result_sha256 is not None else self.result_sha256),
        )

    def cancel(self, reason: str = "cancelled by the user") -> Job:
        """Cancelling a finished job is a no-op, not an error.

        A user pressing cancel as a job completes is racing the worker, and failing their
        request tells them nothing useful about a job that already succeeded.
        """
        if self.is_terminal:
            return self
        return self.advance(JobState.CANCELLED, detail=reason)
