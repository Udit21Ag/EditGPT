"""The job lifecycle. A worker that loses a cancellation is the failure to prevent."""

from __future__ import annotations

import pytest
from editgpt_core import AssetRef, EditOp, EditSpec, MaskSource
from editgpt_core.errors import IllegalTransitionError
from editgpt_core.jobs import (
    TERMINAL,
    TRANSITIONS,
    Job,
    JobState,
    can_transition,
    check_transition,
)

DIGEST = "a" * 64
RESULT = "b" * 64


def spec() -> EditSpec:
    return EditSpec(
        op=EditOp.REMOVE,
        image_ref=AssetRef(bucket="edits", sha256=DIGEST, width=800, height=600),
        mask_source=MaskSource.TEXT,
        target="the car",
    )


def job(state: JobState = JobState.QUEUED) -> Job:
    base = Job(spec=spec())
    if state is JobState.QUEUED:
        return base
    running = base.advance(JobState.PLANNING).advance(JobState.RUNNING)
    if state is JobState.RUNNING:
        return running
    if state is JobState.PLANNING:
        return base.advance(JobState.PLANNING)
    if state is JobState.REVIEW:
        return running.advance(JobState.REVIEW)
    if state is JobState.DONE:
        return running.advance(JobState.REVIEW).advance(JobState.DONE, result_sha256=RESULT)
    if state is JobState.FAILED:
        return running.advance(JobState.FAILED, error="boom")
    return running.advance(JobState.CANCELLED)


def test_a_new_job_starts_queued_with_no_progress() -> None:
    fresh = Job(spec=spec())
    assert fresh.state is JobState.QUEUED
    assert fresh.progress == 0.0
    assert not fresh.is_terminal


@pytest.mark.parametrize("state", list(JobState))
def test_every_state_has_a_transition_rule(state: JobState) -> None:
    """A state missing from the table would raise KeyError at runtime, not a clear error."""
    assert state in TRANSITIONS


@pytest.mark.parametrize("state", sorted(TERMINAL))
def test_terminal_states_permit_nothing(state: JobState) -> None:
    assert TRANSITIONS[state] == frozenset()
    assert job(state).is_terminal


def test_the_happy_path_runs_end_to_end() -> None:
    done = job(JobState.DONE)
    assert done.state is JobState.DONE
    assert [s.state for s in done.steps] == [
        JobState.PLANNING,
        JobState.RUNNING,
        JobState.REVIEW,
        JobState.DONE,
    ]


def test_review_can_send_work_back_to_running() -> None:
    """The retry loop: the critic rejects a result and the job runs again."""
    assert can_transition(JobState.REVIEW, JobState.RUNNING)
    retried = job(JobState.REVIEW).advance(JobState.RUNNING, detail="critic asked for a retry")
    assert retried.state is JobState.RUNNING


def test_running_cannot_loop_directly_back_to_itself() -> None:
    """A retry must pass through review, so the decision to retry is always recorded."""
    assert not can_transition(JobState.RUNNING, JobState.RUNNING)


def test_an_illegal_transition_names_what_is_allowed() -> None:
    with pytest.raises(IllegalTransitionError, match="legal next states are"):
        job(JobState.QUEUED).advance(JobState.DONE, result_sha256=RESULT)


def test_a_terminal_job_says_it_is_terminal_in_the_error() -> None:
    with pytest.raises(IllegalTransitionError, match="terminal"):
        check_transition(JobState.DONE, JobState.RUNNING)


def test_cancelling_a_finished_job_is_a_no_op_not_an_error() -> None:
    """A user pressing cancel as the job completes is racing the worker; failing their
    request tells them nothing useful about a job that already succeeded."""
    done = job(JobState.DONE)
    assert done.cancel() is done


def test_cancelling_a_running_job_records_the_reason() -> None:
    cancelled = job(JobState.RUNNING).cancel("user closed the tab")
    assert cancelled.state is JobState.CANCELLED
    assert cancelled.steps[-1].detail == "user closed the tab"


@pytest.mark.parametrize("state", [JobState.QUEUED, JobState.PLANNING, JobState.RUNNING])
def test_any_live_state_can_be_cancelled(state: JobState) -> None:
    assert job(state).cancel().state is JobState.CANCELLED


def test_a_failed_job_must_say_why() -> None:
    with pytest.raises(ValueError, match="must record why it failed"):
        job(JobState.RUNNING).advance(JobState.FAILED)


def test_a_completed_job_must_reference_its_result() -> None:
    with pytest.raises(ValueError, match="must reference its result"):
        job(JobState.REVIEW).advance(JobState.DONE)


def test_advancing_returns_a_new_job_and_leaves_the_original_alone() -> None:
    """A job is handled by more than one process; in-place mutation that fails halfway
    leaves a record nobody can interpret."""
    original = job(JobState.QUEUED)
    advanced = original.advance(JobState.PLANNING)
    assert original.state is JobState.QUEUED
    assert advanced is not original
    assert advanced.id == original.id


def test_progress_carries_forward_when_a_step_does_not_set_it() -> None:
    partway = job(JobState.RUNNING).advance(JobState.REVIEW, progress=0.6)
    assert partway.progress == pytest.approx(0.6)
    assert partway.advance(JobState.RUNNING).progress == pytest.approx(0.6)


def test_steps_accumulate_as_an_audit_trail() -> None:
    assert len(job(JobState.DONE).steps) == 4


def test_updated_at_moves_forward_on_a_transition() -> None:
    before = job(JobState.QUEUED)
    after = before.advance(JobState.PLANNING)
    assert after.updated_at >= before.updated_at


def test_a_job_round_trips_through_json() -> None:
    original = job(JobState.DONE)
    assert Job.model_validate_json(original.model_dump_json()) == original


def test_an_idempotency_key_is_carried_on_the_job() -> None:
    keyed = Job(spec=spec(), idempotency_key="client-abc-123")
    assert keyed.advance(JobState.PLANNING).idempotency_key == "client-abc-123"
