"""The job lifecycle, driven.

One task drives every job: `run_job`. What actually edits the pixels is a separate
callable looked up by name, so the lifecycle — transitions, progress, cancellation,
artifacts, the ledger — is written and tested exactly once and a new operation is a new
editor rather than a new copy of all of this.

Phase 3 ships one editor, `noop`, which returns the input untouched. That is the point:
it proves upload → queue → progress → artifact end to end with no model involved, so
when a model is attached and something breaks, the pipe is not a suspect.

Cancellation is checked before every transition rather than signalled. A cooperative
check is enough because the work between two checks is one model pass, and the
alternative — killing the process — loses the record of why the job stopped.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from uuid import UUID

from editgpt_core import EditSpec, Job, JobState
from editgpt_store import ANONYMOUS_USER_ID, ProgressEvent, publish, record_artifact, record_cost
from editgpt_store.records import record_image

from editgpt_worker.app import Resources, celery_app, resources

log = logging.getLogger(__name__)

Editor = Callable[[bytes, EditSpec], bytes]
"""Takes the source image's bytes and the spec, returns the edited image's bytes.

Bytes rather than arrays on purpose: the worker's job is orchestration, and keeping the
pixel type out of this module is what stops the lifecycle from acquiring an opinion about
image libraries.
"""


def noop_editor(source: bytes, spec: EditSpec) -> bytes:
    """Return the input unchanged. The pipe-proving editor."""
    log.info("edit.noop", extra={"op": spec.op.value, "bytes": len(source)})
    return source


EDITORS: dict[str, Editor] = {"noop": noop_editor}


class CancelledError(RuntimeError):
    """The job was cancelled between two checkpoints."""


def _announce(res: Resources, job: Job, *, detail: str = "") -> None:
    publish(
        res.redis,
        ProgressEvent(
            job_id=str(job.id),
            state=job.state.value,
            progress=job.progress,
            detail=detail or (job.steps[-1].detail if job.steps else ""),
            terminal=job.is_terminal,
        ),
    )


def _advance(
    res: Resources,
    job: Job,
    to: JobState,
    *,
    owner: UUID,
    progress: float,
    detail: str,
    result_sha256: str | None = None,
) -> Job:
    """Move the job, persist it, and tell anyone watching. In that order.

    Persisting before announcing matters: a client that reacts to the event by fetching
    the job must not see a state older than the one it was just told about.
    """
    current = res.jobs.get(job.id, user_id=owner)
    if current is not None and current.state is JobState.CANCELLED:
        raise CancelledError(f"job {job.id} was cancelled")

    moved = job.advance(to, progress=progress, detail=detail, result_sha256=result_sha256)
    res.jobs.save(moved, user_id=owner)
    _announce(res, moved, detail=detail)
    log.info(
        "job.state",
        extra={"job_id": str(moved.id), "state": to.value, "progress": progress},
    )
    return moved


# Celery ships no stubs, so its decorator erases the signature. The narrow ignore keeps
# the rest of this module strictly typed rather than silencing the file.
@celery_app.task(name="editgpt.run_job")  # type: ignore[untyped-decorator]
def run_job(job_id: str, editor: str = "noop", user_id: str = "") -> dict[str, object]:
    """Drive one job from `queued` to a terminal state.

    `user_id` travels with the message rather than being looked up here, so the worker has
    no way to read a job without knowing whose it is. That is deliberate: a privileged
    "fetch any job" path would be the thing everyone forgets to protect once accounts
    exist. Empty means the anonymous sentinel, which is every job today.

    Returns a small summary rather than the image: the result is in the asset store and
    referenced by digest, and putting pixels in a Celery result backend would put them in
    Redis, which is the one place in this system sized in megabytes.
    """
    res = resources()
    identifier = UUID(job_id)
    owner = UUID(user_id) if user_id else ANONYMOUS_USER_ID
    job = res.jobs.get(identifier, user_id=owner)
    if job is None:
        log.error("job.missing", extra={"job_id": job_id})
        return {"job_id": job_id, "state": "missing"}
    if job.is_terminal:
        # A redelivered message for a job that already finished. Not an error, and
        # re-running it would spend quota to produce what we already have.
        return {"job_id": job_id, "state": job.state.value, "replayed": True}

    started = time.monotonic()
    try:
        job = _advance(
            res, job, JobState.PLANNING, owner=owner, progress=0.1, detail="reading the request"
        )
        source = res.assets.get(job.spec.image_ref.sha256)

        job = _advance(
            res, job, JobState.RUNNING, owner=owner, progress=0.3, detail=f"running {editor}"
        )
        edited = EDITORS[editor](source, job.spec)

        job = _advance(
            res, job, JobState.REVIEW, owner=owner, progress=0.8, detail="checking the result"
        )
        digest = res.assets.put(edited, content_type=job.spec.image_ref.content_type)
        _record_outputs(res, job, digest, editor=editor, byte_size=len(edited), owner=owner)

        job = _advance(
            res,
            job,
            JobState.DONE,
            owner=owner,
            progress=1.0,
            detail="done",
            result_sha256=digest,
        )
        log.info(
            "job.done",
            extra={"job_id": job_id, "seconds": round(time.monotonic() - started, 3)},
        )
        return {"job_id": job_id, "state": job.state.value, "result_sha256": digest}

    except CancelledError:
        log.info("job.cancelled", extra={"job_id": job_id})
        return {"job_id": job_id, "state": JobState.CANCELLED.value}

    except Exception as error:
        # This catches `SoftTimeLimitExceeded` too — verified: it derives from Exception,
        # not BaseException. That matters, because a timeout escaping here would leave
        # the job in `running` with no explanation, which is the exact failure the two
        # time limits exist to avoid.
        reason = f"{type(error).__name__}: {error}"
        log.exception("job.failed", extra={"job_id": job_id})
        current = res.jobs.get(identifier, user_id=owner)
        if current is not None and not current.is_terminal:
            failed = current.advance(JobState.FAILED, progress=current.progress, error=reason)
            res.jobs.save(failed, user_id=owner)
            _announce(res, failed, detail=reason)
        return {"job_id": job_id, "state": JobState.FAILED.value, "error": reason}


def _record_outputs(
    res: Resources, job: Job, digest: str, *, editor: str, byte_size: int, owner: UUID
) -> None:
    """Note the artifact, its metadata, and what producing it cost."""
    session_factory = getattr(res.jobs, "session_factory", None)
    if session_factory is None:
        return  # an in-memory store: nothing to write these to, and nothing depends on them

    # Dimensions come from the spec because every operation shipped so far preserves
    # them. `UPSCALE` will not, and when it lands this must read them from the produced
    # bytes instead — recorded in the debt register rather than guessed at now.
    record_image(
        session_factory,
        sha256=digest,
        width=job.spec.image_ref.width,
        height=job.spec.image_ref.height,
        content_type=job.spec.image_ref.content_type,
        byte_size=byte_size,
        user_id=owner,
    )
    record_artifact(session_factory, job_id=job.id, sha256=digest, kind="result")
    record_cost(
        session_factory,
        provider=f"local:{editor}",
        operation=job.spec.op.value,
        units=1,
        cents=0.0,
        job_id=job.id,
        user_id=owner,
    )
