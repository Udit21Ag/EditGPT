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
from dataclasses import dataclass
from uuid import UUID

from editgpt_core import EditSpec, Job, JobState
from editgpt_store import ANONYMOUS_USER_ID, ProgressEvent, publish, record_artifact, record_cost
from editgpt_store.records import record_image

from editgpt_worker.app import Resources, celery_app, resources

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Produced:
    """What an editor made: the bytes, and their real type and size.

    Reported rather than inferred from the request. `UPSCALE` doubles the dimensions and a
    format Pillow cannot write falls back to PNG, so recording what was *asked for* is how
    the `images` table comes to describe bytes that do not exist — which it did, briefly,
    for an AVIF upload.
    """

    data: bytes
    content_type: str
    width: int
    height: int


Editor = Callable[[bytes, EditSpec], Produced]
"""Takes the source image's bytes and the spec, returns what it produced.

Bytes rather than arrays on purpose: the worker's job is orchestration, and keeping the
pixel type out of this module is what stops the lifecycle from acquiring an opinion about
image libraries.
"""


def noop_editor(source: bytes, spec: EditSpec) -> Produced:
    """Return the input unchanged. The pipe-proving editor."""
    log.info("edit.noop", extra={"op": spec.op.value, "bytes": len(source)})
    return Produced(
        data=source,
        content_type=spec.image_ref.content_type,
        width=spec.image_ref.width,
        height=spec.image_ref.height,
    )


def _real_editor(source: bytes, spec: EditSpec) -> Produced:
    """The shipping editor. Imported lazily so `tasks` stays importable without a model.

    A module-level import would pull ONNX Runtime and OpenCV into anything that touches
    this file — including the gateway's own test suite, which imports the task name.
    """
    from editgpt_worker.editors import edit

    made = edit(source, spec)
    return Produced(
        data=made.data,
        content_type=made.content_type,
        width=made.width,
        height=made.height,
    )


EDITORS: dict[str, Editor] = {"noop": noop_editor, "default": _real_editor}
"""What a job may ask to be run by.

`noop` returns the image untouched and is kept deliberately: it proves the pipe —
upload, queue, progress, artifact — without a model, so when an edit misbehaves the
plumbing is not a suspect. `default` is the real thing.
"""


@celery_app.task(name="editgpt.ground")  # type: ignore[untyped-decorator]
def ground(digest: str, phrase: str) -> dict[str, object]:
    """Resolve a phrase to candidate regions. No edit, no job, no state.

    A task rather than gateway code because models live in workers (invariant 3): the web
    tier must stay small enough to run beside a worker on one 8 GB machine, and grounding
    needs the detector and the SAM encoder — about 2 GB between them.

    The gateway waits on the result rather than streaming it. Grounding is a few seconds
    and the client cannot do anything useful until it lands, so a job with progress would
    be ceremony around a request/response.
    """
    from editgpt_worker.editors import decode_image, ground_phrase, working_size

    res = resources()
    # Bounded, like an edit. `decode_image` keeps the upload's full resolution so a result
    # can be returned at it (TD-021); grounding has no such need and three reasons not to:
    # every candidate mask is produced at this size and travels to the browser, the client
    # decodes each one to draw a thumbnail, and SAM resizes to 1024 internally regardless.
    # A 15.9 MP upload would ship five 15.9 MP masks to render pictures a few hundred
    # pixels wide.
    found = ground_phrase(working_size(decode_image(res.assets.get(digest))), phrase)
    log.info(
        "ground.done",
        extra={
            "candidates": len(found.candidates),
            "ambiguous": found.ambiguous,
            "margin": found.margin,
        },
    )
    return found.model_dump(mode="json")


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
        made = EDITORS[editor](source, job.spec)

        job = _advance(
            res, job, JobState.REVIEW, owner=owner, progress=0.8, detail="checking the result"
        )
        digest = res.assets.put(made.data, content_type=made.content_type)
        _record_outputs(res, job, digest, editor=editor, made=made, owner=owner)

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
    res: Resources, job: Job, digest: str, *, editor: str, made: Produced, owner: UUID
) -> None:
    """Note the artifact, its metadata, and what producing it cost.

    Everything recorded comes from what the editor actually produced. Taking the
    dimensions and the content type from the *request* instead is how a row comes to
    describe bytes that are not there: `UPSCALE` doubles the size, and an unsupported
    output format falls back to PNG.
    """
    session_factory = getattr(res.jobs, "session_factory", None)
    if session_factory is None:
        return  # an in-memory store: nothing to write these to, and nothing depends on them

    record_image(
        session_factory,
        sha256=digest,
        width=made.width,
        height=made.height,
        content_type=made.content_type,
        byte_size=len(made.data),
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
