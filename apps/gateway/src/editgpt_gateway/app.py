"""The gateway application.

The HTTP surface: upload an image, ask for an edit, watch it happen, fetch the result.
No models load here — invariant 3 — so this process stays small enough to run beside a
worker on one 8 GB machine. Everything expensive is a message on a queue.

The job endpoints are deliberately thin. `editgpt_core.jobs` owns which transitions are
legal and `editgpt_store` owns where a job lives; what is left here is translating HTTP
into those, which is all a gateway should be.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from typing import Annotated, Any
from uuid import UUID

from editgpt_core import (
    AssetRef,
    Constraints,
    EditOp,
    EditSpec,
    Job,
    JobState,
    MaskRef,
    MaskSource,
)
from editgpt_store import AssetNotFoundError, ProgressEvent, last_event, record_image, subscribe
from fastapi import FastAPI, Header, HTTPException, Request, Response, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.orm import Session, sessionmaker

from editgpt_gateway import auth, limits, uploads
from editgpt_gateway.auth import PrincipalDep
from editgpt_gateway.deps import Services, ServicesDep, build_services
from editgpt_gateway.settings import Settings, get_settings
from editgpt_gateway.uploads import UploadRejectedError

log = logging.getLogger(__name__)

API_VERSION = "0.2.0"
KEEPALIVE_S = 15.0
"""How often the SSE stream emits a comment when nothing is happening.

Proxies and mobile networks close a connection that has been silent for a while, and a
silently dropped progress stream looks to a user exactly like a hung job.
"""


class Health(BaseModel):
    status: str
    version: str
    environment: str


class Readiness(Health):
    """Readiness reports the *degraded* modes, not just liveness.

    A gateway with no queue still answers every request and never finishes a job. Saying
    so here is what stops that from being discovered by a user instead of by a check.
    """

    storage: str
    jobs: str
    auth: dict[str, Any]
    degraded: list[str]


class Capabilities(BaseModel):
    """What this deployment can actually do.

    Advertised rather than assumed: Phase 0 established that only some of the planned
    operations have a working model behind them, and the frontend should not offer the
    others.
    """

    operations: list[EditOp]
    unsupported: dict[str, str]
    max_megapixels: float
    max_upload_mb: int


class ImageCreated(BaseModel):
    sha256: str
    width: int
    height: int
    content_type: str
    megapixels: float


class MaskPayload(BaseModel):
    width: int
    height: int
    counts: list[int]


class JobRequest(BaseModel):
    """What a client asks for. Translated into an `EditSpec`, which does the validating."""

    op: EditOp
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    """Constrained here rather than downstream. Without the pattern a malformed digest
    reached the asset store, which raises `ValueError` on anything that is not a digest —
    correct for the store, and a 500 for the client. The boundary owns this."""

    mask_source: MaskSource = MaskSource.TEXT
    target: str | None = None
    content: str | None = None
    mask: MaskPayload | None = None
    editor: str = Field(default="noop", pattern=r"^[a-z][a-z0-9_]{0,31}$")
    """Which editor the worker should run. Constrained to a slug because it is used as a
    dictionary key in another process; an unbounded string there is a lookup on
    user-controlled data."""

    max_seconds: float | None = None
    allow_remote: bool | None = None


class JobStepView(BaseModel):
    state: JobState
    at: str
    detail: str
    progress: float


class JobView(BaseModel):
    id: UUID
    state: JobState
    progress: float
    op: EditOp
    result_sha256: str | None
    error: str | None
    created_at: str
    updated_at: str
    steps: list[JobStepView]

    @classmethod
    def of(cls, job: Job) -> JobView:
        return cls(
            id=job.id,
            state=job.state,
            progress=job.progress,
            op=job.spec.op,
            result_sha256=job.result_sha256,
            error=job.error,
            created_at=job.created_at.isoformat(),
            updated_at=job.updated_at.isoformat(),
            steps=[
                JobStepView(
                    state=s.state, at=s.at.isoformat(), detail=s.detail, progress=s.progress
                )
                for s in job.steps
            ],
        )


def create_app(settings: Settings | None = None, services: Services | None = None) -> FastAPI:
    config = settings or get_settings()
    app = FastAPI(title="EditGPT gateway", version=API_VERSION)
    app.state.services = services or build_services(config)

    # ------------------------------------------------------------------ health

    @app.get("/health", response_model=Health)
    def health() -> Health:
        return Health(status="ok", version=API_VERSION, environment=config.environment)

    @app.get("/ready", response_model=Readiness)
    def ready(svc: ServicesDep) -> Readiness:
        """Liveness and readiness are separate so a restart is not mistaken for a rollout."""
        problems = svc.degraded
        return Readiness(
            status="degraded" if problems else "ready",
            version=API_VERSION,
            environment=config.environment,
            storage=svc.storage_kind,
            jobs=svc.job_store_kind,
            auth=auth.describe(config),
            degraded=problems,
        )

    @app.get("/capabilities", response_model=Capabilities)
    def capabilities() -> Capabilities:
        return Capabilities(
            operations=[
                EditOp.REMOVE,
                EditOp.ADD,
                EditOp.REPLACE,
                EditOp.BACKGROUND,
                EditOp.UPSCALE,
            ],
            unsupported={
                EditOp.RESTYLE: "no free instruction-editing model exists",
                EditOp.RETOUCH: "not scoped for v1",
            },
            max_megapixels=config.max_megapixels,
            max_upload_mb=config.max_upload_mb,
        )

    # ------------------------------------------------------------------ images

    @app.post("/v1/images", response_model=ImageCreated, status_code=201)
    async def upload_image(
        request: Request,
        file: UploadFile,
        svc: ServicesDep,
        principal: PrincipalDep,
    ) -> ImageCreated:
        _enforce_rate_limit(request, svc)

        # Refuse an oversized body before buffering it. The declared length is only a
        # claim, so `read_bounded` checks the real one too; this just avoids paying for
        # the obvious case.
        declared = request.headers.get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > _max_body(config):
            raise HTTPException(413, "the upload is larger than this service accepts")

        try:
            data = await uploads.read_bounded(file, config.max_upload_bytes)
            inspected = uploads.inspect(data, max_megapixels=config.max_megapixels)
        except UploadRejectedError as error:
            raise HTTPException(400, str(error)) from error

        digest = svc.assets.put(inspected.data, content_type=inspected.content_type)
        _record(
            svc,
            sha256=digest,
            width=inspected.width,
            height=inspected.height,
            content_type=inspected.content_type,
            byte_size=len(inspected.data),
            user_id=principal,
        )
        log.info(
            "image.uploaded",
            extra={
                "digest": digest,
                "megapixels": round(inspected.megapixels, 2),
                "bytes": len(inspected.data),
            },
        )
        return ImageCreated(
            sha256=digest,
            width=inspected.width,
            height=inspected.height,
            content_type=inspected.content_type,
            megapixels=round(inspected.megapixels, 3),
        )

    @app.get("/v1/images/{digest}")
    def download_image(digest: str, svc: ServicesDep, principal: PrincipalDep) -> Response:
        """Serve an asset by digest.

        Immutable by construction — the name *is* the content — so it is cacheable
        forever, and `private` because that cache is a user's browser and not a CDN.

        **Authenticated but not ownership-checked**, and the distinction is deliberate.
        Requiring a signed-in caller keeps a user's photographs off the open internet.
        Requiring *ownership* would be wrong here: storage is content-addressed, so two
        people uploading the same picture share one digest and one `images` row, and the
        second uploader would be locked out of their own upload. Recorded as TD-019; the
        fix is a user↔image join table, not a check bolted on here.
        """
        del principal  # the dependency is the check; the value is not needed
        try:
            data = svc.assets.get(digest)
        except (AssetNotFoundError, ValueError) as error:
            raise HTTPException(404, "no such image") from error
        return Response(
            content=data,
            media_type=_content_type_of(svc, digest),
            headers={"cache-control": "private, max-age=31536000, immutable"},
        )

    # ------------------------------------------------------------------ jobs

    @app.post("/v1/jobs", response_model=JobView, status_code=202)
    def create_job(
        request: Request,
        body: JobRequest,
        svc: ServicesDep,
        principal: PrincipalDep,
        response: Response,
        idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    ) -> JobView:
        _enforce_rate_limit(request, svc)

        if not svc.assets.exists(body.image_sha256):
            raise HTTPException(404, f"no image {body.image_sha256}; upload it first")

        spec = _to_spec(svc, body)
        job = Job(spec=spec, idempotency_key=idempotency_key)
        stored = svc.jobs.create(job, user_id=principal)
        if stored.id != job.id:
            # The key matched an existing job. Returning it rather than a conflict is the
            # point of idempotency: a client retrying after a dropped response gets the
            # job it already created, not a second one doing the same work twice.
            response.status_code = 200
            log.info("job.idempotent", extra={"job_id": str(stored.id)})
            return JobView.of(stored)

        svc.queue.send(str(stored.id), editor=body.editor, user_id=str(principal))
        log.info(
            "job.created",
            extra={"job_id": str(stored.id), "op": spec.op.value, "editor": body.editor},
        )
        return JobView.of(stored)

    @app.get("/v1/jobs/{job_id}", response_model=JobView)
    def read_job(job_id: UUID, svc: ServicesDep, principal: PrincipalDep) -> JobView:
        job = svc.jobs.get(job_id, user_id=principal)
        if job is None:
            raise HTTPException(404, "no such job")
        return JobView.of(job)

    @app.post("/v1/jobs/{job_id}/cancel", response_model=JobView)
    def cancel_job(job_id: UUID, svc: ServicesDep, principal: PrincipalDep) -> JobView:
        """Cancel a job. Cancelling a finished one succeeds and changes nothing.

        `Job.cancel` makes that a no-op rather than an error on purpose: a user pressing
        cancel as the job completes is racing the worker, and a 409 tells them nothing
        useful about work that already succeeded.
        """
        job = svc.jobs.get(job_id, user_id=principal)
        if job is None:
            raise HTTPException(404, "no such job")
        cancelled = svc.jobs.save(job.cancel(), user_id=principal)
        _publish(svc, cancelled)
        return JobView.of(cancelled)

    @app.get("/v1/jobs/{job_id}/events")
    def job_events(job_id: UUID, svc: ServicesDep, principal: PrincipalDep) -> StreamingResponse:
        """Server-sent events for one job.

        The persisted steps are replayed first, then the live channel is joined. Either
        alone is incomplete — the channel keeps no history, and the table lags by the
        length of a database write — so a client that connects late still sees
        everything, which on a mobile network is the normal case rather than the edge.
        """
        job = svc.jobs.get(job_id, user_id=principal)
        if job is None:
            raise HTTPException(404, "no such job")
        return StreamingResponse(
            _event_stream(svc, job),
            media_type="text/event-stream",
            headers={
                "cache-control": "no-store",
                "connection": "keep-alive",
                # Named for nginx, which otherwise buffers the stream into uselessness.
                "x-accel-buffering": "no",
            },
        )

    @app.get("/")
    def root() -> dict[str, Any]:
        return {"service": "editgpt-gateway", "version": API_VERSION, "docs": "/docs"}

    return app


CHUNK_SLACK = 64 * 1024
"""Headroom on the declared length for multipart framing, which is not part of the file."""


def _max_body(config: Settings) -> int:
    return config.max_upload_bytes + CHUNK_SLACK


def _enforce_rate_limit(request: Request, svc: Services) -> None:
    if request.url.path not in svc.settings.rate_limit_burst_paths:
        return
    identity = limits.identify(
        request.client.host if request.client else None,
        request.headers.get("x-forwarded-for"),
    )
    decision = limits.check(svc.redis, identity, limit=svc.settings.rate_limit_per_minute)
    if not decision.allowed:
        raise HTTPException(
            429,
            "too many requests; this service runs on a free tier and is rate limited",
            headers={"retry-after": str(decision.retry_after_s)},
        )


def _to_spec(svc: Services, body: JobRequest) -> EditSpec:
    """Translate a request into the contract, letting the contract do the rejecting.

    `EditSpec` already refuses a remove with no target and a mask whose size disagrees
    with its image. Re-implementing those checks here would be a second opinion that
    drifts; instead its `ValidationError` becomes a 422 with the reason intact.
    """
    meta = _image_meta(svc, body.image_sha256)
    defaults = Constraints()
    constraints = Constraints(
        max_seconds=defaults.max_seconds if body.max_seconds is None else body.max_seconds,
        allow_remote=defaults.allow_remote if body.allow_remote is None else body.allow_remote,
    )
    try:
        return EditSpec(
            op=body.op,
            image_ref=AssetRef(
                bucket=svc.storage_kind,
                sha256=body.image_sha256,
                width=meta[0],
                height=meta[1],
                content_type=meta[2],
            ),
            mask_source=body.mask_source,
            target=body.target,
            content=body.content,
            mask_ref=(
                MaskRef(width=body.mask.width, height=body.mask.height, counts=body.mask.counts)
                if body.mask is not None
                else None
            ),
            constraints=constraints,
        )
    except ValidationError as error:
        raise HTTPException(422, _first_message(error)) from error


def _first_message(error: ValidationError) -> str:
    for problem in error.errors():
        message = str(problem.get("msg", ""))
        return message.removeprefix("Value error, ")
    return "the request is not a valid edit"


def _session_factory(svc: Services) -> sessionmaker[Session] | None:
    return getattr(svc.jobs, "session_factory", None)


def _record(svc: Services, **fields: Any) -> None:
    factory = _session_factory(svc)
    if factory is not None:
        record_image(factory, **fields)


def _image_meta(svc: Services, digest: str) -> tuple[int, int, str]:
    """Dimensions of a stored image, from the database if it is there and the bytes if not."""
    factory = _session_factory(svc)
    if factory is not None:
        from editgpt_store.models import Image

        with factory() as session:
            row = session.get(Image, digest)
            if row is not None:
                return row.width, row.height, row.content_type

    try:
        inspected = uploads.inspect(svc.assets.get(digest), max_megapixels=float("inf"))
    except (AssetNotFoundError, UploadRejectedError, ValueError) as error:
        raise HTTPException(404, f"no image {digest}") from error
    return inspected.width, inspected.height, inspected.content_type


def _content_type_of(svc: Services, digest: str) -> str:
    factory = _session_factory(svc)
    if factory is None:
        return "application/octet-stream"
    from editgpt_store.models import Image

    with factory() as session:
        row = session.get(Image, digest)
        return row.content_type if row is not None else "application/octet-stream"


def _publish(svc: Services, job: Job) -> None:
    if svc.redis is None:
        return
    from editgpt_store import publish

    publish(
        svc.redis,
        ProgressEvent(
            job_id=str(job.id),
            state=job.state.value,
            progress=job.progress,
            detail=job.steps[-1].detail if job.steps else "",
            terminal=job.is_terminal,
        ),
    )


def _sse(event: ProgressEvent) -> str:
    """One SSE frame. `to_json` is reused so the wire format matches the channel's."""
    return f"event: progress\ndata: {event.to_json()}\n\n"


def _event_stream(svc: Services, job: Job) -> Iterator[str]:
    """Replay what has happened, then follow what happens next.

    A synchronous generator, which Starlette iterates in a thread pool — the Redis client
    here is the blocking one, and using it directly in the event loop would stall every
    other request on this process for the length of a job.
    """
    seen = 0
    for step in job.steps:
        yield _sse(
            ProgressEvent(
                job_id=str(job.id),
                state=step.state.value,
                progress=step.progress,
                detail=step.detail,
                terminal=step.state in {JobState.DONE, JobState.FAILED, JobState.CANCELLED},
            )
        )
        seen += 1

    if job.is_terminal:
        return

    if svc.redis is None:
        # Without Redis there is nothing to follow. Ending the stream is honest; holding
        # it open would look like a job that never progresses.
        yield ": no progress channel available\n\n"
        return

    latest = last_event(svc.redis, job.id)
    if latest is not None and seen == 0:
        yield _sse(latest)
        if latest.terminal:
            return

    last_ping = time.monotonic()
    for event in subscribe(svc.redis, job.id):
        if event is None:
            if time.monotonic() - last_ping >= KEEPALIVE_S:
                last_ping = time.monotonic()
                yield ": keepalive\n\n"
            continue
        yield _sse(event)
        if event.terminal:
            return


app = create_app()
