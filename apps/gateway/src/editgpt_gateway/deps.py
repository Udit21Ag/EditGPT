"""What the routes need, assembled once at startup.

Everything here is chosen by configuration rather than by import, which is what lets the
same application run three ways without a flag in the route code: against Postgres and
Redis for real, against SQLite and a fake queue in tests, and against local disk instead
of object storage when no endpoint is configured.

The degraded modes are explicit and logged. A gateway that silently falls back to
in-memory storage in production would look healthy while losing every job on restart, so
each fallback says so at `WARNING` and the readiness endpoint reports which one is in
use.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Annotated, Any

from editgpt_store import (
    AssetStore,
    InMemoryJobStore,
    JobStore,
    LocalAssetStore,
    S3AssetStore,
    SqlJobStore,
    bootstrap,
    make_engine,
    make_session_factory,
)
from fastapi import Depends, Request

from editgpt_gateway.settings import Settings

log = logging.getLogger(__name__)


class Queue:
    """Somewhere to send a job for execution.

    Deliberately not Celery's `Task` object: the gateway must not import the worker, or
    the worker's model dependencies would end up in the web process — which is
    architectural invariant 3. It sends a message by name.
    """

    def send(self, job_id: str, *, editor: str = "noop", user_id: str = "") -> None:
        raise NotImplementedError

    def call(self, name: str, *args: object, timeout_s: float = 30.0) -> dict[str, object]:
        """Run a task and wait for its answer.

        Only for work that is short and that the caller cannot proceed without — grounding
        a phrase, not editing an image. It occupies a gateway thread for the duration,
        which is why the timeout is not optional.
        """
        raise NotImplementedError


@dataclass
class CeleryQueue(Queue):
    """The real queue: a task sent by name over the broker."""

    app: Any

    def send(self, job_id: str, *, editor: str = "noop", user_id: str = "") -> None:
        # The owner travels with the message. The worker then has no way to read a job
        # without knowing whose it is, which means there is no privileged bypass to
        # forget to protect later.
        self.app.send_task(
            "editgpt.run_job", args=[job_id], kwargs={"editor": editor, "user_id": user_id}
        )

    def call(self, name: str, *args: object, timeout_s: float = 30.0) -> dict[str, object]:
        result: dict[str, object] = self.app.send_task(name, args=list(args)).get(timeout=timeout_s)
        return result


@dataclass
class RecordingQueue(Queue):
    """Records what would have been sent. Used by tests, and when Redis is absent."""

    sent: list[tuple[str, str]]
    called: list[tuple[str, tuple[object, ...]]] = field(default_factory=list)
    answers: dict[str, dict[str, object]] = field(default_factory=dict)

    def send(self, job_id: str, *, editor: str = "noop", user_id: str = "") -> None:
        del user_id
        self.sent.append((job_id, editor))

    def call(self, name: str, *args: object, timeout_s: float = 30.0) -> dict[str, object]:
        del timeout_s  # nothing is really waited on here
        self.called.append((name, args))
        return self.answers.get(name, {})


@dataclass
class Services:
    """The gateway's collaborators. Attached to `app.state` and read by the routes."""

    settings: Settings
    jobs: JobStore
    assets: AssetStore
    queue: Queue
    redis: Any | None
    storage_kind: str
    job_store_kind: str

    @property
    def degraded(self) -> list[str]:
        """Which collaborators are running in a fallback mode, for `/ready`."""
        problems = []
        if self.job_store_kind == "memory":
            problems.append("jobs are in memory and will not survive a restart")
        if self.redis is None:
            problems.append("no redis: progress streaming and rate limiting are off")
        if isinstance(self.queue, RecordingQueue):
            problems.append("no queue: jobs are accepted but never executed")
        if not self.settings.uses_clerk:
            problems.append("no authentication: every request acts as the shared account")
        return problems


def get_services(request: Request) -> Services:
    """The gateway's collaborators, for a route or another dependency that needs them.

    Reached through `request.app.state` rather than a closure over `create_app`. That is
    not a style choice: these modules use `from __future__ import annotations`, so FastAPI
    resolves each parameter's annotation as a *string* against the module's globals. A
    `Depends(closure)` would name something that only exists inside `create_app`, the
    marker would fail to resolve, and FastAPI would quietly treat the parameter as a query
    string field — which is precisely what it did before this was moved out.

    It lives here rather than in `app.py` so `auth.py` can depend on it without the two
    importing each other.
    """
    services = request.app.state.services
    assert isinstance(services, Services)
    return services


ServicesDep = Annotated[Services, Depends(get_services)]


def build_services(settings: Settings) -> Services:
    """Wire everything up, falling back loudly rather than failing to start.

    A gateway that refuses to boot without Postgres is a gateway nobody can run locally
    to look at the frontend. A gateway that pretends to be healthy without it is worse.
    The compromise is to start, degrade, and say so on `/ready`.
    """
    assets, storage_kind = _build_assets(settings)
    jobs, job_store_kind = _build_jobs(settings)
    redis_client = _build_redis(settings)
    queue: Queue = (
        CeleryQueue(app=_celery_client(settings))
        if redis_client is not None
        else RecordingQueue([])
    )
    return Services(
        settings=settings,
        jobs=jobs,
        assets=assets,
        queue=queue,
        redis=redis_client,
        storage_kind=storage_kind,
        job_store_kind=job_store_kind,
    )


def _build_assets(settings: Settings) -> tuple[AssetStore, str]:
    if settings.uses_object_storage:
        store = S3AssetStore.from_settings(
            endpoint_url=settings.s3_endpoint_url,
            access_key_id=settings.s3_access_key_id,
            secret_access_key=settings.s3_secret_access_key,
            bucket=settings.s3_bucket,
            region=settings.s3_region,
        )
        store.ensure_bucket()
        return store, "s3"
    settings.asset_root.mkdir(parents=True, exist_ok=True)
    return LocalAssetStore(root=settings.asset_root), "local"


def _build_jobs(settings: Settings) -> tuple[JobStore, str]:
    try:
        engine = make_engine(settings.database_url)
        # `bootstrap` is idempotent and cheap. It exists so a developer with a fresh
        # container gets a working gateway without remembering to run a migration;
        # a deployment still runs Alembic, which is what can express a change.
        bootstrap(engine)
        return SqlJobStore(session_factory=make_session_factory(engine)), "postgres"
    except Exception as error:
        log.warning("gateway.no_database", extra={"error": str(error)})
        return InMemoryJobStore(), "memory"


def _build_redis(settings: Settings) -> Any | None:
    try:
        import redis

        client = redis.Redis.from_url(settings.redis_url, socket_connect_timeout=1)
        client.ping()
    except Exception as error:
        log.warning("gateway.no_redis", extra={"error": str(error)})
        return None
    return client


def _celery_client(settings: Settings) -> Any:
    from celery import Celery

    return Celery("editgpt-gateway", broker=settings.redis_url, backend=settings.redis_url)
