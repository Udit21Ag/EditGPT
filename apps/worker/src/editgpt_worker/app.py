"""The Celery application and the resources a task needs.

Concurrency is one, deliberately. A worker that runs two edits at once holds two heavy
models resident and breaches the 8 GB budget before it finishes either — the `ModelSlot`
prevents that *within* a process, and `--concurrency=1` is what prevents it between them.
Throughput comes from a bigger host later, not from parallelism here.

Resources are built once per process and cached, not per task. Opening a Postgres pool
and a Redis connection for every edit would dominate the cost of a fast one, and
`max_tasks_per_child` already bounds how long a cached one lives.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from celery import Celery
from editgpt_store import (
    AssetStore,
    JobStore,
    LocalAssetStore,
    S3AssetStore,
    SqlJobStore,
    make_engine,
    make_session_factory,
)

from editgpt_worker.settings import Settings, get_settings

log = logging.getLogger(__name__)


def make_celery(settings: Settings | None = None) -> Celery:
    config = settings or get_settings()
    app = Celery("editgpt", broker=config.redis_url, backend=config.redis_url)
    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
        worker_concurrency=1,
        worker_max_tasks_per_child=config.max_tasks_per_child,
        task_soft_time_limit=config.task_soft_time_limit_s,
        task_time_limit=config.task_time_limit_s,
        task_acks_late=True,
        # With acks_late, a worker killed mid-task would otherwise have its message
        # redelivered and the edit run twice. Rejecting on worker loss makes the job fail
        # visibly instead, which is the honest outcome for work that costs real quota.
        task_reject_on_worker_lost=True,
        broker_connection_retry_on_startup=True,
    )
    # No `autodiscover_tasks` here. It would import `editgpt_worker.tasks` while this
    # module is still executing, and that module imports `celery_app` from here — a cycle.
    # Registration happens in `__init__`, which imports `tasks` *after* this line has run.
    return app


celery_app = make_celery()


@dataclass(frozen=True, slots=True)
class Resources:
    """Everything a task needs, built once per worker process."""

    settings: Settings
    jobs: JobStore
    assets: AssetStore
    redis: Any


@lru_cache(maxsize=1)
def resources() -> Resources:
    config = get_settings()
    import redis

    engine = make_engine(config.database_url)
    store: AssetStore
    if config.uses_object_storage:
        remote = S3AssetStore.from_settings(
            endpoint_url=config.s3_endpoint_url,
            access_key_id=config.s3_access_key_id,
            secret_access_key=config.s3_secret_access_key,
            bucket=config.s3_bucket,
            region=config.s3_region,
        )
        remote.ensure_bucket()
        store = remote
    else:
        config.asset_root.mkdir(parents=True, exist_ok=True)
        store = LocalAssetStore(root=config.asset_root)

    log.info(
        "worker.resources",
        extra={
            "storage": "s3" if config.uses_object_storage else "local",
            "environment": config.environment,
        },
    )
    return Resources(
        settings=config,
        jobs=SqlJobStore(session_factory=make_session_factory(engine)),
        assets=store,
        redis=redis.Redis.from_url(config.redis_url),
    )
