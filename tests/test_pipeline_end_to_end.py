"""The pipe, executed: upload → job → worker → progress → result.

Every job here names `editor: "noop"` deliberately. This file proves the *plumbing* —
transitions, progress, artifacts, idempotency, cancellation — and `noop` returns the image
untouched, so a failure here is never ambiguous between the queue and the model. It also
keeps `make check` hermetic: the real editor loads 550 MB of weights.

This is the only test that drives the gateway and the worker together. It runs the real
FastAPI app over a real HTTP client, a real SQLAlchemy store, a real filesystem asset
store and the **real Celery task function**, with a queue that executes eagerly in-process
and an in-memory Redis. Nothing about the lifecycle is stubbed.

Whether the *edit* is any good is `make eval`'s question, on real photographs.

What it does not cover, and what does cover it instead: the Postgres wire protocol
(`packages/store` runs the same repository against SQLAlchemy), Redis pub/sub transport
(`test_progress.py`), and Celery's own broker (its behaviour is configuration, not ours).
Those three are the parts a container would add, and each is exercised somewhere.
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from editgpt_core import JobState
from editgpt_gateway.app import create_app
from editgpt_gateway.deps import Queue, Services
from editgpt_gateway.settings import Settings
from editgpt_store import LocalAssetStore, ProgressEvent, bootstrap, make_engine
from editgpt_store import make_session_factory as build_session_factory
from editgpt_store.jobs import SqlJobStore
from editgpt_store.records import artifacts_for, spend_since
from editgpt_worker import tasks
from editgpt_worker.app import Resources
from editgpt_worker.settings import Settings as WorkerSettings
from fastapi.testclient import TestClient


class InProcessRedis:
    """Publishes into a list the SSE stream can then read back."""

    def __init__(self) -> None:
        self.published: list[str] = []
        self.values: dict[str, str] = {}

    def publish(self, channel: str, payload: str) -> None:
        self.published.append(payload)

    def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.values[key] = value

    def get(self, key: str) -> str | None:
        return self.values.get(key)

    def incr(self, key: str) -> int:
        return 1

    def expire(self, key: str, seconds: int) -> None:
        return None

    def ttl(self, key: str) -> int:
        return 60

    def pubsub(self, **_: Any) -> InProcessPubSub:
        return InProcessPubSub(self.published)


class InProcessPubSub:
    def __init__(self, queued: list[str]) -> None:
        self.queued = list(queued)

    def subscribe(self, channel: str) -> None:
        return None

    def get_message(self, timeout: float = 0.0) -> dict[str, Any] | None:
        return {"type": "message", "data": self.queued.pop(0)} if self.queued else None

    def close(self) -> None:
        return None


class EagerQueue(Queue):
    """Runs the task inline. This is Celery's own `task_always_eager`, without Celery.

    The point is that `tasks.run_job` really executes — a queue that only recorded the
    message would make this an integration test of nothing. `execute` can be switched off
    for the one test that needs a job created but not yet run.
    """

    def __init__(self, *, execute: bool = True) -> None:
        self.sent: list[tuple[str, str]] = []
        self.execute = execute

    def send(self, job_id: str, *, editor: str = "noop", user_id: str = "") -> None:
        self.sent.append((job_id, editor))
        if self.execute:
            tasks.run_job(job_id, editor=editor, user_id=user_id)


def png_bytes(width: int = 96, height: int = 72) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (12, 200, 90)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def stack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    assets = LocalAssetStore(root=tmp_path / "assets")
    engine = make_engine("sqlite+pysqlite:///:memory:")
    bootstrap(engine)
    session_factory = build_session_factory(engine)
    jobs = SqlJobStore(session_factory=session_factory)
    redis = InProcessRedis()
    queue = EagerQueue()

    monkeypatch.setattr(
        tasks,
        "resources",
        lambda: Resources(
            settings=WorkerSettings(environment="test", asset_root=tmp_path / "assets"),
            jobs=jobs,
            assets=assets,
            redis=redis,
        ),
    )

    settings = Settings(environment="test", asset_root=tmp_path / "assets", max_megapixels=40.0)
    services = Services(
        settings=settings,
        jobs=jobs,
        assets=assets,
        queue=queue,
        redis=redis,
        storage_kind="local",
        job_store_kind="postgres",
    )
    with TestClient(create_app(settings, services)) as client:
        yield {
            "client": client,
            "jobs": jobs,
            "assets": assets,
            "redis": redis,
            "queue": queue,
            "session_factory": session_factory,
        }
    engine.dispose()


def test_upload_to_result_end_to_end(stack: dict[str, Any]) -> None:
    """The Phase 3 exit criterion, in one test."""
    client: TestClient = stack["client"]

    uploaded = client.post("/v1/images", files={"file": ("photo.png", png_bytes(), "image/png")})
    assert uploaded.status_code == 201, uploaded.text
    digest = uploaded.json()["sha256"]

    created = client.post(
        "/v1/jobs",
        json={
            "op": "remove",
            "image_sha256": digest,
            "mask_source": "text",
            "target": "the car",
            "editor": "noop",
        },
        headers={"Idempotency-Key": "e2e-1"},
    )
    assert created.status_code == 202, created.text
    job_id = created.json()["id"]

    # The eager queue ran the real task synchronously, so the job is already finished.
    finished = client.get(f"/v1/jobs/{job_id}").json()
    assert finished["state"] == JobState.DONE
    assert finished["progress"] == 1.0
    assert [s["state"] for s in finished["steps"]] == ["planning", "running", "review", "done"]

    result = client.get(f"/v1/images/{finished['result_sha256']}")
    assert result.status_code == 200
    assert result.content == png_bytes(), "the noop editor returns the image untouched"


def test_the_progress_stream_shows_the_whole_run(stack: dict[str, Any]) -> None:
    client: TestClient = stack["client"]
    digest = client.post(
        "/v1/images", files={"file": ("photo.png", png_bytes(), "image/png")}
    ).json()["sha256"]
    job_id = client.post(
        "/v1/jobs",
        json={"op": "remove", "image_sha256": digest, "target": "the car", "editor": "noop"},
    ).json()["id"]

    with client.stream("GET", f"/v1/jobs/{job_id}/events") as response:
        body = "".join(response.iter_text())
    events = [
        json.loads(line.removeprefix("data: "))
        for line in body.splitlines()
        if line.startswith("data: ")
    ]

    # Replayed from `job_steps`, because the run finished before the stream opened. That
    # is the case the replay exists for.
    assert [e["state"] for e in events] == ["planning", "running", "review", "done"]
    assert events[-1]["terminal"] is True


def test_the_worker_published_every_transition(stack: dict[str, Any]) -> None:
    client: TestClient = stack["client"]
    digest = client.post(
        "/v1/images", files={"file": ("photo.png", png_bytes(), "image/png")}
    ).json()["sha256"]
    client.post(
        "/v1/jobs",
        json={"op": "remove", "image_sha256": digest, "target": "the car", "editor": "noop"},
    )

    published = [ProgressEvent.from_json(p) for p in stack["redis"].published]
    assert [e.state for e in published] == ["planning", "running", "review", "done"]


def test_the_run_is_recorded_as_an_artifact_and_a_ledger_entry(stack: dict[str, Any]) -> None:
    """A job that leaves no trace cannot be audited or billed."""
    client: TestClient = stack["client"]
    digest = client.post(
        "/v1/images", files={"file": ("photo.png", png_bytes(), "image/png")}
    ).json()["sha256"]
    job_id = client.post(
        "/v1/jobs",
        json={"op": "remove", "image_sha256": digest, "target": "the car", "editor": "noop"},
    ).json()["id"]

    artifacts = artifacts_for(stack["session_factory"], UUID(job_id))
    assert [a.kind for a in artifacts] == ["result"]

    calls, cents = spend_since(stack["session_factory"])
    assert calls == 1, "a local edit is free but still counts against a quota"
    assert cents == pytest.approx(0.0)


def test_a_retried_request_does_not_run_the_job_twice(stack: dict[str, Any]) -> None:
    """The whole point of the idempotency key, checked at the level that matters."""
    client: TestClient = stack["client"]
    digest = client.post(
        "/v1/images", files={"file": ("photo.png", png_bytes(), "image/png")}
    ).json()["sha256"]
    body = {"op": "remove", "image_sha256": digest, "target": "the car", "editor": "noop"}
    headers = {"Idempotency-Key": "retry-me"}

    first = client.post("/v1/jobs", json=body, headers=headers)
    second = client.post("/v1/jobs", json=body, headers=headers)

    assert (first.status_code, second.status_code) == (202, 200)
    assert second.json()["id"] == first.json()["id"]
    assert len(stack["queue"].sent) == 1, "the retry must not enqueue a second run"


def test_a_cancelled_job_never_reaches_the_editor(stack: dict[str, Any]) -> None:
    """Cancellation is checked before each transition, so a queued job stops immediately."""
    client: TestClient = stack["client"]
    digest = client.post(
        "/v1/images", files={"file": ("photo.png", png_bytes(), "image/png")}
    ).json()["sha256"]

    # Create the job without running it, cancel, then run the task by hand.
    stack["queue"].execute = False
    job_id = client.post(
        "/v1/jobs",
        json={"op": "remove", "image_sha256": digest, "target": "the car", "editor": "noop"},
    ).json()["id"]
    client.post(f"/v1/jobs/{job_id}/cancel")

    outcome = tasks.run_job(job_id)
    assert outcome["state"] == JobState.CANCELLED
    assert client.get(f"/v1/jobs/{job_id}").json()["state"] == JobState.CANCELLED
