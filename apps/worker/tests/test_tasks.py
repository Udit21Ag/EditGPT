"""The job lifecycle, driven end to end with local doubles.

`run_job` reads its collaborators from `resources()`, which is `lru_cache`d, so these
tests replace that cache entry rather than mocking the task. The task under test is
therefore the real one — the same code path a Celery worker executes — with a real
in-memory job store, a real local asset store and a fake Redis.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from editgpt_core import AssetRef, EditOp, EditSpec, Job, JobState, MaskSource
from editgpt_store import InMemoryJobStore, LocalAssetStore, ProgressEvent
from editgpt_worker import tasks
from editgpt_worker.app import Resources, resources
from editgpt_worker.settings import Settings


class FakeRedis:
    """Records what was published, which is what the SSE stream would have shown."""

    def __init__(self) -> None:
        self.events: list[ProgressEvent] = []
        self.values: dict[str, str] = {}

    def publish(self, channel: str, payload: str) -> None:
        self.events.append(ProgressEvent.from_json(payload))

    def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.values[key] = value


def png_bytes(colour: tuple[int, int, int] = (30, 90, 150)) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), colour).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def redis() -> FakeRedis:
    return FakeRedis()


@pytest.fixture
def res(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, redis: FakeRedis) -> Resources:
    """Real stores, fake Redis, no database.

    `run_job` reaches its collaborators through `tasks.resources()`, so replacing that one
    function is enough to run the genuine task against doubles. Nothing about the task is
    stubbed.
    """
    built = Resources(
        settings=Settings(environment="test", asset_root=tmp_path / "assets"),
        jobs=InMemoryJobStore(),
        assets=LocalAssetStore(root=tmp_path / "assets"),
        redis=redis,
    )
    monkeypatch.setattr(tasks, "resources", lambda: built)
    return built


@pytest.fixture
def queued(res: Resources) -> Job:
    """A job whose source image is in the store, ready to run."""
    data = png_bytes()
    digest = res.assets.put(data, content_type="image/png")
    spec = EditSpec(
        op=EditOp.REMOVE,
        image_ref=AssetRef(bucket="local", sha256=digest, width=32, height=24),
        mask_source=MaskSource.TEXT,
        target="the car",
    )
    return res.jobs.create(Job(spec=spec))


def test_a_job_runs_to_done_through_every_state(res: Resources, queued: Job) -> None:
    result = tasks.run_job(str(queued.id))
    assert result["state"] == JobState.DONE

    finished = res.jobs.get(queued.id)
    assert finished is not None
    assert [s.state for s in finished.steps] == [
        JobState.PLANNING,
        JobState.RUNNING,
        JobState.REVIEW,
        JobState.DONE,
    ]
    assert finished.progress == pytest.approx(1.0)
    assert finished.result_sha256 is not None


def test_the_noop_editor_returns_the_input_untouched(res: Resources, queued: Job) -> None:
    """The whole point of the pipe-proving editor: byte-identical in and out."""
    tasks.run_job(str(queued.id))
    finished = res.jobs.get(queued.id)
    assert finished is not None
    assert finished.result_sha256 == queued.spec.image_ref.sha256
    assert res.assets.get(finished.result_sha256) == res.assets.get(queued.spec.image_ref.sha256)


def test_progress_is_published_for_every_transition(
    res: Resources, queued: Job, redis: FakeRedis
) -> None:
    tasks.run_job(str(queued.id))
    assert [e.state for e in redis.events] == ["planning", "running", "review", "done"]
    assert [e.progress for e in redis.events] == [0.1, 0.3, 0.8, 1.0]
    assert redis.events[-1].terminal is True
    assert all(not e.terminal for e in redis.events[:-1])


def test_a_cancelled_job_stops_at_the_next_checkpoint(
    res: Resources, queued: Job, redis: FakeRedis
) -> None:
    """Cancellation is cooperative, so it takes effect at a transition, not instantly."""
    res.jobs.save(queued.cancel())
    result = tasks.run_job(str(queued.id))

    assert result["state"] == JobState.CANCELLED
    finished = res.jobs.get(queued.id)
    assert finished is not None
    assert finished.state is JobState.CANCELLED
    assert redis.events == [], "no progress should be announced for work that never started"


def test_a_missing_source_image_fails_the_job_with_a_reason(res: Resources, queued: Job) -> None:
    res.assets.delete(queued.spec.image_ref.sha256)
    result = tasks.run_job(str(queued.id))

    assert result["state"] == JobState.FAILED
    finished = res.jobs.get(queued.id)
    assert finished is not None
    assert finished.state is JobState.FAILED
    assert finished.error is not None
    assert "AssetNotFoundError" in finished.error


def test_an_editor_that_raises_fails_the_job_rather_than_leaving_it_running(
    res: Resources, queued: Job, redis: FakeRedis
) -> None:
    """The failure mode the two Celery time limits exist to prevent, forced directly."""

    def explode(source: bytes, spec: EditSpec) -> tasks.Produced:
        raise RuntimeError("the model fell over")

    tasks.EDITORS["exploding"] = explode
    try:
        result = tasks.run_job(str(queued.id), editor="exploding")
    finally:
        del tasks.EDITORS["exploding"]

    assert result["state"] == JobState.FAILED
    finished = res.jobs.get(queued.id)
    assert finished is not None
    assert finished.error == "RuntimeError: the model fell over"
    assert redis.events[-1].terminal is True, "a watching client must be told it ended"


def test_a_redelivered_message_for_a_finished_job_does_not_rerun_it(
    res: Resources, queued: Job, redis: FakeRedis
) -> None:
    """`acks_late` means a message can arrive twice; the work must not be paid for twice."""
    tasks.run_job(str(queued.id))
    published = len(redis.events)

    replay = tasks.run_job(str(queued.id))
    assert replay["replayed"] is True
    assert len(redis.events) == published, "a replay must not re-announce progress"


def test_an_unknown_job_is_reported_rather_than_raised(res: Resources) -> None:
    """Raising would make Celery retry a message that can never succeed."""
    result = tasks.run_job("1c1b1a19-0000-4000-8000-000000000000")
    assert result["state"] == "missing"


def test_the_pipe_prover_is_still_registered(res: Resources) -> None:
    """`noop` must survive the arrival of the real editor.

    It is what proves the queue, the transitions, the artifacts and the ledger without a
    model, so a failure there is never ambiguous between the plumbing and the pixels. The
    Phase 3 version of this test asserted `noop` was the *only* editor; that guard has
    done its job, and this is what replaces it.
    """
    assert "noop" in tasks.EDITORS
    assert "default" in tasks.EDITORS, "the shipping editor should be registered"


def test_resources_are_cached_per_process() -> None:
    """Load-bearing: a Postgres pool and a Redis connection per task would dominate a fast one."""
    assert hasattr(resources, "cache_clear"), "resources() must stay lru_cached"
