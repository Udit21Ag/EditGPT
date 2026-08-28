"""Job intake, idempotency, cancellation and the progress stream."""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

import pytest
from editgpt_core import EditOp, JobState
from editgpt_gateway.app import create_app
from editgpt_gateway.deps import RecordingQueue, Services
from editgpt_gateway.settings import Settings
from editgpt_store import ProgressEvent
from fastapi.testclient import TestClient


def remove_request(digest: str) -> dict[str, Any]:
    return {"op": "remove", "image_sha256": digest, "mask_source": "text", "target": "the car"}


def test_a_job_is_accepted_and_queued(
    client: TestClient, services: Services, uploaded: str
) -> None:
    response = client.post("/v1/jobs", json=remove_request(uploaded))
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["state"] == JobState.QUEUED
    assert body["op"] == EditOp.REMOVE

    queue = services.queue
    assert isinstance(queue, RecordingQueue)
    assert queue.sent == [(body["id"], "default")], "a job with no editor named runs the real one"


def test_a_job_against_an_unknown_image_is_refused(client: TestClient) -> None:
    """The image must exist before the job, or the worker fails on a missing asset."""
    response = client.post("/v1/jobs", json=remove_request("f" * 64))
    assert response.status_code == 404
    assert "upload it first" in response.json()["detail"]


def test_an_unactionable_edit_is_rejected_with_the_contract_s_reason(
    client: TestClient, uploaded: str
) -> None:
    """`EditSpec` does the rejecting; the gateway only translates it into a 422.

    The point of the assertion on the message is that the contract's wording survives:
    re-implementing the rule here would be a second opinion that drifts.
    """
    response = client.post(
        "/v1/jobs",
        json={"op": "remove", "image_sha256": uploaded, "mask_source": "text"},
    )
    assert response.status_code == 422
    assert "target" in response.json()["detail"]


def test_an_add_without_content_is_rejected(client: TestClient, uploaded: str) -> None:
    response = client.post(
        "/v1/jobs",
        json={"op": "add", "image_sha256": uploaded, "mask_source": "text", "target": "the wall"},
    )
    assert response.status_code == 422
    assert "content" in response.json()["detail"]


def test_a_mask_of_the_wrong_size_is_rejected(client: TestClient, uploaded: str) -> None:
    response = client.post(
        "/v1/jobs",
        json={
            "op": "remove",
            "image_sha256": uploaded,
            "mask_source": "brush",
            "mask": {"width": 10, "height": 10, "counts": [50, 50]},
        },
    )
    assert response.status_code == 422
    assert "10x10" in response.json()["detail"]


def test_the_same_idempotency_key_returns_the_first_job_and_queues_once(
    client: TestClient, services: Services, uploaded: str
) -> None:
    """A client retrying after a dropped response must not pay for the work twice."""
    headers = {"Idempotency-Key": "retry-me"}
    first = client.post("/v1/jobs", json=remove_request(uploaded), headers=headers)
    second = client.post("/v1/jobs", json=remove_request(uploaded), headers=headers)

    assert first.status_code == 202
    assert second.status_code == 200, "a repeat is not a new job"
    assert second.json()["id"] == first.json()["id"]

    queue = services.queue
    assert isinstance(queue, RecordingQueue)
    assert len(queue.sent) == 1, "the second request must not enqueue more work"


def test_an_unknown_job_is_a_404(client: TestClient) -> None:
    assert client.get("/v1/jobs/1c1b1a19-0000-4000-8000-000000000000").status_code == 404


def test_a_new_job_reads_back_queued_with_no_steps(client: TestClient, uploaded: str) -> None:
    created = client.post("/v1/jobs", json=remove_request(uploaded)).json()
    body = client.get(f"/v1/jobs/{created['id']}").json()
    assert body["id"] == created["id"]
    assert body["state"] == JobState.QUEUED
    assert body["steps"] == []


def test_cancelling_moves_a_queued_job_to_cancelled(client: TestClient, uploaded: str) -> None:
    created = client.post("/v1/jobs", json=remove_request(uploaded)).json()
    body = client.post(f"/v1/jobs/{created['id']}/cancel").json()
    assert body["state"] == JobState.CANCELLED


def test_cancelling_twice_is_not_an_error(client: TestClient, uploaded: str) -> None:
    """A user racing the worker should not be shown a failure about work that is over."""
    created = client.post("/v1/jobs", json=remove_request(uploaded)).json()
    client.post(f"/v1/jobs/{created['id']}/cancel")
    again = client.post(f"/v1/jobs/{created['id']}/cancel")
    assert again.status_code == 200
    assert again.json()["state"] == JobState.CANCELLED


def test_the_editor_name_is_constrained_to_a_slug(client: TestClient, uploaded: str) -> None:
    """It becomes a dictionary key in another process; unbounded input there is a hazard."""
    response = client.post("/v1/jobs", json=remove_request(uploaded) | {"editor": "../../etc"})
    assert response.status_code == 422


def test_the_rate_limit_refuses_the_fourth_request_in_a_window(
    settings: Settings, services: Services, uploaded: str
) -> None:
    """Configured to 3/minute in the fixture, so the fourth is the one that must fail."""
    services.redis = FakeRedis()
    with TestClient(create_app(settings, services)) as limited:
        codes = [
            limited.post("/v1/jobs", json=remove_request(uploaded)).status_code for _ in range(4)
        ]
    assert codes[:3] == [202, 202, 202]
    assert codes[3] == 429


def test_reading_a_job_is_never_rate_limited(
    settings: Settings, services: Services, uploaded: str
) -> None:
    """Polling costs nothing, and a client we throttle into blindness will just poll harder."""
    services.redis = FakeRedis()
    with TestClient(create_app(settings, services)) as limited:
        created = limited.post("/v1/jobs", json=remove_request(uploaded)).json()
        codes = [limited.get(f"/v1/jobs/{created['id']}").status_code for _ in range(10)]
    assert set(codes) == {200}


# ---------------------------------------------------------------- progress stream


class FakeRedis:
    """Enough Redis to exercise the limiter and the progress channel, in one process.

    A real fake rather than a mock: the SSE test below depends on `publish` and
    `subscribe` actually agreeing about the payload, which a mock would assert away.
    """

    def __init__(self, published: list[str] | None = None) -> None:
        self.counters: dict[str, int] = {}
        self.values: dict[str, str] = {}
        self.published = published or []

    def incr(self, key: str) -> int:
        self.counters[key] = self.counters.get(key, 0) + 1
        return self.counters[key]

    def expire(self, key: str, seconds: int) -> None:
        return None

    def ttl(self, key: str) -> int:
        return 60

    def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.values[key] = value

    def get(self, key: str) -> str | None:
        return self.values.get(key)

    def publish(self, channel: str, payload: str) -> None:
        self.published.append(payload)

    def pubsub(self, **_: Any) -> FakePubSub:
        return FakePubSub(self.published)


class FakePubSub:
    def __init__(self, queued: list[str]) -> None:
        self.queued = list(queued)

    def subscribe(self, channel: str) -> None:
        return None

    def get_message(self, timeout: float = 0.0) -> dict[str, Any] | None:
        if not self.queued:
            return None
        return {"type": "message", "data": self.queued.pop(0)}

    def close(self) -> None:
        return None


def sse_events(body: str) -> list[dict[str, Any]]:
    return [
        json.loads(line.removeprefix("data: "))
        for line in body.splitlines()
        if line.startswith("data: ")
    ]


def test_the_stream_replays_persisted_steps_before_following_the_channel(
    client: TestClient, services: Services, uploaded: str
) -> None:
    """Neither source is complete alone, which is the whole reason both are read."""
    created = client.post("/v1/jobs", json=remove_request(uploaded)).json()
    job = services.jobs.get(UUID(created["id"]))
    assert job is not None

    services.jobs.save(job.advance(JobState.PLANNING, detail="planning", progress=0.1))
    services.redis = FakeRedis(
        published=[
            ProgressEvent(created["id"], "running", 0.5, "erasing").to_json(),
            ProgressEvent(created["id"], "done", 1.0, "done", terminal=True).to_json(),
        ]
    )

    with client.stream("GET", f"/v1/jobs/{created['id']}/events") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = sse_events("".join(response.iter_text()))

    assert [e["state"] for e in events] == ["planning", "running", "done"]
    assert events[-1]["terminal"] is True


def test_the_stream_of_a_finished_job_ends_without_waiting(
    client: TestClient, services: Services, uploaded: str
) -> None:
    """A client opening the stream after the fact must not hang on a message already sent."""
    created = client.post("/v1/jobs", json=remove_request(uploaded)).json()
    job = services.jobs.get(UUID(created["id"]))
    assert job is not None
    services.jobs.save(job.cancel())

    with client.stream("GET", f"/v1/jobs/{created['id']}/events") as response:
        events = sse_events("".join(response.iter_text()))
    assert events[-1]["state"] == "cancelled"
    assert events[-1]["terminal"] is True


def test_the_stream_of_an_unknown_job_is_a_404(client: TestClient) -> None:
    response = client.get("/v1/jobs/1c1b1a19-0000-4000-8000-000000000000/events")
    assert response.status_code == 404


@pytest.mark.parametrize("op", ["restyle", "retouch"])
def test_an_unsupported_operation_is_not_advertised(client: TestClient, op: str) -> None:
    assert op in client.get("/capabilities").json()["unsupported"]


@pytest.mark.parametrize("digest", ["", "not-a-digest", "A" * 64, "f" * 63, "f" * 65])
def test_a_malformed_digest_is_a_422_not_a_500(client: TestClient, digest: str) -> None:
    """Found by running the service by hand: an empty `image_sha256` reached the asset
    store, which rejects anything that is not a digest — correct there, a 500 here."""
    response = client.post("/v1/jobs", json=remove_request(digest))
    assert response.status_code == 422, response.text


# ---------------------------------------------------------------- grounding


def ground_body(digest: str, target: str = "the car") -> dict[str, Any]:
    return {"image_sha256": digest, "target": target}


def test_grounding_returns_candidates_from_the_worker(
    client: TestClient, services: Services, uploaded: str
) -> None:
    """The gateway must not ground anything itself: models live in workers."""
    from editgpt_core import Grounding, MaskCandidate, MaskRef

    answer = Grounding(
        candidates=[
            MaskCandidate(
                box=(0.1, 0.1, 0.4, 0.4),
                score=0.9,
                mask_ref=MaskRef(width=4, height=4, counts=[0, 8, 8]),
            )
        ],
        ambiguous=False,
        margin=0.8,
    )
    queue = services.queue
    assert isinstance(queue, RecordingQueue)
    queue.answers["editgpt.ground"] = answer.model_dump(mode="json")

    response = client.post("/v1/masks", json=ground_body(uploaded))
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["ambiguous"] is False
    assert len(body["candidates"]) == 1
    assert body["candidates"][0]["mask"]["counts"] == [0, 8, 8]
    assert queue.called == [("editgpt.ground", (uploaded, "the car"))]


def test_grounding_reports_ambiguity_so_the_client_can_ask(
    client: TestClient, services: Services, uploaded: str
) -> None:
    from editgpt_core import Grounding, MaskCandidate, MaskRef

    two = [
        MaskCandidate(
            box=(0.1 * (i + 1), 0.1, 0.3 + 0.1 * i, 0.4),
            score=0.55 - 0.05 * i,
            mask_ref=MaskRef(width=4, height=4, counts=[0, 8, 8]),
        )
        for i in range(2)
    ]
    queue = services.queue
    assert isinstance(queue, RecordingQueue)
    queue.answers["editgpt.ground"] = Grounding(
        candidates=two, ambiguous=True, margin=0.05
    ).model_dump(mode="json")

    body = client.post("/v1/masks", json=ground_body(uploaded)).json()
    assert body["ambiguous"] is True
    assert body["margin"] == pytest.approx(0.05)
    assert len(body["candidates"]) == 2


def test_grounding_an_unknown_image_is_refused_before_the_worker_is_bothered(
    client: TestClient, services: Services
) -> None:
    response = client.post("/v1/masks", json=ground_body("f" * 64))
    assert response.status_code == 404

    queue = services.queue
    assert isinstance(queue, RecordingQueue)
    assert not queue.called, "a missing image should not cost a worker round trip"


def test_an_empty_phrase_is_rejected_by_the_contract(client: TestClient, uploaded: str) -> None:
    assert client.post("/v1/masks", json=ground_body(uploaded, "")).status_code == 422


def test_an_unbounded_phrase_is_rejected(client: TestClient, uploaded: str) -> None:
    """It reaches a model with a fixed token budget; an unbounded string there is a slow
    request a caller can simply ask for."""
    assert client.post("/v1/masks", json=ground_body(uploaded, "x" * 5000)).status_code == 422


def test_grounding_a_malformed_digest_is_a_422_not_a_500(client: TestClient) -> None:
    assert client.post("/v1/masks", json=ground_body("not-a-digest")).status_code == 422


def test_no_worker_is_a_503_and_not_a_500(
    settings: Settings, services: Services, uploaded: str
) -> None:
    """Nothing is wrong with the request; the client should retry, not change it."""
    from editgpt_gateway.deps import Queue

    class Dead(Queue):
        def send(self, job_id: str, *, editor: str = "noop", user_id: str = "") -> None:
            return None

        def call(self, name: str, *args: object, timeout_s: float = 30.0) -> dict[str, object]:
            raise TimeoutError("no worker answered")

    services.queue = Dead()
    with TestClient(create_app(settings, services)) as offline:
        assert offline.post("/v1/masks", json=ground_body(uploaded)).status_code == 503


# ---------------------------------------------------------------- backdrop colour


def background_request(digest: str, **over: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "op": "background",
        "image_sha256": digest,
        "mask_source": "whole",
        "colour": "#3366ff",
    }
    body.update(over)
    return body


def test_a_backdrop_colour_is_carried_onto_the_job(
    client: TestClient, services: Services, uploaded: str
) -> None:
    """TD-020. The colour used to stop at the boundary — there was nowhere to put it —
    so `background` painted the same green whatever the request said."""
    response = client.post("/v1/jobs", json=background_request(uploaded))
    assert response.status_code == 202, response.text

    stored = services.jobs.get(UUID(response.json()["id"]))
    assert stored is not None
    assert stored.spec.colour == "#3366ff"
    assert stored.spec.rgb_colour((0, 0, 0)) == (0x33, 0x66, 0xFF)


def test_a_backdrop_with_neither_colour_nor_description_is_refused(
    client: TestClient, uploaded: str
) -> None:
    response = client.post("/v1/jobs", json=background_request(uploaded, colour=None))
    assert response.status_code == 422
    assert "`colour` or `content`" in response.json()["detail"]


def test_a_colour_that_is_not_a_colour_names_the_field(client: TestClient, uploaded: str) -> None:
    """Caught by the request model rather than deep inside, so the 422 says `colour`."""
    response = client.post("/v1/jobs", json=background_request(uploaded, colour="cornflower"))
    assert response.status_code == 422
    assert "colour" in response.text


# ---------------------------------------------------------------- magic select

EMPTY_GROUNDING: dict[str, Any] = {"candidates": [], "ambiguous": False, "margin": 0.0}
"""A phrase matching nothing is a real answer, and the shortest valid one to hand back
while asserting which task was dispatched."""


def test_taps_are_dispatched_to_the_point_task_not_the_phrase_one(
    client: TestClient, services: Services, uploaded: str
) -> None:
    """The routing that keeps a 200 MB detector out of a question that never asks it."""
    queue = services.queue
    assert isinstance(queue, RecordingQueue)
    queue.answers["editgpt.ground_points"] = EMPTY_GROUNDING

    client.post(
        "/v1/masks",
        json={
            "image_sha256": uploaded,
            "points": [{"x": 0.25, "y": 0.5}, {"x": 0.6, "y": 0.4, "include": False}],
        },
    )

    assert queue.called, "nothing was dispatched"
    name, args = queue.called[-1]
    assert name == "editgpt.ground_points"
    # Triples rather than objects: Celery serialises arguments as JSON.
    assert args[1] == [[0.25, 0.5, True], [0.6, 0.4, False]]


def test_a_phrase_still_goes_to_the_phrase_task(
    client: TestClient, services: Services, uploaded: str
) -> None:
    queue = services.queue
    assert isinstance(queue, RecordingQueue)
    queue.answers["editgpt.ground"] = EMPTY_GROUNDING

    client.post("/v1/masks", json={"image_sha256": uploaded, "target": "the car"})
    name, args = queue.called[-1]
    assert name == "editgpt.ground"
    assert args[1] == "the car"
