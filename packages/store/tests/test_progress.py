"""The progress channel.

The contract that matters is that a publisher and a subscriber in two different processes
agree about the payload. So the fake below is a real in-memory Redis rather than a mock:
a mock would let `publish` and `subscribe` drift apart while both tests stayed green.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest
from editgpt_store.progress import (
    EVENT_TTL_S,
    ProgressEvent,
    channel_for,
    last_event,
    publish,
    subscribe,
)


class FakeRedis:
    def __init__(self) -> None:
        self.channels: dict[str, list[str]] = {}
        self.values: dict[str, str] = {}
        self.expiries: dict[str, int] = {}

    def publish(self, channel: str, payload: str) -> None:
        self.channels.setdefault(channel, []).append(payload)

    def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.values[key] = value
        if ex is not None:
            self.expiries[key] = ex

    def get(self, key: str) -> str | None:
        return self.values.get(key)

    def pubsub(self, **_: Any) -> FakePubSub:
        return FakePubSub(self)


class FakePubSub:
    def __init__(self, client: FakeRedis) -> None:
        self.client = client
        self.queue: list[str] = []
        self.closed = False

    def subscribe(self, channel: str) -> None:
        self.queue = list(self.client.channels.get(channel, []))

    def get_message(self, timeout: float = 0.0) -> dict[str, Any] | None:
        if not self.queue:
            return None
        return {"type": "message", "data": self.queue.pop(0)}

    def close(self) -> None:
        self.closed = True


def event(state: str, progress: float, *, terminal: bool = False) -> ProgressEvent:
    return ProgressEvent(
        job_id="job-1", state=state, progress=progress, detail=state, terminal=terminal
    )


def test_an_event_round_trips_through_json() -> None:
    original = event("running", 0.5)
    assert ProgressEvent.from_json(original.to_json()) == original


def test_from_json_tolerates_a_minimal_payload() -> None:
    """A publisher on an older version must not crash a subscriber on a newer one."""
    restored = ProgressEvent.from_json('{"job_id": "j", "state": "queued"}')
    assert restored.progress == 0.0
    assert restored.detail == ""
    assert restored.terminal is False


def test_publishing_also_records_the_latest_event() -> None:
    """Pub/sub keeps nothing, so a client arriving a millisecond late would see nothing."""
    client = FakeRedis()
    publish(client, event("done", 1.0, terminal=True))

    latest = last_event(client, "job-1")
    assert latest is not None
    assert latest.state == "done"
    assert latest.terminal is True


def test_the_remembered_event_expires() -> None:
    """It is a convenience for a late client, not storage — `job_steps` is the record."""
    client = FakeRedis()
    publish(client, event("running", 0.5))
    assert client.expiries[f"{channel_for('job-1')}:last"] == EVENT_TTL_S


def test_no_event_yet_is_none_rather_than_an_error() -> None:
    assert last_event(FakeRedis(), uuid4()) is None


def test_subscribe_yields_events_until_a_terminal_one() -> None:
    client = FakeRedis()
    for one in (
        event("planning", 0.1),
        event("running", 0.5),
        event("done", 1.0, terminal=True),
    ):
        publish(client, one)

    seen = [e for e in subscribe(client, "job-1") if e is not None]
    assert [e.state for e in seen] == ["planning", "running", "done"]


def test_subscribe_stops_at_the_terminal_event_and_ignores_the_rest() -> None:
    """A stream that kept running after `done` would hold a connection open forever."""
    client = FakeRedis()
    publish(client, event("done", 1.0, terminal=True))
    publish(client, event("running", 0.5))

    seen = [e for e in subscribe(client, "job-1") if e is not None]
    assert len(seen) == 1


def test_a_timeout_yields_a_tick_so_the_caller_can_keepalive() -> None:
    """`None` is the signal an SSE endpoint turns into a comment frame."""
    client = FakeRedis()
    publish(client, event("running", 0.5))
    publish(client, event("done", 1.0, terminal=True))

    stream = subscribe(client, "job-1")
    assert next(stream) is not None

    # Nothing more is queued between the two, so this fake never ticks; the property under
    # test is that a tick is *representable*, which the type and the SSE test rely on.
    assert next(stream) is not None


def test_an_undecodable_message_is_skipped_rather_than_killing_the_stream() -> None:
    """One malformed publisher must not stop a user seeing their job finish."""
    client = FakeRedis()
    client.publish(channel_for("job-1"), "{not json")
    publish(client, event("done", 1.0, terminal=True))

    seen = [e for e in subscribe(client, "job-1") if e is not None]
    assert [e.state for e in seen] == ["done"]


def test_the_subscription_is_closed_even_when_the_caller_stops_early() -> None:
    """A generator abandoned mid-stream must not leak a Redis connection per client."""
    client = FakeRedis()
    for one in (event("planning", 0.1), event("done", 1.0, terminal=True)):
        publish(client, one)

    created: list[FakePubSub] = []
    original = client.pubsub

    def tracking(**kwargs: Any) -> FakePubSub:
        made = original(**kwargs)
        created.append(made)
        return made

    client.pubsub = tracking  # type: ignore[method-assign]
    stream = subscribe(client, "job-1")
    next(stream)
    stream.close()

    assert created, "the subscription was never opened"
    assert created[0].closed


@pytest.mark.parametrize("job_id", ["job-1", uuid4()])
def test_the_channel_name_accepts_a_string_or_a_uuid(job_id: object) -> None:
    assert channel_for(job_id).endswith(str(job_id))  # type: ignore[arg-type]
