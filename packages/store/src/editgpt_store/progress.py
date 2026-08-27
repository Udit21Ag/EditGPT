"""The progress channel between the worker and whoever is watching a job.

Redis pub/sub, one channel per job. Pub/sub and not a list, because a progress event has
no value once the job is finished — it is not work to be consumed exactly once, it is a
notification that may be missed.

Missing one is fine and is designed for: **`job_steps` in Postgres is the record, this
channel is only the low-latency path**. A client that connects late gets the persisted
steps replayed first and then joins the live stream, so the two together are complete
even though neither is on its own. That matters more than it sounds — a phone
backgrounding the browser for ten seconds is the common case, not the edge case.

The channel lives here rather than in the worker because both processes speak it, and a
publisher and a subscriber that disagree about the payload shape fail silently.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Generator
from dataclasses import asdict, dataclass
from typing import Any
from uuid import UUID

log = logging.getLogger(__name__)

CHANNEL_PREFIX = "editgpt:job:"
EVENT_TTL_S = 3600
"""How long a terminal event is remembered for a client that connects after the fact.

Pub/sub itself keeps nothing, so the last event is *also* written to a key. Without it a
client that subscribes one millisecond after a job finishes waits forever for a message
that was already sent.
"""


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """One thing that happened to a job, as sent to a browser."""

    job_id: str
    state: str
    progress: float
    detail: str = ""
    terminal: bool = False

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, payload: str | bytes) -> ProgressEvent:
        data = json.loads(payload)
        return cls(
            job_id=str(data["job_id"]),
            state=str(data["state"]),
            progress=float(data.get("progress", 0.0)),
            detail=str(data.get("detail", "")),
            terminal=bool(data.get("terminal", False)),
        )


def channel_for(job_id: UUID | str) -> str:
    return f"{CHANNEL_PREFIX}{job_id}"


def last_key_for(job_id: UUID | str) -> str:
    return f"{CHANNEL_PREFIX}{job_id}:last"


def publish(client: Any, event: ProgressEvent) -> None:
    """Announce an event and remember it as the job's latest.

    Both writes matter: the publish reaches anyone already listening, the key catches
    anyone who arrives afterwards.
    """
    payload = event.to_json()
    client.publish(channel_for(event.job_id), payload)
    client.set(last_key_for(event.job_id), payload, ex=EVENT_TTL_S)


def last_event(client: Any, job_id: UUID | str) -> ProgressEvent | None:
    payload = client.get(last_key_for(job_id))
    return ProgressEvent.from_json(payload) if payload else None


def subscribe(
    client: Any, job_id: UUID | str, *, timeout_s: float = 1.0
) -> Generator[ProgressEvent | None, None, None]:
    """Yield events for a job until a terminal one arrives.

    Typed as a `Generator` rather than an `Iterator` because the `finally` below is the
    contract: a caller that abandons the stream — a browser closing an SSE connection —
    must be able to `close()` it and have the subscription released. An `Iterator`
    promises no such thing, and one connection leaked per disconnected client is the kind
    of bug that only shows up under load.

    Yields `None` when `timeout_s` passes with nothing published. That tick is what lets
    the caller send an SSE keepalive and notice a disconnected client, without this
    module needing to know what SSE is. A caller that does not care can filter it out.
    """
    pubsub = client.pubsub(ignore_subscribe_messages=True)
    pubsub.subscribe(channel_for(job_id))
    try:
        while True:
            message = pubsub.get_message(timeout=timeout_s)
            if message is None or message.get("type") != "message":
                yield None
                continue
            try:
                event = ProgressEvent.from_json(message["data"])
            except (ValueError, KeyError, TypeError):
                log.warning("progress.undecodable", extra={"job_id": str(job_id)})
                continue
            yield event
            if event.terminal:
                return
    finally:
        pubsub.close()
