"""Per-client rate limiting on the endpoints that cost something.

A fixed window in Redis: one `INCR` and one `EXPIRE` on first use. Chosen over a token
bucket because the thing being defended is a free-tier quota against one runaway script,
not fairness at millisecond resolution — and a bucket needs a Lua script to be atomic,
which is a lot of machinery for a boundary that is allowed to be approximate.

**It fails open.** If Redis is unreachable the request is allowed. That is the right
trade here: this limiter protects a quota, not a security boundary, and refusing every
request because a cache is down converts a degraded system into an offline one. When
authentication lands, per-user quota enforcement belongs in the ledger, which is durable,
not here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

WINDOW_S = 60


@dataclass(frozen=True, slots=True)
class Decision:
    allowed: bool
    remaining: int
    retry_after_s: int


def check(client: Any | None, identity: str, *, limit: int) -> Decision:
    """Count this request against `identity`'s window and say whether it may proceed."""
    if client is None or limit <= 0:
        return Decision(allowed=True, remaining=limit, retry_after_s=0)

    key = f"editgpt:rate:{identity}:{WINDOW_S}"
    try:
        used = int(client.incr(key))
        if used == 1:
            client.expire(key, WINDOW_S)
        ttl = int(client.ttl(key))
    except Exception as error:
        log.warning("ratelimit.unavailable", extra={"error": str(error)})
        return Decision(allowed=True, remaining=limit, retry_after_s=0)

    if used > limit:
        return Decision(allowed=False, remaining=0, retry_after_s=max(ttl, 1))
    return Decision(allowed=True, remaining=limit - used, retry_after_s=0)


def identify(client_host: str | None, forwarded_for: str | None) -> str:
    """Who to count against.

    `X-Forwarded-For` is trusted only for its first entry, and only because every
    intended deployment sits behind a proxy that sets it. It is client-controlled when
    the service is exposed directly, so this is a quota key and must never become an
    authorization or audit identity.
    """
    if forwarded_for:
        first = forwarded_for.split(",")[0].strip()
        if first:
            return first
    return client_host or "unknown"
