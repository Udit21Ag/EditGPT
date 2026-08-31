"""Short-lived links to an asset, so a browser can just fetch it.

Every image reached the page through JavaScript: `GET /v1/images/{digest}` needs an
`Authorization` header, a plain `<img src>` cannot send one, so the client fetched each
picture and wrapped it in an object URL. That cost a copy of every image in the tab's
memory until something remembered to revoke it, put the whole download in front of the
first paint, and made the browser's own cache useless.

A signature in the query string is what an `<img>` can carry. It grants **one digest,
until one expiry**, and nothing else — not a session, not another image, not a write.

The key is HMAC-SHA256 over `digest:expires`. Not a token store: the signature is
self-describing, so verifying one costs no round trip and there is nothing to keep in sync
between replicas beyond the key itself.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from base64 import urlsafe_b64encode


def sign(digest: str, expires_at: int, key: str) -> str:
    """A signature for `digest`, valid until `expires_at` (a Unix timestamp)."""
    mac = hmac.new(key.encode(), f"{digest}:{expires_at}".encode(), hashlib.sha256)
    return urlsafe_b64encode(mac.digest()).decode().rstrip("=")


def link(digest: str, *, key: str, ttl_seconds: int, base: str = "") -> tuple[str, int]:
    """A URL for `digest` and the moment it stops working."""
    expires_at = int(time.time()) + ttl_seconds
    signature = sign(digest, expires_at, key)
    return f"{base}/v1/images/{digest}?expires={expires_at}&signature={signature}", expires_at


def verify(digest: str, expires_at: int, signature: str, key: str) -> bool:
    """Whether this signature really covers this digest and has not expired.

    Expiry is checked first and separately: an expired signature is *valid* arithmetic, so
    a comparison alone would accept a link that leaked a month ago.

    `compare_digest` rather than `==`, because the obvious comparison returns as soon as
    two bytes differ and that timing is enough to reconstruct a signature one byte at a
    time.
    """
    if expires_at < int(time.time()):
        return False
    return hmac.compare_digest(sign(digest, expires_at, key), signature)
