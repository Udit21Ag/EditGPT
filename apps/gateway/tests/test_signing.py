"""Short-lived image links, and everything they must not grant.

A signature is a credential. The positive case is one line; the value of this file is the
negatives — what an attacker holding a link, or a stale one, or one for a different
picture, still cannot do.
"""

from __future__ import annotations

import time

from editgpt_gateway.signing import link, sign, verify

KEY = "a-long-random-testing-key"
DIGEST = "a" * 64
OTHER = "b" * 64


def future(seconds: int = 900) -> int:
    return int(time.time()) + seconds


def test_a_signature_verifies_for_the_digest_it_was_made_for() -> None:
    expires = future()
    assert verify(DIGEST, expires, sign(DIGEST, expires, KEY), KEY)


def test_a_signature_does_not_carry_to_another_image() -> None:
    """The property that makes this safe to hand out: one link, one picture."""
    expires = future()
    assert not verify(OTHER, expires, sign(DIGEST, expires, KEY), KEY)


def test_an_expired_signature_is_refused() -> None:
    """Expiry is checked separately from the comparison, because an expired signature is
    still *arithmetically* valid — a link that leaked a month ago would otherwise work."""
    expired = int(time.time()) - 1
    assert not verify(DIGEST, expired, sign(DIGEST, expired, KEY), KEY)


def test_the_expiry_cannot_be_pushed_out_by_the_holder() -> None:
    """It is covered by the signature, so editing it in the URL invalidates the link."""
    expires = future()
    signature = sign(DIGEST, expires, KEY)
    assert not verify(DIGEST, expires + 3600, signature, KEY)


def test_a_signature_from_a_different_key_is_refused() -> None:
    expires = future()
    assert not verify(DIGEST, expires, sign(DIGEST, expires, "another-key"), KEY)


def test_a_tampered_signature_is_refused() -> None:
    expires = future()
    signature = sign(DIGEST, expires, KEY)
    flipped = ("x" if signature[0] != "x" else "y") + signature[1:]
    assert not verify(DIGEST, expires, flipped, KEY)


def test_nonsense_in_the_signature_is_refused_rather_than_raising() -> None:
    # It arrives from a query string, so it is whatever somebody typed.
    for bad in ("", "   ", "!!!!", "a" * 500):
        assert not verify(DIGEST, future(), bad, KEY)


def test_a_link_carries_the_digest_and_its_own_expiry() -> None:
    url, expires_at = link(DIGEST, key=KEY, ttl_seconds=900)
    assert DIGEST in url
    assert f"expires={expires_at}" in url
    assert expires_at > int(time.time())


def test_two_links_for_the_same_image_are_not_the_same_string() -> None:
    """They expire at different moments, so they must sign differently — a fixed link
    would be a permanent credential wearing a timestamp."""
    first, _ = link(DIGEST, key=KEY, ttl_seconds=900)
    second, _ = link(DIGEST, key=KEY, ttl_seconds=1800)
    assert first != second
