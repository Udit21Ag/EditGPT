"""`S3AssetStore` against a real S3 server.

The rest of `test_assets.py` drives this adapter through a hand-written stub, which
proves the arithmetic around the calls and nothing about the calls themselves. A stub
cannot tell you that `head_object` raises the exception you catch, that a `ContentType`
survives a round trip, or that boto3 will even sign the request.

MinIO makes that verifiable with no account, no signup and no payment details — which is
the whole reason it is in `docker-compose.yml`. Every hosted S3-compatible provider
speaks this same API, so what passes here is what will run there.

Skips cleanly when MinIO is not up, so `make check` on a fresh checkout stays green:
`make compose-s3` to start it.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest
from editgpt_store import AssetNotFoundError, AssetStore, S3AssetStore, digest_of

pytestmark = [pytest.mark.service, pytest.mark.enable_socket]

ENDPOINT = "http://localhost:9000"
ACCESS_KEY = "editgpt"
SECRET_KEY = "editgpt-dev-secret"


@pytest.fixture
def bucket() -> Iterator[S3AssetStore]:
    """A throwaway bucket, emptied and dropped afterwards.

    Its own bucket rather than a shared one: these tests delete objects, and a shared
    bucket makes two parallel runs delete each other's.
    """
    boto3 = pytest.importorskip("boto3", reason="needs `editgpt-store[s3]`")
    from botocore.exceptions import BotoCoreError, ClientError

    name = f"editgpt-test-{uuid.uuid4().hex[:12]}"
    store = S3AssetStore.from_settings(
        endpoint_url=ENDPOINT,
        access_key_id=ACCESS_KEY,
        secret_access_key=SECRET_KEY,
        bucket=name,
        region="us-east-1",
    )
    try:
        store.client.list_buckets()
    except (ClientError, BotoCoreError, OSError) as error:
        pytest.skip(f"no S3 endpoint at {ENDPOINT} ({type(error).__name__}); run `make compose-s3`")

    store.ensure_bucket()
    try:
        yield store
    finally:
        listing = store.client.list_objects_v2(Bucket=name).get("Contents", [])
        for item in listing:
            store.client.delete_object(Bucket=name, Key=item["Key"])
        store.client.delete_bucket(Bucket=name)
        del boto3


def test_a_round_trip_survives_a_real_server(bucket: S3AssetStore) -> None:
    data = b"\x89PNG not really, but bytes are bytes"
    digest = bucket.put(data, content_type="image/png")

    assert digest == digest_of(data)
    assert bucket.get(digest) == data
    assert bucket.exists(digest)


def test_the_content_type_is_stored_and_returned_by_the_server(bucket: S3AssetStore) -> None:
    """A stub could only prove we *passed* it. This proves the server kept it.

    It matters because a URL serves whatever the object claims to be: a PNG stored as
    `application/octet-stream` downloads instead of rendering.
    """
    digest = bucket.put(b"webp bytes", content_type="image/webp")
    head = bucket.client.head_object(Bucket=bucket.bucket, Key=digest)
    assert head["ContentType"] == "image/webp"


def test_a_missing_object_raises_the_same_error_as_the_local_store(
    bucket: S3AssetStore,
) -> None:
    """The exception boto3 actually raises is a dynamically built `ClientError`.

    Which is exactly the kind of thing a stub gets wrong: this asserts our `except` really
    catches what a real server produces, so callers can switch adapters by configuration
    without switching their error handling.
    """
    absent = digest_of(b"never stored")
    assert not bucket.exists(absent)
    with pytest.raises(AssetNotFoundError):
        bucket.get(absent)


def test_storing_the_same_bytes_twice_is_idempotent(bucket: S3AssetStore) -> None:
    data = b"the same photograph, uploaded twice"
    first, second = bucket.put(data), bucket.put(data)

    assert first == second
    listing = bucket.client.list_objects_v2(Bucket=bucket.bucket).get("Contents", [])
    assert len(listing) == 1, "content addressing means one key, not two"


def test_delete_removes_it_and_is_idempotent(bucket: S3AssetStore) -> None:
    digest = bucket.put(b"delete me")
    bucket.delete(digest)
    bucket.delete(digest)
    assert not bucket.exists(digest)


def test_ensure_bucket_is_safe_to_call_on_an_existing_bucket(bucket: S3AssetStore) -> None:
    """It runs on every worker and gateway start, so it must not fail the second time."""
    bucket.ensure_bucket()
    bucket.ensure_bucket()
    assert bucket.put(b"still working") == digest_of(b"still working")


def test_a_key_that_is_not_a_digest_never_reaches_the_server(bucket: S3AssetStore) -> None:
    """Traversal is structurally impossible, and that holds for the remote adapter too."""
    with pytest.raises(ValueError, match="not a sha-256 digest"):
        bucket.get("../../etc/passwd")


def test_it_satisfies_the_protocol_the_gateway_depends_on(bucket: S3AssetStore) -> None:
    assert isinstance(bucket, AssetStore)


def test_a_megabyte_survives_intact(bucket: S3AssetStore) -> None:
    """Photographs are not twenty bytes, and chunked transfer is where encodings break."""
    data = bytes(range(256)) * 4096  # 1 MiB, non-repeating enough to catch truncation
    digest = bucket.put(data, content_type="image/jpeg")
    assert bucket.get(digest) == data
