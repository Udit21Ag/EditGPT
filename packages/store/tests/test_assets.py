"""Asset store behaviour, with the emphasis on what makes the naming scheme safe."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest
from editgpt_store import (
    AssetNotFoundError,
    AssetStore,
    LocalAssetStore,
    S3AssetStore,
    digest_of,
)


@pytest.fixture
def store(tmp_path: Path) -> LocalAssetStore:
    return LocalAssetStore(root=tmp_path / "assets")


def test_put_returns_the_digest_of_the_bytes(store: LocalAssetStore) -> None:
    data = b"\x89PNG pretend"
    assert store.put(data) == hashlib.sha256(data).hexdigest()


def test_the_same_bytes_stored_twice_occupy_one_file(store: LocalAssetStore) -> None:
    digest = store.put(b"same")
    assert store.put(b"same") == digest
    assert len(list(store.root.rglob("*"))) == 2  # the fan-out directory and one file


def test_round_trip(store: LocalAssetStore) -> None:
    digest = store.put(b"round trip")
    assert store.get(digest) == b"round trip"
    assert store.exists(digest)


def test_missing_asset_raises_rather_than_returning_empty(store: LocalAssetStore) -> None:
    absent = digest_of(b"never stored")
    assert not store.exists(absent)
    with pytest.raises(AssetNotFoundError):
        store.get(absent)


def test_delete_is_idempotent(store: LocalAssetStore) -> None:
    digest = store.put(b"delete me")
    store.delete(digest)
    store.delete(digest)
    assert not store.exists(digest)


@pytest.mark.parametrize(
    "key",
    [
        "../../etc/passwd",
        "a" * 63,
        "a" * 65,
        "../" + "a" * 61,
        "A" * 64,  # uppercase hex is not what we emit; refusing it keeps one canonical form
        "z" * 64,
    ],
)
def test_a_key_that_is_not_a_digest_is_refused(store: LocalAssetStore, key: str) -> None:
    """Traversal is impossible by construction, and this is the construction.

    Every path is built from a validated digest, so a caller who passes user input
    straight through gets an error rather than a file outside the store.
    """
    with pytest.raises(ValueError, match="not a sha-256 digest"):
        store.get(key)


def test_a_partial_write_is_never_visible_under_its_digest(store: LocalAssetStore) -> None:
    """The staging file is not addressable, so a crash cannot publish truncated bytes."""
    digest = store.put(b"complete")
    staging = store.root / digest[:2] / f".{digest}.partial"
    assert not staging.exists()
    assert store.get(digest) == b"complete"


class StubS3:
    """Enough of boto3's S3 client to exercise the adapter's arithmetic with no server.

    The adapter takes its client by injection precisely so this is possible; constructing
    one inside `put` would make the whole class untestable without credentials.

    This is deliberately *not* the only coverage: a stub cannot tell you which exception a
    real server raises, which is what `test_assets_s3_live.py` is for.
    """

    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, str]] = {}
        self.deleted: list[str] = []

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, ContentType: str) -> None:  # noqa: N803
        self.objects[Key] = (Body, ContentType)

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, object]:  # noqa: N803
        if Key not in self.objects:
            raise RuntimeError("NoSuchKey")
        return {"Body": io.BytesIO(self.objects[Key][0])}

    def head_object(self, *, Bucket: str, Key: str) -> None:  # noqa: N803
        if Key not in self.objects:
            raise RuntimeError("404")

    def delete_object(self, *, Bucket: str, Key: str) -> None:  # noqa: N803
        self.deleted.append(Key)
        self.objects.pop(Key, None)


@pytest.fixture
def remote() -> S3AssetStore:
    return S3AssetStore(bucket="editgpt", client=StubS3())


def test_s3_stores_under_the_digest_and_keeps_the_content_type(remote: S3AssetStore) -> None:
    """Unlike a filesystem, an object store can carry the MIME type, and must — a URL
    serves whatever the object claims to be."""
    digest = remote.put(b"remote bytes", content_type="image/webp")
    assert digest == digest_of(b"remote bytes")
    assert remote.client.objects[digest] == (b"remote bytes", "image/webp")


def test_s3_round_trips(remote: S3AssetStore) -> None:
    digest = remote.put(b"remote bytes")
    assert remote.get(digest) == b"remote bytes"
    assert remote.exists(digest)


def test_a_missing_s3_object_raises_the_same_error_as_a_missing_local_one(
    remote: S3AssetStore,
) -> None:
    """Callers switch adapters by configuration, so the failure must not switch with it."""
    with pytest.raises(AssetNotFoundError):
        remote.get(digest_of(b"never stored"))
    assert not remote.exists(digest_of(b"never stored"))


def test_s3_delete_removes_the_object(remote: S3AssetStore) -> None:
    digest = remote.put(b"delete me")
    remote.delete(digest)
    assert not remote.exists(digest)
    assert remote.client.deleted == [digest]


@pytest.mark.parametrize("method", ["get", "exists", "delete"])
def test_s3_refuses_a_key_that_is_not_a_digest(remote: S3AssetStore, method: str) -> None:
    """The same structural defence as the local store: no user string reaches a key."""
    with pytest.raises(ValueError, match="not a sha-256 digest"):
        getattr(remote, method)("../../secrets")


def test_both_adapters_satisfy_the_protocol(store: LocalAssetStore, remote: S3AssetStore) -> None:
    """The protocol exists because there are two; this is what keeps that true."""
    assert isinstance(store, AssetStore)
    assert isinstance(remote, AssetStore)


def test_from_settings_passes_the_endpoint_through_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The endpoint is configuration, not a vendor baked into the code.

    Worth a test because this is where a wrong assumption would reintroduce vendor
    lock-in, and because `region_name` must be set to *something*: several S3-compatible
    services are region-less, but boto3 signs every request with a region and refuses to
    build a client without one.
    """
    import boto3

    captured: dict[str, object] = {}

    def fake_client(service: str, **kwargs: object) -> object:
        captured.update({"service": service, **kwargs})
        return object()

    monkeypatch.setattr(boto3, "client", fake_client)
    made = S3AssetStore.from_settings(
        endpoint_url="http://localhost:9000",
        access_key_id="key",
        secret_access_key="secret",
        bucket="editgpt",
    )

    assert made.bucket == "editgpt"
    assert captured["service"] == "s3"
    assert captured["endpoint_url"] == "http://localhost:9000"
    assert captured["region_name"] == "auto"
