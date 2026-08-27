"""Content-addressed blob storage.

Every image and artifact is named by the SHA-256 of its bytes. That is not a stylistic
choice: `AssetRef` in the contract already carries a digest, so keys are derived rather
than supplied, and **path traversal is structurally impossible instead of filtered**. It
also makes re-uploading the same photograph free and makes an artifact's identity
independent of which worker produced it.

Two adapters. `LocalAssetStore` writes to a directory and is the default, so a developer
and CI need no credentials. `S3AssetStore` talks to **any S3-compatible endpoint** and is
what a deployment uses; it lives behind the optional `s3` extra so nothing imports boto3
unless it is configured.

The remote adapter is deliberately *not* named after a vendor. The S3 API is the
interface, and the endpoint is configuration — which is what lets the same code run
against MinIO in a container with no account at all, and against a hosted provider later,
without a line changing. See `docs/RUNBOOK.md` for which providers this has been run
against and what each one costs.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

log = logging.getLogger(__name__)

FANOUT = 2
"""Characters of the digest used as a subdirectory.

A flat directory of a hundred thousand files is slow to list on every filesystem worth
supporting. Two hex characters gives 256 buckets, which is enough for this project's
scale without a second level nobody would ever need.
"""


class AssetNotFoundError(KeyError):
    """A digest that is not in the store. Distinct from a malformed digest."""


def digest_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@runtime_checkable
class AssetStore(Protocol):
    """Somewhere bytes can be put and got back by their digest."""

    def put(self, data: bytes, *, content_type: str = "image/png") -> str:
        """Store `data` and return its digest. Storing the same bytes twice is a no-op."""
        ...

    def get(self, digest: str) -> bytes: ...

    def exists(self, digest: str) -> bool: ...

    def delete(self, digest: str) -> None: ...


@dataclass(frozen=True, slots=True)
class LocalAssetStore:
    """Assets on the local filesystem, under `root/ab/abcdef...`.

    Writes go to a temporary name and are then renamed, so a crash mid-write cannot leave
    a truncated file under a digest that claims to describe complete bytes. On the same
    filesystem that rename is atomic.
    """

    root: Path

    def _path(self, digest: str) -> Path:
        _validate(digest)
        return self.root / digest[:FANOUT] / digest

    def put(self, data: bytes, *, content_type: str = "image/png") -> str:
        digest = digest_of(data)
        target = self._path(digest)
        if target.exists():
            return digest
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_name(f".{digest}.partial")
        staging.write_bytes(data)
        staging.replace(target)
        # A filesystem has nowhere to put a MIME type; the `images` row carries it. It is
        # logged rather than dropped so a mismatch between the stored bytes and what the
        # database claims they are is at least reconstructable from the log.
        log.debug(
            "asset.put",
            extra={"digest": digest, "bytes": len(data), "content_type": content_type},
        )
        return digest

    def get(self, digest: str) -> bytes:
        target = self._path(digest)
        if not target.exists():
            raise AssetNotFoundError(f"no asset {digest} under {self.root}")
        return target.read_bytes()

    def exists(self, digest: str) -> bool:
        return self._path(digest).exists()

    def delete(self, digest: str) -> None:
        self._path(digest).unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class S3AssetStore:
    """Any S3-compatible object store. Needs `editgpt-store[s3]` and an endpoint.

    Verified against MinIO, which runs in a container and needs no account. The same
    adapter is what a deployment points at a hosted provider — the endpoint is the only
    thing that differs.
    """

    bucket: str
    client: Any
    """A boto3 S3 client. Injected rather than constructed so a test can pass a stub and
    so credential handling stays in one place — see `from_settings`."""

    @classmethod
    def from_settings(
        cls,
        *,
        endpoint_url: str,
        access_key_id: str,
        secret_access_key: str,
        bucket: str,
        region: str = "auto",
    ) -> S3AssetStore:
        import boto3  # imported here: the s3 extra is optional

        return cls(
            bucket=bucket,
            client=boto3.client(
                "s3",
                endpoint_url=endpoint_url,
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
                # Several S3-compatible services are region-less, but boto3 signs every
                # request with a region and refuses to build a client without one.
                region_name=region,
            ),
        )

    def ensure_bucket(self) -> None:
        """Create the bucket if it is missing. Idempotent.

        A hosted provider hands you a bucket through its console; MinIO starts empty. This
        exists so the local path needs no manual step, and it is safe against a provider
        that already has one.
        """
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except Exception:
            try:
                self.client.create_bucket(Bucket=self.bucket)
            except Exception as error:  # already created by a racing process, or forbidden
                log.warning("asset.bucket_unavailable", extra={"error": str(error)})

    def put(self, data: bytes, *, content_type: str = "image/png") -> str:
        digest = digest_of(data)
        self.client.put_object(Bucket=self.bucket, Key=digest, Body=data, ContentType=content_type)
        return digest

    def get(self, digest: str) -> bytes:
        _validate(digest)
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=digest)
        except Exception as error:  # botocore raises a dynamically built ClientError
            raise AssetNotFoundError(f"no asset {digest} in {self.bucket}") from error
        return bytes(response["Body"].read())

    def exists(self, digest: str) -> bool:
        _validate(digest)
        try:
            self.client.head_object(Bucket=self.bucket, Key=digest)
        except Exception:
            return False
        return True

    def delete(self, digest: str) -> None:
        _validate(digest)
        self.client.delete_object(Bucket=self.bucket, Key=digest)


def _validate(digest: str) -> None:
    """Reject anything that is not a digest before it reaches a path or a key.

    The gateway derives digests from uploaded bytes, so a bad one here means either a
    caller passed a user-supplied string straight through or a stored reference is
    corrupt. Both are worth failing loudly for.
    """
    if len(digest) != 64 or not all(c in "0123456789abcdef" for c in digest):
        raise ValueError(f"not a sha-256 digest: {digest!r}")
