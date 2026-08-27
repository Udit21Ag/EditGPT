"""Configuration, read once from the environment.

The storage and queue fields mirror `editgpt_worker.settings` exactly — same prefix, same
names. Two processes that must agree about which Redis and which Postgres they are using
should read the same variables, not two sets that happen to be filled in consistently.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EDITGPT_", env_file=".env", extra="ignore")

    environment: str = "development"
    redis_url: str = "redis://localhost:6379/0"
    database_url: str = "postgresql+psycopg://editgpt:editgpt@localhost:5432/editgpt"
    max_upload_mb: int = 25
    max_megapixels: float = 40.0
    """Above this an upload is rejected: a 40 MP image would breach the worker's budget."""

    asset_root: Path = Path.home() / ".cache" / "editgpt" / "assets"

    s3_endpoint_url: str = ""
    s3_bucket: str = ""
    s3_access_key_id: str = ""
    s3_secret_access_key: str = ""
    s3_region: str = "auto"
    """Any S3-compatible endpoint. Empty means "keep assets on local disk".

    Not named after a vendor on purpose: the S3 API is the interface and the endpoint is
    configuration, so MinIO in a container and a hosted provider are the same code path.
    Storage switches over only when the endpoint, the bucket *and* an access key are all
    present, so a half-filled `.env` fails at startup rather than writing half a
    deployment's artifacts to a laptop's disk.
    """

    clerk_secret_key: str = Field(
        default="",
        validation_alias=AliasChoices("EDITGPT_CLERK_SECRET_KEY", "CLERK_SECRET_KEY"),
    )
    """Clerk's secret key. Its presence is what turns authentication on.

    Reads the **unprefixed** `CLERK_SECRET_KEY` as well as the prefixed one, and that is
    the point: Clerk's own Next.js SDK requires that exact name, so without the alias the
    same secret would live in two variables and someone would eventually rotate one of
    them. One name, both processes.

    Empty means every request is the anonymous sentinel — which is how a fresh checkout
    and the test suite run, so neither needs a credential. `GET /ready` reports which mode
    is live.
    """

    clerk_jwt_key: str = Field(
        default="",
        validation_alias=AliasChoices("EDITGPT_CLERK_JWT_KEY", "CLERK_JWT_KEY"),
    )
    """The instance's PEM public key, for networkless session verification.

    Optional but strongly preferred. Without it the SDK fetches Clerk's JWKS on a cold
    start and caches it for five minutes, which puts a third party on the path of the
    first request after every deploy. With it, verification is local arithmetic.
    Clerk dashboard → API Keys → Show JWT public key → PEM.
    """

    clerk_authorized_parties: tuple[str, ...] = ()
    """Origins whose tokens this service accepts, e.g. `https://editgpt.example`.

    Empty accepts any party, which is fine locally and is not what you want in
    production: it is the check that stops a token minted for a different application on
    the same Clerk instance from working here.
    """

    rate_limit_per_minute: int = 30
    """Requests one client may make per minute to the mutating endpoints.

    A fixed window rather than a token bucket: the failure this guards against is one
    script hammering a free-tier account, and a window is one Redis `INCR`. A bucket
    would be smoother at the boundary and is not worth a Lua script here.
    """

    rate_limit_burst_paths: tuple[str, ...] = ("/v1/images", "/v1/jobs")
    """Only the endpoints that cost something. Reading a job's state is free and a client
    polling it is a client we would rather have on the SSE stream anyway."""

    @property
    def uses_clerk(self) -> bool:
        return bool(self.clerk_secret_key)

    @property
    def uses_object_storage(self) -> bool:
        return bool(self.s3_endpoint_url and self.s3_bucket and self.s3_access_key_id)

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


def get_settings() -> Settings:
    return Settings()
