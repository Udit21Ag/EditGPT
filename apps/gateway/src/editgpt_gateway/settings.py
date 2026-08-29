"""Configuration, read once from the environment.

The storage and queue fields mirror `editgpt_worker.settings` exactly — same prefix, same
names. Two processes that must agree about which Redis and which Postgres they are using
should read the same variables, not two sets that happen to be filled in consistently.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

LOCAL_ENVIRONMENTS = frozenset({"development", "test", "ci"})
"""Environments where a development default is the right answer, not a warning."""


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

    cors_origins: tuple[str, ...] = ("http://localhost:3000", "http://localhost:3210")
    """Browser origins allowed to call this API, as a JSON list.

    Required for the frontend to work at all, and it did not exist: a preflight `OPTIONS`
    answered 405, so every cross-origin request a browser made was blocked before it was
    sent. Nothing caught it because nothing had ever called this from a browser — curl
    does not preflight, and the component tests replace `fetch`. The full-stack browser
    suite found it on its first real run.

    An explicit list rather than `*`. The wildcard would let any page a user visits call
    this API with their session, and CORS is the only thing standing between those two
    facts. The defaults are the two local development ports — `next dev` and the one
    Playwright starts — because a gateway that needs configuring before it works with the
    app in the same repository is a gateway that appears broken. **A deployment must set
    this to its real origin**, and leaving the defaults in place there allows nothing
    useful rather than allowing everything.
    """

    @property
    def is_deployed(self) -> bool:
        """Whether this is running somewhere other than a developer's machine or CI.

        Named environments rather than "not development", because `test` and `ci` are both
        local in every sense that matters here and neither should be nagged about a setting
        that is correct for them.
        """
        return self.environment not in LOCAL_ENVIRONMENTS

    @property
    def cors_allows_localhost(self) -> bool:
        """Whether any allowed browser origin is a development one.

        Reported by `/ready` outside development. The defaults exist so the app in this
        repository works without configuration, and the cost of that convenience is a
        deployment that keeps them without meaning to.
        """
        return any(
            origin.startswith(("http://localhost", "http://127.0.0.1"))
            for origin in self.cors_origins
        )

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
