"""Worker configuration, read once from the environment.

Deliberately the same `EDITGPT_` prefix and the same field names as the gateway's
settings: the two processes must agree about which Redis and which Postgres they are
talking to, and the cheapest way to guarantee that is for both to read the same
variables.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EDITGPT_", env_file=".env", extra="ignore")

    environment: str = "development"
    redis_url: str = "redis://localhost:6379/0"
    database_url: str = "postgresql+psycopg://editgpt:editgpt@localhost:5432/editgpt"

    asset_root: Path = Path.home() / ".cache" / "editgpt" / "assets"
    """Where `LocalAssetStore` writes. Ignored when object storage is configured."""

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

    asset_grace_hours: float = 24.0
    """How long an unreferenced object is kept before a sweep may delete it.

    An upload that has been stored but whose row is still being committed looks exactly
    like an orphan for as long as that takes. A day is far longer than that window."""

    asset_retention_days: int = 0
    """Age at which a *referenced* object's bytes are deleted. Zero means never.

    Off unless a deployment asks for it: the alternative is a number this project chose
    deleting somebody's photographs on a schedule they never agreed to. The rows survive
    either way, so history stays readable and the fetch answers 404."""

    task_soft_time_limit_s: int = 240
    task_time_limit_s: int = 300
    """Two limits because they do different things: the soft one raises inside the task
    so it can record why it failed, the hard one kills the process. A worker that is only
    hard-killed leaves jobs stuck in `running` forever."""

    max_tasks_per_child: int = 10
    """Recycle the process regularly. ONNX Runtime's arena does not return memory to the
    OS, so on an 8 GB machine a long-lived worker's RSS only ever goes up."""

    @property
    def uses_object_storage(self) -> bool:
        return bool(self.s3_endpoint_url and self.s3_bucket and self.s3_access_key_id)


def get_settings() -> Settings:
    return Settings()
