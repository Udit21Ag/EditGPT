"""Configuration, read once from the environment."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EDITGPT_", env_file=".env", extra="ignore")

    environment: str = "development"
    redis_url: str = "redis://localhost:6379/0"
    database_url: str = "postgresql+psycopg://editgpt:editgpt@localhost:5432/editgpt"
    max_upload_mb: int = 25
    max_megapixels: float = 40.0
    """Above this an upload is rejected: a 40 MP image would breach the worker's budget."""


def get_settings() -> Settings:
    return Settings()
