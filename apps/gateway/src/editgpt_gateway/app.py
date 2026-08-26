"""The gateway application.

Phase 1 ships the skeleton and its health contract only: no AI, no storage. The point
is that the shape is exercised by CI from the first commit, so later phases add
behaviour to something already known to run.
"""

from __future__ import annotations

from typing import Any

from editgpt_core import EditOp
from fastapi import FastAPI
from pydantic import BaseModel

from editgpt_gateway.settings import Settings, get_settings

API_VERSION = "0.1.0"


class Health(BaseModel):
    status: str
    version: str
    environment: str


class Capabilities(BaseModel):
    """What this deployment can actually do.

    Advertised rather than assumed: Phase 0 established that only some of the planned
    operations have a working model behind them, and the frontend should not offer the
    others.
    """

    operations: list[EditOp]
    unsupported: dict[str, str]
    max_megapixels: float


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings or get_settings()
    app = FastAPI(title="EditGPT gateway", version=API_VERSION)

    @app.get("/health", response_model=Health)
    def health() -> Health:
        return Health(status="ok", version=API_VERSION, environment=config.environment)

    @app.get("/ready", response_model=Health)
    def ready() -> Health:
        """Liveness and readiness are separate so a restart is not mistaken for a rollout."""
        return Health(status="ready", version=API_VERSION, environment=config.environment)

    @app.get("/capabilities", response_model=Capabilities)
    def capabilities() -> Capabilities:
        return Capabilities(
            operations=[
                EditOp.REMOVE,
                EditOp.ADD,
                EditOp.REPLACE,
                EditOp.BACKGROUND,
                EditOp.UPSCALE,
            ],
            unsupported={
                EditOp.RESTYLE: "no free instruction-editing model exists",
                EditOp.RETOUCH: "not scoped for v1",
            },
            max_megapixels=config.max_megapixels,
        )

    @app.get("/")
    def root() -> dict[str, Any]:
        return {"service": "editgpt-gateway", "version": API_VERSION, "docs": "/docs"}

    return app


app = create_app()
