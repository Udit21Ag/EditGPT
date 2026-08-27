"""The gateway's contract, exercised from the first commit.

The `client` fixture comes from `conftest.py` and is wired to local doubles. This module
deliberately does not build its own: one that called `create_app` with real settings
would try to reach Postgres and Redis, and a test suite that needs a container is a test
suite that stops being run.
"""

from __future__ import annotations

import pytest
from editgpt_core import EditOp
from editgpt_gateway.app import create_app
from editgpt_gateway.deps import Services
from editgpt_gateway.settings import Settings
from fastapi.testclient import TestClient


def test_health_reports_environment(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["environment"] == "test"


def test_readiness_is_separate_from_liveness(client: TestClient) -> None:
    """Requirement changed in Phase 3: `/ready` now reports *degraded* modes.

    It used to answer a constant "ready", which is worse than nothing — a gateway with no
    queue answers every request and finishes no job. The contract is now that it names
    each fallback it is running in.
    """
    body = client.get("/ready").json()
    assert body["status"] in {"ready", "degraded"}
    assert body["storage"] == "local"


def test_readiness_names_each_missing_collaborator(memory_services: Services) -> None:
    with TestClient(create_app(memory_services.settings, memory_services)) as degraded:
        body = degraded.get("/ready").json()
    assert body["status"] == "degraded"
    assert body["jobs"] == "memory"
    joined = " ".join(body["degraded"])
    assert "restart" in joined, "losing jobs on restart must be said out loud"
    assert "never executed" in joined, "a gateway with no worker must say so"


def test_capabilities_advertises_only_supported_operations(client: TestClient) -> None:
    """The frontend must not offer an operation with no model behind it."""
    body = client.get("/capabilities").json()
    assert EditOp.REMOVE in body["operations"]
    assert EditOp.UPSCALE in body["operations"], "upscaling shipped; advertise it"
    assert EditOp.RESTYLE not in body["operations"]
    assert "no free instruction-editing model" in body["unsupported"][EditOp.RESTYLE]


def test_no_operation_is_both_supported_and_unsupported(client: TestClient) -> None:
    body = client.get("/capabilities").json()
    assert not set(body["operations"]) & set(body["unsupported"])


def test_capabilities_publishes_the_upload_ceiling(client: TestClient, settings: Settings) -> None:
    """An oversized upload would breach the worker's memory budget, so clients are told.

    Compared against the configured value rather than a literal: a hardcoded 40 here would
    pass while the service enforced something else entirely.
    """
    body = client.get("/capabilities").json()
    assert body["max_megapixels"] == pytest.approx(settings.max_megapixels)
    assert body["max_upload_mb"] == settings.max_upload_mb


def test_openapi_schema_is_generated(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert schema["info"]["title"] == "EditGPT gateway"
    assert "/capabilities" in schema["paths"]
