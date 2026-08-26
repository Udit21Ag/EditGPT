"""The gateway's contract, exercised from the first commit."""

from __future__ import annotations

import pytest
from editgpt_core import EditOp
from editgpt_gateway.app import create_app
from editgpt_gateway.settings import Settings
from fastapi.testclient import TestClient


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(Settings(environment="test")))


def test_health_reports_environment(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["environment"] == "test"


def test_readiness_is_separate_from_liveness(client: TestClient) -> None:
    assert client.get("/ready").json()["status"] == "ready"


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


def test_capabilities_publishes_the_upload_ceiling(client: TestClient) -> None:
    """A 40 MP upload would breach the worker's memory budget, so clients are told."""
    assert client.get("/capabilities").json()["max_megapixels"] == pytest.approx(40.0)


def test_openapi_schema_is_generated(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert schema["info"]["title"] == "EditGPT gateway"
    assert "/capabilities" in schema["paths"]
