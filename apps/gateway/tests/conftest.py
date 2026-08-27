"""A gateway wired entirely to local doubles.

No Postgres, no Redis, no Celery, no network — the app is built with `Services`
constructed by hand rather than by `build_services`, which is the whole reason that
function takes its collaborators as data. The doubles are the real in-memory
implementations, not mocks, so these tests exercise the actual code paths.
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path

import pytest
from editgpt_gateway.app import create_app
from editgpt_gateway.deps import RecordingQueue, Services
from editgpt_gateway.settings import Settings
from editgpt_store import InMemoryJobStore, LocalAssetStore, bootstrap, make_engine
from editgpt_store import make_session_factory as _make_session_factory
from editgpt_store.jobs import SqlJobStore
from fastapi.testclient import TestClient


def png_bytes(
    width: int = 64, height: int = 48, colour: tuple[int, int, int] = (10, 120, 200)
) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        environment="test",
        asset_root=tmp_path / "assets",
        max_upload_mb=1,
        max_megapixels=2.0,
        rate_limit_per_minute=3,
    )


@pytest.fixture
def services(settings: Settings) -> Services:
    settings.asset_root.mkdir(parents=True, exist_ok=True)
    engine = make_engine("sqlite+pysqlite:///:memory:")
    bootstrap(engine)
    return Services(
        settings=settings,
        jobs=SqlJobStore(session_factory=_make_session_factory(engine)),
        assets=LocalAssetStore(root=settings.asset_root),
        queue=RecordingQueue([]),
        redis=None,
        storage_kind="local",
        job_store_kind="postgres",
    )


@pytest.fixture
def memory_services(settings: Settings) -> Services:
    """The degraded wiring, so `/ready` has something honest to report."""
    settings.asset_root.mkdir(parents=True, exist_ok=True)
    return Services(
        settings=settings,
        jobs=InMemoryJobStore(),
        assets=LocalAssetStore(root=settings.asset_root),
        queue=RecordingQueue([]),
        redis=None,
        storage_kind="local",
        job_store_kind="memory",
    )


@pytest.fixture
def client(settings: Settings, services: Services) -> Iterator[TestClient]:
    with TestClient(create_app(settings, services)) as made:
        yield made


@pytest.fixture
def uploaded(client: TestClient) -> str:
    """A stored image, returning its digest. Most job tests need one to point at."""
    response = client.post("/v1/images", files={"file": ("photo.png", png_bytes(), "image/png")})
    assert response.status_code == 201, response.text
    digest: str = response.json()["sha256"]
    return digest
