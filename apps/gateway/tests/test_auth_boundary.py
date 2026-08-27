"""Whose data a request can reach.

Authentication is not implemented — every request is the anonymous sentinel — so there is
nothing here about credentials. What *is* testable, and worth testing now, is that the
routes carry an identity at all and pass it down: a route that quietly relies on the
store's default parameter would behave identically today and leak every job the moment a
second user existed.

`dependency_overrides` stands in for the provider that does not exist yet, which is
exactly the seam `current_principal` was extracted to create.
"""

from __future__ import annotations

import io
from typing import Any
from uuid import UUID

import pytest
from editgpt_gateway.app import create_app
from editgpt_gateway.auth import Identity, current_identity
from editgpt_gateway.deps import Services
from editgpt_gateway.settings import Settings
from editgpt_store import ANONYMOUS_USER_ID, User
from fastapi.testclient import TestClient

OTHER_USER = UUID("00000000-0000-0000-0000-0000000000aa")


def png_bytes() -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (48, 32), (7, 90, 160)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def app_and_principal(settings: Settings, services: Services) -> tuple[Any, list[UUID]]:
    """An app whose principal the test can change between requests."""
    acting_as = [ANONYMOUS_USER_ID]
    app = create_app(settings, services)
    app.dependency_overrides[current_identity] = lambda: Identity(
        user_id=acting_as[0], external_id=None
    )

    # The second user needs a row: every `user_id` is a foreign key, which is itself part
    # of the guarantee — an unknown principal cannot own anything.
    factory = getattr(services.jobs, "session_factory", None)
    if factory is not None:
        with factory() as session:
            session.add(User(id=OTHER_USER, external_id="someone-else"))
            session.commit()

    return app, acting_as


def test_a_job_is_invisible_to_a_different_principal(
    app_and_principal: tuple[Any, list[UUID]],
) -> None:
    """The whole point. Before the principal was explicit, this could not be written."""
    app, acting_as = app_and_principal
    with TestClient(app) as client:
        digest = client.post(
            "/v1/images", files={"file": ("a.png", png_bytes(), "image/png")}
        ).json()["sha256"]
        job_id = client.post(
            "/v1/jobs", json={"op": "remove", "image_sha256": digest, "target": "the car"}
        ).json()["id"]

        assert client.get(f"/v1/jobs/{job_id}").status_code == 200

        acting_as[0] = OTHER_USER
        assert client.get(f"/v1/jobs/{job_id}").status_code == 404


def test_another_principal_cannot_cancel_someone_elses_job(
    app_and_principal: tuple[Any, list[UUID]],
) -> None:
    """Cancelling is a write, and a 404 is the right answer — not 403.

    Telling a stranger that a job exists but is not theirs confirms the id is real.
    """
    app, acting_as = app_and_principal
    with TestClient(app) as client:
        digest = client.post(
            "/v1/images", files={"file": ("a.png", png_bytes(), "image/png")}
        ).json()["sha256"]
        job_id = client.post(
            "/v1/jobs", json={"op": "remove", "image_sha256": digest, "target": "the car"}
        ).json()["id"]

        acting_as[0] = OTHER_USER
        assert client.post(f"/v1/jobs/{job_id}/cancel").status_code == 404

        acting_as[0] = ANONYMOUS_USER_ID
        assert client.get(f"/v1/jobs/{job_id}").json()["state"] == "queued"


def test_another_principal_cannot_open_the_progress_stream(
    app_and_principal: tuple[Any, list[UUID]],
) -> None:
    """The stream replays a job's steps, so it is a read of the same data."""
    app, acting_as = app_and_principal
    with TestClient(app) as client:
        digest = client.post(
            "/v1/images", files={"file": ("a.png", png_bytes(), "image/png")}
        ).json()["sha256"]
        job_id = client.post(
            "/v1/jobs", json={"op": "remove", "image_sha256": digest, "target": "the car"}
        ).json()["id"]

        acting_as[0] = OTHER_USER
        assert client.get(f"/v1/jobs/{job_id}/events").status_code == 404


def test_two_principals_may_use_the_same_idempotency_key(
    app_and_principal: tuple[Any, list[UUID]],
) -> None:
    """Idempotency is scoped per user, not global.

    If it were global, one client's key could hand them another's job — which is why the
    unique constraint is on `(user_id, idempotency_key)` and why `user_id` is a real row
    rather than NULL: Postgres treats NULLs in a unique constraint as distinct.
    """
    app, acting_as = app_and_principal
    with TestClient(app) as client:
        digest = client.post(
            "/v1/images", files={"file": ("a.png", png_bytes(), "image/png")}
        ).json()["sha256"]
        body = {"op": "remove", "image_sha256": digest, "target": "the car"}
        headers = {"Idempotency-Key": "shared-key"}

        mine = client.post("/v1/jobs", json=body, headers=headers)
        acting_as[0] = OTHER_USER
        theirs = client.post("/v1/jobs", json=body, headers=headers)

    assert mine.status_code == 202
    assert theirs.status_code == 202, "a different user's key must not be a cache hit"
    assert mine.json()["id"] != theirs.json()["id"]


def test_the_owner_travels_with_the_queue_message(
    app_and_principal: tuple[Any, list[UUID]], services: Services
) -> None:
    """The worker must not need a privileged "fetch any job" path to do its work."""
    from editgpt_gateway.deps import CeleryQueue

    sent: list[dict[str, Any]] = []

    class Recorder:
        def send_task(self, name: str, *, args: list[str], kwargs: dict[str, Any]) -> None:
            sent.append({"name": name, "args": args, "kwargs": kwargs})

    app, acting_as = app_and_principal
    services.queue = CeleryQueue(app=Recorder())
    acting_as[0] = OTHER_USER

    with TestClient(app) as client:
        digest = client.post(
            "/v1/images", files={"file": ("a.png", png_bytes(), "image/png")}
        ).json()["sha256"]
        client.post("/v1/jobs", json={"op": "remove", "image_sha256": digest, "target": "x"})

    assert sent[0]["kwargs"]["user_id"] == str(OTHER_USER)


def test_with_no_provider_configured_every_request_is_the_sentinel(
    services: Services,
) -> None:
    """A real row, not a NULL — see `editgpt_store.models.ANONYMOUS_USER_ID`."""
    from editgpt_gateway.auth import current_identity
    from fastapi import Request

    request = Request({"type": "http", "headers": [], "method": "GET", "path": "/"})
    identity = current_identity(request, services)
    assert identity.user_id == ANONYMOUS_USER_ID
    assert identity.is_anonymous
