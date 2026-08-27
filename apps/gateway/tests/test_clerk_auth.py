"""Authentication with Clerk configured.

Clerk's verification is stubbed at exactly one seam — `auth.verify`, which is the only
function that talks to the SDK — so everything downstream of it is the real code: the
401s, the user provisioning, the ownership filtering, the `/ready` report.

What is *not* tested here is the cryptography. `authenticate_request` checks the
signature, the expiry and the authorized party; reimplementing a check of that here would
test our stub. What is tested is the part we own and can get wrong: **that a rejection is
a rejection**, and never a quiet fall back to the shared account.
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from typing import Any

import pytest
from editgpt_gateway import auth
from editgpt_gateway.app import create_app
from editgpt_gateway.auth import NotAuthenticatedError
from editgpt_gateway.deps import Services
from editgpt_gateway.settings import Settings
from editgpt_store import ANONYMOUS_USER_ID, SqlJobStore, User
from fastapi.testclient import TestClient

ALICE = "user_2abcAliceClerkSubject"
BOB = "user_2defBobClerkSubject"


def png_bytes() -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (40, 30), (200, 40, 90)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def clerk_settings(settings: Settings) -> Settings:
    """The same settings, with authentication switched on.

    A dummy key: its *presence* is what turns Clerk on, and `verify` is stubbed, so no
    real credential is involved and none is needed.
    """
    return settings.model_copy(
        update={"clerk_secret_key": "sk_test_not_a_real_key", "clerk_jwt_key": ""}
    )


@pytest.fixture
def signed_in_as(monkeypatch: pytest.MonkeyPatch) -> list[str | None]:
    """Controls who Clerk says the caller is. `None` means the token was rejected."""
    subject: list[str | None] = [ALICE]

    def fake_verify(request: Any, settings: Settings) -> str:
        if subject[0] is None:
            raise NotAuthenticatedError("token is invalid or expired")
        return subject[0]

    monkeypatch.setattr(auth, "verify", fake_verify)
    return subject


@pytest.fixture
def client(
    clerk_settings: Settings, services: Services, signed_in_as: list[str | None]
) -> Iterator[TestClient]:
    services.settings = clerk_settings
    with TestClient(create_app(clerk_settings, services)) as made:
        yield made


def upload(client: TestClient) -> Any:
    return client.post("/v1/images", files={"file": ("a.png", png_bytes(), "image/png")})


def sessions(services: Services) -> Any:
    """The session factory, narrowed from the protocol these tests are wired with."""
    store = services.jobs
    assert isinstance(store, SqlJobStore)
    return store.session_factory


# ---------------------------------------------------------------- failing closed


def test_a_rejected_token_is_401_and_not_the_shared_account(
    client: TestClient, signed_in_as: list[str | None]
) -> None:
    """The property that matters most. A bad credential must not become no credential."""
    signed_in_as[0] = None
    response = upload(client)

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("post", "/v1/images"),
        ("post", "/v1/jobs"),
        ("get", "/v1/jobs/1c1b1a19-0000-4000-8000-000000000000"),
        ("post", "/v1/jobs/1c1b1a19-0000-4000-8000-000000000000/cancel"),
        ("get", "/v1/jobs/1c1b1a19-0000-4000-8000-000000000000/events"),
        ("get", "/v1/images/" + "a" * 64),
    ],
)
def test_every_v1_endpoint_refuses_an_unauthenticated_caller(
    client: TestClient, signed_in_as: list[str | None], method: str, path: str
) -> None:
    """Enumerated rather than spot-checked: one endpoint missing the dependency is a hole,
    and it is the sort of omission that is invisible in review."""
    signed_in_as[0] = None
    assert getattr(client, method)(path).status_code == 401


@pytest.mark.parametrize("path", ["/health", "/ready", "/capabilities", "/"])
def test_the_operational_endpoints_stay_open(client: TestClient, path: str) -> None:
    """A health check that needs a session cannot be used by the thing checking health."""
    assert client.get(path).status_code == 200


# ---------------------------------------------------------------- provisioning


def test_a_first_request_provisions_a_user_row(client: TestClient, services: Services) -> None:
    """Clerk owns the account; this table owns the foreign key everything else points at."""
    assert upload(client).status_code == 201

    factory = sessions(services)
    with factory() as session:
        rows = {u.external_id: u.id for u in session.query(User).all()}
    assert ALICE in rows
    assert rows[ALICE] != ANONYMOUS_USER_ID


def test_a_second_request_reuses_the_same_row(client: TestClient, services: Services) -> None:
    """Provisioning on every request would grow a row per call and orphan the first."""
    upload(client)
    upload(client)

    factory = sessions(services)
    with factory() as session:
        assert session.query(User).filter(User.external_id == ALICE).count() == 1


def test_two_clerk_subjects_get_two_rows(
    client: TestClient, services: Services, signed_in_as: list[str | None]
) -> None:
    upload(client)
    signed_in_as[0] = BOB
    upload(client)

    factory = sessions(services)
    with factory() as session:
        externals = {u.external_id for u in session.query(User).all()}
    assert {ALICE, BOB} <= externals


# ---------------------------------------------------------------- isolation


def test_one_users_job_is_invisible_to_another(
    client: TestClient, signed_in_as: list[str | None]
) -> None:
    """End to end through HTTP with two real Clerk subjects, not two synthetic UUIDs."""
    digest = upload(client).json()["sha256"]
    job_id = client.post(
        "/v1/jobs", json={"op": "remove", "image_sha256": digest, "target": "the car"}
    ).json()["id"]
    assert client.get(f"/v1/jobs/{job_id}").status_code == 200

    signed_in_as[0] = BOB
    assert client.get(f"/v1/jobs/{job_id}").status_code == 404
    assert client.post(f"/v1/jobs/{job_id}/cancel").status_code == 404
    assert client.get(f"/v1/jobs/{job_id}/events").status_code == 404


def test_the_same_idempotency_key_from_two_users_makes_two_jobs(
    client: TestClient, signed_in_as: list[str | None]
) -> None:
    digest = upload(client).json()["sha256"]
    body = {"op": "remove", "image_sha256": digest, "target": "the car"}
    headers = {"Idempotency-Key": "same-key"}

    mine = client.post("/v1/jobs", json=body, headers=headers)
    signed_in_as[0] = BOB
    theirs = client.post("/v1/jobs", json=body, headers=headers)

    assert (mine.status_code, theirs.status_code) == (202, 202)
    assert mine.json()["id"] != theirs.json()["id"]


def test_the_owner_sent_to_the_worker_is_the_provisioned_row_not_the_sentinel(
    client: TestClient, services: Services
) -> None:
    """A job queued for the sentinel would be invisible to the user who asked for it."""
    from editgpt_gateway.deps import RecordingQueue

    digest = upload(client).json()["sha256"]
    client.post("/v1/jobs", json={"op": "remove", "image_sha256": digest, "target": "x"})

    factory = sessions(services)
    with factory() as session:
        alice = session.query(User).filter(User.external_id == ALICE).one()
        expected = alice.id

    queue = services.queue
    assert isinstance(queue, RecordingQueue)
    job_id = queue.sent[0][0]
    assert services.jobs.get(__import__("uuid").UUID(job_id), user_id=expected) is not None


# ---------------------------------------------------------------- reporting


def test_ready_reports_the_provider_and_never_a_key(
    client: TestClient, clerk_settings: Settings
) -> None:
    body = client.get("/ready").json()
    assert body["auth"] == {"provider": "clerk", "networkless_verification": False}
    assert clerk_settings.clerk_secret_key not in client.get("/ready").text


def test_ready_says_out_loud_when_authentication_is_off(
    settings: Settings, services: Services
) -> None:
    """The one degraded mode that must never be silent."""
    with TestClient(create_app(settings, services)) as anonymous:
        body = anonymous.get("/ready").json()
    assert body["auth"]["provider"] == "none"
    assert any("no authentication" in line for line in body["degraded"])


def test_networkless_verification_is_reported_when_the_public_key_is_set(
    clerk_settings: Settings, services: Services
) -> None:
    """Worth surfacing: without it every cold start depends on Clerk being reachable."""
    with_key = clerk_settings.model_copy(update={"clerk_jwt_key": "-----BEGIN PUBLIC KEY-----"})
    services.settings = with_key
    with TestClient(create_app(with_key, services)) as made:
        assert made.get("/ready").json()["auth"]["networkless_verification"] is True


# ---------------------------------------------------------------- configuration


def test_the_secret_key_is_read_under_clerks_own_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clerk's Next.js SDK requires `CLERK_SECRET_KEY` verbatim.

    Without this alias the same secret would live in two variables — the frontend's and
    ours — and someone would eventually rotate one of them.
    """
    monkeypatch.delenv("EDITGPT_CLERK_SECRET_KEY", raising=False)
    monkeypatch.setenv("CLERK_SECRET_KEY", "sk_test_from_clerks_name")

    made = Settings(_env_file=None)  # type: ignore[call-arg]
    assert made.clerk_secret_key == "sk_test_from_clerks_name"
    assert made.uses_clerk


def test_the_prefixed_name_wins_when_both_are_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit `EDITGPT_` override must beat the shared name, not be shadowed by it."""
    monkeypatch.setenv("CLERK_SECRET_KEY", "sk_shared")
    monkeypatch.setenv("EDITGPT_CLERK_SECRET_KEY", "sk_override")

    assert Settings(_env_file=None).clerk_secret_key == "sk_override"  # type: ignore[call-arg]


def test_no_key_anywhere_means_authentication_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLERK_SECRET_KEY", raising=False)
    monkeypatch.delenv("EDITGPT_CLERK_SECRET_KEY", raising=False)

    assert not Settings(_env_file=None).uses_clerk  # type: ignore[call-arg]
