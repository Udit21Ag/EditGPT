"""The gateway's contract, exercised from the first commit.

The `client` fixture comes from `conftest.py` and is wired to local doubles. This module
deliberately does not build its own: one that called `create_app` with real settings
would try to reach Postgres and Redis, and a test suite that needs a container is a test
suite that stops being run.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from editgpt_core import EditOp, JobState
from editgpt_gateway.app import create_app
from editgpt_gateway.deps import Services
from editgpt_gateway.settings import Settings
from fastapi.testclient import TestClient

from .conftest import png_bytes

REPO_ROOT = Path(__file__).resolve().parents[3]


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


def test_the_web_client_offers_exactly_what_is_advertised(client: TestClient) -> None:
    """The buttons in the browser against the operations the gateway implements.

    `apps/web/lib/edit-request.ts` mirrors `EditSpec`'s rules so the form can grey out a
    button instead of handing the user a 422. A mirror drifts, and both directions hurt:
    an operation offered with nothing behind it is a lie the user acts on, and one
    implemented but never offered is work nobody can reach — which is what the whole
    candidate picker turned out to be.

    Read out of the TypeScript rather than duplicated here, so the assertion fails when
    the list changes rather than when someone remembers to update a copy of it.
    """
    source = (REPO_ROOT / "apps/web/lib/edit-request.ts").read_text()
    offered = set(re.findall(r'^    op: "([a-z]+)",$', source, re.MULTILINE))
    assert offered, "could not find the operation list; the parser or the file moved"

    advertised = set(client.get("/capabilities").json()["operations"])
    assert offered == advertised


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


# ---------------------------------------------------------------- grounding prompts


def test_a_request_must_carry_exactly_one_kind_of_prompt(client: TestClient) -> None:
    """A phrase and a tap need different models, so the request has to say which it is.

    Both is not "use whichever" — it is a caller that has not decided, and answering one
    of them silently would make the other look broken.
    """
    digest = "a" * 64
    for body in (
        {"image_sha256": digest},
        {"image_sha256": digest, "target": "the car", "points": [{"x": 0.5, "y": 0.5}]},
    ):
        response = client.post("/v1/masks", json=body)
        assert response.status_code == 422, response.text
        assert "either `target` or `points`" in response.text

    empty = client.post("/v1/masks", json={"image_sha256": digest, "points": []})
    assert empty.status_code == 422
    assert "at least one tap" in empty.text


def test_a_tap_outside_the_picture_is_refused_at_the_boundary(client: TestClient) -> None:
    """Fractions of the image, so anything outside 0..1 is a client bug worth naming."""
    response = client.post(
        "/v1/masks",
        json={"image_sha256": "a" * 64, "points": [{"x": 1.4, "y": 0.5}]},
    )
    assert response.status_code == 422


# ---------------------------------------------------------------- browser access


def browser_client(services: Services, origins: tuple[str, ...]) -> TestClient:
    """The same app the other tests use, differing only in which origins it admits."""
    return TestClient(create_app(Settings(environment="test", cors_origins=origins), services))


def test_a_browser_preflight_is_answered(services: Services) -> None:
    """The bug the full-stack browser suite found on its first real run.

    There was no CORS middleware at all, so the preflight reached the router, which knows
    nothing about `OPTIONS`, and answered 405 — every cross-origin request a browser made
    was blocked before it was sent. Nothing caught it because nothing had called this from
    a browser: curl does not preflight and the component tests replace `fetch`.
    """
    client = browser_client(services, ("http://localhost:3000",))
    response = client.options(
        "/v1/images",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization",
        },
    )
    assert response.status_code == 200, response.text
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert "authorization" in response.headers["access-control-allow-headers"].lower()


def test_an_origin_that_was_not_allowed_gets_nothing(services: Services) -> None:
    """An explicit list, not a wildcard: otherwise any page a user visits could call this
    API with their session, and CORS is the only thing between those two facts."""
    client = browser_client(services, ("http://localhost:3000",))
    response = client.options(
        "/v1/images",
        headers={
            "Origin": "https://attacker.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.headers.get("access-control-allow-origin") is None


def test_ambient_credentials_are_never_allowed(services: Services) -> None:
    """The session travels as an `Authorization` header, never a cookie, so nothing needs
    the browser to attach credentials of its own — and leaving it off shuts the door on a
    class of cross-site request a cookie session would open."""
    client = browser_client(services, ("http://localhost:3000",))
    response = client.options(
        "/v1/images",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"},
    )
    assert response.headers.get("access-control-allow-credentials") is None


def test_no_origins_configured_means_no_cross_origin_access(services: Services) -> None:
    """A deployment serving the app and the API on one origin needs none of this, and
    should not be handed a header that says otherwise."""
    client = browser_client(services, ())
    response = client.options(
        "/v1/images",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"},
    )
    assert response.headers.get("access-control-allow-origin") is None


def readiness_with(services: Services, **over: object) -> list[str]:
    """`/ready`'s degradations under different settings.

    `Services` carries the settings that `degraded` reads, so both have to be replaced —
    passing new settings to `create_app` alone changes what the routes see and not what
    the readiness check reports, which is a trap worth only falling into once.
    """
    settings = Settings(**({"environment": "test"} | over))  # type: ignore[arg-type]
    swapped = replace(services, settings=settings)
    client = TestClient(create_app(settings, swapped))
    problems: list[str] = client.get("/ready").json()["degraded"]
    return problems


def test_a_deployment_that_kept_the_development_origins_says_so(services: Services) -> None:
    """It fails safe — the real origin is blocked, so somebody notices within a minute —
    but a development default nobody set becomes a setting nobody knows about."""
    problems = readiness_with(
        services, environment="production", cors_origins=("http://localhost:3000",)
    )
    assert any("cors still allows localhost" in problem for problem in problems)


@pytest.mark.parametrize("environment", ["development", "test", "ci"])
def test_a_local_environment_is_not_nagged_about_its_own_defaults(
    services: Services, environment: str
) -> None:
    problems = readiness_with(
        services, environment=environment, cors_origins=("http://localhost:3000",)
    )
    assert not any("cors" in problem for problem in problems)


def test_a_real_origin_is_not_flagged(services: Services) -> None:
    problems = readiness_with(
        services, environment="production", cors_origins=("https://editgpt.example",)
    )
    assert not any("cors" in problem for problem in problems)


def test_an_upload_is_stored_without_the_camera_that_took_it(
    client: TestClient, services: Services
) -> None:
    """The boundary must actually call the scrubber.

    `test_scrub.py` proves the function works; this proves it is wired in. Removing the
    call from `inspect` passed every test in that file, which is the whole reason this one
    exists — a privacy fix nobody invokes is not a fix.
    """
    import io

    from PIL import Image

    tags = Image.Exif()
    tags[0x010F] = "iQOO"
    tags[0x0110] = "iQOO Neo7"
    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), (20, 90, 140)).save(
        buffer, format="JPEG", quality=90, exif=tags.tobytes()
    )
    uploaded = buffer.getvalue()

    with Image.open(io.BytesIO(uploaded)) as image:
        assert dict(image.getexif()), "the fixture carries no metadata; it proves nothing"

    response = client.post("/v1/images", files={"file": ("phone.jpg", uploaded, "image/jpeg")})
    assert response.status_code == 201, response.text

    stored = services.assets.get(response.json()["sha256"])
    with Image.open(io.BytesIO(stored)) as image:
        assert dict(image.getexif()) == {}, "the camera came with it"


def test_every_response_carries_a_request_id(client: TestClient) -> None:
    """The id in a user's console has to be the one in the logs, or it is decoration."""
    response = client.get("/health")
    assert response.headers.get("x-request-id")


def test_an_inbound_request_id_survives_the_hop(client: TestClient) -> None:
    """A trace begun by a proxy or a client should not be renamed here."""
    response = client.get("/health", headers={"X-Request-Id": "trace-abc"})
    assert response.headers["x-request-id"] == "trace-abc"


def test_two_requests_get_different_ids(client: TestClient) -> None:
    first = client.get("/health").headers["x-request-id"]
    second = client.get("/health").headers["x-request-id"]
    assert first != second


# ---------------------------------------------------------------- image links


def test_a_signed_link_serves_the_image_without_a_session(
    client: TestClient, uploaded: str
) -> None:
    created = client.post("/v1/images", files={"file": ("p.png", png_bytes(), "image/png")}).json()
    assert created["url"], "no link was issued"

    # A fresh client with no credentials at all, which is what an `<img>` tag is.
    anonymous = TestClient(client.app)
    anonymous.headers.clear()
    response = anonymous.get(created["url"])
    assert response.status_code == 200, response.text
    assert response.content == png_bytes()


def test_a_link_for_one_image_does_not_serve_another(client: TestClient) -> None:
    created = client.post("/v1/images", files={"file": ("p.png", png_bytes(), "image/png")}).json()
    query = created["url"].split("?", 1)[1]

    other = "b" * 64
    response = client.get(f"/v1/images/{other}?{query}")
    assert response.status_code in (401, 404)


def test_a_request_without_a_valid_signature_still_goes_through_authentication(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The control flow, asserted directly.

    A status code cannot show this here: the test gateway runs in anonymous mode, so
    everything answers 200 whether or not the check ran, and asserting 401 would pass only
    on a configured stack and vacuously everywhere else. What matters is that a missing or
    forged signature falls *through* to the session check rather than around it.
    """
    from editgpt_gateway import auth as auth_module

    created = client.post("/v1/images", files={"file": ("p.png", png_bytes(), "image/png")}).json()
    digest = created["sha256"]
    checked: list[str] = []
    real = auth_module.current_identity

    def watched(request: Any, svc: Any) -> Any:
        checked.append("asked")
        return real(request, svc)

    monkeypatch.setattr(auth_module, "current_identity", watched)

    client.get(f"/v1/images/{digest}")
    assert checked, "no signature, and the session check was skipped"

    checked.clear()
    client.get(f"/v1/images/{digest}?expires=99999999999&signature=forged")
    assert checked, "a forged signature bypassed the session check"

    checked.clear()
    client.get(created["url"])
    assert not checked, "a valid signature should not need a session"


def test_a_finished_job_carries_a_link_to_its_result(
    client: TestClient, services: Services, uploaded: str
) -> None:
    """So the client can show the result without another round trip or an object URL."""
    from uuid import UUID

    created = client.post(
        "/v1/jobs",
        json={
            "op": "remove",
            "image_sha256": uploaded,
            "mask_source": "text",
            "target": "the car",
        },
    ).json()
    job = services.jobs.get(UUID(created["id"]))
    assert job is not None

    digest = services.assets.put(png_bytes(), content_type="image/png")
    # QUEUED -> PLANNING -> RUNNING -> REVIEW -> DONE; the machine has no shortcuts, so a
    # retry is always recorded rather than implied.
    reached = job
    for state in (JobState.PLANNING, JobState.RUNNING, JobState.REVIEW):
        reached = reached.advance(state, detail=state.value)
    services.jobs.save(reached.advance(JobState.DONE, result_sha256=digest, detail="done"))

    view = client.get(f"/v1/jobs/{created['id']}").json()
    assert view["result_sha256"] == digest
    assert view["result_url"], "a result with no link is a result the client cannot show"


# ---------------------------------------------------------------- planning


def test_a_plain_instruction_is_planned_without_a_model(client: TestClient) -> None:
    """The fast-path claim, checkable from outside: no key is configured in tests, and
    "remove the car" still comes back as a plan rather than a question."""
    response = client.post("/v1/plan", json={"instruction": "remove the car"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["route"] == "rule"
    assert body["op"] == "remove"
    assert body["target"] == "car"
    assert body["tokens"] == 0, "a rule answer must not have cost anything"


def test_an_instruction_nobody_can_act_on_comes_back_as_a_question(client: TestClient) -> None:
    response = client.post("/v1/plan", json={"instruction": "make it nicer"})

    body = response.json()
    assert body["route"] == "ask"
    assert body["question"]
    assert body["op"] is None


def test_an_operation_with_no_implementation_is_refused_here_not_in_a_worker(
    client: TestClient,
) -> None:
    """`/capabilities` and the planner read the same tuple, so what the API advertises and
    what it will plan cannot drift apart."""
    advertised = set(client.get("/capabilities").json()["operations"])
    response = client.post("/v1/plan", json={"instruction": "retouch the skin"})

    assert "retouch" not in advertised
    assert response.json()["route"] == "ask"


def test_a_page_of_text_is_refused_before_it_reaches_a_metered_model(
    client: TestClient,
) -> None:
    """A planner prompt is one instruction. Thirty calls in ninety seconds exhausted a
    day's free-tier quota during a benchmark run; an uncapped body is the same failure
    in a single request."""
    response = client.post("/v1/plan", json={"instruction": "remove the car " * 200})
    assert response.status_code == 422


def test_planning_is_rate_limited_like_the_endpoints_that_cost_something(
    settings: Settings,
) -> None:
    assert "/v1/plan" in settings.rate_limit_burst_paths
