"""The schema sent to the model, and the adapter that sends it.

No real call is made: `pytest-socket` disables the network for this tier, and the request
is asserted against a transport mock. A wrong schema does not fail loudly — it decodes
into whatever shape the model preferred — so the conversion is checked rather than trusted.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx
from editgpt_core.errors import ProviderError
from editgpt_planner import Gemini, Intent, response_schema
from editgpt_planner.llm import ENDPOINT

URL = ENDPOINT.format(model="gemini-3.6-flash")


def test_the_schema_carries_every_field_the_model_must_fill() -> None:
    schema = response_schema(Intent)
    assert set(schema["properties"]) == set(Intent.model_fields)


def test_the_schema_contains_nothing_a_response_schema_rejects() -> None:
    """Google's subset has no `$ref`, no `$defs`, no `anyOf`, and no `pattern`."""
    text = json.dumps(response_schema(Intent))
    for rejected in ("$ref", "$defs", "anyOf", "pattern", "additionalProperties"):
        assert rejected not in text, f"{rejected} survived the conversion"


def test_the_enum_is_inlined_rather_than_referenced() -> None:
    """`op` is the field worth constraining: an operation the pipeline has never heard of
    is the one answer that cannot be recovered from."""
    schema = response_schema(Intent)
    assert schema["properties"]["op"]["enum"] == [
        "remove",
        "add",
        "replace",
        "restyle",
        "background",
        "retouch",
        "upscale",
    ]


def test_optional_fields_become_nullable_rather_than_a_union() -> None:
    schema = response_schema(Intent)
    assert schema["properties"]["target"] == {"type": "string", "nullable": True}
    assert schema["properties"]["op"].get("nullable") is None


@respx.mock
def test_the_request_asks_for_json_against_the_schema_at_temperature_zero() -> None:
    """Two identical instructions must not become two different edits."""
    route = respx.post(URL).mock(
        return_value=httpx.Response(
            200, json={"candidates": [{"content": {"parts": [{"text": '{"op": "upscale"}'}]}}]}
        )
    )

    answer = Gemini(api_key="test-key").complete(
        "make it bigger", schema=response_schema(Intent), timeout_s=10
    )

    assert answer == '{"op": "upscale"}'
    body = json.loads(route.calls[0].request.content)
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert body["generationConfig"]["temperature"] == 0.0
    assert body["generationConfig"]["responseSchema"]["properties"]["op"]["enum"]
    assert route.calls[0].request.headers["x-goog-api-key"] == "test-key"


@respx.mock
def test_a_blocked_or_empty_answer_is_an_error_rather_than_a_subscript_crash() -> None:
    """A safety block comes back as a 200 with no candidate — the same shape as success."""
    respx.post(URL).mock(
        return_value=httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}})
    )

    with pytest.raises(ProviderError, match="no answer"):
        Gemini(api_key="k").complete("...", schema={}, timeout_s=5)


@respx.mock
def test_an_http_failure_names_the_status() -> None:
    respx.post(URL).mock(return_value=httpx.Response(429, text="quota"))

    with pytest.raises(ProviderError, match="429"):
        Gemini(api_key="k").complete("...", schema={}, timeout_s=5)


def test_an_unconfigured_client_says_so_before_making_a_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    client = Gemini()

    assert not client.is_configured()
    with pytest.raises(ProviderError, match="GEMINI_API_KEY"):
        client.complete("...", schema={}, timeout_s=5)


def test_the_key_is_not_in_the_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    """It ends up in tracebacks and log lines otherwise, which is how a key leaves a
    process it never left deliberately."""
    assert "secret-key" not in repr(Gemini(api_key="secret-key"))
