"""Structured logging, and the rule that must not be left to memory.

`harness/observability.md` is binding on what may be logged, and one line of it —
credentials never appear — is enforced here rather than trusted to reviewers. The failure
is silent and permanent: a token written to a log file is in that log file forever, and
nothing downstream can undo it.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest
from editgpt_core.logs import REDACTED, JsonFormatter, bound, configure


def render(message: str = "event", **extra: Any) -> dict[str, Any]:
    record = logging.LogRecord("editgpt.test", logging.INFO, __file__, 1, message, None, None)
    for name, value in extra.items():
        setattr(record, name, value)
    parsed: dict[str, Any] = json.loads(JsonFormatter().format(record))
    return parsed


def test_the_fields_a_call_site_recorded_survive() -> None:
    """The whole reason this exists. Thirty-one call sites logged with `extra=` and the
    default formatter rendered the message and dropped every one of them."""
    line = render("erase.pass", index=2, strategy="migan", cost=34.7, kept=True)
    assert line["event"] == "erase.pass"
    assert line["index"] == 2
    assert line["strategy"] == "migan"
    assert line["cost"] == 34.7
    assert line["kept"] is True


def test_the_line_is_one_json_object() -> None:
    text = JsonFormatter().format(
        logging.LogRecord("editgpt.test", logging.INFO, __file__, 1, "e", None, None)
    )
    assert "\n" not in text
    assert json.loads(text)["level"] == "INFO"


@pytest.mark.parametrize(
    "name",
    [
        "clerk_secret_key",
        "CLERK_SECRET_KEY",
        "authorization",
        "api_token",
        "password",
        "jwt_key",
        "session_cookie",
        "credential",
    ],
)
def test_anything_that_looks_like_a_secret_is_replaced(name: str) -> None:
    assert render(**{name: "sk_live_abcdef123456"})[name] == REDACTED


def test_redaction_reaches_inside_a_mapping() -> None:
    # A field can be a dict of provider settings, and the interesting one is nested.
    line = render(provider={"name": "cloudflare", "api_token": "secret"})
    assert line["provider"]["name"] == "cloudflare"
    assert line["provider"]["api_token"] == REDACTED


def test_an_ordinary_field_is_left_alone() -> None:
    assert render(strategy="migan", job_id="abc")["strategy"] == "migan"


def test_an_unserialisable_value_does_not_break_the_line() -> None:
    """A log call must never be the thing that fails."""

    class Awkward:
        def __repr__(self) -> str:
            return "<Awkward>"

    assert render(thing=Awkward())["thing"] == "<Awkward>"


def test_an_exception_is_recorded_by_type_and_message() -> None:
    """A stack trace across a JSON line is unreadable, and the type is what gets searched."""
    try:
        raise ValueError("mask covers 0 px")
    except ValueError:
        import sys

        record = logging.LogRecord(
            "t", logging.ERROR, __file__, 1, "edit.failed", None, sys.exc_info()
        )
        line = json.loads(JsonFormatter().format(record))

    assert line["error"] == "ValueError"
    assert "mask covers 0 px" in line["error_detail"]


# ------------------------------------------------------------------ correlation


def test_bound_fields_reach_every_line_inside_the_block() -> None:
    with bound(request_id="req-1"):
        assert render("job.created")["request_id"] == "req-1"


def test_the_context_unwinds() -> None:
    with bound(request_id="req-1"):
        pass
    assert "request_id" not in render("later")


def test_nesting_merges_and_the_outer_value_wins() -> None:
    # A caller that has already said which job this is should not have it renamed
    # underneath them by something deeper in the stack.
    with bound(job_id="outer"), bound(job_id="inner", stage="erase"):
        line = render("erase.pass")
    assert line["job_id"] == "outer"
    assert line["stage"] == "erase"


def test_an_explicit_field_beats_the_context() -> None:
    with bound(stage="erase"):
        assert render("e", stage="composite")["stage"] == "composite"


# ------------------------------------------------------------------ configuration


def test_configuring_replaces_handlers_rather_than_adding_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Uvicorn and Celery each install their own, and two handlers mean every line twice."""
    root = logging.getLogger()
    original = list(root.handlers)
    try:
        configure(service="test")
        configure(service="test")
        assert len(root.handlers) == 1
    finally:
        root.handlers = original


def test_text_output_is_available_for_a_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    root = logging.getLogger()
    original = list(root.handlers)
    try:
        monkeypatch.setenv("EDITGPT_LOG_FORMAT", "text")
        configure(service="test")
        assert not isinstance(root.handlers[0].formatter, JsonFormatter)
    finally:
        root.handlers = original


def test_the_service_is_on_every_line(monkeypatch: pytest.MonkeyPatch) -> None:
    """One collector holds the gateway and the worker; without this they are one stream."""
    root = logging.getLogger()
    original = list(root.handlers)
    try:
        configure(service="gateway")
        record = logging.LogRecord("t", logging.INFO, __file__, 1, "e", None, None)
        attached = root.handlers[0].filters[0]
        assert isinstance(attached, logging.Filter)
        assert attached.filter(record) is True
        assert record.service == "gateway"  # type: ignore[attr-defined]
    finally:
        root.handlers = original
