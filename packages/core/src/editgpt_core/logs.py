"""Structured logs, so the fields that are already being recorded survive.

Thirty-one call sites in this repository log with `extra={...}` — pass index, strategy,
cost, mask coverage, peak RSS. None of it reached a terminal, because nothing configured
logging and the default formatter renders `%(message)s` and drops every extra. The
instrumentation was written and then thrown away on the way out.

`harness/observability.md` is binding on what may be logged, and one rule needs enforcing
rather than remembering: **credentials never appear**. A field whose name looks like a
secret is replaced rather than trusted to have been omitted, because the failure is silent
and permanent — a token in a log file is a token in a log file forever.

One line of JSON per record, on stderr, so a collector can read it and a human can pipe it
through `jq`. Deliberately not a dependency: this is a formatter and a `dictConfig`, and
a logging library would be a third-party package on the critical path of every process for
no behaviour this does not already have.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

_CONTEXT: ContextVar[dict[str, Any] | None] = ContextVar("editgpt_log_context", default=None)


@contextmanager
def bound(**fields: Any) -> Iterator[None]:
    """Attach `fields` to every record logged inside this block.

    `harness/observability.md` asks every line to carry its correlating identifiers, and
    threading a job id through forty call sites is how that stops happening by the third
    one. A `ContextVar` follows the work instead — including across `await`, and into the
    worker's own task, which is a different process and therefore a different context.

    Nested blocks merge, and the outer values win on a clash: a caller that has already
    said which job this is should not have it renamed underneath them.
    """
    current = _CONTEXT.get() or {}
    token = _CONTEXT.set({**fields, **current})
    try:
        yield
    finally:
        _CONTEXT.reset(token)


def current() -> dict[str, Any]:
    """The fields bound right now, copied.

    For handing a correlating id to something that leaves this process — a queued task
    carries it in its message rather than in a `ContextVar`, because the other end is a
    different process and a `ContextVar` does not cross a broker.
    """
    return dict(_CONTEXT.get() or {})


REDACT = ("token", "secret", "password", "key", "authorization", "credential", "cookie")
"""Substrings that make a field name suspicious. Matched case-insensitively, on the *name*.

Deliberately broad. `jwt_key` and `api_key` both match, and so does an innocent
`mask_key` — a redacted field that did not need to be is a moment of confusion, and the
alternative is a leak nobody notices.
"""

REDACTED = "[redacted]"

# Everything `logging` puts on a record itself. Anything outside this set came from
# `extra=` and is what this formatter exists to preserve.
_STANDARD = frozenset(
    [
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
    ]
)


def _safe(name: str, value: Any) -> Any:
    if any(marker in name.lower() for marker in REDACT):
        return REDACTED
    if isinstance(value, str | int | float | bool | type(None)):
        return value
    if isinstance(value, list | tuple):
        return [_safe(name, item) for item in value]
    if isinstance(value, dict):
        return {str(k): _safe(str(k), v) for k, v in value.items()}
    # A repr rather than the object: a log line must never be the thing that fails, and
    # `default=str` on the whole document would hide which field was unserialisable.
    return repr(value)[:200]


class JsonFormatter(logging.Formatter):
    """One JSON object per record, with everything from `extra` alongside."""

    def format(self, record: logging.LogRecord) -> str:
        document: dict[str, Any] = {
            "at": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for name, value in (_CONTEXT.get() or {}).items():
            document[name] = _safe(name, value)
        for name, value in record.__dict__.items():
            if name in _STANDARD or name.startswith("_"):
                continue
            document[name] = _safe(name, value)

        if record.exc_info:
            # The type and message, not the traceback: a stack trace across a JSON line is
            # unreadable, and the type is what a search is done on.
            kind, error, _ = record.exc_info
            document["error"] = getattr(kind, "__name__", str(kind))
            document["error_detail"] = str(error)[:500]

        return json.dumps(document, default=str)


def configure(*, service: str, level: str | None = None, json_output: bool | None = None) -> None:
    """Set up logging for a process. Call once, at startup, before anything logs.

    `json_output` defaults to on unless `EDITGPT_LOG_FORMAT=text`, because a developer
    reading a terminal wants prose and everywhere else wants fields. `service` is attached
    to every record so one collector can hold the gateway and the worker apart.
    """
    if json_output is None:
        json_output = os.environ.get("EDITGPT_LOG_FORMAT", "json").lower() != "text"
    resolved = (level or os.environ.get("EDITGPT_LOG_LEVEL", "INFO")).upper()

    handler = logging.StreamHandler()
    handler.setFormatter(
        JsonFormatter() if json_output else logging.Formatter("%(levelname)s %(name)s %(message)s")
    )
    handler.addFilter(_Service(service))

    root = logging.getLogger()
    # Replace rather than add: uvicorn and celery install their own, and two handlers mean
    # every line twice.
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(resolved)

    for noisy in ("uvicorn.access", "botocore", "urllib3", "PIL"):
        logging.getLogger(noisy).setLevel(max(logging.INFO, logging.getLevelName(resolved)))


class _Service(logging.Filter):
    def __init__(self, service: str) -> None:
        super().__init__()
        self.service = service

    def filter(self, record: logging.LogRecord) -> bool:
        record.service = self.service
        return True
