"""The one place a language model is called, and the shape it is forced into.

**Constrained decoding, not JSON parsing.** The request carries a response schema derived
from `Intent`, so the model is decoded against a grammar rather than asked politely for
JSON and then regexed. The result is still validated locally: a schema the provider
enforces is a provider's promise, and a planner that trusts it has one unvalidated input
from the internet.

The schema Google accepts is a restricted subset of JSON Schema — no `$ref`, no `$defs`,
no `anyOf`, and nullability is a flag rather than a union with `null`. Pydantic emits all
three, so `response_schema` converts. That conversion is tested rather than trusted,
because a silently wrong schema does not fail: it decodes into a shape the model chose.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx
from editgpt_core.errors import ProviderError
from pydantic import BaseModel

log = logging.getLogger(__name__)

ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

DEFAULT_MODEL = "gemini-3.6-flash"
"""Text only. The image models have no free tier — verified 2026-08-25, see the plan."""

SYSTEM = """You convert one image-editing instruction into one operation.

Operations: remove (erase something and continue the background), add (put something new
in), replace (swap one thing for another), background (change the backdrop), upscale
(increase resolution), restyle, retouch.

Rules:
- `target` is what to act on, copied from the user's own words. Do not resolve it, do not
  guess which one they meant, do not add adjectives they did not write.
- `content` is what should be there instead, for add, replace and background.
- `colour` only when they named a specific colour, as #rrggbb.
- If the instruction is not an edit, or names no operation, choose the closest operation
  only when it is unambiguous. Never invent a target."""


class Completer(Protocol):
    """Anything that can answer with JSON matching a schema."""

    def complete(self, instruction: str, *, schema: dict[str, Any], timeout_s: float) -> str: ...


def response_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Pydantic's JSON schema, reduced to what a response schema may contain."""
    raw = model.model_json_schema()
    defs = raw.get("$defs", {})
    return _reduce(raw, defs)


_KEEP = {"type", "description", "enum", "items", "properties", "required", "nullable"}


def _reduce(node: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any]:
    if ref := node.get("$ref"):
        # `$defs` exists so a schema can repeat itself; a response schema cannot, so the
        # definition is inlined at each use.
        return _reduce(
            {**defs[ref.rsplit("/", 1)[-1]], **{k: v for k, v in node.items() if k != "$ref"}}, defs
        )

    if options := node.get("anyOf"):
        # `str | None` arrives as a union with null. Nullability is a flag here, so the
        # non-null branch is kept and the flag is set — which also means a union of two
        # real types would be silently narrowed, and `Intent` has none for that reason.
        real = [option for option in options if option.get("type") != "null"]
        nullable = len(real) != len(options)
        merged = _reduce(real[0], defs) if real else {"type": "string"}
        return {**merged, **({"nullable": True} if nullable else {})}

    out: dict[str, Any] = {}
    for key, value in node.items():
        if key not in _KEEP:
            continue  # `title`, `default`, `pattern`, `additionalProperties`: all rejected
        if key == "properties":
            out[key] = {name: _reduce(sub, defs) for name, sub in value.items()}
        elif key == "items":
            out[key] = _reduce(value, defs)
        else:
            out[key] = value
    return out


@dataclass(frozen=True)
class Gemini:
    """Google AI Studio's free text tier, over its REST API.

    No SDK: the call is one POST with a JSON body, `httpx` is already in the tree for the
    image provider, and a vendor client would be a second HTTP stack to configure, retry
    and mock. The key is read from the environment at construction, never passed around.
    """

    model: str = DEFAULT_MODEL
    api_key: str | None = field(default=None, repr=False)
    temperature: float = 0.0
    """Zero, because the same instruction twice should not become two different edits."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "api_key", self.api_key or os.environ.get("GEMINI_API_KEY"))

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def complete(self, instruction: str, *, schema: dict[str, Any], timeout_s: float) -> str:
        if not self.api_key:
            raise ProviderError("no GEMINI_API_KEY; the planner has no model to ask")

        payload = {
            "systemInstruction": {"parts": [{"text": SYSTEM}]},
            "contents": [{"role": "user", "parts": [{"text": instruction}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": schema,
                "temperature": self.temperature,
            },
        }
        try:
            response = httpx.post(
                ENDPOINT.format(model=self.model),
                headers={"x-goog-api-key": self.api_key},
                json=payload,
                timeout=timeout_s,
            )
        except httpx.HTTPError as error:
            raise ProviderError(f"transport error: {error}") from error

        if response.status_code != 200:
            raise ProviderError(f"planner model {response.status_code}: {response.text[:300]}")

        body = response.json()
        try:
            text: str = body["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as error:
            # A refusal or a safety block comes back as a 200 with no candidate. Reported
            # as a provider error rather than crashing on a subscript.
            raise ProviderError(f"no answer from the planner model: {str(body)[:300]}") from error
        log.info(
            "planner.completed",
            extra={"model": self.model, "chars": len(text)},
        )
        return text
