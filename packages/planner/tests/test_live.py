"""The planner against the real model. Opt-in: `pytest -m live`.

The mocked tests prove the arithmetic around the call and nothing about the call. They
cannot tell you that the provider accepts the converted schema — and a rejected schema is
the failure mode most likely to survive every local test, because the conversion is
correct-looking JSON either way.

**On credentials.** The root `conftest.py` deletes `GEMINI_*` from the environment so that
no unmarked test can reach a real account. That rule protects the fast tier; the live tier
exists precisely to spend real resources and says so in its markers, so it reads the key
from `.env` itself rather than asking for the guard to be loosened for everybody.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from editgpt_core import EditOp
from editgpt_planner import Gemini, Route, plan

pytestmark = [pytest.mark.live, pytest.mark.enable_socket]

IMPLEMENTED = frozenset(
    {EditOp.REMOVE, EditOp.ADD, EditOp.REPLACE, EditOp.BACKGROUND, EditOp.UPSCALE}
)


def api_key() -> str:
    env = Path(__file__).resolve().parents[3] / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "GEMINI_API_KEY" and value.strip():
                return value.strip().strip("\"'")
    pytest.skip("no GEMINI_API_KEY in .env")


def test_the_provider_accepts_the_converted_schema() -> None:
    """If the conversion were wrong, this is where it shows: a 400 naming the schema."""
    made = plan(
        "get that ugly thing on the right out of the picture",
        available=IMPLEMENTED,
        completer=Gemini(api_key=api_key()),
        # Generous on purpose. In production this call is cut off at ten seconds and the
        # user gets a question instead — measured, the same model has answered in 2.7 s
        # and in 37.8 s. This test is about the schema being accepted, not the deadline.
        timeout_s=60.0,
    )

    assert made.route is Route.MODEL
    assert made.intent is not None
    assert made.intent.op is EditOp.REMOVE
    assert made.intent.target, "the model must copy the user's phrase, not invent one"
    assert made.prompt_tokens > 0, "the call cost something and the plan should say what"


def test_an_instruction_the_rules_answer_costs_nothing_even_with_a_model_present() -> None:
    made = plan("remove the car", available=IMPLEMENTED, completer=Gemini(api_key=api_key()))
    assert made.route is Route.RULE
    assert made.seconds < 0.05, "a rule answer must not be waiting on anything"
