"""Three outcomes and no fourth: a rule answered, the model answered, or the user is asked."""

from __future__ import annotations

from typing import Any

import pytest
from editgpt_core import Constraints, EditOp
from editgpt_core.errors import ProviderError
from editgpt_planner import Route, plan

IMPLEMENTED = frozenset(
    {EditOp.REMOVE, EditOp.ADD, EditOp.REPLACE, EditOp.BACKGROUND, EditOp.UPSCALE}
)


class Fake:
    """A completer that answers with whatever it was given, and counts being asked."""

    def __init__(self, answer: str = '{"op": "remove", "target": "the car"}') -> None:
        self.answer = answer
        self.calls = 0
        self.schemas: list[dict[str, Any]] = []

    def complete(self, instruction: str, *, schema: dict[str, Any], timeout_s: float) -> str:
        self.calls += 1
        self.schemas.append(schema)
        return self.answer


class Broken:
    def complete(self, instruction: str, *, schema: dict[str, Any], timeout_s: float) -> str:
        raise ProviderError("429 quota exhausted")


def test_a_rule_answers_without_the_model_being_asked() -> None:
    """The reliability argument, asserted: the model is not merely unnecessary here, it is
    not called."""
    model = Fake()
    made = plan("remove the car", available=IMPLEMENTED, completer=model)

    assert made.route is Route.RULE
    assert made.actionable
    assert model.calls == 0, "a decidable instruction reached the model"


def test_the_model_is_asked_only_when_no_rule_matches() -> None:
    model = Fake('{"op": "remove", "target": "the thing blocking the view"}')
    made = plan("get that ugly thing out of the way", available=IMPLEMENTED, completer=model)

    assert made.route is Route.MODEL
    assert model.calls == 1
    assert made.intent is not None
    assert made.intent.target == "the thing blocking the view"


def test_the_model_is_constrained_by_a_schema_rather_than_asked_for_json() -> None:
    model = Fake()
    plan("do the needful to that", available=IMPLEMENTED, completer=model)

    schema = model.schemas[0]
    assert schema["properties"]["op"]["enum"], "the operation was not constrained to the enum"
    assert "$ref" not in str(schema), "a response schema cannot carry references"


def test_an_answer_that_is_not_a_usable_edit_becomes_a_question() -> None:
    """Constrained decoding fixes the shape, not the sense: `replace` with no replacement
    satisfies the schema and fails `Intent`."""
    made = plan(
        "swap it", available=IMPLEMENTED, completer=Fake('{"op": "replace", "target": "it"}')
    )

    assert made.route is Route.ASK
    assert made.question
    assert made.intent is None


def test_a_model_that_answers_with_prose_becomes_a_question_not_a_crash() -> None:
    made = plan("hmm", available=IMPLEMENTED, completer=Fake("I think you want to remove the car!"))
    assert made.route is Route.ASK


def test_a_model_that_is_down_degrades_to_a_question() -> None:
    """The model is a dependency, not a foundation. Rules keep working without it."""
    made = plan("that thing over there", available=IMPLEMENTED, completer=Broken())
    assert made.route is Route.ASK

    still_fine = plan("remove the car", available=IMPLEMENTED, completer=Broken())
    assert still_fine.route is Route.RULE


def test_with_no_model_configured_the_rules_are_the_whole_planner() -> None:
    assert plan("remove the car", available=IMPLEMENTED).route is Route.RULE
    assert plan("something vague", available=IMPLEMENTED).route is Route.ASK


def test_a_local_only_request_never_reaches_the_network() -> None:
    """`allow_remote=False` is a constraint on the edit; it has to bind the planner too, or
    an offline job still makes an API call to decide what it is."""
    model = Fake()
    made = plan(
        "unclear instruction",
        available=IMPLEMENTED,
        constraints=Constraints(allow_remote=False),
        completer=model,
    )

    assert made.route is Route.ASK
    assert model.calls == 0


@pytest.mark.parametrize("instruction", ["restyle this as a watercolour", "retouch the skin"])
def test_an_operation_with_no_implementation_is_answered_as_a_sentence(instruction: str) -> None:
    """TD-006: two of the seven operations are unimplemented. Accepting one and failing in
    a worker three steps later is the version of this that wastes a user's minute."""
    model = Fake('{"op": "restyle", "target": "the photo", "content": "a watercolour"}')
    made = plan(instruction, available=IMPLEMENTED, completer=model)

    assert made.route is Route.ASK
    assert made.question is not None
    assert "not implemented" in made.question


def test_the_planner_keeps_its_own_deadline_rather_than_the_edit_s() -> None:
    """Measured live: the same model answered in 2.7 s, 5.3 s and 37.8 s. A text call
    holding the whole job's 60-second budget is a user watching a spinner."""
    seen: list[float] = []

    class Timed:
        def complete(self, instruction: str, *, schema: dict[str, Any], timeout_s: float) -> str:
            seen.append(timeout_s)
            return '{"op": "upscale"}'

    plan("something vague", available=IMPLEMENTED, completer=Timed())
    assert seen == [10.0]

    plan(
        "something vague",
        available=IMPLEMENTED,
        constraints=Constraints(max_seconds=4),
        completer=Timed(),
    )
    assert seen[-1] == 4.0, "a caller asking for less must get less, not the default"


def test_the_plan_records_which_lane_answered_and_how_long_it_took() -> None:
    made = plan("remove the car", available=IMPLEMENTED)
    assert made.route is Route.RULE
    assert made.reason
    assert made.seconds >= 0.0
