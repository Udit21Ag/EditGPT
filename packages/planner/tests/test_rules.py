"""The instructions that must never reach a language model.

Every case here is a network call not made, a quota not burned and an answer that cannot
change between two identical requests. The table is the contract.
"""

from __future__ import annotations

import pytest
from editgpt_core import EditOp
from editgpt_planner import rules


@pytest.mark.parametrize(
    ("instruction", "op", "target", "content"),
    [
        ("remove the car", EditOp.REMOVE, "car", None),
        ("Remove the car from the image", EditOp.REMOVE, "car", None),
        ("erase the person on the left", EditOp.REMOVE, "person on the left", None),
        ("delete the sign", EditOp.REMOVE, "sign", None),
        ("get rid of the wires", EditOp.REMOVE, "wires", None),
        ("please remove the bin", EditOp.REMOVE, "bin", None),
        ("can you remove the bin?", EditOp.REMOVE, "bin", None),
        ("add a moustache", EditOp.ADD, None, "moustache"),
        ("put a hat on him", EditOp.ADD, None, "hat on him"),
        ("replace the sky with a sunset", EditOp.REPLACE, "sky", "a sunset"),
        ("swap the car for a bicycle", EditOp.REPLACE, "car", "a bicycle"),
    ],
)
def test_the_common_instructions_need_no_model(
    instruction: str, op: EditOp, target: str | None, content: str | None
) -> None:
    found = rules.match(instruction)
    assert found is not None, f"{instruction!r} fell through to the model"
    assert found.op is op
    assert found.target == target
    assert found.content == content


def test_replace_is_tried_before_remove() -> None:
    """ "replace the car with a tree" contains a removal. Order is the only thing stopping it."""
    found = rules.match("replace the car with a tree")
    assert found is not None
    assert found.op is EditOp.REPLACE
    assert found.content == "a tree"


@pytest.mark.parametrize(
    ("instruction", "colour"),
    [
        ("change the background to blue", "#3366ff"),
        ("make the background white", "#ffffff"),
        ("set the background to #cb00aa", "#cb00aa"),
        ("change the background to a grey", "#808080"),
    ],
)
def test_a_named_colour_is_resolved_here_rather_than_guessed_downstream(
    instruction: str, colour: str
) -> None:
    """TD-020 was this system accepting the word "blue" and painting green."""
    found = rules.match(instruction)
    assert found is not None
    assert found.op is EditOp.BACKGROUND
    assert found.colour == colour


def test_a_described_backdrop_is_carried_as_text_not_forced_into_a_colour() -> None:
    """ "a sunset" is a real request this operation cannot paint (TD-005). Guessing a hex
    value for it would be answering a different question."""
    found = rules.match("change the background to a sunset")
    assert found is not None
    assert found.colour is None
    assert found.content == "sunset"


@pytest.mark.parametrize(
    "instruction",
    [
        "upscale",
        "enlarge this",
        "increase the resolution",
        "make it sharper",
    ],
)
def test_upscaling_is_recognised_without_a_subject(instruction: str) -> None:
    found = rules.match(instruction)
    assert found is not None
    assert found.op is EditOp.UPSCALE


@pytest.mark.parametrize(
    "instruction",
    [
        "",
        "   ",
        "make it look nicer",
        "do something about the lighting",
        "the car",
        "what can you do?",
        "fix it",
    ],
)
def test_an_instruction_that_does_not_plainly_say_what_to_do_is_passed_on(
    instruction: str,
) -> None:
    """Declining is the point. Rules that guess are worse than no rules: they answer with
    the confidence of a regex and the accuracy of one."""
    assert rules.match(instruction) is None


def test_a_matched_shape_that_is_not_actionable_is_passed_on_rather_than_raised() -> None:
    """ "replace the car with" has the shape and not the substance."""
    assert rules.match("replace the car with") is None


def test_every_documented_example_still_matches() -> None:
    """`rules.examples` is quoted in documentation; it should not rot silently."""
    for instruction, op in rules.examples():
        found = rules.match(instruction)
        assert found is not None, instruction
        assert found.op is op, instruction
