"""The instructions that need no model.

Seven operations, and almost everything a user types about them is decidable by reading
it. "remove the car" is not a planning problem; sending it to a language model buys a
network round trip, a quota burn, a latency floor of about a second and a small chance of
a different answer to the same sentence.

So rules run first and the model sees only the remainder. That is the whole reliability
argument: **every instruction answered here is one that cannot hallucinate, cannot time
out and cannot cost anything.** The rules are deliberately narrow — a phrase that does not
clearly mean one operation is not forced into one, it is passed on.

Ordered, because English overlaps: "replace the car with a tree" contains "the car" and
would match a bare removal if removal were tried first.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from editgpt_core import EditOp

from editgpt_planner.intent import Intent

COLOURS = {
    "white": "#ffffff",
    "black": "#000000",
    "red": "#ff0000",
    "green": "#2ea043",
    "blue": "#3366ff",
    "yellow": "#ffd700",
    "orange": "#ff8c00",
    "purple": "#8a2be2",
    "pink": "#ff69b4",
    "grey": "#808080",
    "gray": "#808080",
    "brown": "#8b4513",
    "cyan": "#00ffff",
    "magenta": "#ff00ff",
    "teal": "#008080",
    "navy": "#000080",
    "beige": "#f5f5dc",
}
"""Named colours a user might actually type, resolved to the hex `EditSpec` carries.

A fixed table rather than a model call: "blue" has no interesting interpretation, and
TD-020 was the version of this system that accepted the word and painted green anyway."""

_ARTICLE = r"(?:the |a |an |my |this )?"


def _clean(text: str) -> str:
    """Trim the politeness and punctuation that carry no meaning for the parse.

    Both ends. "change the background to teal please" put the politeness *inside* the
    captured colour and produced no colour at all — found by `benchmarks.planner`, not by
    the thirty unit tests that came before it.
    """
    text = text.strip().strip(".!?").strip()
    text = re.sub(
        r"^(?:please|can you|could you|would you|i want you to|i'd like you to)\s+",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(r"[,\s]+(?:please|thanks|thank you|for me)$", "", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


VAGUE = re.compile(r"^(?:whatever|something|anything|stuff|it|that thing|the thing)\b", re.I)
"""Subjects that name nothing a detector could ground.

"erase whatever that grey box is at the bottom" matched the removal shape and carried the
filler into the phrase. A rule answering with `whatever that grey box is` has not
understood the sentence, it has matched a verb. These go to the model, which is what the
model is for."""


def _subject(text: str) -> str:
    """The object of the verb, with a trailing modifier left intact.

    "the car on the left" stays whole: narrowing it to "car" would throw away the only
    part of the phrase that says *which* car, which is exactly what the picker exists to
    resolve.
    """
    return re.sub(
        r"\s+(?:from|in|out of)\s+(?:the\s+)?(?:image|photo|picture)$", "", text.strip(), flags=re.I
    )


def _replace(text: str) -> Intent | None:
    # "put a modern sofa where the armchair is" is a replacement wearing an addition's
    # verb. Without this it matched `_add` and quietly dropped the armchair.
    where = re.match(
        rf"^(?:put|place|add)\s+(.+?)\s+where\s+{_ARTICLE}(.+?)(?:\s+(?:is|was|used to be))?$",
        text,
        re.I,
    )
    if where:
        return Intent(
            op=EditOp.REPLACE,
            target=_subject(where.group(2)),
            content=where.group(1).strip(),
        )

    match = re.match(
        rf"^(?:replace|swap(?:\s+out)?|change|turn)\s+{_ARTICLE}(.+?)\s+(?:with|for|into)\s+(.+)$",
        text,
        re.I,
    )
    if not match:
        return None
    return Intent(
        op=EditOp.REPLACE, target=_subject(match.group(1)), content=match.group(2).strip()
    )


def _background(text: str) -> Intent | None:
    match = re.match(
        rf"^(?:change|make|set|turn)\s+{_ARTICLE}background\s+(?:to|into)?\s*(.+)$", text, re.I
    )
    if not match:
        return None
    described = _clean(match.group(1)).lower().removeprefix("a ").removeprefix("the ")
    hex_match = re.fullmatch(r"#[0-9a-fA-F]{6}", described)
    if hex_match:
        return Intent(op=EditOp.BACKGROUND, colour=described.lower())
    if described in COLOURS:
        return Intent(op=EditOp.BACKGROUND, colour=COLOURS[described])
    # A described backdrop — "a sunset" — is not refused here. It is a legitimate request
    # this operation cannot yet paint (TD-005), and the caller decides what to do about a
    # `content` it has no implementation for.
    return Intent(op=EditOp.BACKGROUND, content=described)


def _remove(text: str) -> Intent | None:
    match = re.match(
        rf"^(?:remove|erase|delete|get rid of|take out|clean up)\s+{_ARTICLE}(.+)$", text, re.I
    )
    if not match or VAGUE.match(match.group(1)):
        return None
    return Intent(op=EditOp.REMOVE, target=_subject(match.group(1)))


def _add(text: str) -> Intent | None:
    match = re.match(
        rf"^(?:add|put|place|insert|give (?:him|her|them|it))\s+{_ARTICLE}(.+)$", text, re.I
    )
    if not match or VAGUE.match(match.group(1)):
        return None
    return Intent(op=EditOp.ADD, content=_subject(match.group(1)))


def _upscale(text: str) -> Intent | None:
    if re.match(
        r"^(?:upscale|enlarge|upsample|2x|double the resolution|increase the resolution)\b",
        text,
        re.I,
    ):
        return Intent(op=EditOp.UPSCALE)
    if re.match(
        r"^make (?:it|this|the image|the photo) (?:bigger|larger|sharper|higher resolution)$",
        text,
        re.I,
    ):
        return Intent(op=EditOp.UPSCALE)
    return None


RULES = (_replace, _background, _remove, _add, _upscale)
"""Order matters. `_replace` before `_remove` because "replace the car with a tree"
contains a removal; `_background` before both because it names a region, not an object."""


def match(instruction: str) -> Intent | None:
    """The intent this instruction plainly states, or `None` to send it to the model.

    Returning `None` is not a failure. It is the rules declining to guess, which is the
    only reason a planner behind them is worth having.
    """
    text = _clean(instruction)
    if not text:
        return None
    for rule in RULES:
        try:
            found = rule(text)
        except ValueError:
            # A rule matched the shape but produced something `Intent` refuses — "replace
            # the car with" has a target and no content. Not actionable, and not the
            # model's problem either; the caller asks the user.
            return None
        if found is not None:
            return found
    return None


def examples() -> Iterator[tuple[str, EditOp]]:
    """Instructions the rules are expected to answer, for tests and for documentation."""
    yield from (
        ("remove the car", EditOp.REMOVE),
        ("erase the person on the left", EditOp.REMOVE),
        ("add a moustache", EditOp.ADD),
        ("replace the sky with a sunset", EditOp.REPLACE),
        ("change the background to blue", EditOp.BACKGROUND),
        ("upscale", EditOp.UPSCALE),
    )
