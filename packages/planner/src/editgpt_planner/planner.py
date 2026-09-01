"""Instruction in, typed plan out — with the model asked as little as possible.

Three outcomes and no fourth. The rules answer it, the model answers it, or the user is
asked a question. There is deliberately no "best guess" path: a planner that guesses turns
an ambiguous sentence into a confident wrong edit, and ADR-0003 already established that
this system asks instead.

What is recorded about an answer is **which lane produced it**, not a confidence score.
The lane is a fact; a number attached to a regex match would be decoration.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Collection
from dataclasses import dataclass
from enum import StrEnum

from editgpt_core import Constraints, EditOp
from editgpt_core.errors import ProviderError, ProviderExhaustedError
from pydantic import ValidationError

from editgpt_planner import rules
from editgpt_planner.intent import Intent
from editgpt_planner.llm import Completer, Completion, response_schema

log = logging.getLogger(__name__)


class Route(StrEnum):
    """Who answered."""

    RULE = "rule"
    MODEL = "model"
    ASK = "ask"


@dataclass(frozen=True, slots=True)
class Plan:
    """What to do, or what to ask.

    `intent` and `question` are exclusive: exactly one is set. A caller that reads the
    intent without checking the route cannot be given a half-answer to act on.
    """

    route: Route
    intent: Intent | None = None
    question: str | None = None
    seconds: float = 0.0
    reason: str = ""
    prompt_tokens: int = 0
    output_tokens: int = 0
    """What the model was asked and answered, in tokens. Zero when it was not asked."""

    @property
    def actionable(self) -> bool:
        return self.intent is not None


PLANNING_TIMEOUT_S = 10.0
"""The planner's own deadline, which is not the edit's.

`Constraints.max_seconds` bounds the *whole job* and defaults to 60. Handing that to a
text call as its timeout is how a user waits half a minute to be told what they asked for:
measured live, three instructions took 2.7 s, 5.3 s and **37.8 s** against the same model.
Planning is a sentence, not an edit. Past ten seconds the rules-and-ask path is a better
answer than the one still being generated, and the caller keeps the rest of its budget for
work that produces pixels.
"""

UNSUPPORTED = (
    "{op} is not implemented yet. This build can remove, add, replace, change the "
    "background and upscale."
)
OUT_OF_QUOTA = "the planner model was out of quota"
"""Recorded distinctly so a benchmark can tell "unmeasured" from "wrong"."""

UNCLEAR = (
    "I could not tell which edit that asks for. Try naming the operation and the "
    'subject — for example "remove the car on the left".'
)


def plan(
    instruction: str,
    *,
    available: Collection[EditOp],
    constraints: Constraints | None = None,
    completer: Completer | None = None,
    timeout_s: float | None = None,
) -> Plan:
    """Turn one instruction into one plan.

    `available` is the set of operations the caller can actually run. The planner does not
    know which ops have an implementation — that is a fact about the model package, and
    importing it here would put an imaging stack behind a text parser. Passing it in also
    means an instruction for an unimplemented operation is answered *as a sentence* rather
    than accepted and failed three steps later inside a worker.

    `completer` is optional on purpose: with no model configured the rules still answer
    most instructions, and the rest become a question. The planner degrades to a parser
    rather than to an outage.

    `timeout_s` overrides the planning deadline. For callers that are *measuring* rather
    than serving: a benchmark reporting how long the model takes cannot also be truncating
    it at ten seconds, or it reports its own deadline back to itself.
    """
    limits = constraints or Constraints()
    started = time.monotonic()

    matched = rules.match(instruction)
    if matched is not None:
        return _finish(matched, Route.RULE, started, available, reason="matched a rule")

    if completer is None or not limits.allow_remote:
        return Plan(
            route=Route.ASK,
            question=UNCLEAR,
            seconds=round(time.monotonic() - started, 3),
            reason="no rule matched and no model was available",
        )

    try:
        answer = completer.complete(
            instruction,
            schema=response_schema(Intent),
            timeout_s=timeout_s or min(limits.max_seconds, PLANNING_TIMEOUT_S),
        )
        proposed = Intent.model_validate_json(answer.text)
    except ProviderExhaustedError as error:
        # Out of quota is not the same as wrong, and a measurement that scores it as an
        # error is measuring the free tier rather than the planner.
        log.warning("planner.out_of_quota", extra={"error": str(error)[:200]})
        return _asked(started, OUT_OF_QUOTA, UNCLEAR)
    except ProviderError as error:
        # The model is a dependency, not a foundation. Its being down means the rules are
        # the whole planner for a while, which is a degradation the user can still use.
        log.warning("planner.model_unavailable", extra={"error": str(error)[:200]})
        return _asked(started, "the planner model was unavailable", UNCLEAR)
    except ValidationError as error:
        # Constrained decoding makes this rare, not impossible: the schema constrains the
        # shape, and `Intent` enforces what the shape cannot say — that a replacement
        # names both a target and what replaces it.
        log.info("planner.rejected", extra={"errors": error.error_count()})
        return _asked(started, "the model's answer was not a usable edit", UNCLEAR)

    return _finish(
        proposed,
        Route.MODEL,
        started,
        available,
        reason="answered by the model",
        usage=answer,
    )


def _finish(
    intent: Intent,
    route: Route,
    started: float,
    available: Collection[EditOp],
    *,
    reason: str,
    usage: Completion | None = None,
) -> Plan:
    seconds = round(time.monotonic() - started, 3)
    spent = usage or Completion(text="")
    if intent.op not in available:
        return Plan(
            route=Route.ASK,
            question=UNSUPPORTED.format(op=intent.op.value),
            seconds=seconds,
            reason=f"{intent.op.value} has no implementation here",
            prompt_tokens=spent.prompt_tokens,
            output_tokens=spent.output_tokens,
        )
    log.info(
        "planner.planned",
        extra={
            "route": route.value,
            "op": intent.op.value,
            "seconds": seconds,
            "tokens": spent.prompt_tokens + spent.output_tokens,
        },
    )
    return Plan(
        route=route,
        intent=intent,
        seconds=seconds,
        reason=reason,
        prompt_tokens=spent.prompt_tokens,
        output_tokens=spent.output_tokens,
    )


def _asked(started: float, reason: str, question: str) -> Plan:
    return Plan(
        route=Route.ASK,
        question=question,
        seconds=round(time.monotonic() - started, 3),
        reason=reason,
    )
