"""The replan policy: four rules, and which one wins when two could apply."""

from __future__ import annotations

import pytest
from editgpt_core import Constraints
from editgpt_core.review import IMPLAUSIBLE, UNCHANGED, Action, Verdict, decide


def good() -> Verdict:
    return Verdict(changed=0.8, fill_cost=4.0)


def bad() -> Verdict:
    return Verdict(changed=0.0, fill_cost=40.0, reasons=(UNCHANGED, IMPLAUSIBLE))


def test_a_verdict_with_no_reasons_is_the_definition_of_ok() -> None:
    assert good().ok
    assert not bad().ok


def test_a_good_result_is_accepted_however_many_attempts_it_took() -> None:
    action = decide(
        good(), attempt=9, constraints=Constraints(), seconds_spent=999.0, can_widen=False
    )
    assert action is Action.ACCEPT


def test_a_bad_result_widens_the_selection_while_it_may() -> None:
    action = decide(
        bad(), attempt=1, constraints=Constraints(max_retries=2), seconds_spent=1.0, can_widen=True
    )
    assert action is Action.WIDEN


def test_a_selection_that_cannot_be_widened_is_handed_back_instead() -> None:
    """The user drew it, or it has already been grown once. Either way there is no lever
    left, and returning an edit we believe is wrong is worse than saying so."""
    action = decide(bad(), attempt=1, constraints=Constraints(), seconds_spent=1.0, can_widen=False)
    assert action is Action.ASK


@pytest.mark.parametrize(
    ("attempt", "seconds"),
    [(3, 1.0), (1, 60.0), (3, 60.0)],
)
def test_an_exhausted_budget_stops_before_anything_else_is_tried(
    attempt: int, seconds: float
) -> None:
    """Precedence, not preference: a retry loop that outruns its budget is the failure
    mode the budget exists for, so the check comes before both levers."""
    action = decide(
        bad(),
        attempt=attempt,
        constraints=Constraints(max_retries=2, max_seconds=60.0),
        seconds_spent=seconds,
        can_widen=True,
    )
    assert action is Action.STOP


def test_the_caller_s_own_budget_is_what_is_spent() -> None:
    """A constraint of zero retries means the first verdict is the last word."""
    action = decide(
        bad(),
        attempt=1,
        constraints=Constraints(max_retries=0),
        seconds_spent=0.1,
        can_widen=True,
    )
    assert action is Action.STOP
