"""Judging a finished edit, and deciding what to do about a bad one.

**Not the pass policy.** `editgpt_models.pipeline` already proposes, scores and rolls back
*within* one erase: it asks "did the second pass beat the first over the same region?" and
answers with photometric cost. It cannot ask the question this module exists for, because
it has never seen the instruction: **did the edit do what the user asked?** A fill can
agree perfectly with its surroundings and still leave the car standing.

So the two are stacked deliberately. Passes settle how well the region was filled; the
review settles whether filling that region was the job. The lever the review has that the
pass policy does not is the *selection* — the mask itself, which every pass takes as given.

The contract and the policy live here, in `core`, with no pixels in either: the scoring
needs an imaging stack and `editgpt_models` owns that, while `decide` is a table a person
can read. Keeping the table here also keeps it callable from anywhere — the worker runs it
today, an orchestrator process could run it tomorrow without loading OpenCV to do so.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from editgpt_core.spec import Constraints

UNCHANGED = "the edit changed almost nothing"
IMPLAUSIBLE = "the fill does not agree with what surrounds it"
STILL_THERE = "what you asked to remove is still in the picture"


@dataclass(frozen=True, slots=True)
class Verdict:
    """What the critic found. Empty `reasons` means the edit is good."""

    changed: float
    """Fraction of the edited region whose pixels actually differ.

    Phase 0's silent failure was a six-pixel mask returning the input image while every
    photometric score reported success — an edit agrees beautifully with its surroundings
    when it is its surroundings."""

    fill_cost: float
    """The finished fill's photometric cost, the same measure the pass policy uses."""

    still_there: float | None = None
    """Detector score for the target phrase *in the result*, or `None` if not checked.

    The semantic check, and the only one that can tell a plausible edit from a correct
    one. Costs a model swap, so the caller decides when it is worth paying for."""

    reasons: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.reasons


class Action(StrEnum):
    """What to do about a verdict.

    Deliberately four. There is no "try the other eraser": the pass policy already does
    that inside the edit, and a second copy of it here would be two components disagreeing
    about the same decision.

    There is also no "try the next candidate region". ADR-0003 asks *before* editing when
    two detections are close, which costs one tap; discovering it afterwards costs a whole
    edit and still ends with the same question. Asking early is strictly cheaper, so the
    late version is not worth having.
    """

    ACCEPT = "accept"
    WIDEN = "widen"
    """Grow the selection and edit again. The one lever the pass policy does not hold:
    a poor fill or a surviving object often means the object's boundary spilled outside
    the mask, and every pass took that mask as given.

    Only available for a region the *model* chose. A hand-drawn mask is not ours to grow:
    the user said which pixels, and an agent that quietly edits more than it was pointed
    at is worse than one that admits it could not do the job."""

    ASK = "ask"
    """Hand it back rather than return an edit we believe is wrong. The brush is the
    cheaper path from here, and a user who can see the problem can fix it in one stroke."""

    STOP = "stop"
    """Out of budget. Return the best attempt and say what is wrong with it."""


def decide(
    verdict: Verdict,
    *,
    attempt: int,
    constraints: Constraints,
    seconds_spent: float,
    can_widen: bool,
) -> Action:
    """What to do next. Four rules, total, in order.

    `attempt` counts edits already made, so the first review sees `attempt=1`. Budgets are
    the caller's own `Constraints` rather than numbers invented here — a retry loop that
    does not read the budget it was given is how an agent spends someone else's quota.

    `can_widen` is the caller's answer to "is the selection mine to change?" — false once
    it has already been widened, and false from the start when the user drew it. Asked as
    a question rather than inferred here, because who owns the region is a fact about
    where it came from and this module has never seen it.
    """
    if verdict.ok:
        return Action.ACCEPT
    if attempt > constraints.max_retries or seconds_spent >= constraints.max_seconds:
        return Action.STOP
    if can_widen:
        return Action.WIDEN
    # One structural lever, used once. Widening twice reaches for scene content the user
    # never asked to lose — the failure `residual_max_growth` caps a level down — and a
    # drawn region was never ours to widen at all.
    return Action.ASK
