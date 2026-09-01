"""Scoring a finished edit against what was asked for.

Three checks, cheapest first, and the expensive one is optional.

1. **Did anything change?** Pure arithmetic on the two images. Phase 0's worst failure was
   silent: a six-pixel mask returned the input while every photometric score reported
   success, because an unedited region agrees perfectly with its surroundings.
2. **Is the fill plausible?** `fill_metrics` again, at the end rather than between passes.
   Seeing a cost above `escalate_cost` *here* means the passes ran and did not fix it.
3. **Is the thing still there?** Re-run the detector on the *result* with the same phrase.
   This is the only check that knows what the user asked for, and the only one that can
   tell a beautiful fill of the wrong region from a correct edit.

The third costs a model swap on a machine that holds one model at a time, so it is the
caller's decision: `editors` asks for it when a retry is still affordable, because a check
nobody can act on is seven seconds spent to feel informed.

Thresholds are `Thresholds`, not literals. `escalate_cost` and `min_box_score` already
exist with their provenance recorded; a second set of numbers meaning the same thing is
how two components come to disagree about what "good" is.
"""

from __future__ import annotations

import logging

import numpy as np
from editgpt_core.metrics import fill_metrics
from editgpt_core.review import IMPLAUSIBLE, STILL_THERE, UNCHANGED, Verdict

from editgpt_models.compositing import RGB, Mask
from editgpt_models.config import Thresholds, load_thresholds
from editgpt_models.detect import Detector, detect

log = logging.getLogger(__name__)

CHANGED_LEVELS = 8
"""Per-channel difference, out of 255, at which a pixel counts as changed.

Above the noise floor of a re-encode — JPEG requantisation moves a pixel by a level or
two — and far below anything a person would call an edit."""

MIN_CHANGED = 0.05
"""Fraction of the edited region that must actually differ for an edit to have happened.

Not zero: an eraser that continues a flat sky legitimately reproduces some of what it
replaced, and a mask always includes a dilated margin whose pixels are supposed to survive
untouched. Five percent of the region is well below any real edit and well above both."""

OVERLAP = 0.30
"""Fraction of a detection that must fall inside the edited region to be *that* object.

A detector run on the result finds every instance of the phrase in the picture. Only one
of them is the one we were asked to remove; a second car parked elsewhere is not evidence
of failure."""


def critique(
    source: RGB,
    result: RGB,
    mask: Mask,
    *,
    target: str | None = None,
    detector: Detector | None = None,
    thresholds: Thresholds | None = None,
) -> Verdict:
    """Judge `result` against `source` over `mask`, and against `target` if asked to.

    `mask` is what the edit actually changed — `Edit.mask`, after dilation and any
    residual pass — not what was requested. Scoring the requested region would measure a
    different area than the one that was edited.
    """
    tuning = thresholds or load_thresholds()
    inside = mask > 0
    if not inside.any():
        # Nothing was selected; there is nothing to judge. `MaskTooSmallError` is raised
        # far earlier for this, so reaching here means an operation with no region — the
        # backdrop fill, for one — and it is not this function's place to fail it.
        return Verdict(changed=0.0, fill_cost=0.0)

    difference = np.abs(result.astype(np.int16) - source.astype(np.int16)).max(axis=2)
    changed = float((difference[inside] > CHANGED_LEVELS).mean())
    cost = fill_metrics(result, source, mask).cost

    reasons: list[str] = []
    if changed < MIN_CHANGED:
        reasons.append(UNCHANGED)
    if cost > tuning.escalate_cost:
        reasons.append(IMPLAUSIBLE)

    still_there: float | None = None
    if detector is not None and target:
        still_there = _survives(detector, result, mask, target)
        if still_there >= tuning.min_box_score:
            reasons.append(STILL_THERE)

    verdict = Verdict(
        changed=round(changed, 4),
        fill_cost=round(cost, 2),
        still_there=still_there,
        reasons=tuple(reasons),
    )
    log.info(
        "critic.verdict",
        extra={
            "ok": verdict.ok,
            "changed": verdict.changed,
            "fill_cost": verdict.fill_cost,
            "still_there": verdict.still_there,
            "reasons": list(verdict.reasons),
        },
    )
    return verdict


def _survives(detector: Detector, result: RGB, mask: Mask, phrase: str) -> float:
    """The best detector score for `phrase` inside the edited region, after the edit.

    Zero means the phrase no longer matches anything there, which is what a successful
    removal looks like from the outside.
    """
    height, width = result.shape[:2]
    ys, xs = np.nonzero(mask > 0)
    region = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

    best = 0.0
    for found in detect(detector, result, phrase, top_k=5):
        x0, y0, x1, y1 = found.box
        box = (x0 * width, y0 * height, x1 * width, y1 * height)
        if _inside(box, region) >= OVERLAP:
            best = max(best, found.score)
    return round(best, 4)


def _inside(box: tuple[float, float, float, float], region: tuple[int, int, int, int]) -> float:
    """How much of `box` falls within `region`, as a fraction of the box's own area."""
    width = max(0.0, min(box[2], region[2]) - max(box[0], region[0]))
    height = max(0.0, min(box[3], region[3]) - max(box[1], region[1]))
    area = max((box[2] - box[0]) * (box[3] - box[1]), 1e-6)
    return (width * height) / area
