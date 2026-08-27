"""Doing an edit: one dispatch, one place.

This is the only implementation of "given an operation, an image and a target, produce
the edited image". It was extracted from `evals/run.py`, which held it while there was
exactly one caller. There are now two — the golden set and the Celery worker — and a
second copy of this policy would drift from the first within a week. The repository has
already grown near-duplicate mask utilities twice.

**What lives here and what does not.** This module decides *what happens* for an
operation: which model, in what order, over which mask, and what the result is called.
It does not decide where the pixels came from or where they go — the worker reads and
writes assets, the golden set reads and writes files, and neither concern belongs in a
routing table.

The routing itself is by **operation, not by difficulty** (see `harness/architecture.md`).
That is deliberate and load-bearing: removal never reaches a remote provider, because
Phase 0 measured that the free generative lane fills a masked hole with an *object*
matching the prompt rather than continuing the background. Asked to erase a car it
produced a stone slab, then a different car, then a boulder. ADR-0001 has the images.

Models arrive as a `Models` bundle rather than being loaded here. Loading is a lifetime
concern owned by `ModelSlot`, and a module that both loads and dispatches cannot be
tested without 550 MB of weights.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from editgpt_core import EditOp
from editgpt_core.errors import MaskTooSmallError
from editgpt_core.metrics import MIN_MASK_PX

from editgpt_models.compositing import RGB, Mask, erase_in_place
from editgpt_models.enhance import upscale
from editgpt_models.erase import flat_background_mask, recolour_background
from editgpt_models.pipeline import Erasers, erase, prepare_mask

log = logging.getLogger(__name__)


class Filler(Protocol):
    """Anything that can paint a described thing into a masked hole.

    Structurally identical to `editgpt_providers.Provider.fill`, and typed here rather
    than imported so `models` keeps no dependency on `providers` — the arrow points the
    other way. The worker supplies a bound provider; the golden set supplies its own.
    """

    def __call__(self, rgb: RGB, mask: Mask, prompt: str) -> RGB: ...


@dataclass(frozen=True, slots=True)
class Models:
    """Loaded sessions, supplied by whoever owns their lifetime.

    Each is optional so a caller that only ever erases need not load an upscaler, and a
    deployment with no provider configured still runs every local operation.
    """

    migan: Any = None
    lama: Any = None
    esrgan: Any = None
    fill: Filler | None = None

    def require(self, *names: str) -> None:
        """Fail before touching pixels, naming what is missing and for which operation."""
        missing = [n for n in names if getattr(self, n, None) is None]
        if missing:
            raise ValueError(
                f"this operation needs {', '.join(missing)}, which the caller did not load"
            )


@dataclass(frozen=True, slots=True)
class Edit:
    """A finished edit, and enough of a record to explain it afterwards."""

    image: RGB
    mask: Mask
    """What was actually changed, after dilation and any residual pass — not what was
    asked for. The two differ, and the audit trail should show the one that happened."""

    strategy: str
    """How the result was reached, e.g. `migan -> escalate (rolled back)`."""

    cost: float
    seconds: float
    detail: str = ""
    passes: list[dict[str, object]] = field(default_factory=list)
    """One entry per attempt, kept or rolled back. A rejected strategy is a finding the
    critic loop needs, not noise to discard."""


def remove(models: Models, image: RGB, mask: Mask, *, protect: Mask | None = None) -> Edit:
    """Erase the masked region, multi-pass, keeping only what verifies better.

    The mask is grown first by a fraction of the *object's* longest side. A pixel constant
    that works at 1024 px leaves a visible rectangular outline at 15.9 MP; this is one of
    the two Phase 0 fixes that must not be re-derived.

    `protect` is where that growth must stop: regions belonging to whatever is standing in
    front of, or up against, the target. It arrives from the caller rather than being
    computed here because finding it needs the SAM sessions, and this module dispatches
    rather than segments. `segment.occluder_shield` produces it.
    """
    models.require("migan", "lama")

    # Checked on what was *asked for*, before dilation. Measured: dilation has an 8 px
    # floor, so even a single-pixel mask grows to exactly 64 and clears `MIN_MASK_PX` —
    # a post-dilation check is a guard that can never fire for any non-empty mask.
    selected = int((mask > 0).sum())
    if selected < MIN_MASK_PX:
        raise MaskTooSmallError(
            f"the selected region is {selected} px, below the {MIN_MASK_PX} px floor at "
            "which an edit could be visible; select a larger area"
        )

    grown = prepare_mask(mask, protect=protect)
    started = time.monotonic()
    outcome = erase(Erasers.from_sessions(models.migan, models.lama), image, grown)
    return Edit(
        image=outcome.image,
        mask=outcome.mask,
        strategy=outcome.summary(),
        cost=round(outcome.cost, 2),
        seconds=round(time.monotonic() - started, 2),
        passes=[
            {
                "index": p.index,
                "strategy": p.strategy,
                "kept": p.kept,
                "attempted": p.attempted,
                "delta": p.delta,
                "cost_after": p.cost_after,
            }
            for p in outcome.passes
        ],
    )


def generate(models: Models, image: RGB, mask: Mask, prompt: str, *, via: str = "remote") -> Edit:
    """Paint `prompt` into the masked region using the remote lane.

    Everything around the model call — crop window, chroma match, feather — is shared
    with the local erasers through `erase_in_place`, which is what makes a side-by-side of
    the two lanes a comparison of *models* rather than of plumbing.
    """
    models.require("fill")
    assert models.fill is not None  # narrowed by require, restated for the type checker
    filler = models.fill

    started = time.monotonic()
    result = erase_in_place(lambda crop, m: filler(crop, m, prompt), image, mask)
    return Edit(
        image=result,
        mask=mask,
        # Named, not just "remote". "Which provider produced this" is the first question
        # asked about a bad generative result and the hardest to answer afterwards.
        strategy=f"{via} fill",
        cost=0.0,
        seconds=round(time.monotonic() - started, 2),
        detail=f"prompt: {prompt}",
    )


def recolour(image: RGB, colour: tuple[int, int, int], fallback: Mask | None) -> Edit:
    """Repaint the backdrop a flat colour, keeping the subject.

    Flood-filling inward from the border beats segmenting the subject where it applies:
    it keeps thin structures like table legs, and leaves no fringe of the old colour. It
    only applies to a uniform backdrop, so `fallback` — a segmentation of the subject —
    is used when the border is not uniform. There may be no fallback at all: flood fill
    needs no region, so a caller is entitled to ask for this with nothing selected, and
    the failure is only reached when *both* routes are unavailable. TD-005 has the
    underlying limitation.

    No model runs — which is why this takes no `Models`. A flat colour is compositing,
    not generation.
    """
    started = time.monotonic()
    backdrop = flat_background_mask(image)
    if backdrop is not None:
        subject = np.asarray(255 - backdrop, dtype=np.uint8)
        how = "flood fill from the border"
    elif fallback is not None:
        subject, how = fallback, "semantic mask (border was not uniform)"
    else:
        raise MaskTooSmallError(
            "the border is not a uniform backdrop, so the subject must be selected — "
            "brush it, or name it in the prompt"
        )

    return Edit(
        image=recolour_background(image, subject, colour),
        mask=np.asarray(255 - subject, dtype=np.uint8),
        strategy="composite",
        cost=0.0,
        seconds=round(time.monotonic() - started, 2),
        detail=how,
    )


def enlarge(models: Models, image: RGB) -> Edit:
    """Tiled 2x enhancement of the whole frame.

    Not an interactive operation: ~84 s for a 1024x1024 input on CPU (TD-010). It is only
    tolerable at all because it runs on a queue.
    """
    models.require("esrgan")
    started = time.monotonic()
    enlarged = upscale(models.esrgan, image)
    change = f"{image.shape[1]}x{image.shape[0]} -> {enlarged.shape[1]}x{enlarged.shape[0]}"
    return Edit(
        image=enlarged,
        mask=np.zeros(image.shape[:2], np.uint8),
        # The sizes belong in the strategy, not only in the detail: for an operation whose
        # entire purpose is changing them, a summary that omits them says nothing.
        strategy=f"esrgan-x2 tiled, {change}",
        cost=0.0,
        seconds=round(time.monotonic() - started, 2),
        detail=change,
    )


SUPPORTED: frozenset[EditOp] = frozenset(
    {EditOp.REMOVE, EditOp.ADD, EditOp.REPLACE, EditOp.BACKGROUND, EditOp.UPSCALE}
)
"""Operations with a working model behind them.

Kept beside the dispatch rather than in the gateway so the two cannot disagree about
what is possible — the gateway's `/capabilities` is checked against this.
"""


def execute(
    models: Models,
    op: EditOp,
    image: RGB,
    *,
    mask: Mask | None = None,
    protect: Mask | None = None,
    content: str | None = None,
    colour: tuple[int, int, int] = (0, 255, 0),
    via: str = "remote",
) -> Edit:
    """Run `op` over `image`, returning the edited image and what was done.

    `mask` is required by everything except `UPSCALE`, which acts on the whole frame.
    `content` describes what to paint for the generative operations, and `via` names the
    provider that served them so the audit trail records which one did. `protect` is
    honoured by `REMOVE` alone — the generative lane paints into the hole rather than
    continuing the background, so withholding part of that hole leaves a seam.
    """
    if op not in SUPPORTED:
        available = sorted(o.value for o in SUPPORTED)
        raise ValueError(f"{op} has no implementation; supported: {available}")

    if op is EditOp.UPSCALE:
        return enlarge(models, image)

    # BACKGROUND is the exception: flood-filling the backdrop needs no region, so a
    # missing mask is a legitimate request rather than an incomplete one.
    if op is EditOp.BACKGROUND:
        return recolour(image, colour, mask if mask is not None and mask.any() else None)

    if mask is None or not mask.any():
        raise MaskTooSmallError(f"{op} needs a region to act on and none was given")

    if op is EditOp.REMOVE:
        return remove(models, image, mask, protect=protect)

    # ADD and REPLACE. Both need something to put there; `EditSpec` already refuses a
    # generative op with no content, so reaching here without it is a caller bug.
    if not content:
        raise ValueError(f"{op} needs `content` describing what to put there")
    return generate(models, image, mask, content, via=via)
