"""Tunable parameters, as data rather than constants in source.

Every value here decides pipeline behaviour and every one was originally a literal chosen
by looking at 18 hand-built cases. That is fitting, and fitting without a held-out set
produces numbers that look better than they are.

So they live in a dataclass that can be loaded from a file produced by
`benchmarks.tune`, fitted on one split and reported on another. The defaults below are
the pre-benchmark values, kept so the system runs with no file present — but a default is
a starting point, not a justification.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Thresholds:
    """Decision points in the editing pipeline."""

    min_sam_iou: float = 0.85
    """Below this, SAM's own confidence says its refinement is not trustworthy and the
    coarser seed is kept instead."""

    min_box_score: float = 0.25
    """Below this, a Grounding DINO box is not a match for the phrase.

    The upstream demo's text threshold, used as the starting point rather than the
    answer: `benchmarks.tune` sweeps it on a fit split and reports it on a holdout."""

    ambiguity_margin: float = 0.15
    """Below this gap between the best and second-best detection, ask rather than answer.

    Measured on 250 held-out RefCOCOg samples: answering always hits 0.516, letting the
    user pick from five candidates hits 0.832, and this gate buys most of that for the
    least friction. The value is a starting point fitted on the whole set and reported in
    `docs/adr/0003-ask-when-unsure.md`; `benchmarks.tune` refits it on a split.

    Zero disables asking entirely, which is the behaviour before this existed."""

    candidates: int = 5
    """How many options to offer when asking.

    Held-out hit rate by K: 0.516, 0.692, 0.776, 0.816, 0.832. The curve is still rising
    at five, but a chooser with more than five thumbnails stops being a glance."""

    escalate_cost: float = 25.0
    """Above this photometric cost, the fast eraser's output is poor enough to pay for
    the slower one."""

    accept_cost: float = 15.0
    """Below this the fill agrees with its surroundings well enough to stop working."""

    residual_max_growth: float = 0.50
    """Cap on how much the residual pass may grow the mask."""

    residual_min_growth: float = 0.02
    """Below this the residual is not worth an extra multi-second model call."""

    dilate_frac: float = 0.05
    """Mask dilation as a fraction of the object's longest side."""

    shield: bool = False
    """Whether to withhold neighbouring objects from an erase mask's dilation.

    **Measured on the golden set and not adopted.** It does what it claims — the jumper's
    shoe in `i8` goes from 10.3% erased to 0.5%, and the horse in `i4` is untouched — but
    the erase it produces is worse overall, for a reason the shield cannot fix.

    Dilation had been quietly compensating for SAM *under*-segmenting an object's base.
    Withholding the ring where the Eiffel Tower meets the tree line stops the bleed onto
    the trees and leaves a pale ghost of the tower's base standing, which is far more
    visible than the smeared shoe it prevents. Across eleven removals: one clear
    improvement (`i6`, cost 28.5 -> 15.0), one clear regression (`i8`, and it is the case
    that motivated the work), and the rest unchanged.

    Kept, disabled, and tested rather than deleted — the same treatment `semantic.py`
    gets. The measurement is the asset; re-enabling is this flag. What would make it pay
    is fixing the under-segmentation it exposes, which is the other half of TD-004."""

    shield_max_inside: float = 0.30
    """Above this fraction inside the target, a probed region is one of its own parts.

    The one threshold in occluder shielding that decides quality rather than cost, and
    the one that had to be measured. MobileSAM is part-aware: probing near a horse's
    edge returns *the leg*, and at the original 0.80 that leg was shielded from the
    horse's own erasure — 26% of the animal left standing. At 0.30 the horse is untouched
    and the jumper's shoe in `i8` still goes from 10.3% erased to 5.6%. Values below 0.30
    measured identically on the golden set, so this is the loose end of a plateau rather
    than a peak fitted to eleven pictures."""

    shield_max_area: float = 0.25
    """Above this fraction of the frame, a probed region is scenery rather than an object.

    Probing beside a tower returns the sky; shielding it would withhold most of the
    dilation ring for no benefit."""

    shield_margin_frac: float = 0.25
    """Slack next to the target that is never shielded, as a fraction of the dilation.

    Dilation exists because a mask cut exactly on the silhouette leaves a halo of the
    object's own edge pixels. Shielding a touching neighbour must not take that back, so
    the first quarter of the growth is always allowed.

    This is also what makes the shield safe by construction rather than by a checked
    percentage: the margin covers the selected region, so a shield can only ever withhold
    pixels that dilation added, never the object the user asked to remove."""

    growth_penalty: float = 25.0
    """Cost charged per unit of relative mask growth when comparing candidates."""

    provenance: str = "defaults (not fitted)"
    """Where these values came from. A fitted file overwrites this with its split sizes
    and date, so a reader can tell a measured value from an inherited one."""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> Thresholds:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown threshold(s): {sorted(unknown)}")
        return cls(**data)


def fitted_path() -> Path:
    """Where a fitted set of thresholds is looked for. Override with EDITGPT_THRESHOLDS."""
    override = os.environ.get("EDITGPT_THRESHOLDS")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[4] / "benchmarks" / "fitted_thresholds.json"


def load_thresholds(path: Path | None = None) -> Thresholds:
    """Fitted thresholds if they exist, otherwise the documented defaults.

    Deliberately forgiving about a missing file and strict about a malformed one: running
    without fitted values is normal, while silently ignoring a typo in them is not.
    """
    target = path or fitted_path()
    if not target.exists():
        return Thresholds()
    return Thresholds.from_mapping(json.loads(target.read_text()))
