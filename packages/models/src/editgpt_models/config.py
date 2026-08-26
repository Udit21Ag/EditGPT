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
