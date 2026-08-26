"""Multi-pass erasure: propose, score, keep or roll back.

Every image goes through the models at least twice. That is a deliberate policy, and it
needs a guard, because Phase 0 measured that a *naive* second pass makes things slightly
worse (+0.6 to +3.5 cost on every case tried). So the second pass always runs, and its
output is kept only if it scores better than the first. Mandatory work, verified result.

A third pass runs when the second leaves the fill still above `ACCEPT_COST`, trying a
strategy the second did not.

The three strategies, and what Phase 0 found about each:

* ``escalate`` — the same mask through the other, stronger eraser. Worth it when the
  first fill is poor; the measured crossover is around cost 25.
* ``residual`` — detect what is *still* wrong and erase that. This is the only strategy
  that addresses cast shadows, because the detector needs the object gone before it has
  a clean surface to measure against. Must be capped on mask growth: ungated it flattens
  a scene while scoring better on raw cost.
* ``cross`` — the other eraser, different prior, same mask. Inconsistent on its own, but
  a reasonable third throw when the first two have not settled it.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from editgpt_core.metrics import MIN_MASK_PX, compare, fill_metrics

from editgpt_models.compositing import RGB, Mask, grow
from editgpt_models.erase import erase_lama, erase_migan, residual_mask

log = logging.getLogger(__name__)

Strategy = Literal["escalate", "residual", "cross"]

ESCALATE_COST = 25.0
"""Above this, the fast eraser's output is poor enough to pay for the slower one."""

ACCEPT_COST = 15.0
"""Below this the fill agrees with its surroundings well enough to stop working."""

RESIDUAL_MAX_GROWTH = 0.50
"""Cap on how much the residual pass may grow the mask.

Measured: +35% of the object's area cleaned up a car's cast shadow, while +78% and
+119% meant the detector had latched onto real scene content and the second erase
destroyed it. `fill_metrics` cannot make this call — a larger flat erase scores better
on it — so the cap is structural rather than learned at runtime.
"""

RESIDUAL_MIN_GROWTH = 0.02
"""Below this the residual is not worth an extra multi-second model call."""


@dataclass(frozen=True, slots=True)
class PassRecord:
    """One attempt, whether or not it was kept. The audit trail for a retry loop."""

    index: int
    strategy: str
    kept: bool
    delta: float
    """Score change against the incumbent. Negative is an improvement."""
    cost_after: float
    seconds: float
    mask_growth: float = 0.0
    attempted: bool = True
    """Whether a model actually ran. A strategy that did not apply was never a rollback."""
    note: str = ""


@dataclass
class EraseOutcome:
    image: RGB
    mask: Mask
    passes: list[PassRecord] = field(default_factory=list)

    @property
    def cost(self) -> float:
        return self.passes[-1].cost_after if self.passes else 0.0

    @property
    def kept_passes(self) -> int:
        return sum(1 for p in self.passes if p.kept)

    @property
    def rolled_back(self) -> int:
        """Passes where a model ran and its output lost. Excludes ones that never applied."""
        return sum(1 for p in self.passes if p.attempted and not p.kept)

    def summary(self) -> str:
        parts = []
        for record in self.passes:
            if record.kept:
                parts.append(record.strategy)
            elif record.attempted:
                parts.append(f"{record.strategy} (rolled back)")
            else:
                parts.append(f"{record.strategy} (n/a)")
        return " -> ".join(parts)


@dataclass
class Erasers:
    """The two local erasers, already loaded. Sessions come from a `ModelSlot`."""

    migan: Callable[[RGB, Mask], RGB]
    lama: Callable[[RGB, Mask], RGB]

    @classmethod
    def from_sessions(cls, migan_session: object, lama_session: object) -> Erasers:
        return cls(
            migan=lambda img, m: erase_migan(migan_session, img, m),
            lama=lambda img, m: erase_lama(lama_session, img, m),
        )


def _apply(
    strategy: Strategy,
    erasers: Erasers,
    source: RGB,
    incumbent: RGB,
    mask: Mask,
    used_lama: bool,
) -> tuple[RGB, Mask, str] | None:
    """Produce a candidate for `strategy`, or None when it does not apply."""
    if strategy == "escalate":
        return erasers.lama(source, mask), mask, "same mask, stronger eraser"

    if strategy == "cross":
        eraser = erasers.migan if used_lama else erasers.lama
        return eraser(incumbent, mask), mask, "other eraser over the incumbent"

    residual = residual_mask(incumbent, mask)
    growth = float((residual > 0).sum()) / max(float((mask > 0).sum()), 1)
    if not residual.any() or not (RESIDUAL_MIN_GROWTH <= growth <= RESIDUAL_MAX_GROWTH):
        return None
    return (
        erasers.lama(incumbent, residual),
        np.maximum(mask, residual),
        f"cleared residual, +{growth:.0%} area",
    )


def erase(
    erasers: Erasers,
    image: RGB,
    mask: Mask,
    *,
    min_passes: int = 2,
    max_passes: int = 3,
) -> EraseOutcome:
    """Erase `mask` from `image`, running at least `min_passes` model passes.

    Returns the best-scoring result, along with a record of every pass attempted
    including the ones rolled back — a pass that was tried and rejected is a finding,
    not noise, and the critic loop in Phase 7 needs to see it.
    """
    if int((mask > 0).sum()) < MIN_MASK_PX:
        raise ValueError(
            f"mask covers {int((mask > 0).sum())} px, below the {MIN_MASK_PX} px floor "
            "at which an edit could be visible"
        )

    started = time.monotonic()
    best = erasers.migan(image, mask)
    best_mask = mask
    used_lama = False
    cost = fill_metrics(best, image, mask).cost
    passes = [
        PassRecord(
            index=1,
            strategy="migan",
            kept=True,
            delta=0.0,
            cost_after=round(cost, 2),
            seconds=round(time.monotonic() - started, 3),
            note="fast path",
        )
    ]

    log.info(
        "erase.pass",
        extra={
            "index": 1,
            "strategy": "migan",
            "kept": True,
            "cost": round(cost, 2),
            "mask_coverage": round(float((mask > 0).mean()), 4),
            "seconds": passes[0].seconds,
        },
    )

    tried: set[Strategy] = set()
    for index in range(2, max_passes + 1):
        if index > min_passes and cost <= ACCEPT_COST:
            break

        order: list[Strategy] = (
            ["escalate", "residual", "cross"]
            if cost > ESCALATE_COST
            else ["residual", "escalate", "cross"]
        )
        choice = next((s for s in order if s not in tried), None)
        if choice is None:
            break

        pass_started = time.monotonic()
        produced = _apply(choice, erasers, image, best, best_mask, used_lama)
        tried.add(choice)
        if produced is None:
            passes.append(
                PassRecord(
                    index=index,
                    strategy=choice,
                    kept=False,
                    delta=0.0,
                    cost_after=round(cost, 2),
                    seconds=round(time.monotonic() - pass_started, 3),
                    attempted=False,
                    note="not applicable",
                )
            )
            log.info(
                "erase.pass",
                extra={
                    "index": index,
                    "strategy": choice,
                    "kept": False,
                    "attempted": False,
                    "cost": round(cost, 2),
                    "seconds": passes[-1].seconds,
                },
            )
            continue

        candidate, candidate_mask, note = produced
        region = np.maximum(best_mask, candidate_mask)
        delta = compare(candidate, best, image, region, best_mask, candidate_mask)
        growth = float((candidate_mask > 0).sum() - (best_mask > 0).sum()) / max(
            float((best_mask > 0).sum()), 1
        )

        keep = delta < 0
        if keep:
            best, best_mask = candidate, candidate_mask
            used_lama = used_lama or choice in {"escalate", "residual"}
            cost = fill_metrics(best, image, best_mask).cost

        record = PassRecord(
            index=index,
            strategy=choice,
            kept=keep,
            delta=round(delta, 2),
            cost_after=round(cost, 2),
            seconds=round(time.monotonic() - pass_started, 3),
            mask_growth=round(growth, 3),
            note=note if keep else f"{note}; rolled back",
        )
        passes.append(record)
        log.info(
            "erase.pass",
            extra={
                "index": record.index,
                "strategy": record.strategy,
                "kept": record.kept,
                "attempted": True,
                "delta": record.delta,
                "cost": record.cost_after,
                "mask_growth": record.mask_growth,
                "seconds": record.seconds,
            },
        )

    log.info(
        "erase.done",
        extra={
            "attempted": len(passes),
            "kept": sum(1 for p in passes if p.kept),
            "rolled_back": sum(1 for p in passes if p.attempted and not p.kept),
            "not_applicable": sum(1 for p in passes if not p.attempted),
            "final_cost": round(cost, 2),
            "seconds": round(time.monotonic() - started, 3),
        },
    )
    return EraseOutcome(image=best, mask=best_mask, passes=passes)


def prepare_mask(mask: Mask) -> Mask:
    """Grow a raw segmentation by the object-relative slack the erasers need."""
    return grow(mask)
