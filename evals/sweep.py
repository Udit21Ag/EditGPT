"""Vary a case's prompt or box and score every variant.

Two different optimisation problems wear the same clothes:

* For **removal** the erasers never see text. A phrase only shapes the mask, so a variant
  is judged on mask agreement with the reference box and on the photometric cost of the
  resulting fill.
* For **addition** the prompt reaches the generative model verbatim, and the box decides
  whether the request is even answerable.

    uv run python -m evals.sweep i7 --prompts "the hummingbird" "the small bird on the branch"
    uv run python -m evals.sweep i4 --boxes 0.15,0.37,0.82,0.83 0.12,0.34,0.86,0.90
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np
from editgpt_core import EditOp
from editgpt_core.metrics import box_metrics, fill_metrics
from editgpt_models.pipeline import Erasers, erase

from evals.cases import Box, Case, load
from evals.run import OUT_DIR, Context, load_image, segment_for, strip


@dataclass
class Variant:
    case: str
    kind: str
    value: str
    bbox_iou: float
    inside_frac: float
    coverage: float
    cost: float
    seconds: float
    passes: str


def _mask_for(case: Case, ctx: Context, image: object, rgb: np.ndarray) -> tuple[np.ndarray, str]:
    return segment_for(case, ctx, image, rgb)


def run_variant(base: Case, ctx: Context, *, target: str | None, box: Box | None) -> Variant:
    case = Case(
        id=base.id,
        path=base.path,
        op=base.op,
        target=target if target is not None else base.target,
        content=base.content,
        box=box if box is not None else base.box,
        fill=base.fill,
        note=base.note,
    )
    image = load_image(case.path)
    rgb = np.asarray(image, dtype=np.uint8)

    started = time.monotonic()
    mask, _ = _mask_for(case, ctx, image, rgb)
    erasers = Erasers.from_sessions(ctx.session("migan"), ctx.session("lama"))
    outcome = erase(erasers, rgb, mask)
    elapsed = time.monotonic() - started

    scored = box_metrics(mask // 255, case.box, rgb.shape[:2]) if case.box else {}
    label = target if target is not None else ",".join(f"{v:g}" for v in (box or ()))
    strip(rgb, outcome.mask, outcome.image, f"sweep_{case.id}_{abs(hash(label)) % 10000}.png")

    return Variant(
        case=case.id,
        kind="prompt" if target is not None else "box",
        value=str(label),
        bbox_iou=float(scored.get("bbox_iou", 0.0)),
        inside_frac=float(scored.get("inside_frac", 0.0)),
        coverage=round(float((mask > 0).mean()), 4),
        cost=round(fill_metrics(outcome.image, rgb, outcome.mask).cost, 1),
        seconds=round(elapsed, 2),
        passes=outcome.summary(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_id")
    parser.add_argument("--prompts", nargs="*", default=[])
    parser.add_argument("--boxes", nargs="*", default=[], help="x0,y0,x1,y1 fractions")
    args = parser.parse_args()

    by_id = {c.id: c for c in load(include_deferred=True)}
    if args.case_id not in by_id:
        print(f"unknown case {args.case_id}; known: {sorted(by_id)}", file=sys.stderr)
        return 2
    base = by_id[args.case_id]
    if base.op is not EditOp.REMOVE:
        print(f"{base.id} is {base.op}; sweeping is implemented for removal", file=sys.stderr)
        return 2

    ctx = Context()
    results = [run_variant(base, ctx, target=p, box=None) for p in args.prompts]
    results += [
        run_variant(base, ctx, target=None, box=tuple(float(v) for v in b.split(",")))  # type: ignore[arg-type]
        for b in args.boxes
    ]
    if not results:
        results = [run_variant(base, ctx, target=base.target, box=None)]

    print(f"\n{base.id}: {base.note[:80]}")
    print(f"  {'iou':>5} {'inside':>7} {'cover':>7} {'cost':>7} {'sec':>6}  variant")
    for r in sorted(results, key=lambda r: (-r.bbox_iou, r.cost)):
        print(
            f"  {r.bbox_iou:5.3f} {r.inside_frac:7.3f} {r.coverage * 100:6.1f}% "
            f"{r.cost:7.1f} {r.seconds:6.2f}  {r.value[:58]}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"sweep_{base.id}.json"
    path.write_text(json.dumps([asdict(r) for r in results], indent=2))
    print(f"\n  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
