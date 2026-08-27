"""Fit thresholds on one split, report them on another.

The thresholds in `editgpt_models.config` were originally literals chosen by looking at
18 hand-built cases. That is fitting, and fitting without a held-out split produces
numbers that flatter themselves. This module does it properly:

1. run the models **once** per sample, recording the outcome under *every* branch the
   threshold could take;
2. split deterministically into fit and holdout;
3. sweep the threshold on the fit split only;
4. report the chosen value's performance on holdout, where nothing was fitted.

Because both branches are recorded up front, the sweep itself is arithmetic — no model
runs again, so trying 200 candidate values costs nothing.

**Two thresholds, and only one of them can be fitted this way.**

`min_sam_iou` chooses between two answers — SAM's traced boundary and the detector's
rectangle — and mIoU genuinely prefers one or the other per sample, so the optimum is
interior and the sweep means something.

`min_box_score` chooses between answering and abstaining. Raising it converts answers
into empty masks, which score IoU 0, so mIoU is monotonically non-increasing in it and
the "optimal" value is always zero. That is a real property of the objective, not a
missing feature: mIoU cannot express that a confidently wrong mask erases the wrong
object while an abstention merely shows the user a brush. The sweep below reports its
curve so the trade is visible and a product decision has data, and the value is left at
the upstream default rather than fitted to a number that only looks measured.

    uv run python -m benchmarks.tune --limit 250 --write
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from editgpt_models.config import Thresholds, fitted_path
from editgpt_models.detect import detect, load_detector
from editgpt_models.erase import make_session
from editgpt_models.registry import model_path
from editgpt_models.segment import mask_from_box

from benchmarks.datasets import load_grounding
from benchmarks.grounding import WORKING_SIDE, _fit, mask_iou

OUT_DIR = Path(__file__).resolve().parent / "out"
HOLDOUT_FRACTION = 0.5


@dataclass
class Observation:
    """One sample, scored under every branch the gates could take."""

    id: str
    words: int
    box_score: float
    """The detector's confidence — the signal `min_box_score` reads."""

    sam_confidence: float
    """SAM's own iou_prediction — the signal `min_sam_iou` reads."""

    iou_sam: float
    """IoU if SAM's traced mask is kept."""

    iou_box: float
    """IoU if the detector's rectangle is kept instead."""

    @property
    def split(self) -> str:
        """Deterministic and content-based, so re-running cannot reshuffle the split."""
        digest = hashlib.sha256(self.id.encode()).hexdigest()
        return "holdout" if int(digest[:8], 16) % 100 < HOLDOUT_FRACTION * 100 else "fit"


def box_mask(box: tuple[float, float, float, float], shape: tuple[int, int]) -> np.ndarray:
    """The detector's rectangle, filled — the fallback when SAM is not confident."""
    height, width = shape
    x0, y0, x1, y1 = box
    filled = np.zeros((height, width), np.uint8)
    filled[round(y0 * height) : round(y1 * height), round(x0 * width) : round(x1 * width)] = 255
    return filled


def collect(limit: int) -> list[Observation]:
    detector = load_detector()
    encoder = make_session(model_path("sam-encoder"))
    decoder = make_session(model_path("sam-decoder"))

    observations: list[Observation] = []
    for sample in load_grounding(limit=limit):
        image, truth = _fit(sample.image, sample.mask, WORKING_SIDE)
        height, width = image.shape[:2]

        # min_score=0 records every sample regardless of the gate, which is what lets the
        # `min_box_score` curve be swept afterwards without running the model again.
        found = detect(detector, image, sample.phrase, min_score=0.0, top_k=1)
        if not found:
            continue
        best = found[0]

        refined = mask_from_box(encoder, decoder, image, best.box)
        rectangle = box_mask(best.box, (height, width))

        observations.append(
            Observation(
                id=sample.id,
                words=len(sample.phrase.split()),
                box_score=round(best.score, 4),
                sam_confidence=round(refined.confidence, 4),
                iou_sam=round(mask_iou(refined.mask, truth)[0], 4),
                iou_box=round(mask_iou(rectangle, truth)[0], 4),
            )
        )
    return observations


def score(observations: list[Observation], gate: float) -> float:
    """Mean IoU the pipeline would achieve with this `min_sam_iou`."""
    if not observations:
        return 0.0
    return statistics.mean(
        o.iou_sam if o.sam_confidence >= gate else o.iou_box for o in observations
    )


def sweep(observations: list[Observation], candidates: np.ndarray) -> list[tuple[float, float]]:
    return [(float(g), round(score(observations, float(g)), 4)) for g in candidates]


def abstention_curve(
    observations: list[Observation], sam_gate: float, candidates: np.ndarray
) -> list[dict[str, float]]:
    """What `min_box_score` costs and buys, so the trade-off is visible rather than fitted.

    Reported, not optimised — see the module docstring for why mIoU cannot choose it.
    """
    curve = []
    for gate in candidates:
        answered = [o for o in observations if o.box_score >= gate]
        chosen = [o.iou_sam if o.sam_confidence >= sam_gate else o.iou_box for o in answered]
        curve.append(
            {
                "gate": round(float(gate), 3),
                "abstained": round(1 - len(answered) / max(len(observations), 1), 4),
                "mIoU_overall": round(sum(chosen) / max(len(observations), 1), 4),
                "mIoU_answered": round(statistics.mean(chosen), 4) if chosen else 0.0,
                "precision@0.5_answered": (
                    round(sum(i >= 0.5 for i in chosen) / len(chosen), 4) if chosen else 0.0
                ),
            }
        )
    return curve


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=250)
    parser.add_argument("--write", action="store_true", help="persist the fitted thresholds")
    args = parser.parse_args()

    started = time.monotonic()
    observations = collect(args.limit)
    if len(observations) < 40:
        print(f"only {len(observations)} usable samples; too few to fit", file=sys.stderr)
        return 2

    fit = [o for o in observations if o.split == "fit"]
    holdout = [o for o in observations if o.split == "holdout"]
    candidates = np.round(np.arange(0.0, 1.01, 0.01), 2)
    curve = sweep(fit, candidates)
    best_gate, best_fit = max(curve, key=lambda pair: pair[1])

    current = Thresholds().min_sam_iou
    print(f"\ncollected {len(observations)} samples in {time.monotonic() - started:.0f}s")
    print(f"  fit {len(fit)}   holdout {len(holdout)}   (deterministic, by id hash)\n")
    print("  min_sam_iou: keep SAM's mask above the gate, the detector's box below it")
    print(f"  {'gate':>6} {'fit mIoU':>10} {'holdout mIoU':>14}")
    for gate in sorted({0.0, 0.5, 0.7, 0.8, current, best_gate, 1.01}):
        print(f"  {gate:6.2f} {score(fit, gate):10.4f} {score(holdout, gate):14.4f}")

    print(f"\n  best on fit:      gate {best_gate:.2f} -> mIoU {best_fit:.4f}")
    print(f"  same gate, holdout:            mIoU {score(holdout, best_gate):.4f}")
    print(f"  current default {current:.2f}, holdout: mIoU {score(holdout, current):.4f}")
    delta = score(holdout, best_gate) - score(holdout, current)
    print(f"  improvement on held-out data:  {delta:+.4f}")

    box_gates = np.round(np.arange(0.0, 0.71, 0.05), 2)
    box_curve = abstention_curve(holdout, best_gate, box_gates)
    print("\n  min_box_score: reported, not fitted — mIoU cannot price an abstention")
    print(f"  {'gate':>6} {'abstained':>10} {'mIoU all':>10} {'mIoU ans':>10} {'P@0.5 ans':>10}")
    for row in box_curve:
        print(
            f"  {row['gate']:6.2f} {row['abstained']:10.2%} {row['mIoU_overall']:10.4f} "
            f"{row['mIoU_answered']:10.4f} {row['precision@0.5_answered']:10.4f}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "tuning.json").write_text(
        json.dumps(
            {
                "n": len(observations),
                "fit": len(fit),
                "holdout": len(holdout),
                "curve_on_fit": curve,
                "best_gate": best_gate,
                "holdout_at_best": round(score(holdout, best_gate), 4),
                "holdout_at_default": round(score(holdout, current), 4),
                "box_score_curve_on_holdout": box_curve,
                "observations": [asdict(o) for o in observations],
            },
            indent=2,
        )
    )

    if args.write:
        if delta <= 0:
            print("\n  not writing: the fitted value is no better than the default on holdout")
        else:
            existing = Thresholds()
            fitted = Thresholds(
                **{
                    **asdict(existing),
                    "min_sam_iou": best_gate,
                    "provenance": (
                        f"min_sam_iou fitted on RefCOCOg via the Grounding DINO path, "
                        f"{len(fit)} fit / {len(holdout)} holdout, holdout mIoU "
                        f"{score(holdout, best_gate):.4f} against {score(holdout, current):.4f} "
                        f"at the default; {datetime.now(UTC).date().isoformat()}. "
                        f"Other values are defaults."
                    ),
                }
            )
            fitted_path().parent.mkdir(parents=True, exist_ok=True)
            fitted_path().write_text(fitted.to_json())
            print(f"\n  wrote {fitted_path()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
