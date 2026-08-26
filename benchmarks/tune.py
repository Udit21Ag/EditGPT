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

    uv run python -m benchmarks.tune --limit 400 --write
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
from editgpt_models.erase import make_session
from editgpt_models.registry import model_path
from editgpt_models.segment import load_clipseg, mask_from_seed, seed_from_text
from PIL import Image

from benchmarks.datasets import load_grounding
from benchmarks.grounding import WORKING_SIDE, _fit, mask_iou

OUT_DIR = Path(__file__).resolve().parent / "out"
HOLDOUT_FRACTION = 0.5


@dataclass
class Observation:
    """One sample, scored under both branches of the confidence gate."""

    id: str
    words: int
    confidence: float
    """SAM's own iou_prediction — the signal the gate reads."""
    iou_refined: float
    iou_seed: float

    @property
    def split(self) -> str:
        """Deterministic and content-based, so re-running cannot reshuffle the split."""
        digest = hashlib.sha256(self.id.encode()).hexdigest()
        return "holdout" if int(digest[:8], 16) % 100 < HOLDOUT_FRACTION * 100 else "fit"


def collect(limit: int) -> list[Observation]:
    processor, clipseg = load_clipseg()
    encoder = make_session(model_path("sam-encoder"))
    decoder = make_session(model_path("sam-decoder"))

    observations: list[Observation] = []
    for sample in load_grounding(limit=limit):
        image, truth = _fit(sample.image, sample.mask, WORKING_SIDE)
        heat = seed_from_text(processor, clipseg, Image.fromarray(image), sample.phrase)

        # min_iou=0.0 always accepts the refinement; 1.1 always rejects it. Running both
        # gives the outcome under every possible gate value from a single pass.
        refined = mask_from_seed(encoder, decoder, image, heat, min_iou=0.0)
        seed = mask_from_seed(encoder, decoder, image, heat, min_iou=1.1)
        if refined.source == "none":
            continue  # the phrase matched nothing; no gate value changes that

        observations.append(
            Observation(
                id=sample.id,
                words=len(sample.phrase.split()),
                confidence=round(refined.confidence, 4),
                iou_refined=round(mask_iou(refined.mask, truth)[0], 4),
                iou_seed=round(mask_iou(seed.mask, truth)[0], 4),
            )
        )
    return observations


def score(observations: list[Observation], gate: float) -> float:
    """Mean IoU the pipeline would achieve with this gate value."""
    if not observations:
        return 0.0
    return statistics.mean(
        o.iou_refined if o.confidence >= gate else o.iou_seed for o in observations
    )


def sweep(observations: list[Observation], candidates: np.ndarray) -> list[tuple[float, float]]:
    return [(float(g), round(score(observations, float(g)), 4)) for g in candidates]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=400)
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
    print(f"  {'gate':>6} {'fit mIoU':>10} {'holdout mIoU':>14}")
    for gate in (0.0, 0.5, 0.7, 0.8, current, best_gate, 1.01):
        print(f"  {gate:6.2f} {score(fit, gate):10.4f} {score(holdout, gate):14.4f}")

    print(f"\n  best on fit:      gate {best_gate:.2f} -> mIoU {best_fit:.4f}")
    print(f"  same gate, holdout:            mIoU {score(holdout, best_gate):.4f}")
    print(f"  current default {current:.2f}, holdout: mIoU {score(holdout, current):.4f}")
    delta = score(holdout, best_gate) - score(holdout, current)
    print(f"  improvement on held-out data:  {delta:+.4f}")

    rejected = [o for o in observations if o.confidence < current]
    if rejected:
        better = sum(o.iou_seed > o.iou_refined for o in rejected)
        print(
            f"\n  of {len(rejected)} samples the current gate rejects, the seed was actually "
            f"better in {better} ({better / len(rejected):.0%})"
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
                "observations": [asdict(o) for o in observations],
            },
            indent=2,
        )
    )

    if args.write:
        if delta <= 0:
            print("\n  not writing: the fitted value is no better than the default on holdout")
        else:
            fitted = Thresholds(
                min_sam_iou=best_gate,
                provenance=(
                    f"min_sam_iou fitted on RefCOCOg, {len(fit)} fit / {len(holdout)} holdout, "
                    f"{datetime.now(UTC).date().isoformat()}; other values are defaults"
                ),
            )
            fitted_path().parent.mkdir(parents=True, exist_ok=True)
            fitted_path().write_text(fitted.to_json())
            print(f"\n  wrote {fitted_path()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
