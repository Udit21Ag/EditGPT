"""Is disambiguation worth building, and when should it fire?

TD-015 established that 36% of held-out phrases score below IoU 0.1, identically for two
unrelated grounding models, because both ground the noun and then pick an arbitrary
instance. The proposed remedy is to stop guessing: return the ranked candidates and let
the user tap the right one.

That remedy rests on an assumption nobody has checked — **that a correct candidate is
there to be picked.** If the detector's second and third boxes are no better than its
first, showing them is a modal in front of every edit for nothing. So this measures the
ceiling before anything is built:

* ``hit@1`` — today's accuracy, the baseline.
* ``hit@K`` — how often *any* of the top K candidates is right. The most a perfect
  picker could reach, and therefore the most disambiguation can ever be worth.
* the **margin** between the best and second-best detection score, tested as a predictor
  of the top-1 being wrong. A gate needs a signal; this asks whether there is one.

Deliberately reports, and does not fit. Choosing where the gate sits is `tune.py`'s job,
on a split, once this says there is something to gate on.

    uv run python -m benchmarks.ambiguity --limit 250 --top-k 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from editgpt_models.detect import detect, load_detector
from editgpt_models.erase import make_session
from editgpt_models.registry import model_path
from editgpt_models.segment import mask_from_box

from benchmarks.datasets import load_grounding
from benchmarks.grounding import WORKING_SIDE, _fit, mask_iou

OUT_DIR = Path(__file__).resolve().parent / "out"
HIT = 0.5
"""IoU at which a mask counts as the right object.

The referring-segmentation literature's precision@0.5, so these numbers sit beside the
published ones rather than needing a translation.
"""


@dataclass
class Sample:
    """One phrase, with every candidate the detector offered scored against truth."""

    id: str
    phrase: str
    words: int
    scores: list[float]
    """Detection scores, best first."""

    ious: list[float]
    """Mask IoU of each candidate against ground truth, in the same order."""

    @property
    def margin(self) -> float:
        """How much the best candidate beat the runner-up.

        Zero when there is only one candidate: nothing to be ambiguous *between*.
        """
        return self.scores[0] - self.scores[1] if len(self.scores) > 1 else 0.0

    @property
    def top1_correct(self) -> bool:
        return bool(self.ious and self.ious[0] >= HIT)

    def hit_at(self, k: int) -> bool:
        return any(iou >= HIT for iou in self.ious[:k])

    def best_at(self, k: int) -> float:
        return max(self.ious[:k], default=0.0)


def collect(limit: int, top_k: int) -> list[Sample]:
    detector = load_detector()
    encoder = make_session(model_path("sam-encoder"))
    decoder = make_session(model_path("sam-decoder"))

    samples: list[Sample] = []
    for record in load_grounding(limit=limit):
        image, truth = _fit(record.image, record.mask, WORKING_SIDE)

        # min_score=0 so the ranking is complete: a gate applied here would hide exactly
        # the marginal candidates this is trying to measure.
        found = detect(detector, image, record.phrase, min_score=0.0, top_k=top_k)
        if not found:
            continue

        ious = [
            round(mask_iou(mask_from_box(encoder, decoder, image, c.box).mask, truth)[0], 4)
            for c in found
        ]
        samples.append(
            Sample(
                id=record.id,
                phrase=record.phrase,
                words=len(record.phrase.split()),
                scores=[c.score for c in found],
                ious=ious,
            )
        )
    return samples


def ceiling(samples: list[Sample], top_k: int) -> list[dict[str, float]]:
    """What disambiguation could reach at each K, if the user always picked correctly."""
    return [
        {
            "k": k,
            "hit_rate": round(sum(s.hit_at(k) for s in samples) / len(samples), 4),
            "mIoU": round(statistics.mean(s.best_at(k) for s in samples), 4),
        }
        for k in range(1, top_k + 1)
    ]


def margin_signal(samples: list[Sample]) -> dict[str, object]:
    """Does a narrow margin predict the top guess being wrong?

    Reported as the hit rate on each side of a sweep of cut-points rather than as a single
    correlation: a gate is a decision, and what matters is whether splitting on it
    separates the two groups, not how linear the relationship is.
    """
    rows = []
    for cut in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
        narrow = [s for s in samples if s.margin < cut]
        wide = [s for s in samples if s.margin >= cut]
        if not narrow or not wide:
            continue
        rows.append(
            {
                "margin_below": cut,
                "share_asked": round(len(narrow) / len(samples), 4),
                "top1_hit_when_narrow": round(sum(s.top1_correct for s in narrow) / len(narrow), 4),
                "top1_hit_when_wide": round(sum(s.top1_correct for s in wide) / len(wide), 4),
            }
        )
    return {"cuts": rows}


def report(samples: list[Sample], top_k: int) -> dict[str, object]:
    curve = ceiling(samples, top_k)
    base, best = curve[0], curve[-1]
    return {
        "n": len(samples),
        "ceiling": curve,
        "gain_from_picking": round(best["hit_rate"] - base["hit_rate"], 4),
        "mIoU_gain": round(best["mIoU"] - base["mIoU"], 4),
        "margin": margin_signal(samples),
        "median_candidates": statistics.median(len(s.scores) for s in samples),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=250)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--min-samples", type=int, default=40, help="refuse to conclude from fewer than this"
    )
    args = parser.parse_args()

    started = time.monotonic()
    samples = collect(args.limit, args.top_k)
    if len(samples) < args.min_samples:
        print(f"only {len(samples)} usable samples; too few to conclude", file=sys.stderr)
        return 2

    stats = report(samples, args.top_k)
    print(f"\nRefCOCOg, held out, n={stats['n']}  ({time.monotonic() - started:.0f}s)")
    print("\n  what disambiguation could reach, if the user always picked right:")
    print(f"  {'K':>3} {'hit@K':>8} {'mIoU@K':>9}")
    for row in stats["ceiling"]:  # type: ignore[union-attr]
        print(f"  {row['k']:>3} {row['hit_rate']:>8.3f} {row['mIoU']:>9.4f}")
    print(f"\n  ceiling gain over answering top-1: {stats['gain_from_picking']:+.3f} hit rate")
    print(f"  mIoU gain:                         {stats['mIoU_gain']:+.4f}")

    print("\n  is a narrow score margin a signal that top-1 is wrong?")
    print(f"  {'margin <':>9} {'asked':>7} {'hit|narrow':>11} {'hit|wide':>9}")
    for row in stats["margin"]["cuts"]:  # type: ignore[index]
        print(
            f"  {row['margin_below']:>9.2f} {row['share_asked']:>7.1%} "
            f"{row['top1_hit_when_narrow']:>11.3f} {row['top1_hit_when_wide']:>9.3f}"
        )
    print("\n  (a usable signal means hit|narrow is clearly below hit|wide)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "ambiguity.json"
    path.write_text(
        json.dumps({"summary": stats, "samples": [asdict(s) for s in samples]}, indent=2)
    )
    print(f"\n  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
