"""Held-out grounding benchmark: does text -> mask generalise?

Our own golden set scores a predicted mask against a **box drawn by hand**, which is a
proxy — a correct mask of a non-convex object covers only part of its bounding box. This
benchmark uses RefCOCOg, where the ground truth is a real segmentation mask, so it
reports the field's standard metric (mask IoU) and nothing was fitted to it.

    uv run python -m benchmarks.grounding --limit 200
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from editgpt_models.compositing import RGB
from editgpt_models.erase import make_session
from editgpt_models.registry import model_path
from editgpt_models.segment import load_clipseg, mask_from_seed, seed_from_text
from PIL import Image

from benchmarks.datasets import Mask, load_grounding

OUT_DIR = Path(__file__).resolve().parent / "out"
WORKING_SIDE = 1024


@dataclass
class GroundingResult:
    id: str
    phrase: str
    words: int
    iou: float
    """Mask IoU against ground truth — the standard metric, not a box proxy."""
    precision: float
    recall: float
    source: str
    confidence: float
    gt_coverage: float
    seconds: float


def mask_iou(predicted: Mask, truth: Mask) -> tuple[float, float, float]:
    """IoU, precision and recall between two binary masks."""
    p, t = predicted > 0, truth > 0
    intersection = float(np.logical_and(p, t).sum())
    union = float(np.logical_or(p, t).sum())
    return (
        intersection / union if union else 0.0,
        intersection / float(p.sum()) if p.any() else 0.0,
        intersection / float(t.sum()) if t.any() else 0.0,
    )


def _fit(image: RGB, mask: Mask, side: int) -> tuple[RGB, Mask]:
    """Scale an image and its mask together, so the pair stays aligned."""
    height, width = image.shape[:2]
    if max(height, width) <= side:
        return image, mask
    scale = side / max(height, width)
    size = (round(width * scale), round(height * scale))
    return (
        np.asarray(cv2.resize(image, size, interpolation=cv2.INTER_AREA), dtype=np.uint8),
        np.asarray(cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST), dtype=np.uint8),
    )


def run(limit: int) -> list[GroundingResult]:
    processor, clipseg = load_clipseg()
    encoder = make_session(model_path("sam-encoder"))
    decoder = make_session(model_path("sam-decoder"))

    results: list[GroundingResult] = []
    for sample in load_grounding(limit=limit):
        image, truth = _fit(sample.image, sample.mask, WORKING_SIDE)
        started = time.monotonic()
        heat = seed_from_text(processor, clipseg, Image.fromarray(image), sample.phrase)
        segmentation = mask_from_seed(encoder, decoder, image, heat)
        elapsed = time.monotonic() - started

        iou, precision, recall = mask_iou(segmentation.mask, truth)
        results.append(
            GroundingResult(
                id=sample.id,
                phrase=sample.phrase,
                words=len(sample.phrase.split()),
                iou=round(iou, 4),
                precision=round(precision, 4),
                recall=round(recall, 4),
                source=segmentation.source,
                confidence=round(segmentation.confidence, 3),
                gt_coverage=round(float((truth > 0).mean()), 4),
                seconds=round(elapsed, 2),
            )
        )
    return results


def summarise(results: list[GroundingResult]) -> dict[str, float | int]:
    """Report the metrics the referring-segmentation literature reports.

    mIoU averages per sample; oIoU would pool intersections and unions and is dominated
    by large objects. Both are standard; mIoU is the fairer per-instance view, so the
    hit rates below are computed from it.
    """
    ious = [r.iou for r in results]
    return {
        "n": len(results),
        "mIoU": round(statistics.mean(ious), 4) if ious else 0.0,
        "median_IoU": round(statistics.median(ious), 4) if ious else 0.0,
        "precision@0.5": round(sum(i >= 0.5 for i in ious) / len(ious), 4) if ious else 0.0,
        "precision@0.7": round(sum(i >= 0.7 for i in ious) / len(ious), 4) if ious else 0.0,
        "failures@0.1": round(sum(i < 0.1 for i in ious) / len(ious), 4) if ious else 0.0,
        "median_seconds": round(statistics.median([r.seconds for r in results]), 2)
        if results
        else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    results = run(args.limit)
    if not results:
        print("no samples", file=sys.stderr)
        return 2

    stats = summarise(results)
    print(f"\nRefCOCOg validation, held out, n={stats['n']}")
    for key, value in stats.items():
        if key != "n":
            print(f"  {key:16} {value}")

    by_source: dict[str, list[float]] = {}
    for r in results:
        by_source.setdefault(r.source, []).append(r.iou)
    print("\n  by mask source:")
    for source, ious in sorted(by_source.items()):
        print(f"    {source:16} n={len(ious):4d}  mIoU {statistics.mean(ious):.4f}")

    short = [r.iou for r in results if r.words <= 5]
    long_ = [r.iou for r in results if r.words > 5]
    if short and long_:
        print("\n  by phrase length:")
        print(f"    <= 5 words      n={len(short):4d}  mIoU {statistics.mean(short):.4f}")
        print(f"    >  5 words      n={len(long_):4d}  mIoU {statistics.mean(long_):.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "grounding.json"
    path.write_text(
        json.dumps({"summary": stats, "results": [asdict(r) for r in results]}, indent=2)
    )
    print(f"\n  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
