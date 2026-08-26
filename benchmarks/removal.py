"""Held-out removal benchmark with real paired ground truth.

RemovalBench ships, for each sample, the scene **with** the object, the same scene
**without** it, the object mask, and a reference system's output. That makes SSIM and
PSNR valid here, which they are not on our own fixtures — there we have no image of what
the scene should look like once the object is gone.

It also runs each eraser separately, so every sample yields a labelled example of *which
eraser was better*. That is the training data for `benchmarks.classifier`.

    uv run python -m benchmarks.removal --limit 60
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
from editgpt_core.metrics import fill_metrics
from editgpt_models.compositing import RGB, grow
from editgpt_models.erase import erase_lama, erase_migan, make_session
from editgpt_models.pipeline import Erasers, erase
from editgpt_models.registry import model_path
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from benchmarks.datasets import Mask, load_removal

OUT_DIR = Path(__file__).resolve().parent / "out"
WORKING_SIDE = 768
"""Benchmark inputs are ~3.5 MP. Reduced so a 60-sample run is minutes, not an hour."""


@dataclass
class RemovalResult:
    id: str
    mask_coverage: float
    ssim_migan: float
    ssim_lama: float
    ssim_routed: float
    ssim_baseline: float | None
    cost_migan: float
    cost_lama: float
    """The reference-free score the shipped router decides on. Recorded so it can be
    correlated against SSIM-vs-ground-truth, which is the only way to find out whether
    the proxy the router trusts actually tracks quality."""
    psnr_routed: float
    winner: str
    """Which single eraser scored higher against ground truth."""
    margin: float
    routed_matched_winner: bool
    """Did the shipped router pick the eraser that actually won?"""
    passes: str
    seconds: float


def _fit(image: RGB, side: int) -> RGB:
    height, width = image.shape[:2]
    if max(height, width) <= side:
        return image
    scale = side / max(height, width)
    return np.asarray(
        cv2.resize(
            image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA
        ),
        dtype=np.uint8,
    )


def _fit_mask(mask: Mask, side: int) -> Mask:
    height, width = mask.shape[:2]
    if max(height, width) <= side:
        return mask
    scale = side / max(height, width)
    return np.asarray(
        cv2.resize(
            mask, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_NEAREST
        ),
        dtype=np.uint8,
    )


def score_against(result: RGB, truth: RGB, mask: Mask) -> tuple[float, float]:
    """SSIM and PSNR **inside the edited region only**.

    Scoring the whole frame would be dominated by the untouched majority and would report
    0.99 for every method, including a bad one — the edited region is a few percent of
    the pixels.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 1.0, 99.0
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    a, b = result[y0:y1, x0:x1], truth[y0:y1, x0:x1]
    if min(a.shape[:2]) < 7:
        return 1.0, 99.0
    return (
        float(structural_similarity(a, b, channel_axis=2)),
        float(peak_signal_noise_ratio(b, a, data_range=255)),
    )


def run(limit: int) -> list[RemovalResult]:
    migan = make_session(model_path("migan"))
    lama = make_session(model_path("lama"))
    erasers = Erasers.from_sessions(migan, lama)

    results: list[RemovalResult] = []
    for sample in load_removal(limit=limit):
        image = _fit(sample.image, WORKING_SIDE)
        truth = _fit(sample.ground_truth, WORKING_SIDE)
        mask = grow(_fit_mask(sample.mask, WORKING_SIDE))
        if int(mask.sum()) == 0:
            continue

        started = time.monotonic()
        out_migan = erase_migan(migan, image, mask)
        out_lama = erase_lama(lama, image, mask)
        routed = erase(erasers, image, mask)
        elapsed = time.monotonic() - started

        cost_migan = fill_metrics(out_migan, image, mask).cost
        cost_lama = fill_metrics(out_lama, image, mask).cost
        ssim_migan, _ = score_against(out_migan, truth, mask)
        ssim_lama, _ = score_against(out_lama, truth, mask)
        ssim_routed, psnr_routed = score_against(routed.image, truth, mask)
        ssim_baseline = (
            score_against(_fit(sample.baseline, WORKING_SIDE), truth, mask)[0]
            if sample.baseline is not None
            else None
        )

        winner = "migan" if ssim_migan >= ssim_lama else "lama"
        results.append(
            RemovalResult(
                id=sample.id,
                mask_coverage=round(float((mask > 0).mean()), 4),
                ssim_migan=round(ssim_migan, 4),
                ssim_lama=round(ssim_lama, 4),
                ssim_routed=round(ssim_routed, 4),
                ssim_baseline=round(ssim_baseline, 4) if ssim_baseline is not None else None,
                cost_migan=round(cost_migan, 3),
                cost_lama=round(cost_lama, 3),
                psnr_routed=round(psnr_routed, 2),
                winner=winner,
                margin=round(abs(ssim_migan - ssim_lama), 4),
                routed_matched_winner=bool(abs(ssim_routed - max(ssim_migan, ssim_lama)) < 1e-6),
                passes=routed.summary(),
                seconds=round(elapsed, 2),
            )
        )
    return results


def summarise(results: list[RemovalResult]) -> dict[str, object]:
    if not results:
        return {}
    mean = statistics.mean
    baselines = [r.ssim_baseline for r in results if r.ssim_baseline is not None]
    oracle = mean(max(r.ssim_migan, r.ssim_lama) for r in results)
    return {
        "n": len(results),
        "ssim_migan": round(mean(r.ssim_migan for r in results), 4),
        "ssim_lama": round(mean(r.ssim_lama for r in results), 4),
        "ssim_routed": round(mean(r.ssim_routed for r in results), 4),
        "ssim_oracle": round(oracle, 4),
        "ssim_baseline_omnieraser": round(mean(baselines), 4) if baselines else None,
        "psnr_routed": round(mean(r.psnr_routed for r in results), 2),
        "migan_wins": sum(r.winner == "migan" for r in results),
        "lama_wins": sum(r.winner == "lama" for r in results),
        "router_picked_the_winner": round(
            sum(r.routed_matched_winner for r in results) / len(results), 3
        ),
        "median_seconds": round(statistics.median(r.seconds for r in results), 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=60)
    args = parser.parse_args()

    results = run(args.limit)
    if not results:
        print("no samples", file=sys.stderr)
        return 2

    stats = summarise(results)
    print(f"\nRemovalBench, paired ground truth, n={stats['n']}")
    print("  SSIM inside the edited region, higher is better:")
    for key in (
        "ssim_migan",
        "ssim_lama",
        "ssim_routed",
        "ssim_oracle",
        "ssim_baseline_omnieraser",
    ):
        if stats.get(key) is not None:
            print(f"    {key:26} {stats[key]}")
    print(f"\n  PSNR (routed)              {stats['psnr_routed']}")
    print(f"  MI-GAN wins / LaMa wins    {stats['migan_wins']} / {stats['lama_wins']}")
    print(f"  router picked the winner   {stats['router_picked_the_winner']:.1%}")
    print(f"  median seconds per sample  {stats['median_seconds']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "removal.json"
    path.write_text(
        json.dumps({"summary": stats, "results": [asdict(r) for r in results]}, indent=2)
    )
    print(f"\n  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
