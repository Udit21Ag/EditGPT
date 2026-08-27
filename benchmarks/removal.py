"""Held-out removal benchmark with real paired ground truth.

Each sample ships the scene **with** the object, the same scene **without** it, and the
object mask. That makes SSIM and PSNR valid here, which they are not on our own fixtures
— there we have no image of what the scene should look like once the object is gone.

Two datasets, and running both is the point. TD-013's conclusion — that the photometric
`cost` is a poor proxy for fidelity — was drawn from RemovalBench alone, and its own
resolution says to confirm it elsewhere before acting. RORD-50 is a completely different
distribution: handheld video captures rather than curated stills.

It also scores **two candidate proxies** against ground truth on every fill:

* ``cost``     — `fill_metrics(...).cost`, photometric plausibility. Lower is better.
* ``semantic`` — the ReMOVE score off the MobileSAM encoder. Higher is better.

A proxy that works correlates with SSIM-against-truth in its own direction. Reporting
both, on both datasets, is what turns "which should the router trust?" into a
measurement instead of a preference.

Each eraser also runs separately, so every sample yields a labelled example of *which
eraser was better*. That is the training data for `benchmarks.classifier`.

    uv run python -m benchmarks.removal --limit 69 --dataset both
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from editgpt_core.metrics import fill_metrics
from editgpt_models.compositing import RGB, grow
from editgpt_models.erase import erase_lama, erase_migan, make_session
from editgpt_models.pipeline import Erasers, erase
from editgpt_models.registry import model_path
from editgpt_models.semantic import consistency
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from benchmarks.datasets import Mask, RemovalSample, load_removal, load_rord

OUT_DIR = Path(__file__).resolve().parent / "out"
WORKING_SIDE = 768
"""Benchmark inputs are ~3.5 MP. Reduced so a 60-sample run is minutes, not an hour."""


DATASETS: dict[str, Callable[[int], Iterator[RemovalSample]]] = {
    "removalbench": load_removal,
    "rord": load_rord,
}


@dataclass
class RemovalResult:
    id: str
    group: str
    """The clip a video frame came from. Frames of one clip are not independent."""

    dataset: str
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

    semantic_migan: float
    semantic_lama: float
    """The candidate replacement: ReMOVE off the MobileSAM encoder. Higher is better."""
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


def spearman(xs: list[float], ys: list[float]) -> float:
    """Rank correlation, computed here so the benchmark needs no scipy.

    Rank rather than Pearson because neither proxy is expected to be *linear* in SSIM —
    the question is only whether it orders fills the same way.
    """
    if len(xs) < 3:
        return 0.0

    def ranks(values: list[float]) -> np.ndarray:
        order = np.argsort(np.asarray(values, dtype=np.float64), kind="stable")
        out = np.empty(len(values), dtype=np.float64)
        out[order] = np.arange(len(values), dtype=np.float64)
        return out

    a, b = ranks(xs), ranks(ys)
    a, b = a - a.mean(), b - b.mean()
    denominator = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denominator) if denominator else 0.0


def run(limit: int, dataset: str) -> list[RemovalResult]:
    migan = make_session(model_path("migan"))
    lama = make_session(model_path("lama"))
    encoder = make_session(model_path("sam-encoder"))
    erasers = Erasers.from_sessions(migan, lama)

    results: list[RemovalResult] = []
    for sample in DATASETS[dataset](limit):
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
        semantic_migan = consistency(encoder, out_migan, mask)
        semantic_lama = consistency(encoder, out_lama, mask)
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
                group=sample.group_id,
                dataset=dataset,
                mask_coverage=round(float((mask > 0).mean()), 4),
                ssim_migan=round(ssim_migan, 4),
                ssim_lama=round(ssim_lama, 4),
                ssim_routed=round(ssim_routed, 4),
                ssim_baseline=round(ssim_baseline, 4) if ssim_baseline is not None else None,
                cost_migan=round(cost_migan, 3),
                cost_lama=round(cost_lama, 3),
                semantic_migan=round(semantic_migan, 4),
                semantic_lama=round(semantic_lama, 4),
                psnr_routed=round(psnr_routed, 2),
                winner=winner,
                margin=round(abs(ssim_migan - ssim_lama), 4),
                routed_matched_winner=bool(abs(ssim_routed - max(ssim_migan, ssim_lama)) < 1e-6),
                passes=routed.summary(),
                seconds=round(elapsed, 2),
            )
        )
    return results


def proxy_report(results: list[RemovalResult]) -> dict[str, object]:
    """How well each candidate proxy orders fills against ground truth.

    Every fill counts as one observation, so n is twice the sample count. `cost` is
    lower-is-better and SSIM higher-is-better, so a working `cost` correlates
    **negatively**; `semantic` is higher-is-better, so it should correlate positively.
    """
    ssim = [r.ssim_migan for r in results] + [r.ssim_lama for r in results]
    cost = [r.cost_migan for r in results] + [r.cost_lama for r in results]
    semantic = [r.semantic_migan for r in results] + [r.semantic_lama for r in results]

    def picks(chooser: Callable[[RemovalResult], str]) -> float:
        return round(sum(chooser(r) == r.winner for r in results) / max(len(results), 1), 4)

    majority = "lama" if sum(r.winner == "lama" for r in results) * 2 >= len(results) else "migan"
    return {
        "n_fills": len(ssim),
        "spearman_cost_vs_ssim": round(spearman(cost, ssim), 4),
        "spearman_semantic_vs_ssim": round(spearman(semantic, ssim), 4),
        "cost_picks_the_winner": picks(
            lambda r: "migan" if r.cost_migan <= r.cost_lama else "lama"
        ),
        "semantic_picks_the_winner": picks(
            lambda r: "migan" if r.semantic_migan >= r.semantic_lama else "lama"
        ),
        "always_majority_picks_the_winner": picks(lambda _r: majority),
        "majority_eraser": majority,
        "oracle_gain_over_majority": round(
            statistics.mean(max(r.ssim_migan, r.ssim_lama) for r in results)
            - statistics.mean(
                (r.ssim_lama if majority == "lama" else r.ssim_migan) for r in results
            ),
            4,
        ),
    }


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
    parser.add_argument("--limit", type=int, default=69)
    parser.add_argument("--dataset", choices=[*sorted(DATASETS), "both"], default="both")
    args = parser.parse_args()

    wanted = sorted(DATASETS) if args.dataset == "both" else [args.dataset]
    per_dataset = {name: run(args.limit, name) for name in wanted}
    results = [r for rows in per_dataset.values() for r in rows]
    if not results:
        print("no samples", file=sys.stderr)
        return 2

    for name, rows in per_dataset.items():
        if not rows:
            continue
        _report(name, rows)

    if len(per_dataset) > 1:
        print("\n" + "=" * 68)
        print("  Does the conclusion hold on both? A proxy that only works on one is not")
        print("  a proxy — it is an artefact of that dataset. TD-013 asked for this check.")
        for name, rows in per_dataset.items():
            if rows:
                report = proxy_report(rows)
                print(
                    f"    {name:14} cost {report['spearman_cost_vs_ssim']:+.3f}   "
                    f"semantic {report['spearman_semantic_vs_ssim']:+.3f}"
                )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "removal.json"
    path.write_text(
        json.dumps(
            {
                "by_dataset": {
                    name: {
                        "summary": summarise(rows),
                        "proxies": proxy_report(rows),
                        "results": [asdict(r) for r in rows],
                    }
                    for name, rows in per_dataset.items()
                    if rows
                }
            },
            indent=2,
        )
    )
    print(f"\n  {path}")
    return 0


def _report(name: str, results: list[RemovalResult]) -> None:
    stats = summarise(results)
    print(f"\n{name}, paired ground truth, n={stats['n']}")
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

    report = proxy_report(results)
    print(f"\n  proxy quality, n={report['n_fills']} individual fills:")
    print(
        f"    Spearman cost     vs SSIM   {report['spearman_cost_vs_ssim']:+.4f}"
        "   (a working proxy is NEGATIVE)"
    )
    print(
        f"    Spearman semantic vs SSIM   {report['spearman_semantic_vs_ssim']:+.4f}"
        "   (a working proxy is POSITIVE)"
    )
    print("\n  picking the better eraser, share correct:")
    print(f"    by cost                     {report['cost_picks_the_winner']:.1%}")
    print(f"    by semantic                 {report['semantic_picks_the_winner']:.1%}")
    print(
        f"    always {report['majority_eraser']:<20} "
        f"{report['always_majority_picks_the_winner']:.1%}"
    )
    print(
        f"    the whole choice is worth   {report['oracle_gain_over_majority']:+.4f} SSIM "
        "(oracle over the majority)"
    )


if __name__ == "__main__":
    sys.exit(main())
