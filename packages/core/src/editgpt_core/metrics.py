"""Quality scoring for an edited region. Needs `editgpt-core[metrics]` for OpenCV.

Every constant here was measured in Phase 0 rather than chosen; the numbers that
justify them are in `docs/adr/0001-model-routing.md`.

The central caution: **`fill_cost` compares fills within a fixed mask and must never be
used to compare masks.** A larger flat erase is photometrically consistent with itself,
so it scores better while looking worse. That mistake was made three separate times
during the spike, each time picking the visually worse image. `compare` exists so it
cannot be made a fourth: it charges for area erased beyond the original mask.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

RGB = npt.NDArray[np.uint8]
Mask = npt.NDArray[np.uint8]

RING_PX = 60
"""Width of the ring of real pixels a fill is compared against."""

EDGE_WEIGHT = 12.0
"""How much a detail-level mismatch counts relative to a colour error, in cost units."""

GROWTH_PENALTY = 25.0
"""Cost charged per unit of relative mask growth.

Calibrated on the Phase 0 residual-pass results, where raw cost preferred the variant
that destroyed the image: the laptop desk gained 78% area for a 18.8-point cost
improvement, while the car gained 35% for 11.4. At 25 the desk is correctly rejected
(+0.7) and the car correctly accepted (-2.7).
"""

MIN_MASK_PX = 64
"""Below this a mask cannot produce a visible edit; treat as a failure, not a no-op."""


@dataclass(frozen=True, slots=True)
class FillMetrics:
    """How well an inpainted region agrees with what surrounds it."""

    chroma_delta: float
    """Lab a/b distance to the surrounding ring. ~25 was a visible pale ghost."""

    lum_delta: float
    """Lab L distance. Catches a patch that is too light or too dark."""

    edge_ratio: float
    """Fill edge energy over the ring's. 1.0 matches; << 1 is a smear; >> 1 invents detail."""

    @property
    def cost(self) -> float:
        """One number, lower is better. Only comparable across fills of the same mask."""
        return float(
            self.chroma_delta
            + self.lum_delta
            + EDGE_WEIGHT * abs(np.log(max(self.edge_ratio, 1e-3)))
        )


def fill_metrics(result: RGB, source: RGB, mask: Mask, ring_px: int = RING_PX) -> FillMetrics:
    """Compare the filled region of `result` against real pixels ringing it in `source`."""
    if result.shape != source.shape:
        raise ValueError(f"result {result.shape} and source {source.shape} differ")
    inside = mask > 0
    if not inside.any():
        return FillMetrics(0.0, 0.0, 1.0)

    kernel = np.ones((ring_px, ring_px), np.uint8)
    ring = (cv2.dilate(mask.astype(np.uint8), kernel) > 0) & ~inside
    if int(ring.sum()) < 100:
        return FillMetrics(0.0, 0.0, 1.0)

    lab_out = cv2.cvtColor(result, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_src = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype(np.float32)
    chroma = float(
        np.hypot(
            lab_out[inside, 1].mean() - lab_src[ring, 1].mean(),
            lab_out[inside, 2].mean() - lab_src[ring, 2].mean(),
        )
    )
    lum = float(abs(lab_out[inside, 0].mean() - lab_src[ring, 0].mean()))

    grey_out = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY).astype(np.float32)
    grey_src = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY).astype(np.float32)
    energy_in = float(np.abs(cv2.Laplacian(grey_out, cv2.CV_32F))[inside].mean())
    energy_ring = float(np.abs(cv2.Laplacian(grey_src, cv2.CV_32F))[ring].mean()) + 1e-6

    return FillMetrics(
        chroma_delta=round(chroma, 3),
        lum_delta=round(lum, 3),
        edge_ratio=round(energy_in / energy_ring, 4),
    )


def compare(
    candidate: RGB,
    baseline: RGB,
    source: RGB,
    eval_region: Mask,
    base_mask: Mask,
    candidate_mask: Mask,
) -> float:
    """How much better than `baseline` is `candidate`? Negative means better.

    Both are scored over the *same* `eval_region` — scoring each over its own mask
    compares different areas and rewards whichever erased more. On top of that, the
    candidate is charged `GROWTH_PENALTY` for every unit of area it touched beyond
    `base_mask`, which is what stops "erase more of the scene" from winning.
    """
    base = fill_metrics(baseline, source, eval_region).cost
    cand = fill_metrics(candidate, source, eval_region).cost

    base_area = max(int((base_mask > 0).sum()), 1)
    extra = max(int((candidate_mask > 0).sum()) - int((base_mask > 0).sum()), 0)
    return (cand + GROWTH_PENALTY * extra / base_area) - base


def box_metrics(
    mask: Mask, box: tuple[float, float, float, float], shape: tuple[int, int]
) -> dict[str, float | bool]:
    """Agreement between a predicted mask and a hand-drawn reference box.

    Gates on `inside_frac` with `bbox_iou` — box against box, the only like-for-like
    comparison available without painted ground truth. Mask-against-box is invalid: a
    correct mask of a non-convex object (a lattice tower) covers ~37% of its own
    bounding box and would be scored a failure.
    """
    height, width = shape
    x0, y0, x1, y1 = box
    rx0, ry0 = round(x0 * width), round(y0 * height)
    rx1, ry1 = round(x1 * width), round(y1 * height)

    total = float(mask.sum())
    if total == 0:
        return {"inside_frac": 0.0, "box_recall": 0.0, "bbox_iou": 0.0, "hit": False}

    inside = float(mask[ry0:ry1, rx0:rx1].sum()) / total
    box_area = max((ry1 - ry0) * (rx1 - rx0), 1)
    recall = float(mask[ry0:ry1, rx0:rx1].sum()) / box_area

    ys, xs = np.nonzero(mask)
    mx0, my0, mx1, my1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    inter = max(0, min(mx1, rx1) - max(mx0, rx0)) * max(0, min(my1, ry1) - max(my0, ry0))
    union = (mx1 - mx0) * (my1 - my0) + (rx1 - rx0) * (ry1 - ry0) - inter
    iou = inter / union if union > 0 else 0.0

    return {
        "inside_frac": round(inside, 3),
        "box_recall": round(recall, 3),
        "bbox_iou": round(iou, 3),
        "hit": bool(inside > 0.6 and iou > 0.5),
    }
