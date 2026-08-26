"""The Phase 0 eval set: ten photos, each with a stated intent and, for the
removal cases, a hand-drawn reference box.

The reference boxes are what let us score text-to-mask automatically instead of
squinting at overlays — a CLIPSeg mask is a hit if its mass lands inside the box.
This file is the seed of `evals/` in Phase 2.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from bench.common import PHOTOS

CASES_JSON = PHOTOS.parent / "cases.json"


@dataclass(frozen=True)
class Case:
    id: str
    path: Path
    op: str
    target: str | None
    content: str | None
    box: tuple[float, float, float, float] | None
    fill: str
    note: str

    @property
    def prompt(self) -> str:
        return {
            "remove": f"remove {self.target}",
            "add": f"add {self.content}",
            "background": f"change the background to {self.content}",
        }[self.op]


def load(ops: set[str] | None = None, need_box: bool = False) -> list[Case]:
    if not CASES_JSON.exists():
        raise SystemExit(f"Missing {CASES_JSON}")
    cases = []
    for raw in json.loads(CASES_JSON.read_text()):
        path = PHOTOS / raw["file"]
        if not path.exists():
            raise SystemExit(f"cases.json references a missing photo: {path}")
        case = Case(
            id=raw["id"],
            path=path,
            op=raw["op"],
            target=raw.get("target"),
            content=raw.get("content"),
            box=tuple(raw["box"]) if raw.get("box") else None,
            fill=raw.get("fill", ""),
            note=raw.get("note", ""),
        )
        if ops and case.op not in ops:
            continue
        if need_box and case.box is None:
            continue
        cases.append(case)
    if not cases:
        raise SystemExit(f"No cases matched ops={ops} need_box={need_box}")
    return cases


def box_metrics(mask, box, shape) -> dict:
    """How well does a predicted mask agree with the reference box?

    inside_frac — share of predicted mask mass that falls inside the box.
    bbox_iou    — IoU between the mask's own bounding box and the reference.

    box_recall  — share of the reference box the prediction covers. Reported as a
                  diagnostic only: the reference is a BOUNDING box, so a correct
                  mask of a non-convex object scores low by construction (a perfect
                  Eiffel Tower lattice covers ~37% of its own bounding box).

    A hit needs precision AND agreement of extent, so it gates on inside_frac with
    bbox_iou — box against box, the only apples-to-apples comparison available
    without hand-painted ground-truth masks. inside_frac alone is precision, and
    six pixels in the right place score 1.000: that is how i13 passed a rescore
    while the air conditioner was still visibly in the output.
    """
    import numpy as np

    h, w = shape
    x0, y0, x1, y1 = box
    rx0, ry0, rx1, ry1 = round(x0 * w), round(y0 * h), round(x1 * w), round(y1 * h)

    total = float(mask.sum())
    if total == 0:
        return {"inside_frac": 0.0, "bbox_iou": 0.0, "hit": False}
    inside = float(mask[ry0:ry1, rx0:rx1].sum()) / total

    ys, xs = np.nonzero(mask)
    mx0, my0, mx1, my1 = xs.min(), ys.min(), xs.max(), ys.max()
    ix = max(0, min(mx1, rx1) - max(mx0, rx0))
    iy = max(0, min(my1, ry1) - max(my0, ry0))
    inter = ix * iy
    union = (mx1 - mx0) * (my1 - my0) + (rx1 - rx0) * (ry1 - ry0) - inter
    iou = inter / union if union > 0 else 0.0

    box_area = max((ry1 - ry0) * (rx1 - rx0), 1)
    recall = float(mask[ry0:ry1, rx0:rx1].sum()) / box_area
    return {
        "inside_frac": round(inside, 3),
        "box_recall": round(recall, 3),
        "bbox_iou": round(iou, 3),
        "hit": bool(inside > 0.6 and iou > 0.5),
    }


def fill_metrics(result, src, mask, ring_px: int = 60) -> dict:
    """How well does an inpainted region agree with what surrounds it?

    Measures the defect class Phase 0 actually found — LaMa gets structure right
    and photometry wrong — rather than raw flatness, which would reward a blur.

    chroma_delta — Lab a/b distance between the fill and a ring of real pixels.
                   ~25 was the visible pale-rectangle ghost on i12.
    lum_delta    — L distance. Catches the too-light / too-dark patch.
    edge_ratio   — fill's edge energy over the ring's. 1.0 means it carries the
                   same amount of detail as its surroundings; <<1 is a smear,
                   >>1 is a hallucinated object with invented edges.

    This is the CriticAgent's scorer in Phase 7, where no reference exists.
    """
    import cv2
    import numpy as np

    inside = mask > 0
    if not inside.any():
        return {"chroma_delta": 0.0, "lum_delta": 0.0, "edge_ratio": 1.0}
    ring = (cv2.dilate(mask.astype(np.uint8), np.ones((ring_px, ring_px), np.uint8)) > 0) & ~inside
    if ring.sum() < 100:
        return {"chroma_delta": 0.0, "lum_delta": 0.0, "edge_ratio": 1.0}

    lab_out = cv2.cvtColor(result, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_src = cv2.cvtColor(src, cv2.COLOR_RGB2LAB).astype(np.float32)
    chroma = float(
        np.hypot(
            lab_out[inside, 1].mean() - lab_src[ring, 1].mean(),
            lab_out[inside, 2].mean() - lab_src[ring, 2].mean(),
        )
    )
    lum = float(abs(lab_out[inside, 0].mean() - lab_src[ring, 0].mean()))

    g_out = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g_src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY).astype(np.float32)
    e_out = np.abs(cv2.Laplacian(g_out, cv2.CV_32F))[inside].mean()
    e_ring = np.abs(cv2.Laplacian(g_src, cv2.CV_32F))[ring].mean() + 1e-6
    return {
        "chroma_delta": round(chroma, 2),
        "lum_delta": round(lum, 2),
        "edge_ratio": round(float(e_out / e_ring), 3),
    }
