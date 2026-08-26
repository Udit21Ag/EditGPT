"""The real Phase 0 question: MobileSAM mask -> LaMa erase, end to end.

Benching LaMa on a hardcoded rectangle measures the model but not the product.
This runs the pipeline EditGPT will actually ship — box prompt to mask to
dilate to crop-inpaint to feathered paste-back — and writes a before/after
strip you can judge. It also measures the two models' *combined* peak RSS,
which is the number that decides whether ModelSlot(max_resident=1) is enough.
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from bench import cases
from bench.common import MODELS, OUT, emit, load, session, timed
from bench.run_lama import erase_crop
from bench.run_mobilesam import ENC, preprocess

# LaMa needs slack past the object edge or the edge and its shadow survive as an
# outline. A fixed pixel count cannot work across resolutions: 12 px was ample at
# 1024 px and left a visible rectangle at 15.9 MP. Measured on i12, residual edge
# energy hits the blank-wall floor at ~5% of the object's longest side and buys
# nothing beyond it.
DILATE_FRAC = 0.05
DILATE_MIN, DILATE_MAX = 8, 128


def dilate_px(mask: np.ndarray) -> int:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return DILATE_MIN
    longest = max(xs.max() - xs.min(), ys.max() - ys.min())
    return int(np.clip(round(DILATE_FRAC * longest), DILATE_MIN, DILATE_MAX))


def segment(enc, dec, rgb: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    h, w = rgb.shape[:2]
    x, scale = preprocess(rgb, enc)
    embedding = enc.run(None, {enc.get_inputs()[0].name: x})[0]

    x0, y0, x1, y1 = box
    corners = np.array([[[w * x0, h * y0], [w * x1, h * y1]]], dtype=np.float32) * scale
    feed = {
        "image_embeddings": embedding,
        "point_coords": corners,
        "point_labels": np.array([[2, 3]], dtype=np.float32),
        "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
        "has_mask_input": np.zeros(1, dtype=np.float32),
        "orig_im_size": np.array([h, w], dtype=np.float32),
    }
    names = {i.name for i in dec.get_inputs()}
    masks = dec.run(None, {k: v for k, v in feed.items() if k in names})[0]

    mask = (masks[0, 0] > 0).astype(np.uint8) * 255
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (ENC, ENC), interpolation=cv2.INTER_NEAREST)
        mask = mask[: round(h * scale), : round(w * scale)]
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    px = dilate_px(mask)
    return cv2.dilate(mask, np.ones((px, px), np.uint8), iterations=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="run a single case id, e.g. i8")
    args = ap.parse_args()

    t0 = time.perf_counter()
    enc = session(MODELS / "mobilesam-encoder.onnx")
    dec = session(MODELS / "mobilesam-decoder.onnx")
    lama = session(MODELS / "lama.onnx")
    cold = round(time.perf_counter() - t0, 3)

    todo = [
        c for c in cases.load(ops={"remove"}, need_box=True) if not args.only or c.id == args.only
    ]

    first = todo[0]
    img0 = load(first.path)
    rgb0 = np.array(img0)
    stats = timed(lambda: erase_crop(lama, img0, segment(enc, dec, rgb0, first.box)), repeats=3)

    OUT.mkdir(exist_ok=True)
    per_case = {}
    for case in todo:
        image = load(case.path)
        src = np.array(image)
        started = time.perf_counter()
        mask = segment(enc, dec, src, case.box)
        result = np.array(erase_crop(lama, image, mask))
        elapsed = round(time.perf_counter() - started, 2)

        overlay = src.copy()
        overlay[mask > 0] = (0.55 * overlay[mask > 0] + 0.45 * np.array([255, 40, 40])).astype(
            np.uint8
        )
        strip = np.concatenate([src, overlay, result], axis=1)
        slug = (case.target or "").replace(" ", "_")
        cv2.imwrite(
            str(OUT / f"pipeline_{case.id}_{slug}.png"), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR)
        )

        per_case[case.id] = {
            "prompt": case.prompt,
            "seconds": elapsed,
            "mask_coverage": round(float((mask > 0).mean()), 4),
            "note": case.note,
        }

    emit(
        {
            "cold_load_s": cold,
            "warm_p50_s": stats["warm_p50_s"],
            "detail": stats,
            "dilate_frac": DILATE_FRAC,
            "cases": per_case,
            "note": "all three sessions resident at once — this peak RSS is the worst case",
        }
    )


if __name__ == "__main__":
    main()
