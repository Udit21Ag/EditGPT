"""Settle crop-vs-whole at real camera resolution.

At 15.9 MP the two strategies finally diverge: `whole` downscales the entire
frame 9x to reach LaMa's fixed 512 input and upscales the result back, so every
pixel in the image is destroyed to fix one object. `crop` touches only a window
around the mask. The 1:1 detail strip is where you see it — a downscaled
comparison hides exactly the difference we're measuring.

Also answers: what does a real 15.9 MP upload do to peak RSS?
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from bench import cases
from bench.common import MODELS, OUT, emit, load, session
from bench.run_lama import erase_crop, erase_whole
from bench.run_pipeline import segment

VIEW = 1400  # width of the downscaled overview strip
ZOOM = 700  # side of the 1:1 detail crop


def detail(img: np.ndarray, box, shape) -> np.ndarray:
    """A 1:1 pixel crop centred on the edited region."""
    h, w = shape
    x0, y0, x1, y1 = box
    cx, cy = int((x0 + x1) / 2 * w), int((y0 + y1) / 2 * h)
    half = ZOOM // 2
    cx = int(np.clip(cx, half, w - half))
    cy = int(np.clip(cy, half, h - half))
    return img[cy - half : cy + half, cx - half : cx + half]


def fit(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(img, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)


def main() -> None:
    t0 = time.perf_counter()
    enc = session(MODELS / "mobilesam-encoder.onnx")
    dec = session(MODELS / "mobilesam-decoder.onnx")
    lama = session(MODELS / "lama.onnx")
    cold = round(time.perf_counter() - t0, 3)

    todo = [c for c in cases.load(ops={"remove"}, need_box=True) if c.id in {"i11", "i12", "i13"}]

    OUT.mkdir(exist_ok=True)
    per_case = {}
    for case in todo:
        image = load(case.path, max_side=10_000)  # native resolution
        src = np.array(image)
        h, w = src.shape[:2]

        t = time.perf_counter()
        mask = segment(enc, dec, src, case.box)
        t_mask = round(time.perf_counter() - t, 2)

        t = time.perf_counter()
        crop = np.array(erase_crop(lama, image, mask))
        t_crop = round(time.perf_counter() - t, 2)

        t = time.perf_counter()
        whole = np.array(erase_whole(lama, image, mask))
        t_whole = round(time.perf_counter() - t, 2)

        cv2.imwrite(
            str(OUT / f"fullres_{case.id}_overview.png"),
            cv2.cvtColor(
                np.concatenate([fit(src, VIEW), fit(whole, VIEW), fit(crop, VIEW)], axis=1),
                cv2.COLOR_RGB2BGR,
            ),
        )
        cv2.imwrite(
            str(OUT / f"fullres_{case.id}_detail_1to1.png"),
            cv2.cvtColor(
                np.concatenate(
                    [
                        detail(src, case.box, (h, w)),
                        detail(whole, case.box, (h, w)),
                        detail(crop, case.box, (h, w)),
                    ],
                    axis=1,
                ),
                cv2.COLOR_RGB2BGR,
            ),
        )

        per_case[case.id] = {
            "prompt": case.prompt,
            "megapixels": round(w * h / 1e6, 1),
            "mask_s": t_mask,
            "crop_s": t_crop,
            "whole_s": t_whole,
            "mask_coverage": round(float((mask > 0).mean()), 4),
            "note": case.note,
        }

    emit(
        {
            "cold_load_s": cold,
            "warm_p50_s": per_case[todo[0].id]["crop_s"],
            "cases": per_case,
            "note": "judge fullres_*_detail_1to1.png — order is original | whole | crop",
        }
    )


if __name__ == "__main__":
    main()
