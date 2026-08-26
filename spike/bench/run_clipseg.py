"""CLIPSeg: free text -> coarse mask. This is the model that decides whether
'remove the car' can work without the user touching a brush.

Torch, not ONNX, deliberately: Phase 0 measures the easy path first. If the
RSS is close to budget, Phase 4 exports to ONNX int8 and re-measures.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import torch
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

from bench import cases
from bench.common import OUT, emit, load, timed

MODEL_ID = "CIDAS/clipseg-rd64-refined"
THRESHOLD = 0.4


def main() -> None:
    torch.set_num_threads(4)
    t0 = time.perf_counter()
    proc = CLIPSegProcessor.from_pretrained(MODEL_ID)
    model = CLIPSegForImageSegmentation.from_pretrained(MODEL_ID).eval()
    cold = round(time.perf_counter() - t0, 3)

    todo = cases.load(ops={"remove"}, need_box=True)

    def infer(image, prompts: list[str]):
        inputs = proc(
            text=prompts, images=[image] * len(prompts), padding=True, return_tensors="pt"
        )
        with torch.inference_mode():
            logits = model(**inputs).logits
        if logits.ndim == 2:  # a single prompt comes back without the batch axis
            logits = logits[None, ...]
        return torch.sigmoid(logits).numpy()

    first = load(todo[0].path)
    stats = timed(lambda: infer(first, [todo[0].target]), repeats=3)

    OUT.mkdir(exist_ok=True)
    scored, hits = {}, 0
    for case in todo:
        image = load(case.path)
        rgb = np.array(image)
        heat = infer(image, [case.target])[0]
        m = cv2.resize(heat, image.size, interpolation=cv2.INTER_LINEAR)
        binary = (m > THRESHOLD).astype(np.uint8)

        metrics = cases.box_metrics(binary, case.box, rgb.shape[:2])
        hits += metrics["hit"]
        scored[case.id] = {"prompt": case.target, **metrics}

        overlay = rgb.copy()
        overlay[binary > 0] = (0.55 * overlay[binary > 0] + 0.45 * np.array([255, 40, 40])).astype(
            np.uint8
        )
        h, w = rgb.shape[:2]
        x0, y0, x1, y1 = case.box
        cv2.rectangle(
            overlay,
            (round(x0 * w), round(y0 * h)),
            (round(x1 * w), round(y1 * h)),
            (40, 220, 40),
            2,
        )
        slug = case.target.replace(" ", "_")
        cv2.imwrite(
            str(OUT / f"clipseg_{case.id}_{slug}.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        )

    emit(
        {
            "cold_load_s": cold,
            "warm_p50_s": stats["warm_p50_s"],
            "threshold": THRESHOLD,
            "hit_rate": f"{hits}/{len(todo)}",
            "cases": scored,
            "note": "green box = your stated intent; red = what CLIPSeg found. hit needs "
            "inside_frac>0.6 and bbox_iou>0.3",
        }
    )


if __name__ == "__main__":
    main()
