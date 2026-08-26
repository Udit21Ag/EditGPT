"""The whole product, text in, image out: CLIPSeg seed -> MobileSAM refine -> LaMa erase.

Day 3 rescored CLIPSeg on localisation rather than coverage, on the argument that
a partial hit is a good enough seed for MobileSAM. This runs that argument end to
end and scores the FINAL mask, so the claim stands or falls on evidence.

It also loads torch and both ONNX models in one process on purpose: that is the
worst-case resident footprint the plan's ModelSlot exists to prevent.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import torch
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

from bench import cases
from bench.common import MODELS, OUT, emit, load, session
from bench.run_lama import erase_crop
from bench.run_mobilesam import ENC, preprocess
from bench.run_pipeline import dilate_px

MODEL_ID = "CIDAS/clipseg-rd64-refined"
THRESHOLD = 0.4


def seed_from_text(proc, model, image, target: str) -> np.ndarray:
    inputs = proc(text=[target], images=[image], padding=True, return_tensors="pt")
    with torch.inference_mode():
        logits = model(**inputs).logits
    if logits.ndim == 2:
        logits = logits[None, ...]
    heat = torch.sigmoid(logits)[0].numpy()
    heat = cv2.resize(heat, image.size, interpolation=cv2.INTER_LINEAR)
    return heat


def refine(enc, dec, rgb: np.ndarray, heat: np.ndarray) -> np.ndarray:
    """Turn a coarse heatmap into a SAM prompt: the seed's bounding box, plus its
    hottest pixel as a positive point so a thin seed still names the whole object."""
    h, w = rgb.shape[:2]
    seed = (heat > THRESHOLD).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(seed, 8)
    if n > 1:  # keep only the largest blob; CLIPSeg speckles
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        seed = (labels == largest).astype(np.uint8)

    ys, xs = np.nonzero(seed)
    py, px = np.unravel_index(np.argmax(heat * seed), heat.shape)

    x, scale = preprocess(rgb, enc)
    embedding = enc.run(None, {enc.get_inputs()[0].name: x})[0]
    coords = (
        np.array([[[xs.min(), ys.min()], [xs.max(), ys.max()], [px, py]]], dtype=np.float32) * scale
    )
    feed = {
        "image_embeddings": embedding,
        "point_coords": coords,
        "point_labels": np.array([[2, 3, 1]], dtype=np.float32),
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
    px_ = dilate_px(mask)
    return cv2.dilate(mask, np.ones((px_, px_), np.uint8), iterations=1)


def main() -> None:
    torch.set_num_threads(4)
    t0 = time.perf_counter()
    proc = CLIPSegProcessor.from_pretrained(MODEL_ID)
    clipseg = CLIPSegForImageSegmentation.from_pretrained(MODEL_ID).eval()
    enc = session(MODELS / "mobilesam-encoder.onnx")
    dec = session(MODELS / "mobilesam-decoder.onnx")
    lama = session(MODELS / "lama.onnx")
    cold = round(time.perf_counter() - t0, 3)

    OUT.mkdir(exist_ok=True)
    scored, hits = {}, 0
    for case in cases.load(ops={"remove"}, need_box=True):
        image = load(case.path)
        rgb = np.array(image)
        started = time.perf_counter()
        heat = seed_from_text(proc, clipseg, image, case.target)
        mask = refine(enc, dec, rgb, heat)
        result = np.array(erase_crop(lama, image, mask))
        elapsed = round(time.perf_counter() - started, 2)

        metrics = cases.box_metrics(mask // 255, case.box, rgb.shape[:2])
        hits += metrics["hit"]
        scored[case.id] = {
            "prompt": case.prompt,
            "seconds": elapsed,
            "coverage": round(float((mask > 0).mean()), 4),
            **metrics,
        }

        overlay = rgb.copy()
        overlay[mask > 0] = (0.55 * overlay[mask > 0] + 0.45 * np.array([255, 40, 40])).astype(
            np.uint8
        )
        cv2.imwrite(
            str(OUT / f"text2edit_{case.id}.png"),
            cv2.cvtColor(np.concatenate([rgb, overlay, result], axis=1), cv2.COLOR_RGB2BGR),
        )

    emit(
        {
            "cold_load_s": cold,
            "warm_p50_s": scored[next(iter(scored))]["seconds"],
            "hit_rate": f"{hits}/{len(scored)}",
            "cases": scored,
            "note": "fully text-driven — no brush, no box. torch + both ONNX models resident.",
        }
    )


if __name__ == "__main__":
    main()
