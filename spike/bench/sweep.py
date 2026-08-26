"""Prompt sweeps, scored automatically.

Two different problems wear the same clothes here:

* For **remove**, the eraser never sees text. The prompt only shapes the mask, so a
  variant is judged on mask agreement with the reference box plus the photometric
  cost of the resulting fill.
* For **add**, the prompt goes straight to SD-1.5, so a variant is judged on how well
  the produced region matches what was asked for — CLIP similarity between the edited
  crop and the request. That is the CriticAgent's `score_edit` in Phase 7.

Usage:  uv run python -m bench.sweep i1 i3 i6 i6c i11
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from dotenv import load_dotenv
from transformers import CLIPModel, CLIPProcessor, CLIPSegForImageSegmentation, CLIPSegProcessor

from bench import cases
from bench.common import MODELS, OUT, load, session
from bench.providers import cloudflare_fill
from bench.run_erasers import migan_erase
from bench.run_final import ESCALATE_COST, MIN_SAM_IOU, cost, dilate, refine
from bench.run_lama import erase_crop, erase_crop_with
from bench.run_text2edit import MODEL_ID, THRESHOLD, seed_from_text

CLIP_ID = "openai/clip-vit-base-patch32"

# Mask prompts for removal. The eraser never sees these.
MASK_VARIANTS = {
    "i1": [
        "the car",
        "the yellow car",
        "the yellow car parked on the plaza",
        "the car and its shadow on the ground",
        "the yellow hatchback car including the dark shadow beneath it",
    ],
    "i6": [
        "the laptop",
        "the silver laptop computer",
        "the open laptop with a screen showing a sunflower",
        "the laptop and its screen and keyboard",
        "the laptop computer resting on the wooden desk",
    ],
    "i11": [
        "the air conditioner",
        "the white air conditioner unit mounted on the wall",
        "air conditioner, split AC indoor unit, white appliance",
        "the white split air conditioner box near the ceiling",
        "the wall mounted white air conditioner with vents",
    ],
}

# Fill prompts for additions. These go to SD-1.5 verbatim.
FILL_VARIANTS = {
    "i3": [
        "a realistic moustache on the upper lip",
        "a thick black moustache above the lip, photorealistic, natural hair texture",
        "portrait of a person with a neat black moustache, sharp focus, natural skin",
        "a well groomed dark moustache, fine individual hairs, matching skin tone, photograph",
        "close up of a face with a natural moustache, studio lighting, high detail, realistic",
    ],
    "i6c": [
        "a wooden chair standing on the floor",
        "a wooden chair, product photograph on a plain white background",
        "a single wooden dining chair, isolated on white, studio product shot, full chair visible",
        "wooden chair with four legs and a backrest, centred, plain white background, catalogue photo",
        "a brown wooden chair, complete object, white seamless backdrop, soft studio lighting",
    ],
}
CLIP_TARGETS = {"i3": "a face with a moustache", "i6c": "a wooden chair"}


def clip_score(model, proc, image: np.ndarray, text: str) -> float:
    inputs = proc(text=[text], images=[image], return_tensors="pt", padding=True)
    with torch.inference_mode():
        out = model(**inputs)
    return float(out.logits_per_image[0, 0])


def sweep_remove(case, models, variants) -> list[dict]:
    proc, clipseg, enc, dec, migan, lama = models
    image = load(case.path)
    src = np.array(image)
    rows, panels = [], [src]

    for prompt in variants:
        t0 = time.perf_counter()
        heat = seed_from_text(proc, clipseg, image, prompt)
        mask_sam, sam_iou = refine(enc, dec, src, heat)
        if sam_iou >= MIN_SAM_IOU:
            mask = dilate(mask_sam)
        else:
            mask = dilate((heat > THRESHOLD).astype(np.uint8) * 255)
        if not mask.any():
            rows.append(
                {
                    "prompt": prompt,
                    "sam_iou": 0.0,
                    "bbox_iou": 0.0,
                    "coverage": 0.0,
                    "eraser": "none",
                    "fill_cost": float("inf"),
                    "seconds": round(time.perf_counter() - t0, 2),
                }
            )
            panels += [src, src]
            continue
        met = cases.box_metrics(mask // 255, case.box, src.shape[:2])

        out = np.array(migan_erase(migan, image, mask))
        fm = cases.fill_metrics(out, src, mask)
        eraser = "MI-GAN"
        if cost(fm) > ESCALATE_COST:
            out = np.array(erase_crop(lama, image, mask))
            fm = cases.fill_metrics(out, src, mask)
            eraser = "LaMa"

        overlay = src.copy()
        overlay[mask > 0] = (0.5 * overlay[mask > 0] + 0.5 * np.array([255, 40, 40])).astype(
            np.uint8
        )
        panels += [overlay, out]
        rows.append(
            {
                "prompt": prompt,
                "sam_iou": round(sam_iou, 2),
                "bbox_iou": met["bbox_iou"],
                "coverage": round(float((mask > 0).mean()), 4),
                "eraser": eraser,
                "fill_cost": round(cost(fm), 1),
                "seconds": round(time.perf_counter() - t0, 2),
            }
        )
    _strip(panels, f"sweep_{case.id}_mask.png")
    return rows


def sweep_add(case, clip, variants) -> list[dict]:
    model, proc = clip
    image = load(case.path)
    src = np.array(image)
    h, w = src.shape[:2]
    x0, y0, x1, y1 = case.box
    mask = np.zeros((h, w), np.uint8)
    mask[round(y0 * h) : round(y1 * h), round(x0 * w) : round(x1 * w)] = 255
    box_px = (round(x0 * w), round(y0 * h), round(x1 * w), round(y1 * h))

    rows, panels = [], [src]
    baseline = clip_score(
        model, proc, src[box_px[1] : box_px[3], box_px[0] : box_px[2]], CLIP_TARGETS[case.id]
    )
    for prompt in variants:
        t0 = time.perf_counter()
        try:
            out = np.array(erase_crop_with(cloudflare_fill(prompt), image, mask))
        except Exception as exc:  # noqa: BLE001
            rows.append({"prompt": prompt, "error": f"{type(exc).__name__}: {str(exc)[:120]}"})
            continue
        region = out[box_px[1] : box_px[3], box_px[0] : box_px[2]]
        rows.append(
            {
                "prompt": prompt,
                "clip_score": round(clip_score(model, proc, region, CLIP_TARGETS[case.id]), 2),
                "vs_original": round(
                    clip_score(model, proc, region, CLIP_TARGETS[case.id]) - baseline, 2
                ),
                "seconds": round(time.perf_counter() - t0, 2),
            }
        )
        panels.append(out)
    _strip(panels, f"sweep_{case.id}_fill.png")
    return rows


def _strip(panels: list[np.ndarray], name: str) -> None:
    h = min(p.shape[0] for p in panels)
    panels = [cv2.resize(p, (round(p.shape[1] * h / p.shape[0]), h)) for p in panels]
    OUT.mkdir(exist_ok=True)
    cv2.imwrite(str(OUT / name), cv2.cvtColor(np.concatenate(panels, axis=1), cv2.COLOR_RGB2BGR))


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    wanted = sys.argv[1:] or ["i1", "i3", "i6", "i6c", "i11"]
    by_id = {c.id: c for c in cases.load()}
    torch.set_num_threads(4)

    need_local = any(w in MASK_VARIANTS for w in wanted)
    need_clip = any(w in FILL_VARIANTS for w in wanted)
    models = clip = None
    if need_local:
        models = (
            CLIPSegProcessor.from_pretrained(MODEL_ID),
            CLIPSegForImageSegmentation.from_pretrained(MODEL_ID).eval(),
            session(MODELS / "mobilesam-encoder.onnx"),
            session(MODELS / "mobilesam-decoder.onnx"),
            session(MODELS / "migan-pipeline.onnx"),
            session(MODELS / "lama.onnx"),
        )
    if need_clip:
        clip = (CLIPModel.from_pretrained(CLIP_ID).eval(), CLIPProcessor.from_pretrained(CLIP_ID))

    for cid in wanted:
        case = by_id[cid]
        print(f"\n=== {cid}: {case.prompt} ===")
        if cid in MASK_VARIANTS:
            rows = sweep_remove(case, models, MASK_VARIANTS[cid])
            print(f"  {'sam':>4} {'bbox':>5} {'cover':>6} {'eraser':7} {'cost':>6}  prompt")
            for r in sorted(rows, key=lambda r: (-r["bbox_iou"], r["fill_cost"])):
                print(
                    f"  {r['sam_iou']:4.2f} {r['bbox_iou']:5.3f} {r['coverage'] * 100:5.1f}% "
                    f"{r['eraser']:7} {r['fill_cost']:6.1f}  {r['prompt'][:64]}"
                )
        else:
            rows = sweep_add(case, clip, FILL_VARIANTS[cid])
            print(f"  {'clip':>6} {'delta':>6}  prompt")
            for r in sorted(rows, key=lambda r: -r.get("clip_score", -99)):
                if "error" in r:
                    print(f"  ERROR {r['error'][:70]}  {r['prompt'][:40]}")
                else:
                    print(f"  {r['clip_score']:6.2f} {r['vs_original']:+6.2f}  {r['prompt'][:64]}")


if __name__ == "__main__":
    main()
