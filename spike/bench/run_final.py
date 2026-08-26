"""The pipeline as ADR-0001 decided it. Text in, edited image out.

    text -> CLIPSeg seed
         -> MobileSAM refine, GATED on the decoder's own iou_prediction
         -> MI-GAN erase (fast path)
         -> score the fill
         -> escalate to Big-LaMa only when the score is poor

Every branch here is a Phase 0 finding rather than a guess:

* refinement is conditional, because it improved a weak seed and degraded a strong
  one, and `iou_predictions` is the signal available at runtime where no reference
  box exists;
* MI-GAN goes first because it is 40x faster to load and 3.6x faster at full
  resolution, and LaMa is held back for the cases MI-GAN handles badly;
* the escalation threshold is read off the head-to-head, where LaMa won every case
  whose MI-GAN cost cleared ~25.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import torch
from dotenv import load_dotenv
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

from bench import cases
from bench.common import MODELS, OUT, emit, load, session
from bench.providers import cloudflare_fill
from bench.run_erasers import migan_erase
from bench.run_lama import erase_crop, erase_crop_with
from bench.run_mobilesam import ENC, preprocess
from bench.run_pipeline import dilate_px
from bench.run_text2edit import MODEL_ID, THRESHOLD, seed_from_text

# Below this, SAM's own confidence says its refinement is not trustworthy and the
# CLIPSeg seed is kept instead.
MIN_SAM_IOU = 0.85
# Above this photometric cost, MI-GAN's fill is bad enough to pay LaMa's 2.8 s.
ESCALATE_COST = 25.0
# A second pass over what is still dark near the erased object removes cast shadows.
# It only helps when the leftover is genuinely a shadow: measured, +35% of the object's
# area on i1 cleaned the shadow up, while +78% (i6) and +119% (i2) meant the detector had
# latched onto real scene content and the second erase destroyed it. Cap it well below
# those. fill_metrics cannot make this call — a larger flat erase scores BETTER on it.
RESIDUAL_MAX_GROWTH = 0.50
RESIDUAL_DARK_DROP = 10
RESIDUAL_REACH = 1.1


def _save(src: np.ndarray, mask: np.ndarray, out: np.ndarray, case) -> None:
    overlay = src.copy()
    overlay[mask > 0] = (0.55 * overlay[mask > 0] + 0.45 * np.array([255, 40, 40])).astype(np.uint8)
    label = (case.target or case.content or "").replace(" ", "_")
    cv2.imwrite(
        str(OUT / f"final_{case.id}_{label}.png"),
        cv2.cvtColor(np.concatenate([src, overlay, out], axis=1), cv2.COLOR_RGB2BGR),
    )


def cost(m: dict) -> float:
    return m["chroma_delta"] + m["lum_delta"] + 12 * abs(np.log(max(m["edge_ratio"], 1e-3)))


def refine(enc, dec, rgb: np.ndarray, heat: np.ndarray) -> tuple[np.ndarray, float]:
    """Seed -> SAM mask, returning SAM's confidence so the caller can reject it."""
    h, w = rgb.shape[:2]
    seed = (heat > THRESHOLD).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(seed, 8)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        seed = (labels == largest).astype(np.uint8)
    ys, xs = np.nonzero(seed)
    if len(xs) == 0:
        # CLIPSeg matched nothing above threshold. That is a legitimate outcome for a
        # bad prompt, not an error: return an empty mask and zero confidence so the
        # caller falls through to asking the user rather than crashing.
        return np.zeros_like(seed), 0.0
    py, px = np.unravel_index(np.argmax(heat * seed), heat.shape)

    x, scale = preprocess(rgb, enc)
    embedding = enc.run(None, {enc.get_inputs()[0].name: x})[0]
    feed = {
        "image_embeddings": embedding,
        "point_coords": np.array(
            [[[xs.min(), ys.min()], [xs.max(), ys.max()], [px, py]]], dtype=np.float32
        )
        * scale,
        "point_labels": np.array([[2, 3, 1]], dtype=np.float32),
        "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
        "has_mask_input": np.zeros(1, dtype=np.float32),
        "orig_im_size": np.array([h, w], dtype=np.float32),
    }
    names = {i.name for i in dec.get_inputs()}
    out = dec.run(None, {k: v for k, v in feed.items() if k in names})
    masks, iou = out[0], float(np.ravel(out[1])[0])

    mask = (masks[0, 0] > 0).astype(np.uint8) * 255
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (ENC, ENC), interpolation=cv2.INTER_NEAREST)
        mask = mask[: round(h * scale), : round(w * scale)]
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask, iou


def shadow_pass(fill, image, result: np.ndarray, src: np.ndarray, mask: np.ndarray):
    """Erase what is still darker than the surface near where the object stood.

    The same detector failed on the original image because the reference median was
    polluted by the object and its shadow; run after the object is gone, it has a clean
    surface to compare against.
    """
    from PIL import Image

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None, 0.0
    h, w = mask.shape
    oh = ys.max() - ys.min()
    band = np.zeros_like(mask)
    band[
        ys.min() : min(h, int(ys.max() + RESIDUAL_REACH * oh)),
        max(0, int(xs.min() - 0.4 * oh)) : min(w, int(xs.max() + 0.4 * oh)),
    ] = 255

    lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)[:, :, 0].astype(np.float32)
    ref = cv2.medianBlur(lab.astype(np.uint8), 151).astype(np.float32)
    dark = ((ref - lab) > RESIDUAL_DARK_DROP) & (band > 0)
    dark = cv2.morphologyEx(
        dark.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((11, 11), np.uint8)
    )
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))

    growth = float((dark > 0).sum()) / max(float((mask > 0).sum()), 1)
    if not dark.any() or growth > RESIDUAL_MAX_GROWTH:
        return None, growth
    return np.array(fill(Image.fromarray(result), dark)), growth


def dilate(mask: np.ndarray) -> np.ndarray:
    px = dilate_px(mask)
    return cv2.dilate(mask, np.ones((px, px), np.uint8), iterations=1)


def main() -> None:
    torch.set_num_threads(4)
    proc = CLIPSegProcessor.from_pretrained(MODEL_ID)
    clipseg = CLIPSegForImageSegmentation.from_pretrained(MODEL_ID).eval()
    enc = session(MODELS / "mobilesam-encoder.onnx")
    dec = session(MODELS / "mobilesam-decoder.onnx")
    migan = session(MODELS / "migan-pipeline.onnx")
    lama = session(MODELS / "lama.onnx")

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    OUT.mkdir(exist_ok=True)
    results, escalations, seeds_kept, shadows = {}, 0, 0, 0

    for case in cases.load(ops={"remove", "add"}, need_box=True):
        image = load(case.path)
        src = np.array(image)
        started = time.perf_counter()

        # ADD has no object to segment, so the reference box is the mask — which is
        # what a user's brush would hand us. It goes remote; local erasers cannot
        # synthesise new content.
        if case.op == "add":
            h, w = src.shape[:2]
            x0, y0, x1, y1 = case.box
            mask = np.zeros((h, w), np.uint8)
            mask[round(y0 * h) : round(y1 * h), round(x0 * w) : round(x1 * w)] = 255
            try:
                out = np.array(erase_crop_with(cloudflare_fill(case.fill), image, mask))
                path = "Cloudflare SD-1.5"
            except Exception as exc:  # noqa: BLE001 — the reason is the finding
                results[case.id] = {
                    "prompt": case.prompt,
                    "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                }
                continue
            _save(src, mask, out, case)
            results[case.id] = {
                "prompt": case.prompt,
                "mask_source": "reference box (brush stand-in)",
                "eraser": path,
                "seconds": round(time.perf_counter() - started, 2),
                **cases.fill_metrics(out, src, mask),
            }
            continue

        heat = seed_from_text(proc, clipseg, image, case.target)
        refined, sam_iou = refine(enc, dec, src, heat)
        if sam_iou >= MIN_SAM_IOU:
            mask = dilate(refined)
            source = f"SAM (iou {sam_iou:.2f})"
        else:
            mask = dilate((heat > THRESHOLD).astype(np.uint8) * 255)
            source = f"CLIPSeg seed (SAM iou {sam_iou:.2f} rejected)"
            seeds_kept += 1

        out = np.array(migan_erase(migan, image, mask))
        m = cases.fill_metrics(out, src, mask)
        path, c = "MI-GAN", cost(m)
        if c > ESCALATE_COST:
            out = np.array(erase_crop(lama, image, mask))
            m = cases.fill_metrics(out, src, mask)
            path, escalations = f"MI-GAN -> LaMa (cost {c:.0f})", escalations + 1
        shadow, growth = shadow_pass(lambda im, mk: erase_crop(lama, im, mk), image, out, src, mask)
        if shadow is not None:
            out, path = shadow, f"{path} + shadow pass"
            m = cases.fill_metrics(out, src, mask)
            shadows += 1
        elapsed = round(time.perf_counter() - started, 2)

        _save(src, mask, out, case)
        results[case.id] = {
            "prompt": case.prompt,
            "mask_source": source,
            "eraser": path,
            "seconds": elapsed,
            "residual_growth": round(growth, 2),
            **m,
        }

    emit(
        {
            "warm_p50_s": results[next(iter(results))]["seconds"],
            "escalated_to_lama": f"{escalations}/{len(results)}",
            "shadow_pass_applied": f"{shadows}/{len(results)}",
            "clipseg_seed_kept": f"{seeds_kept}/{len(results)}",
            "cases": results,
            "note": "panels are original | mask | result. Prompt-only, no brush.",
        }
    )


if __name__ == "__main__":
    main()
