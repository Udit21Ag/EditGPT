"""Does feeding the result back through the eraser improve it?

Three strategies, all scored against the ORIGINAL surroundings so pass 2 cannot
flatter itself by comparing to pass 1:

* **again**    — same mask, second pass. Tests whether re-inpainting with a
                 plausible fill as context smooths pass-1 artifacts.
* **cross**    — MI-GAN then LaMa on the same mask. Two different priors.
* **residual** — find what is still wrong in pass 1 and erase THAT. The shadow
                 heuristic failed on the original image because the reference
                 median was polluted by the object and its shadow; after the
                 object is gone the surface is clean, so the same detector has a
                 far better reference to work from.

Usage: uv run python -m bench.refeed i1 i2 i6
"""

from __future__ import annotations

import sys

import cv2
import numpy as np

from bench import cases
from bench.common import MODELS, OUT, load, session
from bench.run_erasers import migan_erase
from bench.run_final import cost, dilate, refine
from bench.run_lama import erase_crop
from bench.run_text2edit import MODEL_ID, seed_from_text

DARK_DROP = 10  # luminance below the local surface, in L units, to count as residual
REACH = 1.1  # search band height, as a multiple of the object's height


def residual_mask(result: np.ndarray, mask1: np.ndarray) -> np.ndarray:
    """What still looks wrong near where the object was."""
    ys, xs = np.nonzero(mask1)
    if len(xs) == 0:
        return np.zeros_like(mask1)
    h, w = mask1.shape
    oh = ys.max() - ys.min()
    band = np.zeros_like(mask1)
    band[
        ys.min() : min(h, int(ys.max() + REACH * oh)),
        max(0, int(xs.min() - 0.4 * oh)) : min(w, int(xs.max() + 0.4 * oh)),
    ] = 255

    lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)[:, :, 0].astype(np.float32)
    ref = cv2.medianBlur(lab.astype(np.uint8), 151).astype(np.float32)
    dark = ((ref - lab) > DARK_DROP) & (band > 0)
    dark = cv2.morphologyEx(
        dark.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((11, 11), np.uint8)
    )
    return cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))


def main() -> None:
    wanted = sys.argv[1:] or ["i1", "i2", "i6"]
    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

    proc = CLIPSegProcessor.from_pretrained(MODEL_ID)
    clipseg = CLIPSegForImageSegmentation.from_pretrained(MODEL_ID).eval()
    enc = session(MODELS / "mobilesam-encoder.onnx")
    dec = session(MODELS / "mobilesam-decoder.onnx")
    migan = session(MODELS / "migan-pipeline.onnx")
    lama = session(MODELS / "lama.onnx")
    by_id = {c.id: c for c in cases.load()}

    for cid in wanted:
        case = by_id[cid]
        image = load(case.path)
        src = np.array(image)
        heat = seed_from_text(proc, clipseg, image, case.target)
        m, _ = refine(enc, dec, src, heat)
        mask1 = dilate(m)

        from PIL import Image

        pass1 = np.array(erase_crop(lama, image, mask1))
        again = np.array(erase_crop(lama, Image.fromarray(pass1), mask1))
        crossed = np.array(migan_erase(migan, Image.fromarray(pass1), mask1))
        res = residual_mask(pass1, mask1)
        grew = float((res > 0).sum()) / max(float((mask1 > 0).sum()), 1)
        fixed = np.array(erase_crop(lama, Image.fromarray(pass1), res)) if res.any() else None

        # Every variant is scored over the SAME region — the union of everything any
        # strategy touched. Scoring each over its own mask compares different areas and
        # penalises the residual pass for cleaning up a larger one, which is how this
        # experiment first read as a failure when the image plainly showed otherwise.
        eval_region = np.maximum(mask1, res) if res.any() else mask1
        base = cost(cases.fill_metrics(pass1, src, eval_region))
        variants = {
            "pass 1 (LaMa)": (pass1, base, mask1),
            "again (LaMa x2)": (again, cost(cases.fill_metrics(again, src, eval_region)), mask1),
            "cross (LaMa->MI-GAN)": (
                crossed,
                cost(cases.fill_metrics(crossed, src, eval_region)),
                mask1,
            ),
        }
        if fixed is not None:
            variants[f"residual (+{grew:.0%} area)"] = (
                fixed,
                cost(cases.fill_metrics(fixed, src, eval_region)),
                res,
            )

        print(f"\n=== {cid}: {case.prompt} ===")
        for name, (_, c, _) in variants.items():
            delta = "" if name.startswith("pass 1") else f"  ({c - base:+.1f} vs pass 1)"
            print(f"  {name:24} cost {c:7.1f}{delta}")

        panels = [src]
        for img_, _, mk in variants.values():
            ov = img_.copy()
            ov[mk > 0] = (0.6 * ov[mk > 0] + 0.4 * np.array([255, 40, 40])).astype(np.uint8)
            panels.append(img_)
        h = min(p.shape[0] for p in panels)
        panels = [cv2.resize(p, (round(p.shape[1] * h / p.shape[0]), h)) for p in panels]
        OUT.mkdir(exist_ok=True)
        cv2.imwrite(
            str(OUT / f"refeed_{cid}.png"),
            cv2.cvtColor(np.concatenate(panels, axis=1), cv2.COLOR_RGB2BGR),
        )
        print("  panels: original | " + " | ".join(variants))


if __name__ == "__main__":
    main()
