from __future__ import annotations

import json
import os
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
PHOTOS = ROOT.parent / "evals" / "photos"   # promoted to evals/ in Phase 1
OUT = ROOT / "out"

WARMUP = 1
REPEATS = 5


def emit(payload: dict) -> None:
    """Hand structured results back to the harness."""
    print("##BENCH##" + json.dumps(payload))


def photos(limit: int = 10) -> list[Path]:
    files = (
        sorted(
            p for p in PHOTOS.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        if PHOTOS.exists()
        else []
    )
    if not files:
        raise SystemExit(
            f"No photos found in {PHOTOS}.\n"
            "Drop 10 real photos there first — at minimum: a car on a street, a portrait,\n"
            "a person on a busy background, a landscape with a signpost, and a group shot."
        )
    return files[:limit]


def load(path: Path, max_side: int = 1024) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    return img


def timed(fn: Callable[[], object], warmup: int = WARMUP, repeats: int = REPEATS) -> dict:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return {
        "warm_p50_s": round(statistics.median(samples), 3),
        "warm_min_s": round(min(samples), 3),
        "warm_max_s": round(max(samples), 3),
        "repeats": repeats,
    }


def box_mask(size: tuple[int, int], frac: tuple[float, float, float, float]) -> np.ndarray:
    """A deterministic rectangular mask, as a stand-in for a user's brush stroke.
    frac is (x0, y0, x1, y1) in 0..1. Returns uint8 HxW, 255 = inpaint here."""
    w, h = size
    x0, y0, x1, y1 = frac
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[round(y0 * h) : round(y1 * h), round(x0 * w) : round(x1 * w)] = 255
    return mask


def providers() -> list[str]:
    """CPU by default. Set EDITGPT_PROVIDERS=CoreMLExecutionProvider,CPUExecutionProvider
    to re-run any benchmark on the Neural Engine and compare."""
    raw = os.environ.get("EDITGPT_PROVIDERS", "CPUExecutionProvider")
    return [p.strip() for p in raw.split(",") if p.strip()]


def session(path: Path, threads: int = 4):
    """ONNX Runtime session tuned for a memory-constrained box."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.enable_mem_pattern = True
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), opts, providers=providers())


def describe_io(sess) -> dict:
    return {
        "inputs": [{"name": i.name, "shape": i.shape, "type": i.type} for i in sess.get_inputs()],
        "outputs": [{"name": o.name, "shape": o.shape, "type": o.type} for o in sess.get_outputs()],
    }
