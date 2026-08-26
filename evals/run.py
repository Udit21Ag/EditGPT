"""Run the golden set end to end and print the quality table.

    make eval                # every runnable case
    make eval ARGS="i1 i6c"  # a subset

Cases that need a provider or weights that are not configured are reported as skipped
rather than failed, so the table is always printable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from dotenv import load_dotenv
from editgpt_core import EditOp
from editgpt_core.errors import ProviderError
from editgpt_core.metrics import box_metrics, fill_metrics
from editgpt_models.compositing import RGB, Mask
from editgpt_models.enhance import downscale_to, upscale
from editgpt_models.erase import flat_background_mask, make_session, recolour_background
from editgpt_models.pipeline import Erasers, erase, prepare_mask
from editgpt_models.registry import model_path
from editgpt_models.segment import load_clipseg, mask_from_seed, seed_from_text
from editgpt_models.slot import ModelSlot
from editgpt_providers import CloudflareWorkersAI
from PIL import Image

from evals.cases import Case, load

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent / "out"
WORKING_SIDE = 1024
UPSCALE_EVAL_SIDE = 320
"""Upscaling is evaluated on a reduced input: 2x of a full frame costs ~84 s on CPU."""
GREEN = (46, 160, 67)


def _detail(image: RGB) -> float:
    """Mean gradient magnitude — a proxy for how much real detail an image carries."""
    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(np.abs(cv2.Laplacian(grey, cv2.CV_32F)).mean())


@dataclass
class Result:
    id: str
    prompt: str
    op: str
    status: str
    seconds: float = 0.0
    passes: str = ""
    kept_passes: int = 0
    cost: float = 0.0
    bbox_iou: float | None = None
    mask_source: str = ""
    detail: str = ""


def load_image(path: Path, max_side: int = WORKING_SIDE) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if max(image.size) > max_side:
        scale = max_side / max(image.size)
        image = image.resize((round(image.width * scale), round(image.height * scale)))
    return image


class Context:
    """Models, loaded lazily and held one at a time."""

    def __init__(self) -> None:
        self.slot = ModelSlot(max_resident=4)  # ONNX sessions here are small; weights are not
        self._clipseg: tuple[Any, Any] | None = None

    def session(self, key: str) -> Any:
        return self.slot.acquire(key, lambda: make_session(model_path(key)))

    def clipseg(self) -> tuple[Any, Any]:
        if self._clipseg is None:
            self._clipseg = load_clipseg()
        return self._clipseg


def segment_for(case: Case, ctx: Context, image: Image.Image, rgb: RGB) -> tuple[Mask, str]:
    """Mask for a case, preferring text so the eval exercises the shipping path."""
    encoder, decoder = ctx.session("sam-encoder"), ctx.session("sam-decoder")
    if case.target:
        processor, model = ctx.clipseg()
        heat = seed_from_text(processor, model, image, case.target)
        seg = mask_from_seed(encoder, decoder, rgb, heat)
        if seg.mask.any():
            return prepare_mask(seg.mask), f"{seg.source} ({seg.confidence:.2f})"

    if case.box is None:
        raise ValueError(f"{case.id}: text found nothing and there is no fallback box")
    height, width = rgb.shape[:2]
    x0, y0, x1, y1 = case.box
    mask = np.zeros((height, width), np.uint8)
    mask[round(y0 * height) : round(y1 * height), round(x0 * width) : round(x1 * width)] = 255
    return mask, "reference box"


def strip(source: RGB, mask: Mask, result: RGB, name: str) -> None:
    overlay = source.copy()
    overlay[mask > 0] = (0.55 * overlay[mask > 0] + 0.45 * np.array([255, 40, 40])).astype(np.uint8)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(OUT_DIR / name),
        cv2.cvtColor(np.concatenate([source, overlay, result], axis=1), cv2.COLOR_RGB2BGR),
    )


def run_case(case: Case, ctx: Context) -> Result:
    started = time.monotonic()
    image = load_image(case.path)
    rgb = np.asarray(image, dtype=np.uint8)

    if case.op is EditOp.REMOVE:
        mask, source = segment_for(case, ctx, image, rgb)
        erasers = Erasers.from_sessions(ctx.session("migan"), ctx.session("lama"))
        outcome = erase(erasers, rgb, mask)
        strip(rgb, outcome.mask, outcome.image, f"{case.id}_{case.op.value}.png")
        iou = None
        if case.box is not None:
            iou = float(box_metrics(mask // 255, case.box, rgb.shape[:2])["bbox_iou"])
        return Result(
            id=case.id,
            prompt=case.prompt,
            op=case.op.value,
            status="ok",
            seconds=round(time.monotonic() - started, 2),
            passes=outcome.summary(),
            kept_passes=outcome.kept_passes,
            cost=round(outcome.cost, 1),
            bbox_iou=iou,
            mask_source=source,
        )

    if case.op is EditOp.UPSCALE:
        # Deliberately driven at a reduced size: 2x enhancement of a full frame takes
        # minutes on CPU, so the eval measures behaviour, not patience.
        small = downscale_to(rgb, UPSCALE_EVAL_SIDE)
        enlarged = upscale(ctx.session("esrgan-x2"), small)
        reference = cv2.resize(small, enlarged.shape[1::-1], interpolation=cv2.INTER_CUBIC).astype(
            np.uint8
        )
        strip(
            reference,
            np.zeros(reference.shape[:2], np.uint8),
            enlarged,
            f"{case.id}_{case.op.value}.png",
        )
        sharper = _detail(enlarged) / max(_detail(reference), 1e-6)
        return Result(
            id=case.id,
            prompt=case.prompt,
            op=case.op.value,
            status="ok",
            seconds=round(time.monotonic() - started, 2),
            passes=f"esrgan-x2 tiled, {small.shape[1]}x{small.shape[0]} -> "
            f"{enlarged.shape[1]}x{enlarged.shape[0]}",
            kept_passes=1,
            cost=round(sharper, 2),
            mask_source="whole image",
            detail=f"detail vs bicubic {sharper:.2f}x",
        )

    if case.op is EditOp.BACKGROUND:
        # A flat colour is compositing, not generation. Keep the subject, repaint the rest.
        # Prefer flood-filling the backdrop over segmenting the subject: it keeps thin
        # structures like table legs and leaves no fringe of the old colour.
        backdrop = flat_background_mask(rgb)
        if backdrop is not None:
            mask = np.asarray(255 - backdrop, dtype=np.uint8)
            source = "flood fill from the border"
        else:
            mask, source = segment_for(case, ctx, image, rgb)
        result = recolour_background(rgb, mask, GREEN)
        strip(rgb, mask, result, f"{case.id}_{case.op.value}.png")
        return Result(
            id=case.id,
            prompt=case.prompt,
            op=case.op.value,
            status="ok",
            seconds=round(time.monotonic() - started, 2),
            passes="composite",
            kept_passes=1,
            cost=0.0,
            mask_source=source,
            detail="deterministic recolour, no generative call",
        )

    # ADD and REPLACE: the reference box is the mask, standing in for a brush stroke.
    # Both go remote — a local eraser cannot synthesise new content.
    provider = CloudflareWorkersAI()
    if not provider.is_configured():
        return Result(
            case.id, case.prompt, case.op.value, "skipped", detail="provider not configured"
        )
    if case.box is None:
        return Result(case.id, case.prompt, case.op.value, "skipped", detail="no placement box")

    height, width = rgb.shape[:2]
    x0, y0, x1, y1 = case.box
    mask = np.zeros((height, width), np.uint8)
    mask[round(y0 * height) : round(y1 * height), round(x0 * width) : round(x1 * width)] = 255

    from editgpt_models.compositing import erase_in_place

    try:
        result = erase_in_place(lambda c, m: provider.fill(c, m, case.fill), rgb, mask)
    except ProviderError as exc:
        return Result(case.id, case.prompt, case.op.value, "failed", detail=str(exc)[:120])

    strip(rgb, mask, result, f"{case.id}_{case.op.value}.png")
    return Result(
        id=case.id,
        prompt=case.prompt,
        op=case.op.value,
        status="ok",
        seconds=round(time.monotonic() - started, 2),
        passes="cloudflare",
        kept_passes=1,
        cost=round(fill_metrics(result, rgb, mask).cost, 1),
        mask_source="reference box (brush stand-in)",
    )


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ids", nargs="*", help="case ids; default is every runnable case")
    args = parser.parse_args()

    cases = [c for c in load() if not args.ids or c.id in args.ids]
    if not cases:
        print("no matching cases", file=sys.stderr)
        return 2

    ctx = Context()
    results: list[Result] = []
    for case in cases:
        try:
            results.append(run_case(case, ctx))
        except FileNotFoundError as exc:
            results.append(
                Result(case.id, case.prompt, case.op.value, "skipped", detail=str(exc)[:90])
            )
        except Exception as exc:
            results.append(
                Result(
                    case.id,
                    case.prompt,
                    case.op.value,
                    "failed",
                    detail=f"{type(exc).__name__}: {exc}"[:120],
                )
            )

    width = max(len(r.prompt) for r in results)
    print(f"\n{'case':6} {'prompt':{width}} {'status':8} {'sec':>6} {'cost':>6} {'iou':>5}  passes")
    for r in results:
        iou = f"{r.bbox_iou:.3f}" if r.bbox_iou is not None else "  -  "
        print(
            f"{r.id:6} {r.prompt:{width}} {r.status:8} {r.seconds:6.2f} {r.cost:6.1f} {iou:>5}  "
            f"{r.passes or r.detail}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "report.json").write_text(json.dumps([asdict(r) for r in results], indent=2))
    print(f"\npeak RSS {ctx.slot.stats.peak_rss_mb:.0f} MB   report: {OUT_DIR / 'report.json'}")
    return 0 if all(r.status != "failed" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
