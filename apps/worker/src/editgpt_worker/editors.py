"""Turning a job's spec into an edited image.

`tasks.py` owns the lifecycle — transitions, progress, cancellation, the ledger. This
owns the one step in the middle: decode the bytes, find the region, run the edit, encode
the result. It is deliberately the only place in the worker that knows what a pixel is.

**The edit itself is not implemented here.** `editgpt_models.execute` decides what each
operation does, and the golden set drives the same function, so a change in editing
behaviour shows up in `make eval` rather than only in production.

## Models are held by a slot, not by this module

Every session comes from a process-wide `ModelSlot`, which keeps one heavy model resident
and evicts on idle time and memory pressure. That is not an optimisation: Grounding DINO
peaks at 1372 MB and MI-GAN at 1150, and this machine has 8 GB. Loading them eagerly, or
holding both, breaches the budget before an edit finishes.

The slot is why editors take a `Resources` and ask for sessions one at a time instead of
receiving a bundle: a bundle is a decision to hold everything at once.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from editgpt_core import EditOp, EditSpec, Grounding
from editgpt_core.errors import MaskTooSmallError
from editgpt_core.rle import decode as decode_rle
from editgpt_models.compositing import RGB, Mask
from editgpt_models.execute import Models, execute
from editgpt_models.registry import model_path
from editgpt_models.slot import ModelSlot

log = logging.getLogger(__name__)

SLOT = ModelSlot()
"""One per worker process. Celery runs `--concurrency=1`, so one slot is one worker."""

GREEN = (46, 160, 67)
"""The stand-in backdrop colour until the request carries one. TD-020."""

MAX_SIDE = 2048
"""Longest side an edit runs at.

The gateway already caps uploads at 40 MP, which bounds *memory*; this bounds *time*.
A 15.9 MP erase is several minutes of model work for a result nobody can see more of
than a 2048px view. The result is returned at this size, which is a visible product
decision rather than an implementation detail — recorded as TD-021.
"""


def session(key: str) -> Any:
    """An ONNX session for `key`, loaded on first use and shared across tasks."""
    from editgpt_models.erase import make_session

    return SLOT.acquire(key, lambda: make_session(model_path(key)))


def detector() -> Any:
    from editgpt_models.detect import DETECTOR_KEY, load_detector

    return SLOT.acquire(DETECTOR_KEY, load_detector)


def decode_image(data: bytes) -> RGB:
    """Bytes to RGB, downscaled to the working size.

    Pillow rather than OpenCV because the gateway already validated the format with it,
    and a second decoder is a second set of format quirks to be surprised by.
    """
    from editgpt_models.enhance import downscale_to
    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return downscale_to(rgb, MAX_SIDE)


FORMATS: dict[str, str] = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
    "image/avif": "AVIF",
}
"""MIME type to the name Pillow knows it by.

Must cover everything `uploads.ALLOWED_FORMATS` accepts. It did not cover AVIF, and the
failure was quiet in the worst way: the encoder fell back to PNG while the `images` row
still said `image/avif`, so a 54 KB photograph came back as 476 KB of PNG bytes served
under a content type they were not. A browser is entitled to refuse that.
"""


def encode_image(image: RGB, content_type: str) -> tuple[bytes, str]:
    """RGB back to bytes, returning **what was actually written** and its type.

    Returning the type rather than assuming it is the point. The caller records this in
    the `images` row, so the row cannot claim a format the bytes are not — which is
    exactly what happened while this returned bytes alone.

    Round-tripping the source format matters on its own: answering a JPEG upload with a
    PNG quietly multiplies the stored size of every photograph by about five.
    """
    from PIL import Image

    fmt = FORMATS.get(content_type, "PNG")
    produced = content_type if fmt != "PNG" or content_type == "image/png" else "image/png"

    buffer = io.BytesIO()
    # 95 rather than Pillow's default 75: this is the output of an edit, and re-encoding
    # artifacts would be attributed to the model rather than to the encoder.
    Image.fromarray(image).save(buffer, format=fmt, quality=95)
    return buffer.getvalue(), produced


def ground_phrase(image: RGB, phrase: str) -> Grounding:
    """Every region a phrase might mean, with the gate on whether to ask.

    One encoder pass regardless of how many candidates come back — see
    `segment.masks_from_boxes`.
    """
    from editgpt_models.segment import candidates_from_phrase

    return candidates_from_phrase(
        detector(), session("sam-encoder"), session("sam-decoder"), image, phrase
    )


def region_for(spec: EditSpec, image: RGB) -> tuple[Mask | None, str]:
    """The region to act on, and where it came from.

    An explicit mask always wins: if the user brushed it, no model gets a vote. Otherwise
    the phrase is grounded. `UPSCALE` needs no region at all.
    """
    if spec.op is EditOp.UPSCALE:
        return None, "whole image"

    if spec.mask_ref is not None:
        # `decode` returns 0/1; everything downstream treats a mask as 0/255, and a mask
        # of ones survives `mask > 0` while looking empty in any image written for review.
        binary = np.asarray(decode_rle(spec.mask_ref) * 255, dtype=np.uint8)
        return _fit_mask(binary, image), f"{spec.mask_source.value} mask"

    if not spec.target:
        if spec.op is EditOp.BACKGROUND:
            # Flood-filling the backdrop needs no region. Only if the border turns out not
            # to be uniform does `execute` need one, and it says so then.
            return None, "backdrop, detected"
        raise MaskTooSmallError(f"{spec.op} needs a target phrase or a mask, and has neither")

    from editgpt_models.segment import mask_from_phrase

    found = mask_from_phrase(
        detector(), session("sam-encoder"), session("sam-decoder"), image, spec.target
    )
    if not found.mask.any():
        if spec.op is EditOp.BACKGROUND:
            return None, "backdrop, detected (nothing matched the phrase)"
        # The phrase names nothing in this picture. A real outcome, not a fault — the
        # user is told, and the brush is the way through.
        raise MaskTooSmallError(f"nothing in this image matches {spec.target!r}")
    return found.mask, f"{found.source} ({found.confidence:.2f})"


def _fit_mask(mask: Mask, image: RGB) -> Mask:
    """Scale a client's mask onto the working image.

    The mask was drawn against the uploaded resolution; the edit runs at `MAX_SIDE`.
    Nearest-neighbour keeps it binary — anything smoother produces grey edges that then
    read as a partially-selected region.
    """
    import cv2

    if mask.shape[:2] == image.shape[:2]:
        return mask
    resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.asarray(resized, dtype=np.uint8)


def models_for(spec: EditSpec) -> Models:
    """Load only what this operation needs, one session at a time.

    Loading everything would hold four heavy models resident to run one edit, which is
    exactly what the slot exists to prevent.
    """
    if spec.op is EditOp.REMOVE:
        return Models(migan=session("migan"), lama=session("lama"))
    if spec.op is EditOp.UPSCALE:
        return Models(esrgan=session("esrgan-x2"))
    if spec.op is EditOp.BACKGROUND:
        return Models()

    from editgpt_providers import CloudflareWorkersAI

    provider = CloudflareWorkersAI()
    if not provider.is_configured():
        raise ValueError(
            f"{spec.op} needs a generative provider and none is configured; "
            "set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN"
        )
    return Models(fill=provider.fill)


@dataclass(frozen=True, slots=True)
class Edited:
    """What an edit produced. Reported rather than assumed by the caller.

    The dimensions and the content type both belong here because both can differ from the
    request: `UPSCALE` doubles the size, and a format Pillow cannot write falls back to
    PNG. Recording the *asked-for* values instead is how the `images` table comes to
    describe bytes that do not exist.
    """

    data: bytes
    content_type: str
    width: int
    height: int


def edit(source: bytes, spec: EditSpec) -> Edited:
    """Run the job's edit and return the encoded result with its real shape."""
    image = decode_image(source)
    mask, origin = region_for(spec, image)

    result = execute(
        models_for(spec),
        spec.op,
        image,
        mask=mask,
        content=spec.content,
        colour=GREEN,
    )
    log.info(
        "edit.done",
        extra={
            "op": spec.op.value,
            "region": origin,
            "strategy": result.strategy,
            "cost": result.cost,
            "seconds": result.seconds,
            "megapixels": round(image.shape[0] * image.shape[1] / 1e6, 2),
        },
    )
    data, content_type = encode_image(result.image, spec.image_ref.content_type)
    return Edited(
        data=data,
        content_type=content_type,
        width=int(result.image.shape[1]),
        height=int(result.image.shape[0]),
    )
