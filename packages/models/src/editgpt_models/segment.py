"""Turning an intent into a mask.

Two stages, because they are good at different things. Something turns free text into a
coarse region, and MobileSAM turns that region — or a box, or a brush stroke — into a
precise boundary. SAM's refinement helps a weak prompt and *hurts* a strong one, so it
is applied conditionally on the decoder's own confidence rather than unconditionally.

The first stage has two implementations and they are not equivalent:

* `mask_from_phrase` (**preferred**) prompts SAM with a Grounding DINO box. See
  `detect.py` and `docs/adr/0002-text-grounding.md` for the measurement that put it
  first.
* `mask_from_seed` prompts SAM with a CLIPSeg heatmap. Retained for "stuff" nouns —
  sky, grass, wall — which a detector trained on objects grounds poorly, and as the
  fallback when the detector finds nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from editgpt_models.compositing import RGB, Mask
from editgpt_models.config import load_thresholds

ENCODER_SIZE = 1024
SAM_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
SAM_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def min_sam_iou() -> float:
    """Below this the decoder's own confidence says its refinement is not trustworthy.

    Read on every call rather than snapshotted at import. It used to be a module constant
    assigned from `load_thresholds()`, which looks wired but is not: the value froze at
    the first import, so pointing `EDITGPT_THRESHOLDS` at a fitted file changed nothing
    unless the process happened to start afterwards. Callers that need to vary it still
    pass `min_iou=` explicitly.
    """
    return load_thresholds().min_sam_iou


CLIPSEG_MODEL = "CIDAS/clipseg-rd64-refined"
CLIPSEG_THRESHOLD = 0.4


@dataclass(frozen=True, slots=True)
class Segmentation:
    mask: Mask
    confidence: float
    source: str
    """Which stage produced the returned mask, for the audit trail."""


def preprocess_for_encoder(rgb: RGB, session: Any) -> tuple[np.ndarray, float]:
    """Shape the encoder input to whatever this export declares, and return the scale.

    Acly's export normalises and PADS to 1024 but does not resize, leaving SAM's
    ResizeLongestSide to the caller. Skipping it silently desynchronises the embedding
    from the point coordinates and the mask tears along the prompt box instead of
    snapping to an object.
    """
    height, width = rgb.shape[:2]
    scale = ENCODER_SIZE / max(height, width)
    declared = session.get_inputs()[0].shape

    resized = cv2.resize(
        rgb, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA
    )
    if len(declared) == 3:  # HWC, the encoder preprocesses internally
        return resized.astype(np.float32), scale

    canvas = np.zeros((ENCODER_SIZE, ENCODER_SIZE, 3), dtype=np.uint8)
    canvas[: resized.shape[0], : resized.shape[1]] = resized
    normalised = (canvas.astype(np.float32) - SAM_MEAN) / SAM_STD
    return normalised.transpose(2, 0, 1)[None, ...], scale


def _decode(
    decoder: Any,
    embedding: np.ndarray,
    coords: np.ndarray,
    labels: np.ndarray,
    shape: tuple[int, int],
    scale: float,
) -> tuple[Mask, float]:
    height, width = shape
    feed = {
        "image_embeddings": embedding,
        "point_coords": coords * scale,
        "point_labels": labels,
        "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
        "has_mask_input": np.zeros(1, dtype=np.float32),
        "orig_im_size": np.array([height, width], dtype=np.float32),
    }
    names = {i.name for i in decoder.get_inputs()}
    missing = names - feed.keys()
    if missing:
        raise ValueError(f"decoder expects unmapped inputs {sorted(missing)}")

    outputs = decoder.run(None, {k: v for k, v in feed.items() if k in names})
    masks, iou = outputs[0], float(np.ravel(outputs[1])[0])

    mask = (masks[0, 0] > 0).astype(np.uint8) * 255
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (ENCODER_SIZE, ENCODER_SIZE), interpolation=cv2.INTER_NEAREST)
        mask = mask[: round(height * scale), : round(width * scale)]
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask, iou


def mask_from_box(
    encoder: Any, decoder: Any, rgb: RGB, box: tuple[float, float, float, float]
) -> Segmentation:
    """Box prompt, in fractions of the image, to a precise mask."""
    height, width = rgb.shape[:2]
    prepared, scale = preprocess_for_encoder(rgb, encoder)
    embedding = encoder.run(None, {encoder.get_inputs()[0].name: prepared})[0]

    x0, y0, x1, y1 = box
    coords = np.array([[[width * x0, height * y0], [width * x1, height * y1]]], dtype=np.float32)
    labels = np.array([[2, 3]], dtype=np.float32)  # SAM encodes a box as two corners
    mask, iou = _decode(decoder, embedding, coords, labels, (height, width), scale)
    return Segmentation(mask=mask, confidence=iou, source="sam-box")


def mask_from_phrase(
    detector: Any,
    encoder: Any,
    decoder: Any,
    rgb: RGB,
    phrase: str,
    *,
    min_score: float | None = None,
    min_iou: float | None = None,
) -> Segmentation:
    """A phrase to a precise mask: detect the region, then let SAM find its boundary.

    When SAM is not confident the detector's box is kept as a filled rectangle. That is
    a worse mask than a traced boundary but a perfectly usable one for an erase — the
    erasers take any shape — and it is a much better answer than a heatmap blob, which
    is what the old fallback returned.
    """
    from editgpt_models.detect import detect

    height, width = rgb.shape[:2]
    found = detect(detector, rgb, phrase, min_score=min_score, top_k=1)
    if not found:
        # The phrase names nothing in this picture. A real outcome, not a failure: the
        # caller shows the brush rather than erasing an arbitrary region.
        return Segmentation(mask=np.zeros((height, width), np.uint8), confidence=0.0, source="none")

    best = found[0]
    refined = mask_from_box(encoder, decoder, rgb, best.box)
    gate = min_sam_iou() if min_iou is None else min_iou
    if refined.confidence >= gate and refined.mask.any():
        return Segmentation(mask=refined.mask, confidence=best.score, source="sam-box")

    x0, y0, x1, y1 = best.box
    rectangle = np.zeros((height, width), np.uint8)
    rectangle[round(y0 * height) : round(y1 * height), round(x0 * width) : round(x1 * width)] = 255
    return Segmentation(mask=rectangle, confidence=best.score, source="detector-box")


def mask_from_seed(
    encoder: Any, decoder: Any, rgb: RGB, heat: np.ndarray, *, min_iou: float | None = None
) -> Segmentation:
    """A coarse heatmap to a precise mask, keeping the seed if SAM is not confident.

    The SAM prompt is the seed's bounding box plus its hottest pixel as a positive
    point, so a thin seed can still name the whole object.
    """
    height, width = rgb.shape[:2]
    seed = (heat > CLIPSEG_THRESHOLD).astype(np.uint8)

    count, labels_img, stats, _ = cv2.connectedComponentsWithStats(seed.astype(np.uint8), 8)
    if count > 1:  # CLIPSeg speckles; keep only the largest blob
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        seed = (labels_img == largest).astype(np.uint8)

    ys, xs = np.nonzero(seed)
    if len(xs) == 0:
        # A prompt matching nothing is a legitimate outcome, not an error. Return an
        # empty mask at zero confidence so the caller can ask the user instead of
        # crashing on a zero-size reduction.
        return Segmentation(mask=np.zeros((height, width), np.uint8), confidence=0.0, source="none")

    peak_y, peak_x = np.unravel_index(int(np.argmax(heat * seed)), heat.shape)
    prepared, scale = preprocess_for_encoder(rgb, encoder)
    embedding = encoder.run(None, {encoder.get_inputs()[0].name: prepared})[0]

    coords = np.array(
        [[[int(xs.min()), int(ys.min())], [int(xs.max()), int(ys.max())], [peak_x, peak_y]]],
        dtype=np.float32,
    )
    labels = np.array([[2, 3, 1]], dtype=np.float32)
    refined, iou = _decode(decoder, embedding, coords, labels, (height, width), scale)

    gate = min_sam_iou() if min_iou is None else min_iou
    if iou < gate:
        return Segmentation(mask=seed * 255, confidence=iou, source="clipseg-seed")
    return Segmentation(mask=refined, confidence=iou, source="sam-refined")


def load_clipseg() -> tuple[Any, Any]:
    """CLIPSeg, still on torch. Needs `editgpt-models[text]`."""
    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

    # transformers ships partial annotations; `.eval()` is untyped there. Widen at the
    # boundary rather than sprinkling ignores through the call sites.
    processor: Any = CLIPSegProcessor.from_pretrained(CLIPSEG_MODEL)
    model: Any = CLIPSegForImageSegmentation.from_pretrained(CLIPSEG_MODEL)
    model.eval()
    return processor, model


def seed_from_text(processor: Any, model: Any, image: Any, target: str) -> np.ndarray:
    """A text phrase to a per-pixel heatmap at the image's resolution."""
    import torch

    inputs = processor(text=[target], images=[image], padding=True, return_tensors="pt")
    with torch.inference_mode():
        logits = model(**inputs).logits
    if logits.ndim == 2:  # a single prompt comes back without the batch axis
        logits = logits[None, ...]
    heat = torch.sigmoid(logits)[0].numpy()
    return np.asarray(cv2.resize(heat, image.size, interpolation=cv2.INTER_LINEAR))
