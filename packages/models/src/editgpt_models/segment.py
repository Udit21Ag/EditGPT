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

import hashlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from editgpt_core.rle import encode as encode_rle
from editgpt_core.spec import Grounding, MaskCandidate

from editgpt_models.compositing import RGB, Mask, dilate_px
from editgpt_models.config import Thresholds, load_thresholds

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


log = logging.getLogger(__name__)

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


_LAST_EMBEDDING: tuple[str, np.ndarray, float] | None = None


def embed(encoder: Any, rgb: RGB) -> tuple[np.ndarray, float]:
    """Encode an image for the decoder, reusing the last result if it is the same image.

    The split in `masks_from_boxes` — expensive encoder, nearly-free decoder — has a
    second consequence once occluder probing exists: one removal encodes the same image
    twice, once to turn a box into a mask and once to probe around that mask. The second
    encode is ~1.5 s of pure waste.

    One entry is enough. The worker runs at concurrency 1 by design (`ModelSlot` holds a
    single heavy model resident), so there is never a second image in flight. The key is
    a hash of the pixels rather than an object identity, so a cache hit means the same
    picture rather than the same array — correctness does not rest on the caller reusing
    a buffer.
    """
    global _LAST_EMBEDDING
    digest = hashlib.blake2b(np.ascontiguousarray(rgb).tobytes(), digest_size=16).hexdigest()
    if _LAST_EMBEDDING is not None and _LAST_EMBEDDING[0] == digest:
        return _LAST_EMBEDDING[1], _LAST_EMBEDDING[2]

    prepared, scale = preprocess_for_encoder(rgb, encoder)
    embedding = encoder.run(None, {encoder.get_inputs()[0].name: prepared})[0]
    _LAST_EMBEDDING = (digest, embedding, scale)
    return embedding, scale


def masks_from_boxes(
    encoder: Any, decoder: Any, rgb: RGB, boxes: Sequence[tuple[float, float, float, float]]
) -> list[Segmentation]:
    """Several box prompts over one image, encoding it **once**.

    The split matters: the encoder is the expensive half (~1.5 s) and the decoder is
    nearly free, and the embedding depends only on the image. Calling `mask_from_box` in
    a loop re-encodes per box, which turned a five-candidate request into five encoder
    passes — measured as the difference between a 40-minute benchmark and a two-hour one.

    Returns one `Segmentation` per box, in the order given.
    """
    if not boxes:
        return []

    height, width = rgb.shape[:2]
    embedding, scale = embed(encoder, rgb)

    found = []
    for x0, y0, x1, y1 in boxes:
        coords = np.array(
            [[[width * x0, height * y0], [width * x1, height * y1]]], dtype=np.float32
        )
        labels = np.array([[2, 3]], dtype=np.float32)  # SAM encodes a box as two corners
        mask, iou = _decode(decoder, embedding, coords, labels, (height, width), scale)
        found.append(Segmentation(mask=mask, confidence=iou, source="sam-box"))
    return found


def mask_from_box(
    encoder: Any, decoder: Any, rgb: RGB, box: tuple[float, float, float, float]
) -> Segmentation:
    """One box prompt, in fractions of the image, to a precise mask."""
    return masks_from_boxes(encoder, decoder, rgb, [box])[0]


def candidates_from_phrase(
    detector: Any,
    encoder: Any,
    decoder: Any,
    rgb: RGB,
    phrase: str,
    *,
    top_k: int | None = None,
    min_score: float | None = None,
    margin: float | None = None,
) -> Grounding:
    """Every region a phrase might mean, and whether we should ask before acting.

    This is the answer to TD-015. Two unrelated grounding models fail on the *same* third
    of held-out phrases, because both ground the noun and then pick an arbitrary instance;
    no better detector fixes that, and every model that resolves relations is GPU-class.
    Offering the alternatives does: measured on 250 held-out RefCOCOg samples, picking
    from five takes the hit rate from **0.516 to 0.832** and mIoU from 0.469 to 0.731 —
    the range published *trained* referring-segmentation models occupy.

    Costs one encoder pass regardless of `top_k`, which is why `masks_from_boxes` exists.
    """
    from editgpt_models.detect import detect

    tuning = load_thresholds()
    wanted = tuning.candidates if top_k is None else top_k
    gate = tuning.ambiguity_margin if margin is None else margin

    found = detect(detector, rgb, phrase, min_score=min_score, top_k=wanted)
    if not found:
        return Grounding(candidates=[], ambiguous=False, margin=0.0)

    segmented = masks_from_boxes(encoder, decoder, rgb, [c.box for c in found])
    candidates = [
        MaskCandidate(box=c.box, score=c.score, mask_ref=encode_rle(s.mask))
        for c, s in zip(found, segmented, strict=True)
        if s.mask.any()
    ]
    if not candidates:
        return Grounding(candidates=[], ambiguous=False, margin=0.0)

    spread = candidates[0].score - candidates[1].score if len(candidates) > 1 else 1.0
    return Grounding(
        candidates=candidates,
        # A lone candidate is never ambiguous: there is nothing to be ambiguous between.
        ambiguous=len(candidates) > 1 and spread < gate,
        margin=round(min(max(spread, 0.0), 1.0), 4),
    )


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
    embedding, scale = embed(encoder, rgb)

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


PROBE_SPACING = 8
"""Grid spacing, in pixels at the working resolution, for occluder probes.

Not a fitted value and not a quality knob: it is how finely the target's edge is walked.
Cost is bounded by `PROBE_CAP` and by the "already explained" skip rather than by this,
so it is set fine enough that a small occluder — a shoe against a tower — is not stepped
over. Measured: at 16 px the jumper's shoe in `i8` is missed on some runs; at 8 px it is
found on all of them, for 32 decoder calls and 0.8 s.
"""

PROBE_CAP = 64
"""Ceiling on decoder calls per shield, so a long boundary cannot dominate an edit."""


def occluder_shield(
    encoder: Any,
    decoder: Any,
    rgb: RGB,
    target: Mask,
    *,
    thresholds: Thresholds | None = None,
) -> Mask:
    """Regions next to `target` that belong to something else, and must not be erased.

    The failure this addresses: erasing the Eiffel Tower also erased the shoe of a man
    jumping in front of it, because the mask is grown by 5% of the object's longest side
    and that slack lands on whatever is adjacent. Dilation is deliberately generous — a
    tight mask leaves a halo of the object's own edge pixels — so the fix is not less
    slack but slack that stops at the next object.

    Occluders are found rather than assumed. Every probe point sits *outside* the target,
    on a grid across the band dilation would reach, and each is one decoder call against
    an embedding that already exists. A returned region is treated as an occluder when it
    is coherent, bounded in size, and mostly **not** the target.

    That last test is the load-bearing one, and it is containment rather than IoU.
    MobileSAM is part-aware: a point on a horse's edge returns *the leg*, which is small,
    confident, and has a low IoU against the whole horse — it passes every other filter
    and shields a quarter of the animal from its own erasure. What separates a part from
    an occluder is that a part lies inside the target and an occluder does not.

    A margin around the target is never shielded, which bounds the damage a wrong shield
    can do and does it structurally rather than by a percentage check. Two things follow.
    The anti-aliasing slack dilation exists to provide survives even where a neighbour is
    pressed against the target. And because the margin covers the selection itself, the
    shield can only ever withhold *grown* pixels — so a shield, however wrong, can never
    leave the subject standing in the result.

    **Known limitation.** This only sees occluders that reach outside the target's
    silhouette. An occluder *within* it — the glove gripping the bat in `i9`, which SAM
    folds into the bat itself — is invisible here, and probing inside is what caused the
    horse regression above. Separating those needs the mask hierarchy that MobileSAM's
    multi-mask decoder exposes and this single-mask export does not. TD-004 carries it.
    """
    limits = thresholds or load_thresholds()
    height, width = target.shape[:2]
    shape = (height, width)
    band = dilate_px(target)
    if band <= 0 or not target.any():
        return np.zeros(shape, np.uint8)

    reach = cv2.dilate(target, np.ones((band, band), np.uint8))
    outward = (reach > 0) & (target == 0)
    ys, xs = np.nonzero(outward)
    if len(xs) == 0:
        return np.zeros(shape, np.uint8)

    seen: set[tuple[int, int]] = set()
    points: list[tuple[int, int]] = []
    for x, y in zip(xs, ys, strict=True):
        cell = (int(x) // PROBE_SPACING, int(y) // PROBE_SPACING)
        if cell in seen:
            continue
        seen.add(cell)
        points.append((int(x), int(y)))

    embedding, scale = embed(encoder, rgb)
    inside = target > 0
    shield = np.zeros(shape, bool)
    # Any region a probe has already returned, whether kept or rejected. Probing the sky
    # around a tower returns the same sky every time; skipping points it already covers
    # is what makes an 8 px grid affordable.
    explained = np.zeros(shape, bool)
    calls = 0

    for x, y in points:
        if explained[y, x] or calls >= PROBE_CAP:
            continue
        found, confidence = _decode(
            decoder,
            embedding,
            np.array([[[x, y], [0.0, 0.0]]], dtype=np.float32),
            np.array([[1, -1]], dtype=np.float32),  # one positive point, one padding
            shape,
            scale,
        )
        calls += 1
        region = found > 0
        if not region.any():
            continue
        explained |= region
        if confidence < limits.min_sam_iou:
            continue
        if region.mean() > limits.shield_max_area:
            continue
        if (region & inside).sum() / region.sum() > limits.shield_max_inside:
            continue
        shield |= region

    if not shield.any():
        return np.zeros(shape, np.uint8)

    # `band` is a kernel size, so the growth it produces is half of it. An odd kernel of
    # 2r+1 dilates by exactly r, which keeps the margin a stated distance rather than an
    # artefact of OpenCV's anchor placement for even kernels.
    margin = max(round(band / 2 * limits.shield_margin_frac), 1)
    keep_clear = cv2.dilate(target, np.ones((2 * margin + 1, 2 * margin + 1), np.uint8))
    shield &= keep_clear == 0

    log.info(
        "shield.applied",
        extra={"probes": calls, "fraction": round(float(shield.mean()), 4), "margin": margin},
    )
    return np.asarray(shield, dtype=np.uint8) * 255
