"""The two local erasers.

They fail differently, which is why both ship. MI-GAN erases a small object against a
flat background perfectly where LaMa leaves a ghost; LaMa is clean on a large object
over texture where MI-GAN produces a crumpled artifact. MI-GAN is 7x smaller, 40x
faster to load and 3.6x faster at full resolution, so it runs first and LaMa is the
escalation.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from editgpt_models.compositing import RGB, Mask, erase_in_place

LAMA_SIZE = 512


def make_session(path: Any, threads: int = 4, providers: list[str] | None = None) -> Any:
    """An ONNX Runtime session tuned for a memory-constrained machine."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.enable_mem_pattern = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), options, providers=providers or ["CPUExecutionProvider"])


def lama_fill(session: Any) -> Any:
    """A 512x512 fill backed by Big-LaMa.

    Inputs are named `image` and `mask`, float32, values divided by 255, mask non-zero
    meaning the region to inpaint.
    """

    def fill(rgb: RGB, mask: Mask) -> RGB:
        image = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        binary = ((mask > 0) * 1).astype(np.float32)[None, ...]
        out = session.run(None, {"image": image[None, ...], "mask": binary[None, ...]})[0]
        return np.asarray(np.clip(out[0].transpose(1, 2, 0), 0, 255), dtype=np.uint8)

    return fill


def erase_lama(session: Any, image: RGB, mask: Mask) -> RGB:
    """Erase with LaMa, which needs our crop and paste-back because its input is fixed."""
    return erase_in_place(lama_fill(session), image, mask)


def erase_migan(session: Any, image: RGB, mask: Mask) -> RGB:
    """Erase with MI-GAN, whose exported graph crops, resizes and blends internally.

    The mask convention is INVERTED relative to LaMa: 255 means keep, 0 means inpaint.
    Getting this backwards yields a plausible-looking image with everything except the
    target repainted, which is easy to miss in review.
    """
    if not mask.any():
        return image.copy()
    rgb = image.transpose(2, 0, 1)[None, ...].astype(np.uint8)
    keep = np.where(mask > 0, 0, 255).astype(np.uint8)[None, None, ...]
    out = session.run(None, {"image": rgb, "mask": keep})[0]
    return np.asarray(out[0].transpose(1, 2, 0), dtype=np.uint8)


def residual_mask(
    result: RGB,
    mask: Mask,
    *,
    dark_drop: int = 10,
    reach: float = 1.1,
) -> npt.NDArray[np.uint8]:
    """What still looks wrong near where an object stood, chiefly its cast shadow.

    This detector fails on the original image, because the reference median is polluted
    by the object and its own shadow. Run *after* the object is gone it has a clean
    surface to measure against, and it finds the shadow. That ordering is the whole
    trick.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros_like(mask)

    height, width = mask.shape
    object_height = int(ys.max() - ys.min())
    band = np.zeros_like(mask)
    band[
        int(ys.min()) : min(height, int(ys.max() + reach * object_height)),
        max(0, int(xs.min() - 0.4 * object_height)) : min(
            width, int(xs.max() + 0.4 * object_height)
        ),
    ] = 255

    lightness = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)[:, :, 0]
    reference = cv2.medianBlur(lightness, 151).astype(np.float32)
    dark = ((reference - lightness.astype(np.float32)) > dark_drop) & (band > 0)

    opened = cv2.morphologyEx(
        dark.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((11, 11), np.uint8)
    )
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    return np.asarray(closed, dtype=np.uint8)


def flat_background_mask(
    image: RGB, *, tolerance: int = 30, border_std_limit: float = 18.0
) -> Mask | None:
    """The backdrop of a product shot, found by flooding inward from the border.

    Semantic segmentation is the wrong tool for a flat backdrop. Asked for "the wooden
    laptop desk" it returns the tabletop and drops the thin legs, so they get painted
    over; and its boundary sits a pixel or two inside the object, leaving a halo of the
    old backdrop. Flooding from the border has neither problem: thin structures are
    preserved because they are simply not background, and the boundary is exactly where
    the colour changes.

    Returns None when the border is not uniform enough to be a backdrop, so the caller
    can fall back to a semantic mask.
    """
    height, width = image.shape[:2]
    border = np.concatenate(
        [
            image[0, :].reshape(-1, 3),
            image[-1, :].reshape(-1, 3),
            image[:, 0].reshape(-1, 3),
            image[:, -1].reshape(-1, 3),
        ]
    )
    if float(border.reshape(-1, 3).std(axis=0).mean()) > border_std_limit:
        return None  # a real scene, not a backdrop

    filled = np.zeros((height + 2, width + 2), np.uint8)
    work = image.copy()
    diff = (tolerance,) * 3
    for seed in ((0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)):
        cv2.floodFill(
            work,
            filled,
            seed,
            (0, 0, 0),
            diff,
            diff,
            cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8),
        )

    background = filled[1:-1, 1:-1]
    # Pull the boundary one pixel into the backdrop so no fringe of it survives.
    return np.asarray(cv2.dilate(background, np.ones((3, 3), np.uint8), iterations=1), np.uint8)


def recolour_background(image: RGB, subject: Mask, colour: tuple[int, int, int]) -> RGB:
    """Replace everything outside `subject` with a flat colour, feathering the boundary.

    A solid background change is compositing, not generation: asking a diffusion model
    to paint a wall green costs a network round trip and returns something less exact.
    """
    blurred = np.asarray(
        cv2.GaussianBlur((subject > 0).astype(np.float32), (0, 0), sigmaX=1.5), dtype=np.float32
    )
    alpha = blurred[..., None]
    flat = np.zeros_like(image)
    flat[:] = colour
    return np.asarray(image * alpha + flat * (1 - alpha), dtype=np.uint8)
