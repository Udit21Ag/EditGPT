"""Putting an inpainted patch back into a full-resolution image without a seam.

Both fixes here were measured in Phase 0 and both are counter-intuitive enough to be
worth stating:

* **Match chroma, never luminance.** LaMa reproduces luminance to within 3/255 and
  drifts chroma about 25 units toward blue; that drift is what reads as a pale ghost
  where an object used to be. Matching luminance as well re-stamps the object wherever
  the reference ring spans two surfaces, such as a table plus the object's own shadow.
* **Dilation scales with the object, never in fixed pixels.** 12 px was ample at
  1024 px and left a visible rectangular outline at 15.9 MP. Residual edge energy
  reaches the blank-surface floor at about 5% of the object's longest side.
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np
import numpy.typing as npt

from editgpt_models.config import load_thresholds

RGB = npt.NDArray[np.uint8]
Mask = npt.NDArray[np.uint8]
Fill = Callable[[RGB, Mask], RGB]

CROP_SIZE = 512
"""LaMa's ONNX export has a fixed 512x512 input, so the window is not negotiable."""

CONTEXT = 1.6
"""Crop window as a multiple of the mask's longest side, to give the model surroundings."""

RING_PX = 60
DILATE_MIN = 8
DILATE_MAX = 128


def dilate_px(mask: Mask, *, frac: float | None = None) -> int:
    """How far to grow a mask past the object edge, relative to the object's size.

    The fraction comes from `editgpt_models.config`, not a constant here, so a value
    fitted on held-out data replaces the Phase 0 one without a code change.
    """
    if frac is None:
        frac = load_thresholds().dilate_frac
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return DILATE_MIN
    longest = max(int(xs.max() - xs.min()), int(ys.max() - ys.min()))
    return int(np.clip(round(frac * longest), DILATE_MIN, DILATE_MAX))


def grow(mask: Mask, *, frac: float | None = None) -> Mask:
    """Dilate a mask by the object-relative amount."""
    size = dilate_px(mask, frac=frac)
    return np.asarray(cv2.dilate(mask, np.ones((size, size), np.uint8), iterations=1), np.uint8)


def match_chroma(patch: RGB, source: RGB, mask: Mask, ring_px: int = RING_PX) -> RGB:
    """Pull a fill's colour onto the surrounding surface, leaving its luminance alone."""
    inside = mask > 0
    if not inside.any():
        return patch
    ring = (cv2.dilate(mask, np.ones((ring_px, ring_px), np.uint8)) > 0) & ~inside
    if int(ring.sum()) < 100:
        return patch

    lab_patch = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_src = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype(np.float32)
    out = lab_patch.copy()
    for channel in (1, 2):  # a and b only; see the module docstring
        p_mean = lab_patch[inside, channel].mean()
        p_std = lab_patch[inside, channel].std() + 1e-6
        r_mean = lab_src[ring, channel].mean()
        r_std = lab_src[ring, channel].std() + 1e-6
        scale = min(r_std / p_std, 2.0)
        out[inside, channel] = (lab_patch[inside, channel] - p_mean) * scale + r_mean
    return np.asarray(
        cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB), dtype=np.uint8
    )


def crop_window(mask: Mask, shape: tuple[int, int]) -> tuple[slice, slice, int]:
    """A square window around the mask with `CONTEXT` times its extent of surroundings."""
    height, width = shape
    ys, xs = np.nonzero(mask)
    centre_x, centre_y = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    extent = max(int(xs.max() - xs.min()), int(ys.max() - ys.min())) * CONTEXT
    side = int(np.clip(max(extent, CROP_SIZE // 2), 32, min(height, width)))
    x0 = int(np.clip(centre_x - side / 2, 0, width - side))
    y0 = int(np.clip(centre_y - side / 2, 0, height - side))
    return slice(y0, y0 + side), slice(x0, x0 + side), side


def erase_in_place(fill: Fill, image: RGB, mask: Mask) -> RGB:
    """Crop around the mask, fill at 512, colour-match, feather back at full resolution.

    Only the fill differs between the local erasers and a remote provider, which is what
    makes a side-by-side of the two lanes a comparison of models rather than of plumbing.
    """
    if not mask.any():
        return image.copy()

    rows, cols, side = crop_window(mask, mask.shape)
    crop = np.asarray(
        cv2.resize(image[rows, cols], (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA),
        dtype=np.uint8,
    )
    crop_mask = np.asarray(
        cv2.resize(mask[rows, cols], (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_NEAREST),
        dtype=np.uint8,
    )

    patch = np.asarray(
        cv2.resize(fill(crop, crop_mask), (side, side), interpolation=cv2.INTER_LANCZOS4),
        dtype=np.uint8,
    )
    patch = match_chroma(patch, image[rows, cols], mask[rows, cols])

    alpha = cv2.GaussianBlur(mask[rows, cols].astype(np.float32) / 255.0, (0, 0), sigmaX=side / 60)
    alpha = np.clip(alpha * 1.6, 0, 1)[..., None]

    out = image.copy()
    out[rows, cols] = (patch * alpha + image[rows, cols] * (1 - alpha)).astype(np.uint8)
    return out
