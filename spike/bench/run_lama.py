"""Big-LaMa erase, ONNX, CPU.

The Carve export has a FIXED 512x512 input, which forces the question Phase 0
exists to answer: do we downscale the whole image (cheap, soft) or crop a
512 window around the mask and paste back at full resolution (sharp, and the
strategy production will use)? We measure both.

I/O confirmed from the Carve demo: inputs 'image' and 'mask', float32,
values /255, mask non-zero = region to inpaint, output CHW uint8-ish.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
from PIL import Image

from bench.common import MODELS, OUT, box_mask, describe_io, emit, load, photos, session, timed

SIZE = 512
CONTEXT = 1.6  # how much surrounding context to include in a crop, as a multiple of the mask box
RING_PX = 60  # width of the reference ring used for colour matching


def color_match(patch: np.ndarray, src: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Pull the fill's colour statistics back onto the surrounding surface.

    Measured on i12: LaMa reproduces luminance to within 3/255 but drifts the
    chroma ~25 units toward blue, which is what reads as a pale rectangle where
    an object used to be. Matching mean and spread in Lab against a ring of real
    pixels around the mask fixes the cast without touching the fill's structure.
    """
    inside = mask > 0
    if not inside.any():
        return patch
    ring = (cv2.dilate(mask, np.ones((RING_PX, RING_PX), np.uint8)) > 0) & ~inside
    if ring.sum() < 100:
        return patch

    lab_patch = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_src = cv2.cvtColor(src, cv2.COLOR_RGB2LAB).astype(np.float32)

    # Chroma only. The measurement that motivated this said luminance was already
    # correct to 3/255 and chroma was off by ~25. Shifting L as well re-stamped the
    # object on i11, where the ring spans both the table and the mug's cast shadow,
    # so the "correct" L to match to is an average of two different surfaces.
    out = lab_patch.copy()
    for ch in (1, 2):
        p_mean, p_std = lab_patch[inside, ch].mean(), lab_patch[inside, ch].std() + 1e-6
        r_mean, r_std = lab_src[ring, ch].mean(), lab_src[ring, ch].std() + 1e-6
        scale = min(r_std / p_std, 2.0)
        out[inside, ch] = (lab_patch[inside, ch] - p_mean) * scale + r_mean
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def _infer(sess, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """rgb HxWx3 uint8, mask HxW uint8 -> inpainted HxWx3 uint8, both already 512x512."""
    image = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    m = ((mask > 0) * 1).astype(np.float32)[None, ...]
    out = sess.run(None, {"image": image[None, ...], "mask": m[None, ...]})[0]
    out = np.clip(out[0].transpose(1, 2, 0), 0, 255).astype(np.uint8)
    return out


def erase_whole(sess, img: Image.Image, mask: np.ndarray) -> Image.Image:
    """Downscale the entire image to 512, inpaint, upscale back."""
    rgb = cv2.resize(np.array(img), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    m = cv2.resize(mask, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    out = _infer(sess, rgb, m)
    out = cv2.resize(out, img.size, interpolation=cv2.INTER_LANCZOS4)
    return Image.fromarray(out)


def erase_crop(sess, img: Image.Image, mask: np.ndarray) -> Image.Image:
    """Crop a context window around the mask, inpaint at 512, feather back in.
    This is the strategy CompositorAgent will inherit in Phase 5."""
    return erase_crop_with(lambda rgb, m: _infer(sess, rgb, m), img, mask)


def erase_crop_with(fill, img: Image.Image, mask: np.ndarray) -> Image.Image:
    """Same compositing, arbitrary fill backend.

    `fill(rgb512, mask512) -> rgb512` is the only thing that differs between the
    local LaMa lane and a remote provider, so routing between them cannot change
    the crop, the colour match or the feather. That is what makes a side-by-side
    comparison of the two lanes mean anything.
    """
    src = np.array(img)
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return img

    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    side = max(xs.max() - xs.min(), ys.max() - ys.min()) * CONTEXT
    side = int(np.clip(max(side, SIZE // 2), 32, min(h, w)))
    x0 = int(np.clip(cx - side / 2, 0, w - side))
    y0 = int(np.clip(cy - side / 2, 0, h - side))
    win = (slice(y0, y0 + side), slice(x0, x0 + side))

    crop = cv2.resize(src[win], (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    cmask = cv2.resize(mask[win], (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    patch = cv2.resize(fill(crop, cmask), (side, side), interpolation=cv2.INTER_LANCZOS4)
    patch = color_match(patch, src[win], mask[win])

    # Feather the seam so the paste-back is invisible.
    alpha = cv2.GaussianBlur(mask[win].astype(np.float32) / 255.0, (0, 0), sigmaX=side / 60)
    alpha = np.clip(alpha * 1.6, 0, 1)[..., None]
    out = src.copy()
    out[win] = (patch * alpha + src[win] * (1 - alpha)).astype(np.uint8)
    return Image.fromarray(out)


def main() -> None:
    t0 = time.perf_counter()
    sess = session(MODELS / "lama.onnx")
    cold = round(time.perf_counter() - t0, 3)

    files = photos()
    img = load(files[0])
    mask = box_mask(img.size, (0.35, 0.45, 0.65, 0.85))

    whole = timed(lambda: erase_whole(sess, img, mask))
    crop = timed(lambda: erase_crop(sess, img, mask))

    OUT.mkdir(exist_ok=True)
    for path in files:
        im = load(path)
        mk = box_mask(im.size, (0.35, 0.45, 0.65, 0.85))
        erase_whole(sess, im, mk).save(OUT / f"lama_whole_{path.stem}.png")
        erase_crop(sess, im, mk).save(OUT / f"lama_crop_{path.stem}.png")

    emit(
        {
            "cold_load_s": cold,
            "warm_p50_s": crop["warm_p50_s"],
            "modes": {"whole_image_512": whole, "crop_512_pasteback": crop},
            "io": describe_io(sess),
            "n_photos": len(files),
            "note": "eyeball out/lama_*.png before trusting any of these numbers",
        }
    )


if __name__ == "__main__":
    main()
