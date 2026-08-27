"""Carrying an edit made at a working size back onto the full-resolution original.

The property that matters is not that the output is large — that is trivial — but that
the parts nobody edited are the *original's own pixels*, bit for bit, rather than a
downscale-and-upscale round trip of them. TD-021 was the reverse: a 15.9 MP photograph
returned at roughly 3 MP because the bound on the work had been put on the answer too.
"""

from __future__ import annotations

import cv2
import numpy as np
from editgpt_models.compositing import reproject


def photo(height: int = 400, width: int = 600) -> np.ndarray:
    """Textured, so a resample of it is detectable. A flat image hides every artefact."""
    generator = np.random.default_rng(seed=5)
    return generator.integers(0, 255, (height, width, 3), dtype=np.uint8)


def working(image: np.ndarray, side: int = 150) -> np.ndarray:
    height, width = image.shape[:2]
    scale = side / max(height, width)
    return np.asarray(
        cv2.resize(image, (round(width * scale), round(height * scale))), dtype=np.uint8
    )


def test_the_result_comes_back_at_the_originals_size() -> None:
    original = photo()
    small = working(original)
    mask = np.zeros(small.shape[:2], np.uint8)
    mask[20:40, 20:40] = 255

    out = reproject(original, small, mask)
    assert out.shape == original.shape


def test_pixels_far_from_the_edit_are_the_originals_own() -> None:
    """The whole point. A round trip through the working size would leave every one of
    these slightly softened, across the ~90% of the frame that was never edited."""
    original = photo()
    small = working(original)
    edited = small.copy()
    edited[20:40, 20:40] = 0

    mask = np.zeros(small.shape[:2], np.uint8)
    mask[20:40, 20:40] = 255

    out = reproject(original, small, mask)
    corner = (slice(300, 400), slice(500, 600))  # opposite corner from the mask
    assert np.array_equal(out[corner], original[corner])


def test_the_edit_itself_survives_the_journey() -> None:
    original = photo()
    small = working(original)
    edited = small.copy()
    edited[20:40, 20:40] = 0  # a black square where the edit happened

    mask = np.zeros(small.shape[:2], np.uint8)
    mask[20:40, 20:40] = 255

    out = reproject(original, edited, mask)
    scale = original.shape[0] / small.shape[0]
    centre = out[int(30 * scale), int(30 * scale)]
    assert centre.max() < 40, f"the edit did not arrive: {centre}"


def test_an_image_already_at_full_size_is_returned_untouched() -> None:
    """The common case once an upload is smaller than the working bound: no resample at
    all, rather than a resize to the size it already is."""
    original = photo(120, 160)
    edited = original.copy()
    edited[10:20, 10:20] = 0
    mask = np.zeros(original.shape[:2], np.uint8)
    mask[10:20, 10:20] = 255

    out = reproject(original, edited, mask)
    assert np.array_equal(out, edited)


def test_an_empty_mask_changes_nothing() -> None:
    original = photo()
    small = working(original)
    out = reproject(original, small, np.zeros(small.shape[:2], np.uint8))
    assert np.array_equal(out, original)


def test_only_the_neighbourhood_of_the_edit_is_touched() -> None:
    """Measured regression: blending the whole frame left the Gaussian's tail faintly
    resampling pixels nowhere near the edit — 0.24 grey levels above the JPEG floor on a
    15.9 MP photograph — and multiplied two float arrays the size of the upload to do it.
    """
    original = photo(400, 600)
    small = working(original)
    edited = small.copy()
    edited[20:40, 20:40] = 0

    mask = np.zeros(small.shape[:2], np.uint8)
    mask[20:40, 20:40] = 255

    out = reproject(original, edited, mask)
    scale = original.shape[0] / small.shape[0]
    # Everything below the mask's reach, in full-resolution rows.
    below = slice(int(60 * scale), original.shape[0])
    assert np.array_equal(out[below], original[below])
