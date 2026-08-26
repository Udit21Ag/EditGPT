"""Backdrop detection and recolouring.

The case that motivated this: asked to segment "the wooden laptop desk", CLIPSeg plus
SAM returned the tabletop and dropped the thin legs, so recolouring painted over them,
and the boundary left a fringe of the old backdrop. Flooding from the border has
neither failure.
"""

from __future__ import annotations

import numpy as np
from editgpt_models.erase import flat_background_mask, recolour_background

WHITE = 245
GREEN = (46, 160, 67)


def product_shot() -> np.ndarray:
    """A dark object with a thin leg, on a near-white backdrop."""
    image = np.full((200, 200, 3), WHITE, dtype=np.uint8)
    image[60:120, 60:140] = (120, 90, 40)  # the body
    image[120:180, 96:100] = (30, 30, 30)  # a thin leg, 4 px wide
    return image


def busy_scene() -> np.ndarray:
    rng = np.random.default_rng(5)
    return rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)


def test_a_flat_backdrop_is_detected() -> None:
    mask = flat_background_mask(product_shot())
    assert mask is not None
    assert mask[5, 5] > 0, "a corner must be background"
    assert mask[90, 100] == 0, "the object must not be"


def test_thin_structures_survive() -> None:
    """The failure that semantic segmentation had: a 4 px leg is not background."""
    mask = flat_background_mask(product_shot())
    assert mask is not None
    assert mask[150, 98] == 0, "the leg was swallowed by the backdrop"


def test_a_busy_scene_is_refused_so_the_caller_can_fall_back() -> None:
    assert flat_background_mask(busy_scene()) is None


def test_recolour_replaces_the_backdrop_and_keeps_the_subject() -> None:
    image = product_shot()
    background = flat_background_mask(image)
    assert background is not None
    subject = np.asarray(255 - background, dtype=np.uint8)

    out = recolour_background(image, subject, GREEN)
    assert tuple(int(v) for v in out[5, 5]) == GREEN
    assert out[90, 100].mean() < 200, "the object should be untouched"


def test_no_fringe_of_the_old_backdrop_survives() -> None:
    """The halo defect: a boundary sitting inside the object leaves a rim of old colour."""
    image = product_shot()
    background = flat_background_mask(image)
    assert background is not None
    out = recolour_background(image, np.asarray(255 - background, np.uint8), GREEN)

    # Sample a ring two pixels outside the object; none of it should still be white.
    ring = np.concatenate([out[57:59, 58:142].reshape(-1, 3), out[121:123, 58:142].reshape(-1, 3)])
    assert not np.any(np.all(ring > WHITE - 10, axis=1)), "old backdrop survived at the edge"


def test_tolerance_controls_how_much_is_treated_as_backdrop() -> None:
    image = product_shot()
    image[0:200, 0:200][image[..., 0] == WHITE] = (WHITE - 6, WHITE - 6, WHITE - 6)
    tight = flat_background_mask(image, tolerance=2)
    loose = flat_background_mask(image, tolerance=60)
    assert tight is not None
    assert loose is not None
    assert int((loose > 0).sum()) >= int((tight > 0).sum())
