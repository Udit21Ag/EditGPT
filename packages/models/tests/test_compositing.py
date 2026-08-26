"""Mask growth and colour matching, the two fixes that made large edits usable."""

from __future__ import annotations

import numpy as np
import pytest
from editgpt_models.compositing import (
    DILATE_MAX,
    DILATE_MIN,
    crop_window,
    dilate_px,
    grow,
    match_chroma,
)


def box(shape: tuple[int, int], y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    m = np.zeros(shape, dtype=np.uint8)
    m[y0:y1, x0:x1] = 255
    return m


def test_dilation_scales_with_the_object_not_the_image() -> None:
    """12 px was ample at 1024 px and left a visible outline at 15.9 MP."""
    small = box((200, 200), 90, 110, 90, 110)  # 20 px object
    large = box((4000, 4000), 1000, 3000, 1000, 3000)  # 2000 px object
    assert dilate_px(large) > dilate_px(small)


def test_dilation_is_clamped_at_both_ends() -> None:
    speck = box((100, 100), 50, 51, 50, 51)
    assert dilate_px(speck) == DILATE_MIN

    huge = box((20000, 20000), 0, 19999, 0, 19999)
    assert dilate_px(huge) == DILATE_MAX


def test_dilation_of_an_empty_mask_is_the_floor_not_a_crash() -> None:
    assert dilate_px(np.zeros((50, 50), dtype=np.uint8)) == DILATE_MIN


def test_grow_expands_the_masked_area() -> None:
    mask = box((200, 200), 90, 110, 90, 110)
    assert int((grow(mask) > 0).sum()) > int((mask > 0).sum())


def test_crop_window_is_square_and_inside_the_image() -> None:
    mask = box((300, 400), 100, 140, 200, 260)
    rows, cols, side = crop_window(mask, (300, 400))
    assert rows.stop - rows.start == side
    assert cols.stop - cols.start == side
    assert rows.start >= 0
    assert rows.stop <= 300
    assert cols.start >= 0
    assert cols.stop <= 400


def test_crop_window_contains_the_mask() -> None:
    mask = box((300, 400), 100, 140, 200, 260)
    rows, cols, _ = crop_window(mask, (300, 400))
    assert mask[rows, cols].sum() == mask.sum()


def test_chroma_match_corrects_a_blue_cast() -> None:
    rng = np.random.default_rng(3)
    source = np.zeros((120, 120, 3), dtype=np.uint8)
    source[..., 0], source[..., 1], source[..., 2] = 176, 173, 163
    source = np.clip(source + rng.normal(0, 3, (120, 120, 3)), 0, 255).astype(np.uint8)

    mask = box((120, 120), 40, 80, 40, 80)
    patch = source.copy()
    patch[mask > 0] = [176, 180, 195]  # the measured drift: luminance right, chroma blue

    corrected = match_chroma(patch, source, mask)
    before = abs(int(patch[mask > 0][:, 2].mean()) - int(source[mask == 0][:, 2].mean()))
    after = abs(int(corrected[mask > 0][:, 2].mean()) - int(source[mask == 0][:, 2].mean()))
    assert after < before


def test_chroma_match_leaves_luminance_alone() -> None:
    """Matching luminance too re-stamps the object where the ring spans two surfaces."""
    source = np.full((120, 120, 3), 120, dtype=np.uint8)
    mask = box((120, 120), 40, 80, 40, 80)
    patch = source.copy()
    patch[mask > 0] = [60, 60, 60]  # much darker, but neutral in chroma

    corrected = match_chroma(patch, source, mask)
    assert corrected[mask > 0].mean() == pytest.approx(60, abs=12)


def test_chroma_match_is_a_no_op_without_a_usable_ring() -> None:
    source = np.full((30, 30, 3), 120, dtype=np.uint8)
    everything = np.full((30, 30), 255, dtype=np.uint8)
    assert np.array_equal(match_chroma(source.copy(), source, everything), source)
