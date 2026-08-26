"""The mask codec must survive any shape, including the degenerate ones."""

from __future__ import annotations

import numpy as np
from editgpt_core.rle import decode, encode
from hypothesis import given, settings
from hypothesis import strategies as st


@given(
    width=st.integers(min_value=1, max_value=40),
    height=st.integers(min_value=1, max_value=40),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=150, deadline=None)
def test_round_trip_is_lossless(width: int, height: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    mask = (rng.random((height, width)) > 0.5).astype(np.uint8)
    assert np.array_equal(decode(encode(mask)), mask)


def test_all_zeros_round_trips() -> None:
    mask = np.zeros((7, 5), dtype=np.uint8)
    ref = encode(mask)
    assert ref.area_px == 0
    assert np.array_equal(decode(ref), mask)


def test_all_ones_round_trips() -> None:
    """The encoding opens with a zero run, so a fully-set mask needs a leading 0."""
    mask = np.ones((7, 5), dtype=np.uint8)
    ref = encode(mask)
    assert ref.counts[0] == 0
    assert ref.area_px == 35
    assert np.array_equal(decode(ref), mask)


def test_single_pixel() -> None:
    mask = np.zeros((3, 3), dtype=np.uint8)
    mask[1, 1] = 1
    ref = encode(mask)
    assert ref.area_px == 1
    assert np.array_equal(decode(ref), mask)


def test_non_zero_values_are_treated_as_set() -> None:
    """Callers hand us 0/255 masks from OpenCV as often as 0/1."""
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0, :] = 255
    assert decode(encode(mask)).sum() == 4


def test_area_matches_the_decoded_mask() -> None:
    rng = np.random.default_rng(7)
    mask = (rng.random((32, 24)) > 0.7).astype(np.uint8)
    ref = encode(mask)
    assert ref.area_px == int(mask.sum())
    assert ref.coverage == ref.area_px / (32 * 24)
