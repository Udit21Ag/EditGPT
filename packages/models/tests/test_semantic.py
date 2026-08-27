"""The ReMOVE score, with the encoder replaced by a stub.

MobileSAM is 44 MB and `make check` is hermetic, so the encoder here is a stand-in that
emits a controllable embedding grid. That is enough, because what needs testing is the
arithmetic around it: which grid cells count as inside, which as outside, which as
padding, and the honest zeros for cases where the score is undefined.

Whether the *real* encoder's embeddings make this score track fidelity is an entirely
different question, and no unit test can answer it. `benchmarks/removal.py` does, against
paired ground truth on two datasets.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest
from editgpt_models.semantic import MIN_PATCHES, consistency

GRID = 64
CHANNELS = 8


class _EncoderInput:
    """What MobileSAM's export declares: NCHW, padded to 1024, preprocessed by the caller."""

    name = "images"
    shape: ClassVar[list[int]] = [1, 3, 1024, 1024]


class StubEncoder:
    """Emits a fixed embedding grid, and reports the input shape MobileSAM's export does."""

    def __init__(self, grid: np.ndarray) -> None:
        self.grid = grid[None, ...].astype(np.float32)
        self.calls = 0

    def get_inputs(self) -> list[object]:
        return [_EncoderInput()]

    def run(self, _outputs: object, _feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.calls += 1
        return [self.grid]


def uniform_grid(value: float = 1.0) -> np.ndarray:
    """Every patch identical, so inside and outside must embed the same."""
    return np.full((CHANNELS, GRID, GRID), value, dtype=np.float32)


def split_grid() -> np.ndarray:
    """Top and bottom halves orthogonal, so a region confined to one is unlike the other."""
    grid = np.zeros((CHANNELS, GRID, GRID), dtype=np.float32)
    grid[0, : GRID // 2, :] = 1.0
    grid[1, GRID // 2 :, :] = 1.0
    return grid


def image(size: int = 512) -> np.ndarray:
    return np.full((size, size, 3), 100, dtype=np.uint8)


def band(size: int = 512, top: int = 0, bottom: int = 256) -> np.ndarray:
    mask = np.zeros((size, size), np.uint8)
    mask[top:bottom, :] = 255
    return mask


def test_a_region_embedding_like_its_surroundings_scores_high() -> None:
    """The success case: the object is gone and the fill reads as background."""
    encoder = StubEncoder(uniform_grid())
    assert consistency(encoder, image(), band()) == pytest.approx(1.0, abs=1e-5)


def test_a_region_embedding_unlike_its_surroundings_scores_low() -> None:
    """The failure case: something is still there, and it is not background."""
    encoder = StubEncoder(split_grid())
    score = consistency(encoder, image(), band(top=0, bottom=256))
    assert score == pytest.approx(0.0, abs=1e-5)
    assert score < 1.0


def test_an_empty_mask_is_an_honest_zero_not_a_pass() -> None:
    """Nothing was erased, so there is nothing to judge. Zero says "cannot tell"."""
    encoder = StubEncoder(uniform_grid())
    assert consistency(encoder, image(), np.zeros((512, 512), np.uint8)) == 0.0
    assert encoder.calls == 0, "an undefined case must not pay for a forward pass"


def test_a_full_frame_mask_is_an_honest_zero() -> None:
    """With nothing outside the mask there is no context to compare against."""
    encoder = StubEncoder(uniform_grid())
    assert consistency(encoder, image(), np.full((512, 512), 255, np.uint8)) == 0.0
    assert encoder.calls == 0


def test_a_region_too_small_to_fill_the_minimum_patches_returns_zero() -> None:
    """One patch's mean is noise, not a measurement — see MIN_PATCHES."""
    encoder = StubEncoder(uniform_grid())
    tiny = np.zeros((512, 512), np.uint8)
    tiny[10:12, 10:12] = 255
    assert consistency(encoder, image(), tiny) == 0.0


def test_the_minimum_is_a_stated_number_not_a_magic_one() -> None:
    assert MIN_PATCHES >= 4, "below four the mean is dominated by the boundary patch"


def test_a_zero_embedding_cannot_divide_by_zero() -> None:
    """A degenerate encoder output must return zero rather than a NaN nobody notices."""
    encoder = StubEncoder(np.zeros((CHANNELS, GRID, GRID), dtype=np.float32))
    score = consistency(encoder, image(), band())
    assert score == 0.0
    assert not np.isnan(score)


def test_the_score_never_leaves_the_cosine_range() -> None:
    generator = np.random.default_rng(seed=3)
    for _ in range(5):
        grid = generator.normal(size=(CHANNELS, GRID, GRID)).astype(np.float32)
        score = consistency(StubEncoder(grid), image(), band())
        assert -1.0 <= score <= 1.0


def test_the_encoder_runs_once_per_call() -> None:
    """It is the expensive part; a second pass would double the cost of every erase."""
    encoder = StubEncoder(uniform_grid())
    consistency(encoder, image(), band())
    assert encoder.calls == 1
