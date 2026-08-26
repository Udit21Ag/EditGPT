"""Task features must be computable from inputs alone and must separate real cases."""

from __future__ import annotations

import numpy as np
import pytest
from editgpt_models.features import TaskFeatures, extract


def textured(side: int = 200) -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.asarray(rng.integers(0, 255, (side, side, 3), dtype=np.uint8))


def flat(side: int = 200) -> np.ndarray:
    return np.full((side, side, 3), 180, dtype=np.uint8)


def box(shape: tuple[int, int], y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    m[y0:y1, x0:x1] = 255
    return m


def test_an_empty_mask_yields_zeros_rather_than_a_crash() -> None:
    features = extract(textured(), np.zeros((200, 200), np.uint8))
    assert features.mask_coverage == 0.0
    assert len(features.as_vector()) == len(TaskFeatures.names())


def test_coverage_tracks_mask_area() -> None:
    small = extract(textured(), box((200, 200), 90, 110, 90, 110))
    large = extract(textured(), box((200, 200), 20, 180, 20, 180))
    assert large.mask_coverage > small.mask_coverage * 10


def test_a_thin_shape_is_less_compact_than_a_square() -> None:
    square = extract(textured(), box((200, 200), 60, 140, 60, 140))
    sliver = extract(textured(), box((200, 200), 20, 180, 98, 102))
    assert sliver.mask_compactness < square.mask_compactness


def test_aspect_ratio_detects_elongation() -> None:
    square = extract(textured(), box((200, 200), 60, 140, 60, 140))
    tall = extract(textured(), box((200, 200), 20, 180, 90, 110))
    assert square.aspect_ratio == pytest.approx(1.0, abs=0.05)
    assert tall.aspect_ratio > 3.0


def test_ring_edge_density_separates_flat_from_textured_surroundings() -> None:
    mask = box((200, 200), 80, 120, 80, 120)
    assert extract(textured(), mask).ring_edge_density > extract(flat(), mask).ring_edge_density * 5


def test_border_contact_detects_an_object_touching_the_frame() -> None:
    inside = extract(textured(), box((200, 200), 60, 140, 60, 140))
    touching = extract(textured(), box((200, 200), 0, 80, 0, 80))
    assert inside.border_contact == pytest.approx(0.0, abs=1e-6)
    assert touching.border_contact > 0.2


def test_features_use_no_information_beyond_image_and_mask() -> None:
    """The signature is the guarantee: anything else would make these unusable at
    inference time, which is the only time they matter."""
    import inspect

    assert list(inspect.signature(extract).parameters) == ["image", "mask"]


def test_the_vector_matches_the_declared_names() -> None:
    features = extract(textured(), box((200, 200), 80, 120, 80, 120))
    assert len(features.as_vector()) == len(TaskFeatures.names())
    assert set(features.as_dict()) == set(TaskFeatures.names())


def test_features_are_finite_for_degenerate_masks() -> None:
    """A one-pixel or full-frame mask must not produce NaN, which would poison training."""
    for mask in (box((200, 200), 100, 101, 100, 101), np.full((200, 200), 255, np.uint8)):
        assert np.all(np.isfinite(extract(textured(), mask).as_vector()))
