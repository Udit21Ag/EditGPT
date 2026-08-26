"""Scoring behaves the way Phase 0 measured, including where it must refuse to.

The regression that matters here is `compare`: raw fill cost prefers the variant that
erases more of the scene, because a large flat region is photometrically consistent
with itself. These tests pin that down so the mistake cannot come back.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from editgpt_core.metrics import GROWTH_PENALTY, box_metrics, compare, fill_metrics


def scene(shape: tuple[int, int] = (200, 200)) -> np.ndarray:
    """A textured beige surface, standing in for a wall or a desk."""
    rng = np.random.default_rng(0)
    base = np.zeros((*shape, 3), dtype=np.uint8)
    base[..., 0], base[..., 1], base[..., 2] = 176, 173, 163
    noise = rng.normal(0, 4, (*shape, 3))
    return np.asarray(np.clip(base + noise, 0, 255), dtype=np.uint8)


def box_mask(shape: tuple[int, int], y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255
    return mask


def test_an_untouched_region_scores_near_zero() -> None:
    src = scene()
    mask = box_mask(src.shape[:2], 80, 120, 80, 120)
    metrics = fill_metrics(src.copy(), src, mask)
    assert metrics.chroma_delta < 1.0
    assert metrics.lum_delta < 1.0
    assert metrics.edge_ratio == pytest.approx(1.0, abs=0.25)


def test_a_blue_cast_shows_up_as_chroma_error() -> None:
    """The exact defect measured on the framed picture.

    The shift is applied in Lab so luminance is provably untouched, reproducing what
    the eraser actually does: reproduce brightness, drift the colour toward blue. If
    this ever starts reporting a luminance error too, the chroma-only correction in
    `match_chroma` is no longer the right fix.
    """
    src = scene()
    mask = box_mask(src.shape[:2], 80, 120, 80, 120)
    lab = cv2.cvtColor(src, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[mask > 0, 2] -= 25.0  # b channel toward blue; L and a untouched
    out = np.asarray(
        cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB), dtype=np.uint8
    )

    metrics = fill_metrics(out, src, mask)
    assert metrics.chroma_delta > 20.0
    assert metrics.lum_delta < 1.0
    assert metrics.edge_ratio == pytest.approx(1.0, abs=0.2), "texture must be preserved"


def test_a_flat_fill_reads_as_a_smear() -> None:
    src = scene()
    out = src.copy()
    mask = box_mask(src.shape[:2], 80, 120, 80, 120)
    out[mask > 0] = [176, 173, 163]  # texture removed entirely
    assert fill_metrics(out, src, mask).edge_ratio < 0.6


def test_empty_and_tiny_masks_are_neutral_not_crashes() -> None:
    src = scene()
    empty = np.zeros(src.shape[:2], dtype=np.uint8)
    assert fill_metrics(src.copy(), src, empty).cost == pytest.approx(0.0, abs=1e-6)

    speck = np.zeros(src.shape[:2], dtype=np.uint8)
    speck[0, 0] = 255
    fill_metrics(src.copy(), src, speck)  # must not raise


def test_mismatched_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match="differ"):
        fill_metrics(scene((100, 100)), scene((200, 200)), np.zeros((100, 100), np.uint8))


def test_compare_rejects_a_bigger_erase_that_looks_better_on_raw_cost() -> None:
    """The i6 regression, in miniature.

    A candidate that flattens a much larger area scores better on raw fill cost. It must
    lose once the growth penalty is applied, or the pipeline destroys images while
    reporting an improvement.
    """
    src = scene()
    base_mask = box_mask(src.shape[:2], 90, 110, 90, 110)
    wide_mask = box_mask(src.shape[:2], 90, 126, 90, 110)  # +80%, as the desk case grew

    baseline = src.copy()
    baseline[base_mask > 0] = [168, 166, 182]  # a mild blue ghost, as LaMa leaves

    # A *plausible* fill over the larger area. Flat colour would be caught by the edge
    # term on its own; the failure mode that matters is a believable fill that simply
    # erased more of the user's image than was asked for.
    candidate = src.copy()

    eval_region = np.maximum(base_mask, wide_mask)
    raw = (
        fill_metrics(candidate, src, eval_region).cost
        - fill_metrics(baseline, src, eval_region).cost
    )
    penalised = compare(candidate, baseline, src, eval_region, base_mask, wide_mask)

    assert raw < 0, "precondition: raw cost prefers the larger flat erase"
    assert penalised > 0, "growth penalty must overturn it"


def test_compare_accepts_a_genuine_improvement_at_equal_area() -> None:
    src = scene()
    mask = box_mask(src.shape[:2], 90, 110, 90, 110)

    baseline = src.copy()
    baseline[mask > 0] = [150, 150, 200]  # badly wrong
    candidate = src.copy()
    candidate[mask > 0] = [176, 173, 163]  # close to the surroundings

    assert compare(candidate, baseline, src, mask, mask, mask) < 0


def test_growth_penalty_is_proportional_to_the_original_mask() -> None:
    src = scene()
    base_mask = box_mask(src.shape[:2], 95, 105, 95, 105)
    grown = box_mask(src.shape[:2], 95, 115, 95, 105)  # exactly double the area
    identical = src.copy()

    delta = compare(identical, identical, src, base_mask, base_mask, grown)
    assert delta == pytest.approx(GROWTH_PENALTY * 1.0, abs=0.5)


def test_box_metrics_scores_a_well_placed_mask_as_a_hit() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70, 30:70] = 1
    result = box_metrics(mask, (0.3, 0.3, 0.7, 0.7), (100, 100))
    assert result["hit"] is True
    assert result["inside_frac"] == pytest.approx(1.0)
    assert result["bbox_iou"] > 0.9


def test_box_metrics_rejects_a_mask_on_the_wrong_object() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[5:15, 5:15] = 1
    assert box_metrics(mask, (0.3, 0.3, 0.7, 0.7), (100, 100))["hit"] is False


def test_box_metrics_rejects_a_correctly_placed_speck() -> None:
    """A six-pixel mask inside the box has perfect precision and is still useless."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[49:52, 49:52] = 1
    result = box_metrics(mask, (0.3, 0.3, 0.7, 0.7), (100, 100))
    assert result["inside_frac"] == pytest.approx(1.0)
    assert result["hit"] is False


def test_box_metrics_tolerates_a_non_convex_mask() -> None:
    """A lattice covers little of its own bounding box but is still the right mask."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70:4, 30:70] = 1  # sparse horizontal bars
    result = box_metrics(mask, (0.3, 0.3, 0.7, 0.7), (100, 100))
    assert result["box_recall"] < 0.4
    assert result["hit"] is True


def test_box_metrics_handles_an_empty_mask() -> None:
    empty = np.zeros((100, 100), dtype=np.uint8)
    assert box_metrics(empty, (0.3, 0.3, 0.7, 0.7), (100, 100)) == {
        "inside_frac": 0.0,
        "box_recall": 0.0,
        "bbox_iou": 0.0,
        "hit": False,
    }
