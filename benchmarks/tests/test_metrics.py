"""Mask IoU is the metric the whole grounding benchmark rests on."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.grounding import _fit, mask_iou


def box(shape: tuple[int, int], y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    m[y0:y1, x0:x1] = 255
    return m


def test_identical_masks_score_one() -> None:
    mask = box((100, 100), 20, 60, 20, 60)
    iou, precision, recall = mask_iou(mask, mask)
    assert iou == pytest.approx(1.0)
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(1.0)


def test_disjoint_masks_score_zero() -> None:
    a = box((100, 100), 0, 20, 0, 20)
    b = box((100, 100), 60, 80, 60, 80)
    assert mask_iou(a, b)[0] == pytest.approx(0.0)


def test_half_overlap_is_a_third() -> None:
    """Two equal boxes sharing half their area: intersection 1, union 3."""
    a = box((100, 100), 0, 40, 0, 20)
    b = box((100, 100), 20, 60, 0, 20)
    assert mask_iou(a, b)[0] == pytest.approx(1 / 3, abs=1e-6)


def test_precision_and_recall_are_not_symmetric() -> None:
    """A prediction covering the truth plus more has full recall and poor precision."""
    truth = box((100, 100), 40, 60, 40, 60)
    over = box((100, 100), 20, 80, 20, 80)
    _, precision, recall = mask_iou(over, truth)
    assert recall == pytest.approx(1.0)
    assert precision < 0.2


def test_an_empty_prediction_scores_zero_without_dividing_by_zero() -> None:
    truth = box((100, 100), 40, 60, 40, 60)
    assert mask_iou(np.zeros((100, 100), np.uint8), truth) == (0.0, 0.0, 0.0)


def test_both_empty_is_zero_not_nan() -> None:
    empty = np.zeros((100, 100), np.uint8)
    assert all(np.isfinite(v) for v in mask_iou(empty, empty))


def test_non_binary_masks_are_thresholded_consistently() -> None:
    """Callers pass 0/1 and 0/255 interchangeably; the metric must not care."""
    a = box((100, 100), 20, 60, 20, 60)
    assert mask_iou(a, (a > 0).astype(np.uint8))[0] == pytest.approx(1.0)


def test_fit_scales_image_and_mask_together() -> None:
    """If the pair desynchronised, every IoU would be quietly wrong."""
    rng = np.random.default_rng(0)
    image = np.asarray(rng.integers(0, 255, (400, 800, 3), dtype=np.uint8))
    mask = box((400, 800), 100, 300, 200, 600)
    scaled_image, scaled_mask = _fit(image, mask, 200)

    assert scaled_image.shape[:2] == scaled_mask.shape[:2]
    assert max(scaled_image.shape[:2]) == 200
    before = mask.mean() / 255
    after = scaled_mask.mean() / 255
    assert after == pytest.approx(before, abs=0.02), "mask area fraction should survive scaling"


def test_fit_is_a_no_op_when_already_small() -> None:
    image = np.zeros((50, 50, 3), np.uint8)
    mask = box((50, 50), 10, 20, 10, 20)
    out_image, out_mask = _fit(image, mask, 200)
    assert out_image is image
    assert out_mask is mask
