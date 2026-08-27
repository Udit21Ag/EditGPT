"""Dataset loaders, exercised without downloading anything.

The network is disabled in this tier, so what is tested here is the frame arithmetic and
the mask cleanup — the parts that were got wrong by hand rather than by the hub.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.datasets import GroundingSample, RemovalSample, _polygons_to_mask


def sample(identifier: str, group: str = "") -> RemovalSample:
    blank = np.zeros((4, 4, 3), np.uint8)
    return RemovalSample(
        id=identifier,
        image=blank,
        ground_truth=blank,
        mask=np.zeros((4, 4), np.uint8),
        group=group,
    )


def test_a_sample_with_no_group_is_its_own_group() -> None:
    """RemovalBench stills are independent; each is its own unit for a split."""
    assert sample("still-7").group_id == "still-7"


def test_frames_of_one_clip_share_a_group() -> None:
    """The property that stops frame 60 being scored after fitting on frame 20."""
    first = sample("clip-a#20", group="clip-a")
    second = sample("clip-a#60", group="clip-a")
    assert first.group_id == second.group_id == "clip-a"
    assert first.id != second.id


def test_polygons_are_rasterised_into_a_filled_mask() -> None:
    mask = _polygons_to_mask([[1.0, 1.0, 9.0, 1.0, 9.0, 9.0, 1.0, 9.0]], 12, 12)
    assert mask.shape == (12, 12)
    assert mask[5, 5] == 255
    assert mask[0, 0] == 0


def test_several_polygons_of_one_object_are_all_filled() -> None:
    """A COCO object split by occlusion arrives as more than one polygon."""
    mask = _polygons_to_mask(
        [[0.0, 0.0, 4.0, 0.0, 4.0, 4.0, 0.0, 4.0], [8.0, 8.0, 12.0, 8.0, 12.0, 12.0, 8.0, 12.0]],
        16,
        16,
    )
    assert mask[2, 2] == 255
    assert mask[10, 10] == 255
    assert mask[6, 6] == 0


@pytest.mark.parametrize("degenerate", [[], [[1.0, 2.0]], ["not a polygon"], [[1.0, 2.0, 3.0]]])
def test_a_polygon_with_too_few_points_is_skipped_not_crashed(degenerate: object) -> None:
    """Fewer than three points cannot bound an area; the dataset contains a few."""
    mask = _polygons_to_mask(degenerate, 8, 8)
    assert int(mask.sum()) == 0


def test_a_grounding_sample_carries_the_datasets_mask_not_a_box() -> None:
    """The reason IoU here is the field's metric rather than our box proxy."""
    truth = np.zeros((8, 8), np.uint8)
    truth[2:6, 2:6] = 255
    made = GroundingSample(
        id="1", image=np.zeros((8, 8, 3), np.uint8), phrase="the car", mask=truth
    )
    assert int(made.mask.sum()) == 16 * 255
