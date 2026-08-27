"""Threshold fitting: the split must be honest and the sweep must be arithmetic.

The `Observation` fields changed when grounding moved from CLIPSeg to Grounding DINO —
the fallback below a low SAM confidence is now the detector's box rather than a heatmap
seed, and there is a second signal (`box_score`) that this module reports but does not
fit. Behaviour intentionally changed; the properties asserted here did not.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from benchmarks.tune import Observation, abstention_curve, box_mask, score, sweep


def obs(
    ident: str, sam_confidence: float, iou_sam: float, iou_box: float, box_score: float = 0.9
) -> Observation:
    return Observation(
        id=ident,
        words=5,
        box_score=box_score,
        sam_confidence=sam_confidence,
        iou_sam=iou_sam,
        iou_box=iou_box,
    )


def test_the_split_is_deterministic_across_runs() -> None:
    """A split that reshuffles would let a value be fitted and reported on the same data."""
    first = [obs(str(i), 0.5, 0.5, 0.5).split for i in range(200)]
    second = [obs(str(i), 0.9, 0.1, 0.2).split for i in range(200)]
    assert first == second, "the split must depend on the id alone"


def test_the_split_depends_on_the_id_not_the_order() -> None:
    a = obs("sample-a", 0.5, 0.5, 0.5).split
    assert obs("sample-a", 0.1, 0.9, 0.2).split == a


def test_both_splits_are_populated() -> None:
    splits = [obs(str(i), 0.5, 0.5, 0.5).split for i in range(400)]
    assert 100 < splits.count("fit") < 300
    assert 100 < splits.count("holdout") < 300


def test_a_gate_of_zero_always_keeps_sams_mask() -> None:
    data = [obs("a", 0.1, 1.0, 0.0), obs("b", 0.9, 1.0, 0.0)]
    assert score(data, 0.0) == pytest.approx(1.0)


def test_a_gate_above_one_always_falls_back_to_the_box() -> None:
    data = [obs("a", 0.1, 1.0, 0.0), obs("b", 0.9, 1.0, 0.0)]
    assert score(data, 1.01) == pytest.approx(0.0)


def test_the_gate_selects_per_sample_by_confidence() -> None:
    data = [obs("a", 0.4, 0.0, 1.0), obs("b", 0.8, 1.0, 0.0)]
    # At 0.6, "a" falls back to its box (1.0) and "b" keeps SAM's mask (1.0).
    assert score(data, 0.6) == pytest.approx(1.0)


def test_an_empty_set_scores_zero_rather_than_raising() -> None:
    assert score([], 0.5) == 0.0


def test_the_sweep_covers_every_candidate() -> None:
    data = [obs(str(i), i / 10, 1.0, 0.0) for i in range(10)]
    candidates = np.round(np.arange(0.0, 1.01, 0.1), 2)
    curve = sweep(data, candidates)
    assert len(curve) == len(candidates)
    assert curve[0][1] >= curve[-1][1], "raising the gate can only reject SAM's mask here"


def test_the_sweep_is_monotonic_when_sams_mask_always_wins() -> None:
    """A sanity check on the sweep itself: if SAM is always better, a higher gate cannot
    score higher."""
    data = [obs(str(i), i / 20, 1.0, 0.0) for i in range(20)]
    scores = [s for _, s in sweep(data, np.round(np.arange(0, 1.01, 0.05), 2))]
    assert all(a >= b - 1e-9 for a, b in pairwise(scores))


def test_raising_the_box_gate_only_ever_abstains_more() -> None:
    """The property that makes `min_box_score` unfittable by mIoU, asserted rather than
    only argued in the docstring."""
    data = [obs(str(i), 0.99, 0.8, 0.4, box_score=i / 10) for i in range(10)]
    curve = abstention_curve(data, sam_gate=0.5, candidates=np.round(np.arange(0, 1.01, 0.1), 2))
    abstained = [row["abstained"] for row in curve]
    overall = [row["mIoU_overall"] for row in curve]
    assert all(a <= b for a, b in pairwise(abstained)), "abstention must rise with the gate"
    assert all(a >= b - 1e-9 for a, b in pairwise(overall)), "so overall mIoU can only fall"


def test_the_answered_subset_improves_as_the_gate_rises() -> None:
    """The other half of the trade: what is left is better, there is just less of it.

    Built so confidence and correctness agree, which is the case a gate is for.
    """
    data = [obs(str(i), 0.99, i / 10, 0.0, box_score=i / 10) for i in range(10)]
    curve = abstention_curve(data, sam_gate=0.5, candidates=np.round(np.arange(0, 0.91, 0.1), 2))
    answered = [row["mIoU_answered"] for row in curve]
    assert all(a <= b + 1e-9 for a, b in pairwise(answered))


def test_the_box_mask_covers_exactly_the_detected_rectangle() -> None:
    mask = box_mask((0.25, 0.5, 0.75, 1.0), (100, 200))
    assert mask.shape == (100, 200)
    assert mask[:50].sum() == 0, "nothing above the box"
    assert mask[50:, 50:150].all(), "the box itself is filled"
    assert mask[50:, :50].sum() == 0, "nothing left of the box"
