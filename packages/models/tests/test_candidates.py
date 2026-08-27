"""Grounding a phrase to candidates, and the gate on whether to ask.

The models are stubbed; what is under test is the arithmetic that turns ranked detections
into a decision. Whether the candidates are any *good* is measured in
`benchmarks/ambiguity.py` against ground truth, and that measurement is what justifies
the feature existing at all.
"""

from __future__ import annotations

from collections.abc import Callable, Collection

import numpy as np
import pytest
from editgpt_core import Grounding
from editgpt_models.config import Thresholds
from editgpt_models.detect import Detection
from editgpt_models.segment import Segmentation, candidates_from_phrase


def image(size: int = 64) -> np.ndarray:
    return np.zeros((size, size, 3), np.uint8)


def blob(size: int = 64, at: int = 10) -> np.ndarray:
    mask = np.zeros((size, size), np.uint8)
    mask[at : at + 20, at : at + 20] = 255
    return mask


Arrange = Callable[..., Grounding]


@pytest.fixture
def stubbed(monkeypatch: pytest.MonkeyPatch) -> Arrange:
    """Drive `candidates_from_phrase` with scripted detections and masks."""

    def arrange(
        scores: list[float], *, empty: Collection[int] = (), margin: float = 0.15
    ) -> Grounding:
        found = [
            Detection(box=(0.1 + i * 0.01, 0.1, 0.4 + i * 0.01, 0.4), score=score)
            for i, score in enumerate(scores)
        ]
        monkeypatch.setattr("editgpt_models.detect.detect", lambda *_a, **_k: found)
        monkeypatch.setattr(
            "editgpt_models.segment.masks_from_boxes",
            lambda *_a, **_k: [
                Segmentation(
                    np.zeros((64, 64), np.uint8) if i in empty else blob(at=10 + i),
                    0.9,
                    "sam-box",
                )
                for i in range(len(scores))
            ],
        )
        monkeypatch.setattr(
            "editgpt_models.segment.load_thresholds",
            lambda *_a, **_k: Thresholds(ambiguity_margin=margin, candidates=5),
        )
        return candidates_from_phrase(object(), object(), object(), image(), "the car")

    return arrange


def test_candidates_come_back_ranked_best_first(stubbed: Arrange) -> None:
    found = stubbed([0.9, 0.6, 0.3])
    assert [round(c.score, 1) for c in found.candidates] == [0.9, 0.6, 0.3]


def test_a_clear_winner_is_not_ambiguous(stubbed: Arrange) -> None:
    """0.95 against 0.07 is the shape of a confident detection; asking there is friction."""
    found = stubbed([0.95, 0.07])
    assert not found.ambiguous
    assert found.margin == pytest.approx(0.88, abs=0.01)


def test_two_close_candidates_are_ambiguous(stubbed: Arrange) -> None:
    """Measured: when the margin is narrow, top-1 is right 42% of the time against 57%
    when it is wide. That gap is the whole justification for asking."""
    found = stubbed([0.55, 0.50])
    assert found.ambiguous
    assert found.margin == pytest.approx(0.05, abs=0.01)


def test_a_lone_candidate_is_never_ambiguous(stubbed: Arrange) -> None:
    """There is nothing to be ambiguous *between*; a chooser with one option is a dialog
    box that wastes a click."""
    found = stubbed([0.4])
    assert not found.ambiguous
    assert found.margin == 1.0


def test_a_zero_gate_never_asks(stubbed: Arrange) -> None:
    """The behaviour before this existed, kept reachable by configuration."""
    assert not stubbed([0.51, 0.50], margin=0.0).ambiguous


def test_nothing_matching_is_an_empty_answer_not_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A phrase naming something absent is a real outcome; the caller offers the brush."""
    monkeypatch.setattr("editgpt_models.detect.detect", lambda *_a, **_k: [])
    found = candidates_from_phrase(object(), object(), object(), image(), "a unicorn")

    assert found.candidates == []
    assert not found.ambiguous
    assert found.best is None


def test_a_candidate_whose_mask_came_back_empty_is_dropped(stubbed: Arrange) -> None:
    """An empty mask is an option that would edit nothing — offering it is offering a
    no-op that reports as success."""
    found = stubbed([0.9, 0.8, 0.7], empty={1})
    assert len(found.candidates) == 2
    assert [round(c.score, 1) for c in found.candidates] == [0.9, 0.7]


def test_every_candidate_carries_its_mask(stubbed: Arrange) -> None:
    """The mask travels with the candidate so picking one costs no second segmentation —
    which would also be free to return a *different* mask."""
    found = stubbed([0.9, 0.6])
    for candidate in found.candidates:
        assert candidate.mask_ref.area_px > 0
        assert candidate.mask_ref.width == 64
