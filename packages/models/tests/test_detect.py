"""Grounding DINO's caller, with the graph replaced by a stub.

The model itself is not under test here — it is 204 MB and `make check` must stay
hermetic. What is under test is everything around it, which is where the bugs were: the
preprocessing convention, the token range the score is taken over, and the mapping from
the model's centre-width-height boxes to the corner form the rest of the pipeline uses.
"""

from __future__ import annotations

import numpy as np
import pytest
from editgpt_models.detect import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_SIDE,
    Detection,
    Detector,
    _preprocess,
    detect,
    normalise_phrase,
)


class StubTokenizer:
    """Encodes to a fixed number of ids: [CLS] + one per word + [SEP]."""

    def __init__(self, words: int = 2) -> None:
        self.words = words
        self.seen: list[str] = []

    def encode(self, text: str) -> StubTokenizer:
        self.seen.append(text)
        return self

    @property
    def ids(self) -> list[int]:
        return [101, *range(1000, 1000 + self.words), 102]


class StubSession:
    """Returns scripted logits and boxes, and records what it was fed."""

    def __init__(self, logits: np.ndarray, boxes: np.ndarray) -> None:
        self.logits = logits
        self.boxes = boxes
        self.feeds: list[dict[str, np.ndarray]] = []

    def run(self, _outputs: object, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.feeds.append(feed)
        return [self.logits, self.boxes]


def logit(probability: float) -> float:
    return float(np.log(probability / (1 - probability)))


def make_detector(scores: list[float], boxes: list[tuple[float, float, float, float]]) -> Detector:
    """A detector whose queries have the given per-token scores and centre-form boxes."""
    tokens = 4  # [CLS] + 2 words + [SEP]
    logits = np.full((1, len(scores), 256), logit(0.001), dtype=np.float32)
    for index, score in enumerate(scores):
        logits[0, index, 1 : tokens - 1] = logit(score)
    return Detector(
        session=StubSession(logits, np.asarray([boxes], dtype=np.float32)),
        tokenizer=StubTokenizer(words=tokens - 2),
    )


@pytest.fixture
def image() -> np.ndarray:
    return np.full((480, 640, 3), 128, dtype=np.uint8)


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("the car", "the car."),
        ("The Car", "the car."),
        ("  the car.  ", "the car."),
        ("the car...", "the car."),
        ("", ""),
        ("   ", ""),
    ],
)
def test_a_phrase_is_lowercased_and_period_terminated(given: str, expected: str) -> None:
    """The form the model was trained on. An unterminated prompt scores lower silently."""
    assert normalise_phrase(given) == expected


def test_preprocessing_squashes_to_the_declared_input(image: np.ndarray) -> None:
    """Measured, not assumed: squash beats letterbox 0.511 to 0.230 on RefCOCOg boxes."""
    tensor, pixel_mask = _preprocess(image)
    assert tensor.shape == (1, 3, INPUT_SIDE, INPUT_SIDE)
    assert pixel_mask.shape == (1, INPUT_SIDE, INPUT_SIDE)
    assert pixel_mask.all(), "nothing is padded, so every pixel is real"


def test_preprocessing_applies_imagenet_normalisation(image: np.ndarray) -> None:
    tensor, _ = _preprocess(image)
    expected = (128 / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    assert tensor[0, :, 0, 0] == pytest.approx(expected, abs=1e-4)


def test_the_best_box_comes_back_in_corner_fractions(image: np.ndarray) -> None:
    detector = make_detector(scores=[0.9], boxes=[(0.5, 0.5, 0.2, 0.4)])
    found = detect(detector, image, "the car", min_score=0.25)

    assert len(found) == 1
    assert found[0].box == pytest.approx((0.4, 0.3, 0.6, 0.7))
    assert found[0].score == pytest.approx(0.9, abs=1e-3)


def test_results_are_sorted_best_first(image: np.ndarray) -> None:
    detector = make_detector(
        scores=[0.4, 0.95, 0.6],
        boxes=[(0.1, 0.1, 0.1, 0.1), (0.5, 0.5, 0.2, 0.2), (0.9, 0.9, 0.1, 0.1)],
    )
    found = detect(detector, image, "the car", min_score=0.25)
    assert [round(f.score, 2) for f in found] == [0.95, 0.6, 0.4]


def test_detections_below_the_gate_are_dropped(image: np.ndarray) -> None:
    detector = make_detector(scores=[0.9, 0.3, 0.1], boxes=[(0.5, 0.5, 0.2, 0.2)] * 3)
    assert len(detect(detector, image, "the car", min_score=0.5)) == 1


def test_nothing_above_the_gate_is_an_empty_list_not_an_error(image: np.ndarray) -> None:
    """A phrase naming something absent is a real outcome; the caller offers the brush."""
    detector = make_detector(scores=[0.05], boxes=[(0.5, 0.5, 0.2, 0.2)])
    assert detect(detector, image, "a unicorn", min_score=0.25) == []


def test_an_empty_phrase_never_reaches_the_model(image: np.ndarray) -> None:
    detector = make_detector(scores=[0.9], boxes=[(0.5, 0.5, 0.2, 0.2)])
    assert detect(detector, image, "   ", min_score=0.25) == []
    assert detector.session.feeds == []


def test_a_box_predicted_outside_the_frame_is_clipped(image: np.ndarray) -> None:
    """The model happily predicts a percent or two past the edge; downstream indexes with it."""
    detector = make_detector(scores=[0.9], boxes=[(0.5, 0.5, 1.4, 1.4)])
    found = detect(detector, image, "the sky", min_score=0.25)
    assert found[0].box == (0.0, 0.0, 1.0, 1.0)


def test_a_degenerate_box_is_discarded(image: np.ndarray) -> None:
    """Zero width after clipping would produce an empty mask that reports as a success."""
    detector = make_detector(scores=[0.9], boxes=[(0.0, 0.5, 0.0, 0.4)])
    assert detect(detector, image, "the edge", min_score=0.25) == []


def test_top_k_bounds_how_many_candidates_come_back(image: np.ndarray) -> None:
    detector = make_detector(scores=[0.9, 0.8, 0.7, 0.6], boxes=[(0.5, 0.5, 0.2, 0.2)] * 4)
    assert len(detect(detector, image, "the car", min_score=0.25, top_k=2)) == 2


def test_the_special_tokens_are_excluded_from_the_score() -> None:
    """[CLS] and [SEP] activate for every query and would flatten the ranking.

    Built so the special-token positions are maximal and the phrase positions are not: if
    they were included, both queries would score ~1.0 and the ordering would be lost.
    """
    tokens = 4
    logits = np.full((1, 2, 256), logit(0.999), dtype=np.float32)
    logits[0, 0, 1 : tokens - 1] = logit(0.8)
    logits[0, 1, 1 : tokens - 1] = logit(0.2)
    detector = Detector(
        session=StubSession(logits, np.asarray([[(0.5, 0.5, 0.2, 0.2)] * 2], dtype=np.float32)),
        tokenizer=StubTokenizer(words=tokens - 2),
    )

    found = detect(detector, np.zeros((10, 10, 3), np.uint8), "two words", min_score=0.0)
    assert [round(f.score, 1) for f in found] == [0.8, 0.2]


def test_the_prompt_reaching_the_tokenizer_is_normalised(image: np.ndarray) -> None:
    detector = make_detector(scores=[0.9], boxes=[(0.5, 0.5, 0.2, 0.2)])
    detect(detector, image, "  The CAR ", min_score=0.25)
    assert detector.tokenizer.seen == ["the car."]


def test_detection_area_is_derived_not_stored() -> None:
    assert Detection(box=(0.2, 0.1, 0.6, 0.6), score=0.9).area == pytest.approx(0.2)
