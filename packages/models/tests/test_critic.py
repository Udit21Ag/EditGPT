"""Judging a finished edit, with no model loaded.

The two cheap checks are arithmetic and are tested against images built here. The
semantic check needs a detector, which is 1372 MB of ONNX; the *decision it drives* is
what matters and that is testable with a stub, so `detect` is replaced. The real detector
is exercised in the eval tier, not in the fast one.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from editgpt_core.review import IMPLAUSIBLE, STILL_THERE, UNCHANGED
from editgpt_models import critic
from editgpt_models.config import Thresholds
from editgpt_models.detect import Detection, Detector


def stub_detector() -> Detector:
    """A `Detector` with nothing loaded. `detect` is replaced, so neither field is used."""
    return Detector(session=None, tokenizer=None)


def photo(width: int = 128, height: int = 96) -> np.ndarray:
    """A textured surface, so a fill has something to agree or disagree with."""
    rng = np.random.default_rng(7)
    base = np.full((height, width, 3), (90, 110, 130), np.uint8)
    noise = rng.integers(-12, 12, size=(height, width, 3), dtype=np.int16)
    return np.asarray(np.clip(base.astype(np.int16) + noise, 0, 255), dtype=np.uint8)


def region(width: int = 128, height: int = 96) -> np.ndarray:
    mask = np.zeros((height, width), np.uint8)
    mask[30:60, 40:80] = 255
    return mask


def test_an_edit_that_changed_nothing_is_caught_however_well_it_scores() -> None:
    """Phase 0's silent failure: the input returned unedited, and every photometric score
    reporting success — because an unedited region agrees perfectly with its surroundings."""
    source = photo()
    verdict = critic.critique(source, source.copy(), region())

    assert UNCHANGED in verdict.reasons
    assert verdict.changed == 0.0


def test_a_fill_that_matches_its_surroundings_passes() -> None:
    source = photo()
    result = source.copy()
    # A different draw of the same surface: changed everywhere, and still plausible.
    rng = np.random.default_rng(11)
    patch = np.clip(
        np.full((30, 40, 3), (90, 110, 130), np.int16) + rng.integers(-12, 12, (30, 40, 3)), 0, 255
    )
    result[30:60, 40:80] = patch.astype(np.uint8)

    verdict = critic.critique(source, result, region())

    assert verdict.ok, verdict.reasons
    assert verdict.changed > 0.5


def test_a_fill_that_does_not_belong_is_reported() -> None:
    source = photo()
    result = source.copy()
    result[30:60, 40:80] = (255, 0, 200)

    verdict = critic.critique(source, result, region())

    assert IMPLAUSIBLE in verdict.reasons
    assert verdict.fill_cost > Thresholds().escalate_cost


def test_no_region_means_nothing_to_judge_rather_than_a_failure() -> None:
    """`BACKGROUND` floods the backdrop with no mask at all. Not every op has a region."""
    source = photo()
    verdict = critic.critique(source, source.copy(), np.zeros((96, 128), np.uint8))

    assert verdict.ok
    assert verdict.changed == 0.0


# ---------------------------------------------------------------- the semantic check


def stub_detect(monkeypatch: pytest.MonkeyPatch, found: list[Detection]) -> list[str]:
    asked: list[str] = []

    def fake(detector: Any, rgb: Any, phrase: str, **kwargs: Any) -> list[Detection]:
        asked.append(phrase)
        return found

    monkeypatch.setattr(critic, "detect", fake)
    return asked


def edited() -> tuple[np.ndarray, np.ndarray]:
    source = photo()
    result = source.copy()
    rng = np.random.default_rng(3)
    result[30:60, 40:80] = np.clip(
        np.full((30, 40, 3), (90, 110, 130), np.int16) + rng.integers(-12, 12, (30, 40, 3)), 0, 255
    ).astype(np.uint8)
    return source, result


def test_a_target_still_detected_where_it_was_means_the_edit_did_not_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The check no photometric score can make: the fill is plausible and the car is
    still standing in it."""
    asked = stub_detect(monkeypatch, [Detection(box=(0.32, 0.32, 0.61, 0.61), score=0.91)])
    source, result = edited()

    verdict = critic.critique(source, result, region(), target="the car", detector=stub_detector())

    assert STILL_THERE in verdict.reasons
    assert verdict.still_there == 0.91
    assert asked == ["the car"], "the detector must be asked about the user's own phrase"


def test_the_same_object_somewhere_else_in_the_picture_is_not_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second car parked across the frame is not evidence that this one survived."""
    stub_detect(monkeypatch, [Detection(box=(0.80, 0.80, 0.98, 0.95), score=0.95)])
    source, result = edited()

    verdict = critic.critique(source, result, region(), target="the car", detector=stub_detector())

    assert verdict.ok, verdict.reasons
    assert verdict.still_there == 0.0


def test_a_weak_detection_is_not_enough_to_call_it_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`min_box_score` is the same gate the grounding path uses; a smudge that scores 0.1
    is not the object."""
    stub_detect(monkeypatch, [Detection(box=(0.32, 0.32, 0.61, 0.61), score=0.10)])
    source, result = edited()

    verdict = critic.critique(source, result, region(), target="the car", detector=stub_detector())

    assert verdict.ok, verdict.reasons
    assert verdict.still_there == 0.10


def test_without_a_detector_the_semantic_check_is_skipped_rather_than_guessed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """It costs a model swap, so the caller decides. `None` means not checked, which is
    different from checked and clean."""
    asked = stub_detect(monkeypatch, [])
    source, result = edited()

    verdict = critic.critique(source, result, region(), target="the car")

    assert verdict.still_there is None
    assert not asked
