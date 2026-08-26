"""The multi-pass policy: always run twice, keep only what verifies as better."""

from __future__ import annotations

import numpy as np
import pytest
from editgpt_models.pipeline import ACCEPT_COST, Erasers, erase


def scene(shape: tuple[int, int] = (200, 200)) -> np.ndarray:
    rng = np.random.default_rng(1)
    base = np.zeros((*shape, 3), dtype=np.uint8)
    base[..., 0], base[..., 1], base[..., 2] = 176, 173, 163
    return np.clip(base + rng.normal(0, 4, (*shape, 3)), 0, 255).astype(np.uint8)


def box(shape: tuple[int, int], y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    m = np.zeros(shape, dtype=np.uint8)
    m[y0:y1, x0:x1] = 255
    return m


def perfect(source: np.ndarray):  # type: ignore[no-untyped-def]
    """An eraser that reproduces the surroundings exactly."""

    def fill(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out = image.copy()
        out[mask > 0] = source[mask > 0]
        return out

    return fill


def tinted(colour: tuple[int, int, int]):  # type: ignore[no-untyped-def]
    """An eraser that leaves a flat, wrongly-coloured patch."""

    def fill(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out = image.copy()
        out[mask > 0] = colour
        return out

    return fill


def test_a_tiny_mask_is_refused_rather_than_silently_no_op() -> None:
    src = scene()
    speck = np.zeros(src.shape[:2], dtype=np.uint8)
    speck[0:3, 0:3] = 255
    erasers = Erasers(migan=perfect(src), lama=perfect(src))

    with pytest.raises(ValueError, match=r"below the \d+ px floor"):
        erase(erasers, src, speck)


def test_at_least_two_passes_always_run() -> None:
    """The policy is mandatory second-pass, even when the first result is already good."""
    src = scene()
    mask = box(src.shape[:2], 80, 120, 80, 120)
    erasers = Erasers(migan=perfect(src), lama=perfect(src))

    outcome = erase(erasers, src, mask)
    assert len(outcome.passes) >= 2
    assert outcome.passes[0].strategy == "migan"


def test_a_worse_second_pass_is_rolled_back() -> None:
    """Phase 0 measured naive second passes making things worse. They must not stick.

    Fixtures are calibrated so the branch under test is actually taken: mid-grey costs
    45.2, above ESCALATE_COST, so LaMa is tried; strong blue costs 142.0, so it loses.
    An earlier version of this test used a *perfect* first fill, which meant no strategy
    ever applied — it asserted "nothing was kept" and passed without a rollback occurring.
    """
    src = scene()
    mask = box(src.shape[:2], 80, 120, 80, 120)
    erasers = Erasers(migan=tinted((150, 150, 150)), lama=tinted((90, 90, 200)))

    outcome = erase(erasers, src, mask)
    ran = [p for p in outcome.passes if p.index > 1 and p.attempted]

    assert ran, "a second pass must have actually run"
    assert all(not p.kept for p in ran), "a degrading pass must be rolled back"
    assert outcome.rolled_back >= 1
    assert np.array_equal(outcome.image, tinted((150, 150, 150))(src, mask))


def test_a_better_second_pass_is_kept() -> None:
    src = scene()
    mask = box(src.shape[:2], 80, 120, 80, 120)
    erasers = Erasers(migan=tinted((90, 90, 200)), lama=perfect(src))

    outcome = erase(erasers, src, mask)
    assert outcome.kept_passes >= 2
    assert outcome.cost < ACCEPT_COST


def test_a_third_pass_only_runs_while_the_result_is_unacceptable() -> None:
    src = scene()
    mask = box(src.shape[:2], 80, 120, 80, 120)

    good = Erasers(migan=perfect(src), lama=perfect(src))
    assert len(erase(good, src, mask).passes) == 2, "a good result stops at the minimum"

    bad = Erasers(migan=tinted((60, 200, 60)), lama=tinted((60, 200, 60)))
    assert len(erase(bad, src, mask).passes) == 3, "a poor result earns a third throw"


def test_max_passes_is_honoured() -> None:
    src = scene()
    mask = box(src.shape[:2], 80, 120, 80, 120)
    erasers = Erasers(migan=tinted((60, 200, 60)), lama=tinted((60, 200, 60)))
    assert len(erase(erasers, src, mask, min_passes=2, max_passes=2).passes) == 2


def test_every_attempt_is_recorded_including_rejected_ones() -> None:
    """A pass that was tried and rejected is a finding the critic loop needs to see."""
    src = scene()
    mask = box(src.shape[:2], 80, 120, 80, 120)
    erasers = Erasers(migan=tinted((150, 150, 150)), lama=tinted((90, 90, 200)))

    outcome = erase(erasers, src, mask)
    assert "rolled back" in outcome.summary()
    assert all(p.seconds >= 0 for p in outcome.passes)


def test_a_strategy_that_did_not_apply_is_not_reported_as_a_rollback() -> None:
    """The summary is the primary record of what the pipeline tried.

    Conflating "never applied" with "tried and rejected" misreports the run: for weeks
    the eval table showed `residual (rolled back)` on cases where the residual detector
    had found nothing and no model had run.
    """
    src = scene()
    mask = box(src.shape[:2], 80, 120, 80, 120)
    # A perfect first fill leaves no residual, so that strategy cannot apply.
    outcome = erase(Erasers(migan=perfect(src), lama=perfect(src)), src, mask)

    not_applicable = [p for p in outcome.passes if not p.attempted]
    assert not_applicable, "the residual strategy should have been inapplicable here"
    assert all(not p.kept for p in not_applicable)
    assert outcome.rolled_back == 0, "nothing ran, so nothing was rolled back"
    assert "(n/a)" in outcome.summary()
    assert "rolled back" not in outcome.summary()


def test_a_genuinely_rejected_pass_is_reported_as_rolled_back() -> None:
    src = scene()
    mask = box(src.shape[:2], 80, 120, 80, 120)
    outcome = erase(Erasers(migan=tinted((150, 150, 150)), lama=tinted((90, 90, 200))), src, mask)

    assert outcome.rolled_back >= 1
    assert "rolled back" in outcome.summary()
    assert "(n/a)" not in outcome.summary().split("->")[1]


def test_every_pass_emits_a_log_record(caplog: pytest.LogCaptureFixture) -> None:
    """A decision that is not logged is invisible in production, whatever it returns."""
    import logging

    src = scene()
    mask = box(src.shape[:2], 80, 120, 80, 120)
    with caplog.at_level(logging.INFO, logger="editgpt_models.pipeline"):
        outcome = erase(Erasers(migan=perfect(src), lama=perfect(src)), src, mask)

    pass_logs = [r for r in caplog.records if r.getMessage() == "erase.pass"]
    assert len(pass_logs) == len(outcome.passes), "a pass ran without a log record"
    assert any(r.getMessage() == "erase.done" for r in caplog.records)


def test_logs_carry_no_image_data(caplog: pytest.LogCaptureFixture) -> None:
    """harness/observability.md: never log raw user content."""
    import logging

    src = scene()
    mask = box(src.shape[:2], 80, 120, 80, 120)
    with caplog.at_level(logging.INFO, logger="editgpt_models.pipeline"):
        erase(Erasers(migan=perfect(src), lama=perfect(src)), src, mask)

    for record in caplog.records:
        for value in vars(record).values():
            assert not isinstance(value, np.ndarray), "an array reached the log record"
