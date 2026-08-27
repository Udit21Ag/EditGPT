"""The edit dispatch, with the models replaced by callables.

`execute` decides *what happens* for an operation; the models decide what the pixels
become. Substituting the latter leaves the former fully testable without 550 MB of
weights, and the substitution is honest because `Models` was designed to take injected
sessions for exactly this reason.

What the golden set covers instead, and this cannot: whether the result looks right.
"""

from __future__ import annotations

import numpy as np
import pytest
from editgpt_core import EditOp
from editgpt_core.errors import MaskTooSmallError
from editgpt_models.execute import SUPPORTED, Models, execute


def image(size: int = 256) -> np.ndarray:
    """Textured rather than flat: a flat image makes every fill look perfect."""
    generator = np.random.default_rng(seed=11)
    return generator.integers(0, 255, (size, size, 3), dtype=np.uint8)


def square(size: int = 256, box: int = 80) -> np.ndarray:
    mask = np.zeros((size, size), np.uint8)
    mask[60 : 60 + box, 60 : 60 + box] = 255
    return mask


class RecordingEraser:
    """Stands in for a loaded eraser and records that it was asked."""

    def __init__(self, name: str, calls: list[str]) -> None:
        self.name, self.calls = name, calls

    def __call__(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        self.calls.append(self.name)
        out = rgb.copy()
        out[mask > 0] = 128
        return out


def erasers() -> tuple[Models, list[str]]:
    calls: list[str] = []
    return (
        Models(migan=RecordingEraser("migan", calls), lama=RecordingEraser("lama", calls)),
        calls,
    )


# `Erasers.from_sessions` wraps sessions in `erase_migan`/`erase_lama`, which expect ONNX
# sessions. These tests bypass that by constructing `Erasers` directly through the models
# bundle, so `execute` is exercised and the ONNX boundary is not.
@pytest.fixture(autouse=True)
def _direct_erasers(monkeypatch: pytest.MonkeyPatch) -> None:
    from editgpt_models import pipeline

    monkeypatch.setattr(
        pipeline.Erasers,
        "from_sessions",
        classmethod(lambda cls, migan, lama: cls(migan=migan, lama=lama)),
    )


def test_remove_runs_the_erasers_and_reports_what_it_did() -> None:
    models, calls = erasers()
    edit = execute(models, EditOp.REMOVE, image(), mask=square())

    assert calls, "no eraser ran"
    assert "migan" in edit.strategy
    assert edit.passes, "a pass record is the audit trail; an empty one is a reporting bug"
    assert edit.seconds >= 0.0


def test_remove_dilates_the_mask_before_erasing() -> None:
    """One of the two Phase 0 fixes: dilation scales with the object, not in pixels.

    Asserted through the *reported* mask rather than by inspecting the call, because the
    reported mask is what the audit trail and the compositor both use.
    """
    models, _ = erasers()
    asked = square()
    edit = execute(models, EditOp.REMOVE, image(), mask=asked)

    assert int((edit.mask > 0).sum()) > int((asked > 0).sum())


def test_a_region_too_small_to_see_is_refused_rather_than_silently_returned() -> None:
    """Returning the input unchanged looks like success and reports as success.

    This project lost real time to a mask so small the output was identical to the input
    while every numeric check passed.

    Calibrated against the real numbers rather than guessed: dilation has an 8 px floor,
    so a *single* pixel grows to exactly 64 and clears `MIN_MASK_PX`. A post-dilation
    check could never fire, which is why the floor applies to what was selected.
    """
    models, calls = erasers()
    tiny = np.zeros((256, 256), np.uint8)
    tiny[10:17, 10:17] = 255  # 49 px selected, below the 64 px floor

    with pytest.raises(MaskTooSmallError, match="visible"):
        execute(models, EditOp.REMOVE, image(), mask=tiny)
    assert not calls, "no model should run for a region that cannot produce a visible edit"


def test_an_empty_mask_is_refused_for_every_operation_that_needs_one() -> None:
    models, _ = erasers()
    with pytest.raises(MaskTooSmallError):
        execute(models, EditOp.REMOVE, image(), mask=np.zeros((256, 256), np.uint8))


def test_remove_never_reaches_a_remote_provider() -> None:
    """Architectural invariant 6, and ADR-0001's central measured finding.

    The generative lane fills a masked hole with an *object* matching the prompt rather
    than continuing the background: asked to erase a car it produced a stone slab, then a
    different car, then a boulder.
    """
    called: list[str] = []

    def provider(rgb: np.ndarray, mask: np.ndarray, prompt: str) -> np.ndarray:
        called.append(prompt)
        return rgb

    models, _ = erasers()
    execute(
        Models(migan=models.migan, lama=models.lama, fill=provider),
        EditOp.REMOVE,
        image(),
        mask=square(),
        content="a car",
    )
    assert not called, "removal reached the remote lane"


def test_add_paints_through_the_filler_it_was_given() -> None:
    seen: list[str] = []

    def provider(rgb: np.ndarray, mask: np.ndarray, prompt: str) -> np.ndarray:
        seen.append(prompt)
        out = rgb.copy()
        out[mask > 0] = 200
        return out

    edit = execute(Models(fill=provider), EditOp.ADD, image(), mask=square(), content="a moustache")
    assert seen == ["a moustache"]
    assert "a moustache" in edit.detail


def test_the_provider_that_served_a_generative_edit_is_named(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ "Which provider produced this" is the first question asked about a bad result."""
    edit = execute(
        Models(fill=lambda rgb, *_: rgb),
        EditOp.ADD,
        image(),
        mask=square(),
        content="a hat",
        via="cloudflare",
    )
    assert edit.strategy == "cloudflare fill"


def test_a_generative_operation_without_content_is_a_caller_bug() -> None:
    """`EditSpec` already refuses this, so reaching here means something bypassed it."""
    with pytest.raises(ValueError, match="content"):
        execute(Models(fill=lambda rgb, *_: rgb), EditOp.ADD, image(), mask=square())


def test_upscale_needs_no_mask_and_changes_the_size(monkeypatch: pytest.MonkeyPatch) -> None:
    def doubler(session: object, rgb: np.ndarray) -> np.ndarray:
        return np.repeat(np.repeat(rgb, 2, axis=0), 2, axis=1)

    monkeypatch.setattr("editgpt_models.execute.upscale", doubler)
    edit = execute(Models(esrgan=object()), EditOp.UPSCALE, image(64))

    assert edit.image.shape[:2] == (128, 128)
    assert "64x64 -> 128x128" in edit.strategy, "the sizes belong in the summary"


def test_background_needs_no_model_at_all() -> None:
    """A flat colour is compositing, not generation."""
    edit = execute(Models(), EditOp.BACKGROUND, image(), mask=square(), colour=(0, 255, 0))
    assert edit.strategy == "composite"
    assert edit.cost == 0.0


def test_a_missing_model_is_named_before_any_pixel_is_touched() -> None:
    """The error should say which model and for which operation, not fail deep inside."""
    with pytest.raises(ValueError, match="migan"):
        execute(Models(), EditOp.REMOVE, image(), mask=square())


def test_an_unimplemented_operation_says_what_is_supported() -> None:
    with pytest.raises(ValueError, match="supported"):
        execute(Models(), EditOp.RESTYLE, image(), mask=square())


def test_supported_matches_what_the_gateway_advertises() -> None:
    """The two must not drift: advertising an operation with no implementation is a lie
    the frontend acts on."""
    assert EditOp.RESTYLE not in SUPPORTED
    assert EditOp.RETOUCH not in SUPPORTED
    assert {EditOp.REMOVE, EditOp.ADD, EditOp.REPLACE, EditOp.BACKGROUND, EditOp.UPSCALE} == set(
        SUPPORTED
    )


# ---------------------------------------------------------------- occluder shielding


def test_protection_is_applied_after_dilation_not_before() -> None:
    """The order is the whole point.

    The shield exists to stop the mask's growth reaching a neighbour. Subtracting it
    before growing would let the dilation walk straight back over it, which reads as
    working and is not — the produced mask is the audit trail, so it is asserted there.
    """
    models, _ = erasers()
    asked = square()
    # A strip immediately outside the selection, well inside where dilation will reach.
    shield = np.zeros((256, 256), np.uint8)
    shield[140:170, 60:140] = 255

    edit = execute(models, EditOp.REMOVE, image(), mask=asked, protect=shield)

    assert not (edit.mask > 0)[shield > 0].any(), "the growth reached into the shield"
    assert int((edit.mask > 0).sum()) > int((asked > 0).sum()), "nothing was dilated at all"


def test_the_selected_region_is_erased_even_under_a_shield() -> None:
    """A shield may withhold what dilation added; it may never withhold what was asked
    for. Otherwise the subject is left standing and the result looks like a no-op."""
    models, _ = erasers()
    asked = square()
    everything = np.full((256, 256), 255, np.uint8)

    edit = execute(models, EditOp.REMOVE, image(), mask=asked, protect=everything)
    assert (edit.mask > 0)[asked > 0].all()


def test_the_generative_lane_ignores_a_shield() -> None:
    """It paints into the hole rather than continuing the background, so a hole with a
    bite out of it comes back with a seam."""
    painted: list[str] = []

    def provider(rgb: np.ndarray, mask: np.ndarray, prompt: str) -> np.ndarray:
        painted.append(prompt)
        return rgb

    shield = np.zeros((256, 256), np.uint8)
    shield[60:140, 60:140] = 255
    edit = execute(
        Models(fill=provider),
        EditOp.ADD,
        image(),
        mask=square(),
        protect=shield,
        content="a hat",
    )
    assert painted == ["a hat"]
    assert (edit.mask > 0)[shield > 0].any(), "the shield was applied to a generative edit"
