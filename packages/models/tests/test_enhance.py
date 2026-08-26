"""Tiled upscaling: correct size, no lattice, bounded memory."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from editgpt_models.enhance import OVERLAP, SCALE, TILE, downscale_to, upscale


class FakeUpscaler:
    """A stand-in that doubles a tile exactly, so geometry can be tested without weights."""

    def __init__(self) -> None:
        self.calls = 0

    def get_inputs(self):  # type: ignore[no-untyped-def]
        return [type("I", (), {"name": "input"})()]

    def run(self, _outputs, feed):  # type: ignore[no-untyped-def]
        self.calls += 1
        array = next(iter(feed.values()))[0]  # CHW, 0..1
        hwc = array.transpose(1, 2, 0)
        doubled = cv2.resize(
            hwc, (hwc.shape[1] * SCALE, hwc.shape[0] * SCALE), interpolation=cv2.INTER_NEAREST
        )
        return [doubled.transpose(2, 0, 1)[None, ...]]


def textured(side: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 255, (side, side, 3), dtype=np.uint8)
    return np.asarray(cv2.GaussianBlur(raw, (0, 0), 1.2), dtype=np.uint8)


def test_a_small_image_is_processed_in_one_pass() -> None:
    session = FakeUpscaler()
    out = upscale(session, textured(TILE // 2))
    assert session.calls == 1
    assert out.shape[:2] == (TILE, TILE)


def test_a_large_image_is_tiled_and_keeps_its_dimensions() -> None:
    session = FakeUpscaler()
    src = textured(TILE * 2)
    out = upscale(session, src)
    assert session.calls > 1, "a large image must be tiled"
    assert out.shape[:2] == (src.shape[0] * SCALE, src.shape[1] * SCALE)
    assert out.dtype == np.uint8


def test_non_square_images_keep_their_aspect_ratio() -> None:
    session = FakeUpscaler()
    src = np.asarray(textured(TILE * 2)[:, : TILE * 3 // 2], dtype=np.uint8)
    out = upscale(session, src)
    assert out.shape[:2] == (src.shape[0] * SCALE, src.shape[1] * SCALE)


def test_tiling_leaves_no_lattice() -> None:
    """A plain grid shows as periodic edge energy at the tile pitch. The taper prevents it."""
    session = FakeUpscaler()
    out = upscale(session, textured(TILE * 2))

    grey = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY).astype(np.float32)
    energy = np.abs(cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)).mean(axis=0)
    pitch = (TILE - OVERLAP) * SCALE
    at_seams = energy[pitch::pitch].mean()
    assert at_seams / energy.mean() < 1.35, "visible tile lattice"


def test_output_is_not_blank_or_saturated() -> None:
    session = FakeUpscaler()
    out = upscale(session, textured(TILE * 2))
    assert 20 < float(out.mean()) < 235
    assert float(out.std()) > 5, "detail was lost entirely"


def test_upscale_memory_stays_bounded_across_repeats() -> None:
    """Tiling exists so activations do not scale with image area. Prove it does not leak."""
    import gc

    import psutil

    session = FakeUpscaler()
    src = textured(TILE * 2)
    upscale(session, src)  # warm allocator
    gc.collect()
    baseline = psutil.Process().memory_info().rss / 1e6

    for _ in range(3):
        upscale(session, src)
    gc.collect()
    grew = psutil.Process().memory_info().rss / 1e6 - baseline
    assert grew < 200, f"RSS grew {grew:.0f} MB over three upscales"


@pytest.mark.parametrize("side", [64, 200, 1024])
def test_downscale_respects_the_ceiling(side: int) -> None:
    out = downscale_to(textured(1024), side)
    assert max(out.shape[:2]) <= side


def test_downscale_is_a_no_op_when_already_small() -> None:
    src = textured(100)
    assert downscale_to(src, 500) is src
