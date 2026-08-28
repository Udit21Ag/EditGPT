"""Keeping the eraser's dilation off whatever is standing in front of the target.

The sessions are fakes. That is not a shortcut here: the thing under test is a *policy* —
which of SAM's returned regions count as something else in the way — and the policy is
the part that got it wrong. Real weights would test MobileSAM, which is not the code that
erased a man's shoe along with the Eiffel Tower.

Each fake decoder answers a point prompt with a scripted region, so a test can stage the
exact geometry it is about: an occluder that overhangs the target, a part that lies
inside it, a neighbour touching its edge.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from editgpt_models import segment
from editgpt_models.compositing import Mask
from editgpt_models.config import Thresholds
from editgpt_models.segment import occluder_shield

SIZE = 256


class FakeEncoder:
    """Declares an HWC input so `preprocess_for_encoder` takes the short path."""

    def __init__(self) -> None:
        self.calls = 0

    def get_inputs(self) -> list[Any]:
        return [type("In", (), {"name": "image", "shape": [3, 1024, 1024]})()]

    def run(self, _outputs: Any, _feed: dict[str, Any]) -> list[np.ndarray]:
        self.calls += 1
        return [np.zeros((1, 256, 64, 64), np.float32)]


class ScriptedDecoder:
    """Returns whichever scripted region contains the prompted point."""

    NAMES = (
        "image_embeddings",
        "point_coords",
        "point_labels",
        "mask_input",
        "has_mask_input",
        "orig_im_size",
    )

    def __init__(self, regions: list[tuple[np.ndarray, float]], default: float = 0.95) -> None:
        self.regions, self.default = regions, default
        self.points: list[tuple[int, int]] = []

    def get_inputs(self) -> list[Any]:
        return [type("In", (), {"name": n})() for n in self.NAMES]

    def run(self, _outputs: Any, feed: dict[str, Any]) -> list[np.ndarray]:
        # `_decode` multiplies by the encoder scale; at 256 px that is 1024/256 = 4.
        x, y = (feed["point_coords"][0][0] / (segment.ENCODER_SIZE / SIZE)).astype(int)
        self.points.append((int(x), int(y)))
        for region, confidence in self.regions:
            if region[y, x]:
                return [np.where(region, 1.0, -1.0)[None, None], np.array([[confidence]])]
        return [np.full((1, 1, SIZE, SIZE), -1.0), np.array([[self.default]])]


def box(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    out = np.zeros((SIZE, SIZE), bool)
    out[y0:y1, x0:x1] = True
    return out


def as_mask(region: np.ndarray) -> Mask:
    return np.asarray(region, np.uint8) * 255


@pytest.fixture(autouse=True)
def _clear_embedding_cache() -> None:
    segment._LAST_EMBEDDING = None


def image() -> np.ndarray:
    return np.zeros((SIZE, SIZE, 3), np.uint8)


# ------------------------------------------------------------------ what is shielded


def test_a_neighbour_overhanging_the_target_is_shielded() -> None:
    """The `i8` failure: the mask is grown by 5% of the object's longest side and that
    slack lands on the shoe of a man jumping in front of the tower."""
    target = as_mask(box(60, 200, 60, 140))
    occluder = box(150, 240, 140, 200)  # beside the target, reaching into its dilation

    decoder = ScriptedDecoder([(occluder, 0.95)])
    shield = occluder_shield(FakeEncoder(), decoder, image(), target)

    assert shield.any(), "the neighbour was not found"
    covered = (shield > 0)[occluder].mean()
    assert covered > 0.5, f"only {covered:.0%} of the neighbour was shielded"


def test_a_part_of_the_target_is_not_shielded() -> None:
    """The regression this threshold exists for.

    MobileSAM is part-aware: probing near a horse's edge returns *the leg*, which is
    small, confident, and has a low IoU against the whole horse. Shielding it withheld
    26% of the animal from its own erasure and left it standing in the result.

    The staged leg is 44% inside the target — deliberately in the band between the 0.30
    that ships and the 0.80 the first version used, because a part that barely pokes out
    is rejected either way and would let the original bug back in unnoticed.
    """
    target = as_mask(box(60, 140, 60, 140))
    leg = box(100, 190, 70, 95)  # a limb of the target, extending well past its edge

    inside_fraction = (leg & (target > 0)).sum() / leg.sum()
    assert 0.30 < inside_fraction < 0.80, (
        f"fixture drifted out of the disputed band: {inside_fraction:.2f}"
    )

    decoder = ScriptedDecoder([(leg, 0.95)])
    shield = occluder_shield(FakeEncoder(), decoder, image(), target)

    assert not shield.any(), "a part of the target was shielded from its own erasure"


def test_scenery_is_too_large_to_be_an_occluder() -> None:
    """Probing beside a tower returns the sky. Shielding it would withhold the whole
    dilation ring to protect nothing."""
    target = as_mask(box(60, 200, 60, 140))
    sky = ~box(60, 200, 60, 140)

    decoder = ScriptedDecoder([(sky, 0.95)])
    assert not occluder_shield(FakeEncoder(), decoder, image(), target).any()


def test_a_region_the_decoder_is_unsure_of_is_not_shielded() -> None:
    target = as_mask(box(60, 200, 60, 140))
    occluder = box(150, 240, 140, 200)

    decoder = ScriptedDecoder([(occluder, 0.10)])
    assert not occluder_shield(FakeEncoder(), decoder, image(), target).any()


# ------------------------------------------------------------------ guards


def test_the_slack_next_to_the_target_survives_a_touching_neighbour() -> None:
    """Dilation exists because a mask cut on the silhouette leaves a halo of the object's
    own edge pixels. A neighbour pressed against the target must not take that back."""
    target = as_mask(box(60, 200, 60, 140))
    touching = box(60, 200, 140, 220)  # flush against the target's right edge

    decoder = ScriptedDecoder([(touching, 0.95)])
    shield = occluder_shield(FakeEncoder(), decoder, image(), target)

    assert shield.any(), "the neighbour was not found at all"
    assert not shield[120, 140], "the pixel next to the target was shielded away"
    assert shield[120, 150], "the neighbour beyond the margin should still be shielded"


def test_a_shield_can_never_withhold_the_selected_region() -> None:
    """The guarantee that replaced a percentage guard.

    An earlier version abandoned the shield when it would withhold more than 15% of the
    target — a check that could never fire, because the margin already excludes the
    target from the shield. This asserts the property directly instead: whatever a probe
    returns, the region the user asked to remove is still removed. The staged region is
    deliberately pathological, covering almost the whole frame including the target.
    """
    target = as_mask(box(60, 200, 60, 140))
    swallowing = box(0, 256, 0, 256) & ~box(60, 120, 60, 140)

    decoder = ScriptedDecoder([(swallowing, 0.95)])
    limits = Thresholds(shield_max_area=1.0)
    shield = occluder_shield(FakeEncoder(), decoder, image(), target, thresholds=limits)

    assert not (shield > 0)[target > 0].any(), "the shield reached into the selected region"


def test_an_empty_target_is_not_probed() -> None:
    decoder = ScriptedDecoder([])
    assert not occluder_shield(
        FakeEncoder(), decoder, image(), np.zeros((SIZE, SIZE), np.uint8)
    ).any()
    assert not decoder.points, "an empty mask has no edge to walk"


def test_probing_is_bounded_by_the_cap() -> None:
    """A long boundary must not turn one edit into hundreds of decoder calls."""
    target = as_mask(box(10, 246, 10, 246) & ~box(20, 236, 20, 236))  # a thin, long ring

    decoder = ScriptedDecoder([])
    occluder_shield(FakeEncoder(), decoder, image(), target)
    assert len(decoder.points) <= segment.PROBE_CAP


def test_every_probe_starts_outside_the_target() -> None:
    """Probing inside is what returned the horse's leg; the invariant is worth asserting
    directly rather than only through the case that broke."""
    target = as_mask(box(60, 200, 60, 140))

    decoder = ScriptedDecoder([])
    occluder_shield(FakeEncoder(), decoder, image(), target)

    assert decoder.points
    assert not any(target[y, x] for x, y in decoder.points)


# ------------------------------------------------------------------ the embedding cache


def test_the_same_image_is_encoded_once() -> None:
    """One removal encodes the image to turn a box into a mask and again to probe around
    it. The encoder is ~1.5 s; the decoder is nearly free."""
    encoder = FakeEncoder()
    rgb = image()

    segment.embed(encoder, rgb)
    segment.embed(encoder, rgb.copy())
    assert encoder.calls == 1


def test_a_different_image_is_encoded_again() -> None:
    """The key is the pixels, so a hit means the same picture rather than the same array."""
    encoder = FakeEncoder()
    other = image()
    other[0, 0] = 255

    segment.embed(encoder, image())
    segment.embed(encoder, other)
    assert encoder.calls == 2


# ------------------------------------------------------------------ tapping


def test_a_tap_returns_the_region_under_it() -> None:
    """Magic select: SAM alone, no detector. The scripted decoder answers the prompted
    point, which is exactly the contract this function relies on."""
    from editgpt_models.segment import mask_from_points

    region = box(40, 120, 40, 120)
    decoder = ScriptedDecoder([(region, 0.97)])
    found = mask_from_points(FakeEncoder(), decoder, image(), [(0.3, 0.3, True)])

    assert found.source == "sam-point"
    assert found.confidence == pytest.approx(0.97)
    assert (found.mask > 0)[region].all()


def test_a_tap_that_finds_nothing_is_a_real_answer() -> None:
    """Not an error: the caller says "nothing there" rather than erasing a guess."""
    from editgpt_models.segment import mask_from_points

    found = mask_from_points(FakeEncoder(), ScriptedDecoder([]), image(), [(0.3, 0.3, True)])
    assert found.source == "none"
    assert found.confidence == 0.0
    assert not found.mask.any()


def test_no_points_asks_the_decoder_nothing() -> None:
    from editgpt_models.segment import mask_from_points

    decoder = ScriptedDecoder([])
    found = mask_from_points(FakeEncoder(), decoder, image(), [])
    assert not found.mask.any()
    assert not decoder.points, "an empty prompt should not reach the model"


def test_an_excluded_point_is_sent_as_a_negative_prompt() -> None:
    """The second half of the interaction — tap what came along and should not have.

    Asserted on the labels the decoder receives, because that is the only place the
    distinction exists; a positive and a negative tap are the same coordinates otherwise.
    """
    from editgpt_models import segment

    seen: dict[str, np.ndarray] = {}
    real = segment._decode

    def capture(
        decoder: Any, embedding: Any, coords: Any, labels: Any, shape: Any, scale: Any
    ) -> tuple[Any, float]:
        seen["labels"] = labels
        seen["coords"] = coords
        return real(decoder, embedding, coords, labels, shape, scale)

    segment._decode, original = capture, segment._decode
    try:
        segment.mask_from_points(
            FakeEncoder(),
            ScriptedDecoder([]),
            image(),
            [(0.3, 0.3, True), (0.6, 0.6, False)],
        )
    finally:
        segment._decode = original

    labels = seen["labels"][0].tolist()
    assert labels[:2] == [1.0, 0.0], f"include/exclude were not encoded as 1/0: {labels}"
    # SAM's export reads the final prompt as a box corner without a padding point, and the
    # mask tears along it.
    assert labels[-1] == -1.0, "the padding point is missing"
    assert seen["coords"].shape[1] == 3
