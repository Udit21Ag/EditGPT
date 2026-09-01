"""Binding an intent to an image, and the rules that stop an unusable one getting there."""

from __future__ import annotations

import pytest
from editgpt_core import AssetRef, EditOp, MaskRef, MaskSource
from editgpt_planner import Intent
from pydantic import ValidationError


def image() -> AssetRef:
    return AssetRef(bucket="local", sha256="a" * 64, width=640, height=480)


def mask() -> MaskRef:
    """Half the frame set, as the codec writes it: a run of zeros, then a run of ones."""
    return MaskRef(width=640, height=480, counts=[153_600, 153_600])


def test_a_phrase_becomes_a_text_masked_spec() -> None:
    spec = Intent(op=EditOp.REMOVE, target="the car").to_spec(image())
    assert spec.mask_source is MaskSource.TEXT
    assert spec.target == "the car"


def test_a_drawn_region_outranks_the_phrase_beside_it() -> None:
    """A brush stroke is the most specific thing a user can give; the sentence is context."""
    spec = Intent(op=EditOp.REMOVE, target="the car").to_spec(image(), mask=mask())
    assert spec.mask_source is MaskSource.BRUSH
    assert spec.mask_ref is not None


def test_an_operation_with_no_subject_acts_on_the_whole_image() -> None:
    spec = Intent(op=EditOp.UPSCALE).to_spec(image())
    assert spec.mask_source is MaskSource.WHOLE


@pytest.mark.parametrize(
    "kwargs",
    [
        {"op": EditOp.ADD},
        {"op": EditOp.REPLACE, "target": "the sky"},
        {"op": EditOp.REMOVE},
        {"op": EditOp.BACKGROUND},
    ],
)
def test_an_intent_that_could_not_become_a_spec_is_refused_at_the_planner(
    kwargs: dict[str, object],
) -> None:
    """The same rules `EditSpec` enforces, one step earlier, so the failure is "the model
    did not answer the question" rather than a validation error from inside job creation."""
    with pytest.raises(ValidationError):
        Intent(**kwargs)  # type: ignore[arg-type]


def test_a_colour_must_be_a_colour() -> None:
    with pytest.raises(ValidationError):
        Intent(op=EditOp.BACKGROUND, colour="blue")
