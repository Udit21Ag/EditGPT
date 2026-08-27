"""EditSpec rejects unactionable work at construction, not three agents later."""

from __future__ import annotations

import pytest
from editgpt_core import AssetRef, Constraints, EditOp, EditSpec, MaskRef, MaskSource
from pydantic import ValidationError

DIGEST = "a" * 64


def image(width: int = 800, height: int = 600) -> AssetRef:
    return AssetRef(bucket="edits", sha256=DIGEST, width=width, height=height)


def mask(width: int = 800, height: int = 600) -> MaskRef:
    return MaskRef(width=width, height=height, counts=[0, width * height])


def test_asset_ref_rejects_a_non_digest() -> None:
    with pytest.raises(ValidationError, match="64 lowercase hex"):
        AssetRef(bucket="edits", sha256="not-a-digest", width=10, height=10)


def test_asset_ref_exposes_uri_and_size() -> None:
    ref = image(2000, 1000)
    assert ref.uri == f"asset://edits/{DIGEST}"
    assert ref.megapixels == pytest.approx(2.0)


def test_mask_ref_counts_must_cover_the_image() -> None:
    with pytest.raises(ValidationError, match="RLE counts sum to"):
        MaskRef(width=10, height=10, counts=[0, 50])


def test_mask_ref_reports_area_and_coverage() -> None:
    ref = MaskRef(width=10, height=10, counts=[40, 20, 40])
    assert ref.area_px == 20
    assert ref.coverage == pytest.approx(0.2)


def test_remove_needs_a_target_or_a_mask() -> None:
    with pytest.raises(ValidationError, match="either a `target` phrase or an explicit mask"):
        EditSpec(op=EditOp.REMOVE, image_ref=image(), mask_source=MaskSource.AUTO)


def test_remove_by_text_is_accepted() -> None:
    spec = EditSpec(
        op=EditOp.REMOVE, image_ref=image(), mask_source=MaskSource.TEXT, target="the car"
    )
    assert spec.op is EditOp.REMOVE
    assert not spec.is_generative


@pytest.mark.parametrize("op", [EditOp.ADD, EditOp.REPLACE])
def test_content_ops_need_content(op: EditOp) -> None:
    """`BACKGROUND` is deliberately absent: a hex `colour` answers "what goes there" as
    completely as a description does, so it has its own rule below."""
    with pytest.raises(ValidationError, match="needs `content`"):
        EditSpec(op=op, image_ref=image(), mask_source=MaskSource.BRUSH, mask_ref=mask())


def test_text_mask_source_needs_a_target() -> None:
    with pytest.raises(ValidationError, match="needs a `target` phrase"):
        EditSpec(
            op=EditOp.ADD, image_ref=image(), mask_source=MaskSource.TEXT, content="a moustache"
        )


@pytest.mark.parametrize("source", [MaskSource.BRUSH, MaskSource.POINT])
def test_hand_drawn_mask_sources_need_a_mask(source: MaskSource) -> None:
    with pytest.raises(ValidationError, match="needs a `mask_ref`"):
        EditSpec(op=EditOp.REMOVE, image_ref=image(), mask_source=source, target="the car")


def test_a_mask_of_a_differently_shaped_image_is_refused() -> None:
    """What the rule is for: a mask that belongs to some other picture."""
    with pytest.raises(ValidationError, match="not the shape of"):
        EditSpec(
            op=EditOp.REMOVE,
            image_ref=image(800, 600),
            mask_source=MaskSource.BRUSH,
            mask_ref=mask(640, 640),
        )


def test_a_mask_at_a_smaller_resolution_is_accepted() -> None:
    """A candidate from `POST /v1/masks` arrives at the resolution grounding ran at, not
    the upload's. Requiring an exact match made the client upscale a mask the worker
    scales straight back down — lossy work to satisfy a check that gained nothing."""
    spec = EditSpec(
        op=EditOp.REMOVE,
        image_ref=image(800, 600),
        mask_source=MaskSource.BRUSH,
        mask_ref=mask(640, 480),
    )
    assert spec.mask_ref is not None
    assert (spec.mask_ref.width, spec.mask_ref.height) == (640, 480)


def test_a_mask_a_few_pixels_out_of_shape_is_still_refused() -> None:
    """The tolerance absorbs one rescale's rounding and nothing more: five pixels wrong
    on an 800x600 image scores 3000 against a budget of 1400."""
    with pytest.raises(ValidationError, match="not the shape of"):
        EditSpec(
            op=EditOp.REMOVE,
            image_ref=image(800, 600),
            mask_source=MaskSource.BRUSH,
            mask_ref=mask(645, 480),
        )


def brushed(op: EditOp, *, target: str | None = None, content: str | None = None) -> EditSpec:
    return EditSpec(
        op=op,
        image_ref=image(),
        mask_source=MaskSource.BRUSH,
        mask_ref=mask(),
        target=target,
        content=content,
    )


def test_generative_ops_are_flagged_and_removal_is_not() -> None:
    """Phase 0 measured that removal is worse on the free remote lane, so it never routes there."""
    assert brushed(EditOp.ADD, content="a hat").is_generative
    assert not brushed(EditOp.REMOVE, target="the car").is_generative
    assert not brushed(EditOp.UPSCALE).is_generative


def test_spec_round_trips_through_json() -> None:
    spec = EditSpec(
        op=EditOp.REMOVE,
        image_ref=image(),
        mask_source=MaskSource.TEXT,
        target="the car",
        constraints=Constraints(max_seconds=12.5, allow_remote=False),
        confidence=0.82,
    )
    assert EditSpec.model_validate_json(spec.model_dump_json()) == spec


def test_spec_is_immutable() -> None:
    spec = EditSpec(
        op=EditOp.REMOVE, image_ref=image(), mask_source=MaskSource.TEXT, target="the car"
    )
    with pytest.raises(ValidationError):
        spec.target = "the van"


def test_unknown_fields_are_rejected() -> None:
    """Typos in an agent-to-agent payload should fail loudly, not be silently dropped."""
    with pytest.raises(ValidationError):
        EditSpec(
            op=EditOp.REMOVE,
            image_ref=image(),
            mask_source=MaskSource.TEXT,
            target="the car",
            targt="the car",  # type: ignore[call-arg]
        )


# ---------------------------------------------------------------- background colour


def background(**over: object) -> EditSpec:
    fields: dict[str, object] = {
        "op": EditOp.BACKGROUND,
        "image_ref": image(),
        "mask_source": MaskSource.WHOLE,
        "content": "a green wall",
    }
    fields.update(over)
    return EditSpec(**fields)  # type: ignore[arg-type]


def test_a_colour_is_enough_to_say_what_the_backdrop_should_be() -> None:
    """TD-020. `content` used to be the only way to answer, and nothing read it — the
    operation painted the same green whatever it said."""
    spec = background(content=None, colour="#3366ff")
    assert spec.rgb_colour((0, 0, 0)) == (0x33, 0x66, 0xFF)


def test_free_text_is_still_enough_on_its_own() -> None:
    """An intent agent will one day turn "a sunset" into something paintable; until then
    the field is accepted and the fallback colour is used."""
    assert background(colour=None).rgb_colour((46, 160, 67)) == (46, 160, 67)


def test_a_backdrop_described_by_neither_is_refused() -> None:
    with pytest.raises(ValidationError, match="`colour` or `content`"):
        background(content=None, colour=None)


def test_a_colour_that_is_not_a_colour_is_refused_at_the_boundary() -> None:
    for bad in ("blue", "#12345", "#gggggg", "3366ff"):
        with pytest.raises(ValidationError):
            background(colour=bad)


def test_the_channels_are_not_swapped() -> None:
    """OpenCV is BGR and everything else here is RGB; a swap survives every test that
    only checks a grey value."""
    assert background(colour="#ff0000").rgb_colour((0, 0, 0)) == (255, 0, 0)
    assert background(colour="#0000ff").rgb_colour((0, 0, 0)) == (0, 0, 255)


def test_case_does_not_matter() -> None:
    assert background(colour="#AABBCC").rgb_colour((0, 0, 0)) == background(
        colour="#aabbcc"
    ).rgb_colour((0, 0, 0))
