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


@pytest.mark.parametrize("op", [EditOp.ADD, EditOp.REPLACE, EditOp.BACKGROUND])
def test_content_ops_need_content(op: EditOp) -> None:
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


def test_mask_dimensions_must_match_the_image() -> None:
    with pytest.raises(ValidationError, match="but image is"):
        EditSpec(
            op=EditOp.REMOVE,
            image_ref=image(800, 600),
            mask_source=MaskSource.BRUSH,
            mask_ref=mask(640, 480),
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
