"""The worker's edit step, with the models replaced.

`editors.edit` does four things: decode the bytes, find the region, run the edit, encode
the result. Three of those are the worker's own and are tested here. The fourth —
`execute` — is tested in `packages/models`, and stubbing it keeps this file about the
worker rather than about erasure.

Nothing here loads a model, so this runs in the fast tier. `make eval` is where the
result is judged on real photographs.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pytest
from editgpt_core import AssetRef, EditOp, EditSpec, MaskSource
from editgpt_core.errors import MaskTooSmallError, ProviderUnavailableError
from editgpt_core.rle import encode as encode_rle
from editgpt_models.execute import Edit
from editgpt_worker import editors


def png(width: int = 64, height: int = 48, colour: tuple[int, int, int] = (20, 140, 90)) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(buffer, format="PNG")
    return buffer.getvalue()


def jpeg(width: int = 64, height: int = 48) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (200, 30, 60)).save(buffer, format="JPEG")
    return buffer.getvalue()


def spec(
    op: EditOp = EditOp.REMOVE,
    *,
    target: str | None = "the car",
    mask: np.ndarray | None = None,
    content: str | None = None,
    content_type: str = "image/png",
    width: int = 64,
    height: int = 48,
) -> EditSpec:
    return EditSpec(
        op=op,
        image_ref=AssetRef(
            bucket="local", sha256="a" * 64, width=width, height=height, content_type=content_type
        ),
        mask_source=(
            MaskSource.BRUSH
            if mask is not None
            else (MaskSource.TEXT if target else MaskSource.WHOLE)
        ),
        target=target,
        content=content,
        mask_ref=encode_rle(mask) if mask is not None else None,
    )


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace `execute` and record what the worker handed it."""
    seen: dict[str, Any] = {}

    def fake_execute(models: Any, op: EditOp, image: np.ndarray, **kwargs: Any) -> Edit:
        seen.update({"op": op, "image": image, "models": models, **kwargs})
        painted = image.copy()
        if kwargs.get("mask") is not None:
            painted[kwargs["mask"] > 0] = 0
        return Edit(
            image=painted,
            mask=np.zeros(image.shape[:2], np.uint8),
            strategy="stub",
            cost=1.0,
            seconds=0.1,
        )

    monkeypatch.setattr(editors, "execute", fake_execute)
    monkeypatch.setattr(editors, "models_for", lambda _s: object())
    return seen


# ---------------------------------------------------------------- regions


def test_a_brushed_mask_wins_over_the_phrase(
    captured: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the user drew it, no model gets a vote."""
    grounded: list[str] = []

    def spy_detector() -> object:
        grounded.append("asked")
        return object()

    monkeypatch.setattr(editors, "detector", spy_detector)

    brushed = np.zeros((48, 64), np.uint8)
    brushed[10:30, 10:30] = 255
    editors.edit(png(), spec(mask=brushed))

    assert not grounded, "a brushed mask must not be second-guessed by the detector"
    assert captured["mask"] is not None
    assert int((captured["mask"] > 0).sum()) > 0


def test_a_clients_mask_is_scaled_onto_the_working_image(captured: dict[str, Any]) -> None:
    """The mask is drawn at the uploaded size; the edit runs at `MAX_SIDE`."""
    brushed = np.zeros((48, 64), np.uint8)
    brushed[10:30, 10:30] = 255

    editors.edit(png(64, 48), spec(mask=brushed))
    assert captured["mask"].shape == captured["image"].shape[:2]


def test_a_rescaled_mask_stays_binary(captured: dict[str, Any]) -> None:
    """Anything smoother than nearest-neighbour produces grey edges, which then read as
    a partially-selected region."""
    brushed = np.zeros((48, 64), np.uint8)
    brushed[10:30, 10:30] = 255

    editors.edit(png(64, 48), spec(mask=brushed))
    assert set(np.unique(captured["mask"])) <= {0, 255}


def test_upscale_needs_no_region(captured: dict[str, Any]) -> None:
    editors.edit(png(), spec(EditOp.UPSCALE, target=None, mask=None))
    assert captured["mask"] is None


def test_a_phrase_matching_nothing_is_refused_with_the_phrase_in_the_message(
    captured: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real outcome, not a fault — and the user needs to know which words failed."""
    from editgpt_models.segment import Segmentation

    monkeypatch.setattr(editors, "detector", lambda: object())
    monkeypatch.setattr(editors, "session", lambda _key: object())
    monkeypatch.setattr(
        "editgpt_models.segment.mask_from_phrase",
        lambda *_a, **_k: Segmentation(np.zeros((48, 64), np.uint8), 0.0, "none"),
    )

    with pytest.raises(MaskTooSmallError, match="unicorn"):
        editors.edit(png(), spec(target="a unicorn"))


# ---------------------------------------------------------------- encoding


def test_the_result_comes_back_in_the_format_it_arrived_as(captured: dict[str, Any]) -> None:
    """Returning a PNG for a JPEG upload multiplies the stored size of every photograph."""
    from PIL import Image

    brushed = np.zeros((48, 64), np.uint8)
    brushed[10:30, 10:30] = 255
    made = editors.edit(jpeg(), spec(mask=brushed, content_type="image/jpeg"))

    with Image.open(io.BytesIO(made.data)) as image:
        assert image.format == "JPEG"
    assert made.content_type == "image/jpeg"


def test_an_unknown_content_type_falls_back_to_a_lossless_format(
    captured: dict[str, Any],
) -> None:
    """Guessing JPEG for an unknown type would silently degrade the result."""
    from PIL import Image

    brushed = np.zeros((48, 64), np.uint8)
    brushed[10:30, 10:30] = 255
    made = editors.edit(png(), spec(mask=brushed, content_type="application/octet-stream"))

    with Image.open(io.BytesIO(made.data)) as image:
        assert image.format == "PNG"
    assert made.content_type == "image/png", "the row must not claim a type the bytes are not"


def test_a_large_image_is_worked_on_at_a_bounded_size(captured: dict[str, Any]) -> None:
    """The gateway's megapixel cap bounds memory; this bounds *time*.

    A 15.9 MP erase is minutes of model work for a result nobody views at that size.
    """
    editors.edit(png(4000, 3000), spec(EditOp.UPSCALE, target=None))
    assert max(captured["image"].shape[:2]) == editors.MAX_SIDE


def test_a_small_image_is_left_alone(captured: dict[str, Any]) -> None:
    """Upscaling to the working size would invent detail and then edit the invention."""
    editors.edit(png(64, 48), spec(EditOp.UPSCALE, target=None))
    assert captured["image"].shape[:2] == (48, 64)


# ---------------------------------------------------------------- model selection


def test_each_operation_asks_only_for_the_models_it_needs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loading everything would hold four heavy models resident to run one edit."""
    asked: list[str] = []

    def spy_session(key: str) -> object:
        asked.append(key)
        return object()

    monkeypatch.setattr(editors, "session", spy_session)

    editors.models_for(spec(EditOp.REMOVE))
    assert set(asked) == {"migan", "lama"}

    asked.clear()
    editors.models_for(spec(EditOp.UPSCALE, target=None))
    assert asked == ["esrgan-x2"]

    asked.clear()
    editors.models_for(spec(EditOp.BACKGROUND, target=None, content="green"))
    assert not asked, "recolouring is compositing; it needs no model at all"


def test_a_generative_operation_without_a_provider_says_which_keys_are_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An error should tell a human what to do next without opening the source."""
    from editgpt_providers import CloudflareWorkersAI

    monkeypatch.setattr(CloudflareWorkersAI, "is_configured", lambda _self: False)
    with pytest.raises(ProviderUnavailableError, match="CLOUDFLARE_ACCOUNT_ID"):
        editors.models_for(spec(EditOp.ADD, content="a hat"))


def test_a_generative_job_is_refused_before_the_image_is_ever_grounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The saving, not the message: grounding is the expensive part of a job.

    Before this, a request with no provider configured ran the detector and the segmenter
    to completion — seconds of CPU and most of the memory budget — and only then found
    there was nowhere to send the result.
    """
    from editgpt_providers import CloudflareWorkersAI

    monkeypatch.setattr(CloudflareWorkersAI, "is_configured", lambda _self: False)
    grounded: list[str] = []
    monkeypatch.setattr(editors, "region_for", lambda *_a, **_k: grounded.append("ran"))

    with pytest.raises(ProviderUnavailableError):
        editors.edit(png(), spec(EditOp.ADD, content="a hat"))
    assert not grounded, "the image was segmented for an edit that could never run"


def test_a_provider_that_is_backing_off_says_how_long_to_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wait is actionable; "unavailable" is not."""
    from editgpt_providers import CircuitBreaker, ProviderChain

    class Down:
        name = "cloudflare"

        def is_configured(self) -> bool:
            return True

        def fill(self, rgb: Any, mask: Any, prompt: str) -> Any:
            raise AssertionError("an open breaker must not be called")

    chain = ProviderChain([Down()], breakers={"cloudflare": CircuitBreaker(threshold=1)})
    chain.breakers["cloudflare"].record_failure("Workers AI 429")
    monkeypatch.setattr(editors, "provider_chain", lambda: chain)

    with pytest.raises(ProviderUnavailableError, match="retry in about"):
        editors.check_provider(spec(EditOp.ADD, content="a hat"))


def test_local_operations_never_ask_about_a_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Erase, upscale and recolour finish on this machine and must not depend on a key."""
    monkeypatch.setattr(
        editors, "provider_chain", lambda: pytest.fail("a local edit consulted the provider")
    )
    for op in (EditOp.REMOVE, EditOp.UPSCALE, EditOp.BACKGROUND):
        editors.check_provider(spec(op, target="the car", content="green"))


def test_the_chain_is_built_once_so_the_breaker_survives_a_job() -> None:
    """A breaker rebuilt per job is not a breaker: the count has to outlive the failure."""
    assert editors.provider_chain() is editors.provider_chain()


def test_avif_round_trips_instead_of_silently_becoming_png(captured: dict[str, Any]) -> None:
    """The bug this closes: AVIF was missing from the format map, so a 54 KB photograph
    came back as 476 KB of PNG bytes recorded under `image/avif`.
    """
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), (10, 90, 200)).save(buffer, format="AVIF")

    brushed = np.zeros((48, 64), np.uint8)
    brushed[10:30, 10:30] = 255
    made = editors.edit(buffer.getvalue(), spec(mask=brushed, content_type="image/avif"))

    assert made.content_type == "image/avif"
    with Image.open(io.BytesIO(made.data)) as image:
        assert image.format == "AVIF"


def test_every_accepted_upload_format_can_also_be_written() -> None:
    """A format the gateway takes in but cannot write out falls back to PNG, and the
    fallback was silent. Asserted across the boundary so the two lists cannot drift."""
    from editgpt_gateway.uploads import ALLOWED_FORMATS

    writable = {editors.FORMATS[mime] for mime in editors.FORMATS}
    assert set(ALLOWED_FORMATS.values()) <= set(editors.FORMATS), (
        "the gateway accepts a format the worker cannot write back"
    )
    assert "AVIF" in writable


def test_the_reported_size_is_what_was_produced_not_what_was_asked_for(
    captured: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """TD-016: `UPSCALE` changes the dimensions, so recording the request's would describe
    an image that does not exist."""

    def doubling_execute(models: Any, op: EditOp, image: np.ndarray, **kwargs: Any) -> Edit:
        bigger = np.repeat(np.repeat(image, 2, axis=0), 2, axis=1)
        return Edit(
            image=bigger,
            mask=np.zeros(bigger.shape[:2], np.uint8),
            strategy="stub",
            cost=0.0,
            seconds=0.1,
        )

    monkeypatch.setattr(editors, "execute", doubling_execute)
    made = editors.edit(png(64, 48), spec(EditOp.UPSCALE, target=None))

    assert (made.width, made.height) == (128, 96)


# ---------------------------------------------------------------- occluder shielding


def test_shielding_is_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """The measured default. `Thresholds.shield` records why."""
    from editgpt_models.segment import Segmentation

    found = np.zeros((48, 64), np.uint8)
    found[10:30, 10:30] = 255
    asked: list[str] = []

    monkeypatch.setattr(editors, "detector", lambda: object())
    monkeypatch.setattr(editors, "session", lambda _key: object())
    monkeypatch.setattr(
        "editgpt_models.segment.mask_from_phrase",
        lambda *_a, **_k: Segmentation(found, 0.9, "sam-box"),
    )
    monkeypatch.setattr(
        "editgpt_models.segment.occluder_shield", lambda *_a, **_k: asked.append("probed")
    )

    region = editors.region_for(spec(EditOp.REMOVE), np.zeros((48, 64, 3), np.uint8))
    assert region.protect is None
    assert not asked, "shielding ran despite being off"


def test_a_brushed_region_carries_no_shield(monkeypatch: pytest.MonkeyPatch) -> None:
    """ "If the user drew it, no model gets a vote" extends to shielding. Trimming a
    hand-drawn selection because a model disagrees about its edges is the same override
    the brush exists to escape."""
    called: list[str] = []
    monkeypatch.setattr(
        "editgpt_models.segment.occluder_shield", lambda *_a, **_k: called.append("asked")
    )

    brushed = np.zeros((48, 64), np.uint8)
    brushed[10:30, 10:30] = 255
    region = editors.region_for(spec(mask=brushed), np.zeros((48, 64, 3), np.uint8))

    assert region.protect is None
    assert not called


def test_a_grounded_removal_is_shielded_when_shielding_is_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Off by default — the golden set says it makes the erase worse overall — so the
    test states the setting it depends on rather than inheriting whatever is on disk."""
    from editgpt_models.config import Thresholds
    from editgpt_models.segment import Segmentation

    monkeypatch.setattr(editors, "load_thresholds", lambda: Thresholds(shield=True))

    found = np.zeros((48, 64), np.uint8)
    found[10:30, 10:30] = 255
    shield = np.zeros((48, 64), np.uint8)
    shield[30:40, 10:30] = 255

    monkeypatch.setattr(editors, "detector", lambda: object())
    monkeypatch.setattr(editors, "session", lambda _key: object())
    monkeypatch.setattr(
        "editgpt_models.segment.mask_from_phrase",
        lambda *_a, **_k: Segmentation(found, 0.9, "sam-box"),
    )
    monkeypatch.setattr("editgpt_models.segment.occluder_shield", lambda *_a, **_k: shield)

    region = editors.region_for(spec(EditOp.REMOVE), np.zeros((48, 64, 3), np.uint8))
    assert region.protect is not None
    assert int((region.protect > 0).sum()) > 0


def test_only_removal_pays_for_a_shield(monkeypatch: pytest.MonkeyPatch) -> None:
    """Probing is ~30 decoder calls; the generative lane cannot use the answer."""
    from editgpt_models.config import Thresholds
    from editgpt_models.segment import Segmentation

    monkeypatch.setattr(editors, "load_thresholds", lambda: Thresholds(shield=True))

    found = np.zeros((48, 64), np.uint8)
    found[10:30, 10:30] = 255
    asked: list[str] = []

    monkeypatch.setattr(editors, "detector", lambda: object())
    monkeypatch.setattr(editors, "session", lambda _key: object())
    monkeypatch.setattr(
        "editgpt_models.segment.mask_from_phrase",
        lambda *_a, **_k: Segmentation(found, 0.9, "sam-box"),
    )
    monkeypatch.setattr(
        "editgpt_models.segment.occluder_shield", lambda *_a, **_k: asked.append("probed")
    )

    region = editors.region_for(
        spec(EditOp.REPLACE, content="a sheep"), np.zeros((48, 64, 3), np.uint8)
    )
    assert region.protect is None
    assert not asked


def test_the_shield_reaches_execute(
    captured: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The plumbing between the two, which is where a shield would silently go missing."""
    shield = np.zeros((48, 64), np.uint8)
    shield[30:40, 10:30] = 255
    monkeypatch.setattr(
        editors, "region_for", lambda _s, _i: editors.Region(shield.copy(), "stub", shield)
    )

    editors.edit(png(), spec(EditOp.REMOVE))
    assert captured["protect"] is not None


# ---------------------------------------------------------------- returned resolution


def test_a_large_upload_comes_back_at_the_size_it_arrived_at(captured: dict[str, Any]) -> None:
    """TD-021. The bound belongs on the work, not on the answer: a 4000x3000 photograph
    was returning at 2048x1536 and the API never said so."""
    brushed = np.zeros((3000, 4000), np.uint8)
    brushed[1000:2000, 1000:2000] = 255

    made = editors.edit(png(4000, 3000), spec(mask=brushed, width=4000, height=3000))
    assert (made.width, made.height) == (4000, 3000)


def test_the_edit_still_runs_at_the_bounded_size(captured: dict[str, Any]) -> None:
    """The other half of TD-021, and the half that must not regress: `fill_metrics` runs
    two full-frame colour conversions per pass, so the *work* stays bounded."""
    brushed = np.zeros((3000, 4000), np.uint8)
    brushed[1000:2000, 1000:2000] = 255

    editors.edit(png(4000, 3000), spec(mask=brushed, width=4000, height=3000))
    assert max(captured["image"].shape[:2]) == editors.MAX_SIDE


def test_upscale_is_exempt_from_reprojection(captured: dict[str, Any]) -> None:
    """It produces its own geometry, and asking Real-ESRGAN for a 63 MP output would meet
    the RSS ceiling long before it produced a picture."""
    made = editors.edit(png(4000, 3000), spec(EditOp.UPSCALE, target=None, width=4000, height=3000))
    assert max(made.width, made.height) == editors.MAX_SIDE


# ---------------------------------------------------------------- backdrop colour


def test_the_requested_backdrop_colour_reaches_the_compositor(captured: dict[str, Any]) -> None:
    """TD-020: `BACKGROUND` painted the same green whatever was asked for, and reported
    success. The failure was silent, which is what made it worth a contract field."""
    request = EditSpec(
        op=EditOp.BACKGROUND,
        image_ref=AssetRef(
            bucket="local", sha256="a" * 64, width=64, height=48, content_type="image/png"
        ),
        mask_source=MaskSource.WHOLE,
        colour="#3366ff",
    )
    editors.edit(png(), request)
    assert captured["colour"] == (0x33, 0x66, 0xFF)


def test_a_request_with_no_colour_still_gets_one(captured: dict[str, Any]) -> None:
    request = EditSpec(
        op=EditOp.BACKGROUND,
        image_ref=AssetRef(
            bucket="local", sha256="a" * 64, width=64, height=48, content_type="image/png"
        ),
        mask_source=MaskSource.WHOLE,
        content="a green wall",
    )
    editors.edit(png(), request)
    assert captured["colour"] == editors.GREEN
