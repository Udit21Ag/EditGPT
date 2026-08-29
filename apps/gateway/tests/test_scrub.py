"""Removing metadata without touching the picture.

Two properties, and the second is as important as the first. The metadata must be gone —
that is the privacy fix. And the pixels must be *identical*, because this is an image
editor and a boundary that silently recompresses every upload would compound its loss
through every edit made afterwards.

The fixtures are built with real metadata written by Pillow rather than asserted against a
hand-made byte string, so the parser is checked against files a camera could actually
produce.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from editgpt_gateway.scrub import has_metadata, scrub
from PIL import Image


def photo(
    fmt: str, *, exif: bool = False, size: tuple[int, int] = (64, 48), **save: object
) -> bytes:
    """A textured image, so a recompression is visible in the pixels rather than hidden in
    a flat colour."""
    generator = np.random.default_rng(seed=3)
    array = generator.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    image = Image.fromarray(array)

    options: dict[str, object] = dict(save)
    if exif:
        tags = Image.Exif()
        tags[0x010F] = "iQOO"  # Make
        tags[0x0110] = "iQOO Neo7"  # Model
        tags[0x0132] = "2026:08:25 18:27:01"  # DateTime
        options["exif"] = tags.tobytes()

    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **options)
    return buffer.getvalue()


def pixels(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def tags(data: bytes) -> dict[int, object]:
    with Image.open(io.BytesIO(data)) as image:
        return dict(image.getexif())


# ------------------------------------------------------------------ JPEG


def test_a_jpeg_loses_its_camera() -> None:
    original = photo("JPEG", exif=True, quality=90)
    assert tags(original), "the fixture carries no metadata; it proves nothing"

    cleaned = scrub(original, "JPEG")
    assert tags(cleaned) == {}


def test_a_jpeg_keeps_every_pixel() -> None:
    """The reason this is done at the segment level rather than by re-saving.

    Re-encoding at quality 90 moves pixels by tens of levels; here the compressed scan is
    copied byte for byte and the decode is identical.
    """
    original = photo("JPEG", exif=True, quality=90)
    assert np.array_equal(pixels(scrub(original, "JPEG")), pixels(original))


def test_a_jpeg_keeps_its_colour_profile() -> None:
    """ICC says how to interpret the pixels. Dropping it changes what the picture looks
    like, which is not what "remove metadata" is supposed to mean."""
    profile = b"\x00" * 128  # opaque to us; it only has to survive
    original = photo("JPEG", exif=True, quality=90, icc_profile=profile)

    with Image.open(io.BytesIO(scrub(original, "JPEG"))) as image:
        assert image.info.get("icc_profile") == profile


def test_a_jpeg_with_nothing_to_remove_is_returned_untouched() -> None:
    # Not "equivalent" — the same bytes. Anything else is a silent rewrite of a file
    # somebody trusted us with.
    original = photo("JPEG", quality=90)
    assert scrub(original, "JPEG") is original or scrub(original, "JPEG") == original
    assert not has_metadata(original, "JPEG")


def test_a_stripped_jpeg_is_smaller_than_what_arrived() -> None:
    original = photo("JPEG", exif=True, quality=90)
    assert len(scrub(original, "JPEG")) < len(original)


# ------------------------------------------------------------------ PNG


def test_a_png_loses_its_text_chunks() -> None:
    from PIL import PngImagePlugin

    info = PngImagePlugin.PngInfo()
    info.add_text("Author", "someone")
    info.add_text("Comment", "taken at home")
    original = photo("PNG", pnginfo=info)

    with Image.open(io.BytesIO(original)) as image:
        assert image.info.get("Author") == "someone", "the fixture carries no text"

    cleaned = scrub(original, "PNG")
    with Image.open(io.BytesIO(cleaned)) as image:
        assert "Author" not in image.info
        assert "Comment" not in image.info


def test_a_png_keeps_every_pixel() -> None:
    from PIL import PngImagePlugin

    info = PngImagePlugin.PngInfo()
    info.add_text("Author", "someone")
    original = photo("PNG", pnginfo=info)
    assert np.array_equal(pixels(scrub(original, "PNG")), pixels(original))


def test_a_png_with_nothing_to_remove_is_returned_untouched() -> None:
    original = photo("PNG")
    assert scrub(original, "PNG") == original


# ------------------------------------------------------------------ WebP and AVIF


def test_a_webp_loses_its_exif_and_keeps_its_pixels() -> None:
    original = photo("WEBP", exif=True, lossless=True)
    if not tags(original):
        pytest.skip("this Pillow build does not write EXIF into WebP")

    cleaned = scrub(original, "WEBP")
    assert tags(cleaned) == {}
    assert np.array_equal(pixels(cleaned), pixels(original))


def test_an_avif_loses_its_exif() -> None:
    original = photo("AVIF", exif=True)
    if not tags(original):
        pytest.skip("this Pillow build does not write EXIF into AVIF")

    assert tags(scrub(original, "AVIF")) == {}


# ------------------------------------------------------------------ robustness


def test_a_truncated_file_is_returned_rather_than_raising() -> None:
    """`inspect` decides what is readable; this is not a second opinion. Half a JPEG is
    already going to be refused, and raising here would turn that into a 500."""
    original = photo("JPEG", exif=True, quality=90)
    for cut in (2, 10, len(original) // 2):
        assert isinstance(scrub(original[:cut], "JPEG"), bytes)


def test_something_that_is_not_the_format_it_claims_is_passed_through() -> None:
    assert scrub(b"not an image at all", "JPEG") == b"not an image at all"
    assert scrub(b"not an image at all", "PNG") == b"not an image at all"
    assert scrub(b"not an image at all", "WEBP") == b"not an image at all"
