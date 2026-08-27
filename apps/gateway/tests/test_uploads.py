"""The input boundary. Every case here is an attack or a mistake that must not get past it."""

from __future__ import annotations

import io
import struct
import zlib

import pytest
from editgpt_gateway import uploads
from editgpt_gateway.uploads import UploadRejectedError
from fastapi.testclient import TestClient

from .conftest import png_bytes


def bomb_png(width: int, height: int) -> bytes:
    """A tiny PNG whose header claims enormous dimensions.

    Hand-built rather than produced by Pillow, because Pillow will not write a header it
    has no pixels for — which is exactly why this file is dangerous and why the check has
    to happen at the header rather than after a decode.
    """

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IEND", b"")


def test_a_real_png_is_accepted_and_measured() -> None:
    inspected = uploads.inspect(png_bytes(320, 240), max_megapixels=40)
    assert (inspected.width, inspected.height) == (320, 240)
    assert inspected.content_type == "image/png"
    assert inspected.megapixels == pytest.approx(0.0768)


def test_a_decompression_bomb_is_refused_from_the_header() -> None:
    """The defence that matters: 100 bytes on the wire, 10 GB if decoded."""
    data = bomb_png(60000, 60000)
    assert len(data) < 200
    with pytest.raises(UploadRejectedError, match="MP limit"):
        uploads.inspect(data, max_megapixels=40)


def test_a_bomb_just_over_our_limit_is_caught_by_our_own_check() -> None:
    """Below Pillow's ceiling but above ours, so this is *our* check firing, not its.

    Worth separating: without it, the test above would pass on Pillow's defence alone and
    the megapixel limit this service actually advertises would be untested.
    """
    data = bomb_png(8000, 8000)  # 64 MP: under Pillow's ~178 MP refusal, over our 40
    with pytest.raises(UploadRejectedError, match=r"64\.0 MP, over the 40 MP limit"):
        uploads.inspect(data, max_megapixels=40)


def test_a_non_image_is_refused() -> None:
    with pytest.raises(UploadRejectedError, match="not an image"):
        uploads.inspect(b"#!/bin/sh\nrm -rf /\n", max_megapixels=40)


def test_svg_is_refused_even_though_it_is_an_image_format() -> None:
    """Not on the allowlist: it is markup, and markup can reference and script."""
    svg = b'<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"></svg>'
    with pytest.raises(UploadRejectedError):
        uploads.inspect(svg, max_megapixels=40)


def test_a_gif_is_refused_by_the_allowlist() -> None:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buffer, format="GIF")
    with pytest.raises(UploadRejectedError, match="not supported"):
        uploads.inspect(buffer.getvalue(), max_megapixels=40)


def test_the_declared_content_type_does_not_decide_the_format(client: TestClient) -> None:
    """A PNG announced as a JPEG is stored as a PNG. The bytes are the authority."""
    response = client.post("/v1/images", files={"file": ("lying.jpg", png_bytes(), "image/jpeg")})
    assert response.status_code == 201
    assert response.json()["content_type"] == "image/png"


def test_an_oversized_image_is_refused_with_a_usable_message(client: TestClient) -> None:
    response = client.post(
        "/v1/images", files={"file": ("big.png", png_bytes(2000, 2000), "image/png")}
    )
    assert response.status_code == 400
    assert "MP limit" in response.json()["detail"]


def test_an_empty_upload_is_refused(client: TestClient) -> None:
    response = client.post("/v1/images", files={"file": ("empty.png", b"", "image/png")})
    assert response.status_code == 400
    assert "empty" in response.json()["detail"]


def test_uploading_the_same_image_twice_returns_the_same_digest(client: TestClient) -> None:
    data = png_bytes(100, 100)
    first = client.post("/v1/images", files={"file": ("a.png", data, "image/png")}).json()
    second = client.post("/v1/images", files={"file": ("b.png", data, "image/png")}).json()
    assert first["sha256"] == second["sha256"]


def test_an_uploaded_image_can_be_fetched_back(client: TestClient, uploaded: str) -> None:
    response = client.get(f"/v1/images/{uploaded}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert "immutable" in response.headers["cache-control"]


def test_a_traversal_key_is_a_404_not_a_file(client: TestClient) -> None:
    response = client.get("/v1/images/..%2F..%2Fetc%2Fpasswd")
    assert response.status_code == 404


def test_avif_is_accepted() -> None:
    """A phone or a browser save produces AVIF routinely, and it arrives named `.jpg`.

    Uses the repository's own `i1.jpg`, which is genuinely an AVIF file — the fixture that
    exposed the omission when the service was driven by hand.
    """
    from pathlib import Path

    photo = Path(__file__).resolve().parents[3] / "evals/photos/i1.jpg"
    inspected = uploads.inspect(photo.read_bytes(), max_megapixels=40)
    assert inspected.content_type == "image/avif"
    assert inspected.width > 0


def test_the_allowlist_has_no_animated_or_markup_formats() -> None:
    """A guard on the allowlist itself: these are excluded for reasons, not by accident."""
    assert "GIF" not in uploads.ALLOWED_FORMATS
    assert "SVG" not in uploads.ALLOWED_FORMATS
    assert "TIFF" not in uploads.ALLOWED_FORMATS
