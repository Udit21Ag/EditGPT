"""Removing what a photograph says about the person who took it.

An upload was stored and served back byte for byte, so everything the camera wrote came
with it. Measured on a phone photograph in this repository's own fixtures: make `iQOO`,
model `iQOO Neo7`, and the second it was taken. A picture taken outdoors carries the
coordinates too, and the gateway would hand them to anyone the digest reached.

**Nothing is re-encoded that does not have to be.** This is an image editor; degrading a
photograph on the way in would compound through every edit made afterwards, and it would
do so silently. So an image with no metadata is returned untouched, JPEG is stripped at
the segment level with the compressed data never decoded, and only the container formats
that carry metadata *and* cannot be edited in place are re-encoded.

**Colour profiles are not metadata for this purpose.** ICC describes how to interpret the
pixels; dropping it changes what the picture looks like. JFIF density likewise. Both are
kept, and only the segments that describe the *photographer* are removed.
"""

from __future__ import annotations

import io
import logging

log = logging.getLogger(__name__)

# JPEG markers. APP1 carries EXIF and XMP, APP13 carries IPTC, COM is a free-text comment.
# APP0 (JFIF) and APP2 (ICC) are kept: they describe the image, not its author.
JPEG_KEEP = {0xE0, 0xE2}
JPEG_DROP = {*range(0xE1, 0xF0), 0xFE} - JPEG_KEEP

# PNG ancillary chunks that carry text, timestamps or EXIF. Everything else — including
# `iCCP`, `gAMA` and `sRGB` — describes the pixels and stays.
PNG_DROP = {b"tEXt", b"zTXt", b"iTXt", b"eXIf", b"tIME"}

# RIFF chunks in a WebP container.
WEBP_DROP = {b"EXIF", b"XMP "}


def _strip_jpeg(data: bytes) -> bytes:
    """Drop metadata segments without touching the entropy-coded image data.

    A JPEG is a sequence of marker segments until `SOS`, after which the compressed scan
    runs to the end. Copying every byte except the segments we drop leaves the picture
    bit-identical — no decode, no quantisation, no generation loss.
    """
    if not data.startswith(b"\xff\xd8"):
        return data

    out = bytearray(b"\xff\xd8")
    at = 2
    while at < len(data) - 1:
        if data[at] != 0xFF:
            break  # not where a marker should be; hand back what we have rather than guess
        marker = data[at + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            out += data[at : at + 2]  # standalone marker, no payload
            at += 2
            continue
        if marker == 0xDA:  # start of scan: the rest is compressed image data
            out += data[at:]
            return bytes(out)
        if at + 4 > len(data):
            break
        length = int.from_bytes(data[at + 2 : at + 4], "big")
        end = at + 2 + length
        if length < 2 or end > len(data):
            break  # malformed; `inspect` already accepted it, so keep what we have
        if marker not in JPEG_DROP:
            out += data[at:end]
        at = end
    return bytes(out)


def _strip_png(data: bytes) -> bytes:
    """Drop text, time and EXIF chunks. PNG is lossless, so this is too."""
    signature = b"\x89PNG\r\n\x1a\n"
    if not data.startswith(signature):
        return data

    out = bytearray(signature)
    at = len(signature)
    while at + 8 <= len(data):
        length = int.from_bytes(data[at : at + 4], "big")
        kind = data[at + 4 : at + 8]
        end = at + 12 + length  # length + type + payload + crc
        if end > len(data):
            break
        if kind not in PNG_DROP:
            out += data[at:end]
        at = end
        if kind == b"IEND":
            break
    return bytes(out)


def _strip_webp(data: bytes) -> bytes:
    """Drop the EXIF and XMP chunks from a RIFF container, and fix the size it declares."""
    if not (data.startswith(b"RIFF") and data[8:12] == b"WEBP"):
        return data

    body = bytearray()
    at = 12
    while at + 8 <= len(data):
        fourcc = data[at : at + 4]
        size = int.from_bytes(data[at + 4 : at + 8], "little")
        end = at + 8 + size + (size % 2)  # chunks are padded to an even length
        if end > len(data):
            break
        if fourcc not in WEBP_DROP:
            body += data[at:end]
        at = end

    out = bytearray(b"RIFF")
    out += (len(body) + 4).to_bytes(4, "little")
    out += b"WEBP"
    out += body
    return bytes(out)


def _reencode(data: bytes, image_format: str) -> bytes:
    """The fallback: decode and write again, which drops everything Pillow was not asked
    to keep.

    Only reached for a container this module cannot edit in place *and* that actually
    carries metadata — in practice, AVIF. It is the one path that can cost quality, so it
    says so in the log rather than doing it quietly.
    """
    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        image.load()
        buffer = io.BytesIO()
        keep = {"icc_profile": image.info["icc_profile"]} if "icc_profile" in image.info else {}
        image.save(buffer, format=image_format, quality=95, **keep)
    written = buffer.getvalue()
    log.info(
        "scrub.reencoded",
        extra={"format": image_format, "before": len(data), "after": len(written)},
    )
    return written


def has_metadata(data: bytes, image_format: str) -> bool:
    """Whether there is anything here worth removing.

    Asked first so an image carrying nothing is returned exactly as it arrived. Most
    uploads are in that case, and it is the difference between a boundary that protects
    people and one that quietly rewrites their files.
    """
    if image_format == "JPEG":
        return _strip_jpeg(data) != data
    if image_format == "PNG":
        return _strip_png(data) != data
    if image_format == "WEBP":
        return _strip_webp(data) != data

    from PIL import Image

    try:
        with Image.open(io.BytesIO(data)) as image:
            return len(image.getexif()) > 0 or "xmp" in image.info
    except Exception:
        return False


def scrub(data: bytes, image_format: str) -> bytes:
    """`data` without the metadata that identifies its author, camera or location."""
    if not has_metadata(data, image_format):
        return data
    if image_format == "JPEG":
        return _strip_jpeg(data)
    if image_format == "PNG":
        return _strip_png(data)
    if image_format == "WEBP":
        return _strip_webp(data)
    return _reencode(data, image_format)
