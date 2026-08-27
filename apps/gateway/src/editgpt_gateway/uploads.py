"""The input boundary.

Architectural invariant 7: validation happens once, here, and everything downstream may
assume it is dealing with a real image of a bounded size. That makes this the file where
being paranoid is correct.

What is checked, and why each one is not optional:

* **Declared size, before reading.** A `Content-Length` over the limit is refused without
  buffering the body, so a 2 GB upload costs a header parse rather than 2 GB of memory.
* **Actual size, while reading.** The declared length is a claim by the client. The real
  one is counted as chunks arrive and the read stops the moment it exceeds the limit.
* **Format from the bytes, not the filename or the declared type.** Both are attacker
  controlled. Pillow identifies the format from the magic bytes.
* **Pixel count from the header, before decoding.** A 200 KB PNG can declare 60000x60000
  and cost 10 GB to decode — the decompression bomb. `Image.open` parses the header
  without decoding, so the dimensions are known while the file is still 200 KB.
* **An allowlist of formats**, not a denylist. SVG and anything with a scripting or
  external-reference capability is simply not on it.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Any

from editgpt_core import AssetRef

log = logging.getLogger(__name__)

ALLOWED_FORMATS: dict[str, str] = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
    "AVIF": "image/avif",
}
"""Formats accepted, keyed by what Pillow calls them.

Allowlist, not denylist. Notably absent: SVG (markup, can reference and script), GIF
(animation the pipeline has no meaning for), and TIFF (a container format with a long
history of parser bugs).

**AVIF is on the list because real files are AVIF.** A phone or a browser save produces
one routinely, and it arrives named `.jpg` — one of this repository's own golden-set
photos is exactly that, which is how the omission was found. Listing it costs nothing:
the check runs *after* `Image.open` succeeds, so on a Pillow build without AVIF the file
is rejected earlier with a clearer message. HEIC is deliberately still absent — Pillow
needs a plugin for it, so accepting it here would promise something the decoder cannot
deliver.
"""

CHUNK = 64 * 1024


class UploadRejectedError(ValueError):
    """The upload is not something this system will accept. The message is user-facing."""


@dataclass(frozen=True, slots=True)
class Upload:
    data: bytes
    width: int
    height: int
    content_type: str

    @property
    def megapixels(self) -> float:
        return self.width * self.height / 1e6


async def read_bounded(stream: Any, limit: int) -> bytes:
    """Read at most `limit` bytes, refusing rather than truncating past it.

    Truncation would be worse than refusal: a half-read JPEG still decodes, so the system
    would accept a corrupted image and produce a confusing result instead of a clear
    error.
    """
    read = getattr(stream, "read", None)
    if read is None:
        raise UploadRejectedError("no file was uploaded")

    buffer = bytearray()
    while True:
        chunk = await read(CHUNK)
        if not chunk:
            break
        buffer.extend(chunk)
        if len(buffer) > limit:
            raise UploadRejectedError(
                f"file is larger than the {limit // (1024 * 1024)} MB limit; "
                "resize it or crop it before uploading"
            )
    if not buffer:
        raise UploadRejectedError("the uploaded file is empty")
    return bytes(buffer)


def inspect(data: bytes, *, max_megapixels: float) -> Upload:
    """Identify an image from its bytes and check it is within bounds.

    Raises `UploadRejectedError` with a message a user can act on. Never raises anything
    that would surface as a 500 — a malformed upload is expected, not a fault.
    """
    from PIL import Image, UnidentifiedImageError

    try:
        # `open` parses the header and stops. `.size` is available without decoding a
        # single pixel, which is the whole defence against a decompression bomb.
        with Image.open(io.BytesIO(data)) as image:
            image_format = image.format or ""
            width, height = image.size
    except UnidentifiedImageError as error:
        raise UploadRejectedError("that file is not an image this system can read") from error
    except Image.DecompressionBombError as error:
        # Pillow refuses outright above ~2x its own ceiling, which is well above ours, so
        # this fires only on the truly absurd. Translated rather than passed through: the
        # size check below produces a message a user can act on, and a bomb should get
        # the same one instead of a library exception name.
        raise UploadRejectedError(
            f"the image declares more pixels than can be decoded safely, over the "
            f"{max_megapixels:.0f} MP limit"
        ) from error
    except Exception as error:
        raise UploadRejectedError(f"the image could not be read: {type(error).__name__}") from error

    if image_format not in ALLOWED_FORMATS:
        accepted = ", ".join(sorted(ALLOWED_FORMATS))
        raise UploadRejectedError(
            f"{image_format or 'unknown'} is not supported; send one of {accepted}"
        )

    if width <= 0 or height <= 0:
        raise UploadRejectedError(
            f"the image declares a {width}x{height} size, which cannot be edited"
        )

    megapixels = width * height / 1e6
    if megapixels > max_megapixels:
        raise UploadRejectedError(
            f"the image is {megapixels:.1f} MP, over the {max_megapixels:.0f} MP limit; "
            "editing it would exceed the worker's memory budget"
        )

    return Upload(
        data=data,
        width=width,
        height=height,
        content_type=ALLOWED_FORMATS[image_format],
    )


def to_ref(upload: Upload, *, bucket: str, digest: str) -> AssetRef:
    return AssetRef(
        bucket=bucket,
        sha256=digest,
        width=upload.width,
        height=upload.height,
        content_type=upload.content_type,
    )
