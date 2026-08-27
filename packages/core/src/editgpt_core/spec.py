"""The contract every agent speaks.

Two rules are enforced here rather than by convention:

1. **Images travel by reference.** `AssetRef` is content-addressed; no field on this
   model can hold pixels. That keeps agent-to-agent payloads small enough that the
   orchestrator's context is unaffected by image size.
2. **An operation must be actionable.** A `REMOVE` with neither a target phrase nor a
   mask, or an `ADD` with nothing to add, is rejected at construction instead of
   failing three agents downstream.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class EditOp(StrEnum):
    REMOVE = "remove"
    ADD = "add"
    REPLACE = "replace"
    RESTYLE = "restyle"
    BACKGROUND = "background"
    RETOUCH = "retouch"
    UPSCALE = "upscale"


class MaskSource(StrEnum):
    TEXT = "text"
    BRUSH = "brush"
    POINT = "point"
    AUTO = "auto"
    WHOLE = "whole"


class AssetRef(BaseModel):
    """A content-addressed pointer to an image in object storage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    bucket: str = Field(min_length=1)
    sha256: str
    width: Annotated[int, Field(gt=0, le=100_000)]
    height: Annotated[int, Field(gt=0, le=100_000)]
    content_type: str = "image/png"

    @model_validator(mode="after")
    def _check_digest(self) -> Self:
        if not SHA256_RE.match(self.sha256):
            raise ValueError(f"sha256 must be 64 lowercase hex characters, got {self.sha256!r}")
        return self

    @property
    def uri(self) -> str:
        """A stable, vendor-neutral name for this asset.

        `asset://` rather than a provider's scheme: the same digest is the same asset
        whether it lives on local disk, in MinIO or behind a hosted endpoint, and a
        reference that names the storage backend stops being true when the backend
        changes.
        """
        return f"asset://{self.bucket}/{self.sha256}"

    @property
    def megapixels(self) -> float:
        return self.width * self.height / 1e6


class MaskRef(BaseModel):
    """A binary mask, run-length encoded. Never raw pixels on the wire."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    width: Annotated[int, Field(gt=0)]
    height: Annotated[int, Field(gt=0)]
    counts: list[int] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_counts(self) -> Self:
        total = sum(self.counts)
        if total != self.width * self.height:
            raise ValueError(
                f"RLE counts sum to {total}, expected {self.width * self.height} "
                f"for a {self.width}x{self.height} mask"
            )
        return self

    @property
    def area_px(self) -> int:
        """Set pixels. The encoding starts with a run of zeros, so odd runs are ones."""
        return sum(self.counts[1::2])

    @property
    def coverage(self) -> float:
        return self.area_px / (self.width * self.height)


class MaskCandidate(BaseModel):
    """One region a phrase might have meant, with the mask already computed.

    Grounding returns several of these when it cannot tell which instance was meant. The
    mask travels with the candidate rather than being fetched later, because the
    alternative — the client sending a box back and the server re-segmenting — pays for
    the expensive half of SAM twice and can return a *different* mask the second time.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    box: tuple[float, float, float, float]
    """(x0, y0, x1, y1) as fractions of the image, so a candidate survives every resize
    between the model and the browser."""

    score: float = Field(ge=0.0, le=1.0)
    """The detector's confidence for this candidate."""

    mask_ref: MaskRef
    label: str = ""
    """What the user should see for this option. Empty when the phrase is the only name
    we have, which is the common case — the picture does the disambiguating."""

    @model_validator(mode="after")
    def _box_is_ordered(self) -> Self:
        x0, y0, x1, y1 = self.box
        if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
            raise ValueError(
                f"box must be (x0, y0, x1, y1) in [0,1] with x0<x1 and y0<y1, got {self.box}"
            )
        return self


class Grounding(BaseModel):
    """What a phrase resolved to, and whether we are confident enough to act on it.

    `ambiguous` is the whole point. Answering with a confident wrong mask erases the wrong
    object; asking every time puts a modal in front of every edit. Measured on 250 held-out
    RefCOCOg samples, letting the user pick from five candidates takes the hit rate from
    0.516 to 0.832 — so it is worth asking, but only when the guess is actually shaky.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidates: list[MaskCandidate]
    """Best first. Empty means the phrase matched nothing in this image, which is a real
    answer and not a failure — the caller offers the brush."""

    ambiguous: bool
    """Whether the caller should ask before editing."""

    margin: float = Field(ge=0.0, le=1.0)
    """How far the best candidate beat the runner-up. Zero with fewer than two."""

    @property
    def best(self) -> MaskCandidate | None:
        return self.candidates[0] if self.candidates else None


class Constraints(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    preserve_identity: bool = True
    max_seconds: Annotated[float, Field(gt=0, le=600)] = 60.0
    max_cost_cents: Annotated[float, Field(ge=0)] = 5.0
    max_retries: Annotated[int, Field(ge=0, le=5)] = 2
    allow_remote: bool = True


class EditSpec(BaseModel):
    """A single, fully-specified edit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    op: EditOp
    image_ref: AssetRef
    mask_source: MaskSource
    target: str | None = None
    content: str | None = None
    colour: str | None = Field(default=None, pattern=r"^#[0-9a-fA-F]{6}$")
    """The exact colour for `BACKGROUND`, as `#rrggbb`.

    Carried as a value rather than parsed out of `content` downstream. TD-020 was the
    absence of this: the operation always painted the same green, so "change the
    background to blue" succeeded and returned green — a capability advertised, silently
    not delivered.

    An exact colour is also the honest shape for what this operation *does*. It composites
    a flat backdrop rather than generating one (TD-005), so a hex value loses nothing a
    description would have carried, and it removes the guessing entirely: a client that
    knows the user picked a colour should say which one, not describe it and hope.

    Free text stays in `content` for the day an intent agent can turn "a sunset" into
    something this operation could paint."""

    mask_ref: MaskRef | None = None
    constraints: Constraints = Constraints()
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0

    @model_validator(mode="after")
    def _actionable(self) -> Self:
        if self.op in {EditOp.ADD, EditOp.REPLACE} and not self.content:
            raise ValueError(f"{self.op} needs `content` describing what to put there")
        if self.op is EditOp.BACKGROUND and not (self.content or self.colour):
            # Either is a complete answer to "what goes there": a hex value says it
            # exactly, and free text says it for an intent agent to resolve later.
            raise ValueError("background needs a `colour` or `content` describing the backdrop")
        if self.op is EditOp.REMOVE and not (self.target or self.mask_ref):
            raise ValueError("remove needs either a `target` phrase or an explicit mask")
        if self.mask_source is MaskSource.TEXT and not self.target:
            raise ValueError("mask_source=text needs a `target` phrase to segment")
        if self.mask_source in {MaskSource.BRUSH, MaskSource.POINT} and self.mask_ref is None:
            raise ValueError(f"mask_source={self.mask_source} needs a `mask_ref`")
        if self.mask_ref is not None:
            self._check_mask_covers_the_image()
        return self

    def _check_mask_covers_the_image(self) -> None:
        """The mask must describe *this* image, at any resolution.

        Shape, not size. A mask is a region of a picture, and pinning it to one
        rasterisation was over-specification with a cost: a candidate from `POST /v1/masks`
        arrives at the resolution grounding ran at, so an exact-size rule forced the client
        to upscale a mask that the worker's `_fit_mask` immediately scales back down —
        lossy work in the browser to satisfy a check, and nothing gained.

        Matching aspect keeps what the rule was actually for, which is catching a mask
        belonging to a different image. The tolerance is derived rather than picked:
        cross-multiplying makes this integer arithmetic, and rescaling by `s` moves each
        side by less than a pixel, so the skew of a legitimate rescale cannot exceed
        `width + height`. A mask five pixels wrong on an 800x600 image scores 3000 against
        a budget of 1400, so this is far tighter than it looks.
        """
        assert self.mask_ref is not None  # narrowed by the caller, restated for mypy
        mask_w, mask_h = self.mask_ref.width, self.mask_ref.height
        image_w, image_h = self.image_ref.width, self.image_ref.height
        if abs(mask_w * image_h - mask_h * image_w) > image_w + image_h:
            raise ValueError(
                f"mask is {mask_w}x{mask_h}, which is not the shape of a "
                f"{image_w}x{image_h} image; a mask may be any size but must match the "
                "image's aspect ratio"
            )

    def rgb_colour(self, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
        """`colour` as RGB, or `fallback` when the request did not name one.

        Here rather than in the worker so the parsing sits next to the pattern that
        validates it. Two places that both turn `#rrggbb` into three integers is two
        places that can disagree about which end is red.
        """
        if self.colour is None:
            return fallback
        raw = self.colour.lstrip("#")
        return (int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16))

    @property
    def is_generative(self) -> bool:
        """Whether this needs a remote model.

        Phase 0 measured that removal is strictly better served locally: the free
        generative lane fills a masked hole with an object rather than continuing the
        background, so removal never routes remote. See ADR-0001.
        """
        return self.op in {EditOp.ADD, EditOp.REPLACE, EditOp.RESTYLE, EditOp.BACKGROUND}
