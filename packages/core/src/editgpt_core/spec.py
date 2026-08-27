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
    mask_ref: MaskRef | None = None
    constraints: Constraints = Constraints()
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0

    @model_validator(mode="after")
    def _actionable(self) -> Self:
        if self.op in {EditOp.ADD, EditOp.REPLACE, EditOp.BACKGROUND} and not self.content:
            raise ValueError(f"{self.op} needs `content` describing what to put there")
        if self.op is EditOp.REMOVE and not (self.target or self.mask_ref):
            raise ValueError("remove needs either a `target` phrase or an explicit mask")
        if self.mask_source is MaskSource.TEXT and not self.target:
            raise ValueError("mask_source=text needs a `target` phrase to segment")
        if self.mask_source in {MaskSource.BRUSH, MaskSource.POINT} and self.mask_ref is None:
            raise ValueError(f"mask_source={self.mask_source} needs a `mask_ref`")
        if self.mask_ref is not None and (self.mask_ref.width, self.mask_ref.height) != (
            self.image_ref.width,
            self.image_ref.height,
        ):
            raise ValueError(
                f"mask is {self.mask_ref.width}x{self.mask_ref.height} but image is "
                f"{self.image_ref.width}x{self.image_ref.height}"
            )
        return self

    @property
    def is_generative(self) -> bool:
        """Whether this needs a remote model.

        Phase 0 measured that removal is strictly better served locally: the free
        generative lane fills a masked hole with an object rather than continuing the
        background, so removal never routes remote. See ADR-0001.
        """
        return self.op in {EditOp.ADD, EditOp.REPLACE, EditOp.RESTYLE, EditOp.BACKGROUND}
