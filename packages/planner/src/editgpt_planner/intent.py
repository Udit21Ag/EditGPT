"""What a user asked for, before it is bound to a particular image.

`EditSpec` is the contract the pipeline runs on, and it needs things an instruction does
not contain: which image, which mask, what the constraints are. `Intent` is the part a
sentence *can* answer — the operation and its subject — so the planner has something small
and total to produce, and the caller supplies the rest.

Small matters here. This is the schema a language model is constrained to emit, and every
optional field is another way for it to be creative in a place where creativity is a bug.
"""

from __future__ import annotations

from typing import Self

from editgpt_core import AssetRef, EditOp, EditSpec, MaskRef, MaskSource
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Intent(BaseModel):
    """One operation and its subject. The model fills this or fails validation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    op: EditOp
    target: str | None = None
    """What to act on, in the user's words — "the red car", "the minaret on the right".

    Left as the user's phrase rather than resolved here: grounding it needs the image, and
    ADR-0003 says an ambiguous phrase becomes candidates for a person to choose between,
    not a guess made in a planner that has never seen the pixels."""

    content: str | None = None
    """What should be there instead, for the operations that put something there."""

    colour: str | None = Field(default=None, pattern=r"^#[0-9a-fA-F]{6}$")
    """An exact backdrop colour, when the instruction named one."""

    @model_validator(mode="after")
    def _actionable(self) -> Self:
        """The same rules `EditSpec` enforces, applied one step earlier.

        A plan that cannot become a spec is not a plan. Catching it here means the failure
        is "the model did not answer the question" — recoverable by asking the user —
        rather than a validation error surfacing from inside job creation.
        """
        if self.op in {EditOp.ADD, EditOp.REPLACE} and not self.content:
            raise ValueError(f"{self.op} needs `content` describing what to put there")
        if self.op is EditOp.BACKGROUND and not (self.content or self.colour):
            raise ValueError("background needs a colour or a description of the backdrop")
        if self.op in {EditOp.REMOVE, EditOp.REPLACE} and not self.target:
            raise ValueError(f"{self.op} needs a `target` naming what to act on")
        return self

    def to_spec(
        self,
        image: AssetRef,
        *,
        mask: MaskRef | None = None,
        confidence: float = 1.0,
    ) -> EditSpec:
        """Bind this intent to an image, and to a mask if the user drew one.

        The mask source follows from what is present rather than from what was said: a
        drawn region is the most specific thing a user can give and outranks the phrase
        they typed alongside it.
        """
        if mask is not None:
            source = MaskSource.BRUSH
        elif self.target:
            source = MaskSource.TEXT
        else:
            source = MaskSource.WHOLE
        return EditSpec(
            op=self.op,
            image_ref=image,
            mask_source=source,
            target=self.target,
            content=self.content,
            colour=self.colour,
            mask_ref=mask,
            confidence=confidence,
        )
