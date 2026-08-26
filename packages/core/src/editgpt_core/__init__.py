"""Shared contracts for EditGPT.

Only the pure-Python contract layer is re-exported here. `editgpt_core.metrics` and
`editgpt_core.rle` are imported explicitly by the callers that need them, so importing
this package never drags OpenCV or an imaging stack into a process that only speaks
in `EditSpec` objects — the orchestrator being the case that matters.
"""

from editgpt_core.errors import (
    EditGPTError,
    IllegalTransitionError,
    MaskTooSmallError,
    ProviderError,
    ProviderExhaustedError,
    ValidationError,
)
from editgpt_core.jobs import Job, JobState, JobStep, can_transition, check_transition
from editgpt_core.spec import (
    AssetRef,
    Constraints,
    EditOp,
    EditSpec,
    MaskRef,
    MaskSource,
)

__all__ = [
    "AssetRef",
    "Constraints",
    "EditGPTError",
    "EditOp",
    "EditSpec",
    "IllegalTransitionError",
    "Job",
    "JobState",
    "JobStep",
    "MaskRef",
    "MaskSource",
    "MaskTooSmallError",
    "ProviderError",
    "ProviderExhaustedError",
    "ValidationError",
    "can_transition",
    "check_transition",
]
