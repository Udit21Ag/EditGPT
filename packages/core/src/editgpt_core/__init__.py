"""Shared contracts for EditGPT.

Only the pure-Python contract layer is re-exported here. `editgpt_core.metrics` and
`editgpt_core.rle` are imported explicitly by the callers that need them, so importing
this package never drags OpenCV or an imaging stack into a process that only speaks
in `EditSpec` objects — the orchestrator being the case that matters.
"""

from editgpt_core.errors import (
    EditGPTError,
    EditRejectedError,
    IllegalTransitionError,
    MaskTooSmallError,
    ProviderError,
    ProviderExhaustedError,
    ProviderUnavailableError,
    ValidationError,
)
from editgpt_core.jobs import Job, JobState, JobStep, can_transition, check_transition
from editgpt_core.review import Action, Verdict, decide
from editgpt_core.spec import (
    AssetRef,
    Constraints,
    EditOp,
    EditSpec,
    Grounding,
    MaskCandidate,
    MaskRef,
    MaskSource,
)

__all__ = [
    "Action",
    "AssetRef",
    "Constraints",
    "EditGPTError",
    "EditOp",
    "EditRejectedError",
    "EditSpec",
    "Grounding",
    "IllegalTransitionError",
    "Job",
    "JobState",
    "JobStep",
    "MaskCandidate",
    "MaskRef",
    "MaskSource",
    "MaskTooSmallError",
    "ProviderError",
    "ProviderExhaustedError",
    "ProviderUnavailableError",
    "ValidationError",
    "Verdict",
    "can_transition",
    "check_transition",
    "decide",
]
