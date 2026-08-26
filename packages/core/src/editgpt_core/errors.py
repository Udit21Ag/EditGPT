"""Error taxonomy. Every failure an agent can report is one of these."""

from __future__ import annotations


class EditGPTError(Exception):
    """Base for every error this system raises deliberately."""


class ValidationError(EditGPTError):
    """An EditSpec or mask failed a semantic check that Pydantic cannot express."""


class MaskTooSmallError(EditGPTError):
    """A mask covers so few pixels that editing it cannot produce a visible result.

    Raised rather than silently returning the input, because in Phase 0 a six-pixel
    mask on the air conditioner produced an output identical to the input while every
    numeric score reported success.
    """


class ProviderError(EditGPTError):
    """A remote provider returned an error or an unusable response."""


class ProviderExhaustedError(ProviderError):
    """Every configured provider failed or is out of quota."""


class IllegalTransitionError(EditGPTError):
    """A job state change the lifecycle does not permit.

    Defined here rather than in `jobs` so every deliberate failure in the system shares a
    base class — a caller can catch `EditGPTError` and know it has caught ours.
    """
