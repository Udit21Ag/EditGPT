"""Local model execution.

Nothing here is imported eagerly: loading this package must not load a model. The
entry points are `ModelSlot` for lifetime management and `pipeline.edit` for the
routing decided in ADR-0001.
"""

from editgpt_models.registry import ModelSpec, model_path, registry
from editgpt_models.slot import ModelSlot, SlotFullError

__all__ = ["ModelSlot", "ModelSpec", "SlotFullError", "model_path", "registry"]
