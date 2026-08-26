"""Remote generative providers behind one interface.

A provider is a fill: ``(rgb, mask) -> rgb`` at a fixed working size. Everything around
it — the crop window, the chroma match, the feather — is shared with the local erasers,
so comparing lanes compares models rather than plumbing.

Removal never routes here. Phase 0 measured that the free generative lane fills a masked
hole with an object matching the prompt rather than continuing the background: asked to
erase a car it produced a stone slab, then a different car, then a boulder. See ADR-0001.
"""

from editgpt_providers.base import (
    CircuitBreaker,
    Provider,
    ProviderChain,
    ProviderHealth,
)
from editgpt_providers.cloudflare import CloudflareWorkersAI

__all__ = [
    "CircuitBreaker",
    "CloudflareWorkersAI",
    "Provider",
    "ProviderChain",
    "ProviderHealth",
]
