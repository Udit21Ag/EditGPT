"""Provider interface, circuit breaker and failover chain."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
from editgpt_core.errors import ProviderError, ProviderExhaustedError

RGB = npt.NDArray[np.uint8]
Mask = npt.NDArray[np.uint8]

log = logging.getLogger(__name__)


@runtime_checkable
class Provider(Protocol):
    """Anything that can fill a masked region from a text description."""

    name: str

    def fill(self, rgb: RGB, mask: Mask, prompt: str) -> RGB: ...

    def is_configured(self) -> bool: ...


@dataclass
class ProviderHealth:
    calls: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    opened_at: float | None = None
    last_error: str = ""


@dataclass
class CircuitBreaker:
    """Stop calling a provider that is failing, and probe it again after a cooldown.

    A quota exhaustion returns the same HTTP shape as a transient fault, so the breaker
    treats them alike: back off, then retry once the window has passed.
    """

    threshold: int = 3
    cooldown_s: float = 60.0
    health: ProviderHealth = field(default_factory=ProviderHealth)

    @property
    def is_open(self) -> bool:
        if self.health.opened_at is None:
            return False
        if time.monotonic() - self.health.opened_at >= self.cooldown_s:
            self.health.opened_at = None
            self.health.consecutive_failures = 0
            return False
        return True

    def record_success(self) -> None:
        self.health.calls += 1
        self.health.consecutive_failures = 0
        self.health.opened_at = None

    @property
    def seconds_until_probe(self) -> float:
        """How long until this provider is tried again. Zero when it is not backing off."""
        if self.health.opened_at is None:
            return 0.0
        return max(0.0, self.cooldown_s - (time.monotonic() - self.health.opened_at))

    def record_failure(self, error: str) -> None:
        self.health.calls += 1
        self.health.failures += 1
        self.health.consecutive_failures += 1
        self.health.last_error = error
        if self.health.consecutive_failures >= self.threshold:
            self.health.opened_at = time.monotonic()


@dataclass(frozen=True, slots=True)
class Availability:
    """Whether the chain could be called at all — answered without calling it.

    Exists so a caller can refuse a job *before* the expensive part of it. The worker
    grounds an image with a detector and a segmenter before it ever reaches a provider;
    finding out at that point that there was nowhere to send the result wastes the whole
    run and tells the user nothing they can act on.
    """

    ready: bool
    configured: bool = True
    """Whether any provider has credentials. False is a different problem from a failing
    one — it is fixed by setting a variable, not by waiting — and only the caller knows
    which variables to name."""

    reason: str = ""
    retry_after_s: float = 0.0
    """Seconds until the soonest provider is probed again. Zero when waiting will not
    help."""


@dataclass
class ProviderChain:
    """Try providers in order, skipping any whose breaker is open."""

    providers: list[Provider]
    breakers: dict[str, CircuitBreaker] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for provider in self.providers:
            self.breakers.setdefault(provider.name, CircuitBreaker())

    def availability(self) -> Availability:
        """Whether a call would reach anything, without making one."""
        waits: list[float] = []
        configured = False
        for provider in self.providers:
            if not provider.is_configured():
                continue
            configured = True
            breaker = self.breakers[provider.name]
            if breaker.is_open:
                waits.append(breaker.seconds_until_probe)
                continue
            return Availability(ready=True)

        if waits:
            wait = min(waits)
            return Availability(
                ready=False,
                reason=(
                    "every provider is backing off after repeated failures; "
                    f"retry in about {max(1, round(wait))}s"
                ),
                retry_after_s=wait,
            )
        return Availability(ready=False, configured=configured, reason="no provider is configured")

    def fill(self, rgb: RGB, mask: Mask, prompt: str) -> tuple[RGB, str]:
        """Returns the filled image and the name of the provider that produced it."""
        errors: dict[str, str] = {}
        started = time.monotonic()
        for provider in self.providers:
            breaker = self.breakers[provider.name]
            if not provider.is_configured():
                errors[provider.name] = "not configured"
                continue
            if breaker.is_open:
                errors[provider.name] = "circuit open"
                continue
            try:
                out = provider.fill(rgb, mask, prompt)
            except ProviderError as exc:
                breaker.record_failure(str(exc))
                errors[provider.name] = str(exc)[:200]
                log.warning("provider %s failed: %s", provider.name, exc)
                continue
            breaker.record_success()
            log.info(
                "provider.filled",
                extra={
                    "provider": provider.name,
                    "seconds": round(time.monotonic() - started, 3),
                    "skipped": list(errors),
                    "mask_coverage": round(float((mask > 0).mean()), 4),
                },
            )
            return out, provider.name

        raise ProviderExhaustedError(f"every provider failed or was unavailable: {errors}")
