"""Failover, circuit breaking and the refusal to send an empty prompt."""

from __future__ import annotations

import numpy as np
import pytest
from editgpt_core.errors import ProviderError, ProviderExhaustedError
from editgpt_providers.base import CircuitBreaker, ProviderChain
from editgpt_providers.cloudflare import CloudflareWorkersAI


class Stub:
    def __init__(self, name: str, *, fails: bool = False, configured: bool = True) -> None:
        self.name = name
        self.fails = fails
        self.configured = configured
        self.calls = 0

    def is_configured(self) -> bool:
        return self.configured

    def fill(self, rgb, mask, prompt):  # type: ignore[no-untyped-def]
        self.calls += 1
        if self.fails:
            raise ProviderError(f"{self.name} is down")
        out = rgb.copy()
        out[mask > 0] = 255
        return out


def image() -> np.ndarray:
    return np.zeros((16, 16, 3), dtype=np.uint8)


def mask() -> np.ndarray:
    m = np.zeros((16, 16), dtype=np.uint8)
    m[4:12, 4:12] = 255
    return m


def test_first_healthy_provider_wins() -> None:
    primary, secondary = Stub("a"), Stub("b")
    _, used = ProviderChain([primary, secondary]).fill(image(), mask(), "a hat")
    assert used == "a"
    assert secondary.calls == 0


def test_failure_falls_through_to_the_next() -> None:
    primary, secondary = Stub("a", fails=True), Stub("b")
    _, used = ProviderChain([primary, secondary]).fill(image(), mask(), "a hat")
    assert used == "b"


def test_unconfigured_providers_are_skipped_without_being_called() -> None:
    primary, secondary = Stub("a", configured=False), Stub("b")
    _, used = ProviderChain([primary, secondary]).fill(image(), mask(), "a hat")
    assert used == "b"
    assert primary.calls == 0


def test_all_failing_raises_with_every_reason() -> None:
    chain = ProviderChain([Stub("a", fails=True), Stub("b", fails=True)])
    with pytest.raises(ProviderExhaustedError) as excinfo:
        chain.fill(image(), mask(), "a hat")
    assert "a is down" in str(excinfo.value)
    assert "b is down" in str(excinfo.value)


def test_breaker_opens_after_repeated_failures_and_stops_calling() -> None:
    flaky = Stub("a", fails=True)
    chain = ProviderChain([flaky, Stub("b")])
    for _ in range(4):
        chain.fill(image(), mask(), "a hat")

    assert chain.breakers["a"].is_open
    assert flaky.calls == 3, "the breaker should stop calls once open"


def test_breaker_recovers_after_the_cooldown() -> None:
    breaker = CircuitBreaker(threshold=1, cooldown_s=0.01)
    breaker.record_failure("boom")
    assert breaker.is_open

    import time

    time.sleep(0.02)
    assert not breaker.is_open


def test_success_resets_the_consecutive_failure_count() -> None:
    breaker = CircuitBreaker(threshold=2)
    breaker.record_failure("boom")
    breaker.record_success()
    breaker.record_failure("boom")
    assert not breaker.is_open


def test_cloudflare_refuses_an_empty_prompt() -> None:
    """The API rejects it, which is precisely why this lane cannot do removal."""
    provider = CloudflareWorkersAI(account_id="a" * 32, api_token="token")
    with pytest.raises(ProviderError, match="non-empty prompt"):
        provider.fill(image(), mask(), "   ")


def test_cloudflare_reports_missing_configuration_with_the_permissions_needed() -> None:
    provider = CloudflareWorkersAI(account_id=None, api_token=None)
    assert not provider.is_configured()
    with pytest.raises(ProviderError, match="Workers AI Edit"):
        provider.fill(image(), mask(), "a hat")


# ---------------------------------------------------------------- blank returns


class TestBlankFills:
    """A 200 carrying a black frame must not be composited as a result.

    Stable Diffusion returns black when its safety checker fires. This happened once on
    the golden set — a black rectangle stamped on a face, reported as a completed edit —
    and the identical call minutes later produced a correct moustache. Transient, silent
    and destructive.

    The thresholds below are the ones measured on that incident and on correct fills, so
    these fixtures are calibrated rather than invented.
    """

    def image(self, value: int = 128) -> np.ndarray:
        generator = np.random.default_rng(seed=5)
        base = generator.integers(value - 40, value + 40, (64, 64, 3))
        return np.asarray(np.clip(base, 0, 255), dtype=np.uint8)

    def mask(self) -> np.ndarray:
        mask = np.zeros((64, 64), np.uint8)
        mask[16:48, 16:48] = 255
        return mask

    def test_an_all_black_frame_is_refused(self) -> None:
        from editgpt_providers.cloudflare import _reject_blank

        with pytest.raises(ProviderError, match="blank fill"):
            _reject_blank(np.zeros((64, 64, 3), np.uint8), self.mask())

    def test_the_message_says_what_to_do_next(self) -> None:
        from editgpt_providers.cloudflare import _reject_blank

        with pytest.raises(ProviderError, match="retry or rephrase"):
            _reject_blank(np.zeros((64, 64, 3), np.uint8), self.mask())

    def test_a_real_fill_is_accepted(self) -> None:
        """Measured on the live provider: mean 122.5, spread 61.7 inside the mask."""
        from editgpt_providers.cloudflare import _reject_blank

        _reject_blank(self.image(122), self.mask())

    def test_a_legitimately_dark_fill_is_accepted(self) -> None:
        """Darkness alone is not a fault. The golden set's `i4r` measured mean 37.2 and
        was a correct result; rejecting on brightness would have thrown it away."""
        from editgpt_providers.cloudflare import _reject_blank

        _reject_blank(self.image(37), self.mask())

    def test_only_the_masked_region_decides(self) -> None:
        """The model returns the whole frame. A dark *background* with a real fill in the
        mask is a normal night-time photograph, not a failure."""
        from editgpt_providers.cloudflare import _reject_blank

        frame = np.zeros((64, 64, 3), np.uint8)
        frame[16:48, 16:48] = self.image(120)[16:48, 16:48]
        _reject_blank(frame, self.mask())

    def test_a_blank_frame_with_no_mask_is_still_refused(self) -> None:
        """Falling back to the whole frame keeps the check meaningful when the caller
        passes an empty mask rather than skipping it silently."""
        from editgpt_providers.cloudflare import _reject_blank

        with pytest.raises(ProviderError):
            _reject_blank(np.zeros((64, 64, 3), np.uint8), np.zeros((64, 64), np.uint8))
