"""ModelSlot keeps exactly one heavy model alive. That is the 8 GB budget in code."""

from __future__ import annotations

import time

import pytest
from editgpt_models.slot import ModelSlot


class Fake:
    """Stands in for an InferenceSession; identity is what we assert on."""

    def __init__(self, name: str) -> None:
        self.name = name


def test_a_miss_loads_and_a_hit_does_not() -> None:
    slot = ModelSlot()
    calls: list[str] = []

    def loader() -> Fake:
        calls.append("migan")
        return Fake("migan")

    first = slot.acquire("migan", loader)
    second = slot.acquire("migan", loader)

    assert first is second
    assert calls == ["migan"]
    assert slot.stats.loads == 1
    assert slot.stats.hits == 1


def test_a_second_model_evicts_the_first() -> None:
    slot = ModelSlot(max_resident=1)
    slot.acquire("lama", lambda: Fake("lama"))
    slot.acquire("migan", lambda: Fake("migan"))

    assert slot.resident == ["migan"]
    assert slot.stats.evictions == 1


def test_capacity_above_one_is_honoured() -> None:
    slot = ModelSlot(max_resident=2)
    slot.acquire("a", lambda: Fake("a"))
    slot.acquire("b", lambda: Fake("b"))
    assert sorted(slot.resident) == ["a", "b"]

    slot.acquire("c", lambda: Fake("c"))
    assert len(slot.resident) == 2
    assert "a" not in slot.resident, "least recently used should go first"


def test_lru_order_follows_use_not_load() -> None:
    slot = ModelSlot(max_resident=2)
    slot.acquire("a", lambda: Fake("a"))
    slot.acquire("b", lambda: Fake("b"))
    slot.acquire("a", lambda: Fake("a"))  # refresh a
    slot.acquire("c", lambda: Fake("c"))

    assert "b" not in slot.resident
    assert sorted(slot.resident) == ["a", "c"]


def test_idle_models_expire() -> None:
    slot = ModelSlot(idle_ttl_s=0.05)
    slot.acquire("lama", lambda: Fake("lama"))
    time.sleep(0.08)
    slot.acquire("migan", lambda: Fake("migan"))

    assert slot.resident == ["migan"]
    assert slot.stats.ttl_evictions >= 1


def test_release_and_clear() -> None:
    slot = ModelSlot(max_resident=3)
    slot.acquire("a", lambda: Fake("a"))
    assert slot.release("a") is True
    assert slot.release("a") is False

    slot.acquire("b", lambda: Fake("b"))
    slot.clear()
    assert slot.resident == []


def test_pressure_evicts_before_loading() -> None:
    """With the ceiling below current RSS, an existing model is dropped to make room."""
    slot = ModelSlot(max_resident=2, rss_ceiling_mb=1)
    slot.acquire("a", lambda: Fake("a"))
    slot.acquire("b", lambda: Fake("b"))
    assert slot.stats.pressure_evictions >= 1


def test_loader_failure_leaves_the_slot_clean() -> None:
    slot = ModelSlot()

    def boom() -> Fake:
        raise RuntimeError("weights are corrupt")

    with pytest.raises(RuntimeError, match="corrupt"):
        slot.acquire("broken", boom)

    assert slot.resident == []
    assert slot.acquire("ok", lambda: Fake("ok")).name == "ok"
