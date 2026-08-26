"""The 8 GB budget, enforced.

Two tests at two costs. The first is synthetic and runs on every commit: it proves the
slot actually releases memory rather than merely forgetting a reference. The second
drives the real models and is marked `memory` and `slow`, so it runs on demand and
nightly rather than on every push — it needs ~285 MB of weights.

Both assert against `DEFAULT_RSS_CEILING_MB` **over repeated iterations**. Peak RSS
varies about +/-15% run to run because the ONNX arena allocator is not deterministic,
and a single sample once set this ceiling 20% too low.
"""

from __future__ import annotations

import gc

import numpy as np
import psutil
import pytest
from editgpt_models.slot import DEFAULT_RSS_CEILING_MB, ModelSlot

BLOCK_MB = 120


def rss_mb() -> float:
    return float(psutil.Process().memory_info().rss) / 1e6


class HeavyBlock:
    """Stands in for a loaded model: it owns real, resident memory."""

    def __init__(self, megabytes: int = BLOCK_MB) -> None:
        # Touched, not just allocated — an untouched allocation may never be resident.
        self.buffer = np.ones(megabytes * 1_000_000, dtype=np.uint8)

    def __len__(self) -> int:
        return int(self.buffer.nbytes)


def test_the_slot_actually_frees_memory_not_just_references() -> None:
    """Swap models repeatedly; RSS must not accumulate.

    A slot that drops references but leaves memory resident would pass every unit test
    in this suite and still exhaust an 8 GB machine in production.
    """
    slot = ModelSlot(max_resident=1, idle_ttl_s=3600)
    gc.collect()
    baseline = rss_mb()

    for cycle in range(8):
        slot.acquire(f"model-{cycle % 2}", HeavyBlock)

    slot.clear()
    gc.collect()
    settled = rss_mb()

    # One block may still be held by the allocator; several would mean a leak.
    assert settled - baseline < BLOCK_MB * 1.5, (
        f"RSS grew {settled - baseline:.0f} MB over 8 swaps of a {BLOCK_MB} MB model, "
        "which means eviction is not releasing memory"
    )


def test_only_one_heavy_model_is_resident_across_a_mixed_workload() -> None:
    slot = ModelSlot(max_resident=1, idle_ttl_s=3600)
    for key in ["sam", "migan", "lama", "sam", "clipseg", "migan"]:
        slot.acquire(key, HeavyBlock)
        assert len(slot.resident) == 1, f"{slot.resident} resident while loading {key}"

    assert slot.stats.loads == 6, "every switch should be a real load"
    assert slot.stats.evictions == 5


def test_peak_stays_under_the_ceiling_for_a_synthetic_workload() -> None:
    slot = ModelSlot(max_resident=1, idle_ttl_s=3600)
    for cycle in range(12):
        block = slot.acquire(f"model-{cycle % 3}", HeavyBlock)
        assert len(block) > 0
    assert slot.stats.peak_rss_mb < DEFAULT_RSS_CEILING_MB


@pytest.mark.memory
@pytest.mark.slow
def test_real_pipeline_peak_rss_over_repeated_runs() -> None:
    """Drive the real erasers and assert the ceiling holds over several iterations.

    Skipped unless the weights are already present, so a fresh checkout does not
    silently download 285 MB during a test run.
    """
    from editgpt_models.erase import erase_migan, make_session
    from editgpt_models.registry import model_path

    try:
        migan = model_path("migan", download=False)
    except FileNotFoundError:
        pytest.skip("model weights not present; run `make models` first")

    slot = ModelSlot(max_resident=1)
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (1024, 1024, 3), dtype=np.uint8)
    mask = np.zeros((1024, 1024), dtype=np.uint8)
    mask[300:600, 300:600] = 255

    peak = 0.0
    for _ in range(3):
        session = slot.acquire("migan", lambda: make_session(migan))
        erase_migan(session, image, mask)
        peak = max(peak, slot.rss_mb())

    assert peak < DEFAULT_RSS_CEILING_MB, (
        f"peak RSS {peak:.0f} MB exceeded the {DEFAULT_RSS_CEILING_MB} MB ceiling"
    )
