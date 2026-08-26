"""One heavy model resident at a time.

Phase 0 measured a full pipeline at 1.47-2.12 GB across repeated runs on an 8 GB
machine. That leaves no room for two erasers to be alive at once, so residency is a
managed resource rather than a side effect of imports.

The implementation is synchronous and thread-safe because it runs inside Celery
workers, which are processes with a small thread pool — an async lock would be
ceremony around a problem that does not exist here.
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import psutil

log = logging.getLogger(__name__)

DEFAULT_MAX_RESIDENT = 1
DEFAULT_IDLE_TTL_S = 90.0
DEFAULT_RSS_CEILING_MB = 2200
"""Set from the worst case observed over repeated Phase 0 runs, not an average.

Peak RSS varies about +/-15% run to run because the ONNX arena allocator is not
deterministic. A single sample put this at 1900 MB, which was 20% too low.
"""


class SlotFullError(RuntimeError):
    """A model could not be admitted without breaching the RSS ceiling."""


@dataclass
class _Entry:
    value: Any
    loaded_at: float
    last_used: float
    load_seconds: float


@dataclass
class SlotStats:
    loads: int = 0
    hits: int = 0
    evictions: int = 0
    ttl_evictions: int = 0
    pressure_evictions: int = 0
    peak_rss_mb: float = 0.0


@dataclass
class ModelSlot:
    """An LRU of at most `max_resident` heavy models, with idle and pressure eviction.

    Usage::

        slot = ModelSlot()
        session = slot.acquire("migan", lambda: make_session(path))

    The loader is only called on a miss, so callers may pass an expensive closure
    unconditionally.
    """

    max_resident: int = DEFAULT_MAX_RESIDENT
    idle_ttl_s: float = DEFAULT_IDLE_TTL_S
    rss_ceiling_mb: int = DEFAULT_RSS_CEILING_MB
    stats: SlotStats = field(default_factory=SlotStats)

    _entries: OrderedDict[str, _Entry] = field(default_factory=OrderedDict, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)

    # ------------------------------------------------------------------ public

    def acquire(self, key: str, loader: Callable[[], Any]) -> Any:
        with self._lock:
            self._expire_idle()
            entry = self._entries.get(key)
            if entry is not None:
                entry.last_used = time.monotonic()
                self._entries.move_to_end(key)
                self.stats.hits += 1
                return entry.value

            while len(self._entries) >= self.max_resident:
                self._evict_oldest("capacity")

            if self.rss_mb() > self.rss_ceiling_mb and self._entries:
                self._evict_oldest("pressure")
                self.stats.pressure_evictions += 1

            started = time.monotonic()
            value = loader()
            elapsed = time.monotonic() - started

            now = time.monotonic()
            self._entries[key] = _Entry(value, now, now, elapsed)
            self.stats.loads += 1
            self._note_rss()
            log.info(
                "model.loaded",
                extra={
                    "model": key,
                    "load_seconds": round(elapsed, 3),
                    "rss_mb": round(self.rss_mb()),
                    "resident": list(self._entries),
                },
            )

            if self.rss_mb() > self.rss_ceiling_mb:
                log.warning(
                    "%s loaded but RSS is %.0f MB, over the %d MB ceiling",
                    key,
                    self.rss_mb(),
                    self.rss_ceiling_mb,
                )
            return value

    def release(self, key: str) -> bool:
        """Drop a model now. Returns whether it was resident."""
        with self._lock:
            if key not in self._entries:
                return False
            self._drop(key)
            return True

    def clear(self) -> None:
        with self._lock:
            for key in list(self._entries):
                self._drop(key)

    @property
    def resident(self) -> list[str]:
        with self._lock:
            return list(self._entries)

    def rss_mb(self) -> float:
        return float(psutil.Process().memory_info().rss) / 1e6

    # ----------------------------------------------------------------- internal

    def _note_rss(self) -> None:
        self.stats.peak_rss_mb = max(self.stats.peak_rss_mb, self.rss_mb())

    def _expire_idle(self) -> None:
        now = time.monotonic()
        for key, entry in list(self._entries.items()):
            if now - entry.last_used > self.idle_ttl_s:
                self._drop(key)
                self.stats.ttl_evictions += 1

    def _evict_oldest(self, reason: str) -> None:
        key = next(iter(self._entries))
        log.debug("evicting %s (%s)", key, reason)
        self._drop(key)
        self.stats.evictions += 1

    def _drop(self, key: str) -> None:
        self._entries.pop(key, None)
        gc.collect()
        self._note_rss()
