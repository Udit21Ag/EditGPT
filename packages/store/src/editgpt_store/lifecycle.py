"""Deleting stored bytes that nothing needs any more.

Two different problems, one pass.

**Orphans.** A blob whose row was never written. Every upload that was never turned into
a job leaves one, and no query can find them — the database only knows what it wrote
down, and that is exactly the set an orphan is not in. Only a scan of the store can see
them, which is what `AssetStore.scan` exists for.

**Expiry.** Signed links already expire; the objects behind them did not. A deployment
that wants pixels to stop existing after a month sets `EDITGPT_ASSET_RETENTION_DAYS` and
gets that, while the rows stay: a job's history remains readable, and fetching its result
answers 404 rather than pretending the image was never made.

Deleting is not undoable, so three things hold. Nothing is deleted without a reference
check, and a sweep with no database **refuses to run** rather than treating every object
as an orphan. Objects younger than a grace period are always kept, because an upload
whose row is still being committed is indistinguishable from an orphan for exactly as
long as that takes. And `dry_run` is the default everywhere a human can invoke it — the
report is the same either way, so "what would this delete" costs one command.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker

from editgpt_store.assets import AssetStore, StoredObject
from editgpt_store.models import Artifact, Image

log = logging.getLogger(__name__)

ORPHAN = "orphan"
EXPIRED = "expired"


@dataclass(frozen=True, slots=True)
class Policy:
    """When bytes may go."""

    grace_hours: float = 24.0
    """How long an unreferenced object is kept anyway.

    An upload that has been stored but not yet recorded looks exactly like an orphan. A
    day is far longer than that window and short enough that a failed upload does not
    live for a week."""

    retention_days: int = 0
    """Age after which even a referenced object's bytes are deleted. Zero means never.

    Off by default because the alternative is a number this project invented deleting
    somebody's photographs on a schedule they never chose."""


@dataclass(frozen=True, slots=True)
class SweepReport:
    """What a sweep did, or would have done."""

    scanned: int = 0
    orphans: int = 0
    expired: int = 0
    bytes_freed: int = 0
    dry_run: bool = True

    @property
    def deleted(self) -> int:
        return self.orphans + self.expired


def verdict(obj: StoredObject, *, referenced: bool, now: datetime, policy: Policy) -> str | None:
    """Whether this object may be deleted, and under which rule. `None` keeps it.

    Pure, and separate from the deleting, so the decision can be tested exhaustively
    without a filesystem and so a dry run and a real run cannot diverge: both call this.
    """
    age = now - obj.modified_at
    if not referenced:
        return ORPHAN if age >= timedelta(hours=policy.grace_hours) else None
    if policy.retention_days and age >= timedelta(days=policy.retention_days):
        return EXPIRED
    return None


def referenced_digests(session_factory: sessionmaker[Session]) -> set[str]:
    """Every digest the database knows about, from both tables that name one.

    Read in one go rather than asked per object: the store is scanned in the thousands
    and a query each would be a thousand round trips, while the whole set of digests is
    kilobytes.
    """
    with session_factory() as session:
        images = session.execute(sa.select(Image.sha256)).scalars().all()
        artifacts = session.execute(sa.select(Artifact.sha256)).scalars().all()
    return {*images, *artifacts}


def sweep(
    assets: AssetStore,
    session_factory: sessionmaker[Session] | None,
    *,
    policy: Policy | None = None,
    now: datetime | None = None,
    dry_run: bool = True,
) -> SweepReport:
    """Delete what `verdict` says may go, and report what happened either way.

    Raises rather than sweeping when there is no database: without one every object is
    unreferenced, and the sweep would delete the entire store.
    """
    if session_factory is None:
        raise ValueError("a sweep needs the database that says which objects are referenced")

    rules = policy or Policy()
    moment = now or datetime.now(UTC)
    keep = referenced_digests(session_factory)

    scanned = orphans = expired = freed = 0
    for obj in assets.scan():
        scanned += 1
        why = verdict(obj, referenced=obj.digest in keep, now=moment, policy=rules)
        if why is None:
            continue
        if why == ORPHAN:
            orphans += 1
        else:
            expired += 1
        freed += obj.size
        if not dry_run:
            assets.delete(obj.digest)
        log.info(
            "asset.swept",
            extra={
                "digest": obj.digest,
                "reason": why,
                "bytes": obj.size,
                "age_hours": round((moment - obj.modified_at).total_seconds() / 3600, 1),
                "dry_run": dry_run,
            },
        )

    report = SweepReport(
        scanned=scanned,
        orphans=orphans,
        expired=expired,
        bytes_freed=freed,
        dry_run=dry_run,
    )
    log.info(
        "asset.sweep",
        extra={
            "scanned": report.scanned,
            "deleted": report.deleted,
            "orphans": report.orphans,
            "expired": report.expired,
            "bytes_freed": report.bytes_freed,
            "dry_run": dry_run,
        },
    )
    return report
