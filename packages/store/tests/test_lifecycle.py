"""Deleting bytes nothing needs, and — more importantly — not deleting anything else.

Age is supplied by moving the *clock* rather than the files: `sweep(now=...)` takes the
moment to judge against, so a test about a thirty-day-old object does not have to forge
an mtime, and the same call is what a scheduled run makes.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from editgpt_store import (
    Image,
    LocalAssetStore,
    Policy,
    SqlJobStore,
    StoredObject,
    record_artifact,
    record_image,
    sweep,
)
from editgpt_store.lifecycle import EXPIRED, ORPHAN, verdict
from sqlalchemy.orm import Session, sessionmaker


@pytest.fixture
def store(tmp_path: Path) -> LocalAssetStore:
    return LocalAssetStore(root=tmp_path / "assets")


def kept(store: LocalAssetStore) -> set[str]:
    return {obj.digest for obj in store.scan()}


def recorded(session_factory: sessionmaker[Session], store: LocalAssetStore, data: bytes) -> str:
    digest = store.put(data)
    record_image(
        session_factory,
        sha256=digest,
        width=1,
        height=1,
        content_type="image/png",
        byte_size=len(data),
    )
    return digest


LATER = datetime.now(UTC) + timedelta(days=2)


# ---------------------------------------------------------------- the decision


def obj(*, hours_old: float, size: int = 10) -> StoredObject:
    return StoredObject(
        digest="a" * 64, size=size, modified_at=datetime.now(UTC) - timedelta(hours=hours_old)
    )


def test_an_unreferenced_object_past_the_grace_period_is_an_orphan() -> None:
    assert (
        verdict(obj(hours_old=25), referenced=False, now=datetime.now(UTC), policy=Policy())
        == ORPHAN
    )


def test_a_fresh_unreferenced_object_is_kept() -> None:
    """An upload whose row is still being committed is indistinguishable from an orphan.

    Only for as long as the commit takes — which is why the grace period exists and why
    it is a day rather than a minute.
    """
    assert (
        verdict(obj(hours_old=1), referenced=False, now=datetime.now(UTC), policy=Policy()) is None
    )


def test_a_referenced_object_is_kept_forever_by_default() -> None:
    """Retention off means off: no schedule this project invented deletes a photograph."""
    old = obj(hours_old=24 * 365)
    assert verdict(old, referenced=True, now=datetime.now(UTC), policy=Policy()) is None


def test_retention_expires_a_referenced_object_once_configured() -> None:
    old = obj(hours_old=24 * 31)
    policy = Policy(retention_days=30)
    assert verdict(old, referenced=True, now=datetime.now(UTC), policy=policy) == EXPIRED


def test_retention_does_not_reach_an_object_younger_than_it() -> None:
    policy = Policy(retention_days=30)
    assert (
        verdict(obj(hours_old=24 * 29), referenced=True, now=datetime.now(UTC), policy=policy)
        is None
    )


# ---------------------------------------------------------------- the sweep


def test_an_orphan_is_deleted_and_a_recorded_image_is_not(
    store: LocalAssetStore, session_factory: sessionmaker[Session]
) -> None:
    """The case this exists for: an upload that was never turned into a job.

    No query can find it — the database only knows what it wrote down, and an orphan is
    precisely what is not in it.
    """
    keep = recorded(session_factory, store, b"a photograph")
    orphan = store.put(b"an upload nobody finished")

    report = sweep(store, session_factory, now=LATER, dry_run=False)

    assert kept(store) == {keep}
    assert report.orphans == 1
    assert report.scanned == 2
    assert report.bytes_freed == len(b"an upload nobody finished")
    assert not store.exists(orphan)


def test_an_artifact_reference_keeps_bytes_no_image_row_mentions(
    store: LocalAssetStore, session_factory: sessionmaker[Session], job: object
) -> None:
    """Results are recorded as artifacts, not images. Both tables are references."""
    stored = SqlJobStore(session_factory)
    saved = stored.save(job)  # type: ignore[arg-type]
    digest = store.put(b"the result of an edit")
    record_artifact(session_factory, job_id=saved.id, sha256=digest, kind="result")

    sweep(store, session_factory, now=LATER, dry_run=False)
    assert store.exists(digest), "a job's own result was swept"


def test_retention_takes_the_bytes_and_leaves_the_history(
    store: LocalAssetStore, session_factory: sessionmaker[Session]
) -> None:
    """A row that outlives its pixels: the job stays readable, the fetch answers 404."""
    digest = recorded(session_factory, store, b"an old photograph")

    sweep(
        store,
        session_factory,
        policy=Policy(retention_days=1),
        now=LATER,
        dry_run=False,
    )

    assert not store.exists(digest)
    with session_factory() as session:
        assert session.get(Image, digest) is not None, "history was deleted with the bytes"


def test_a_dry_run_reports_what_it_would_delete_and_deletes_nothing(
    store: LocalAssetStore, session_factory: sessionmaker[Session]
) -> None:
    """The two runs share `verdict`, so the report cannot promise one thing and do another."""
    orphan = store.put(b"an upload nobody finished")

    planned = sweep(store, session_factory, now=LATER)
    assert planned.dry_run is True
    assert store.exists(orphan), "a dry run deleted something"

    done = sweep(store, session_factory, now=LATER, dry_run=False)
    assert (done.orphans, done.expired, done.bytes_freed) == (
        planned.orphans,
        planned.expired,
        planned.bytes_freed,
    )
    assert not store.exists(orphan)


def test_a_sweep_without_a_database_refuses_instead_of_deleting_everything(
    store: LocalAssetStore,
) -> None:
    """Fail closed. With no references, every object looks like an orphan."""
    store.put(b"a photograph")
    with pytest.raises(ValueError, match="database"):
        sweep(store, None, now=LATER, dry_run=False)


def test_nothing_is_deleted_before_the_grace_period_has_passed(
    store: LocalAssetStore, session_factory: sessionmaker[Session]
) -> None:
    digest = store.put(b"an upload arriving right now")
    report = sweep(store, session_factory, dry_run=False)

    assert store.exists(digest)
    assert report.deleted == 0


# ---------------------------------------------------------------- what a scan sees


def test_a_scan_reports_the_size_and_leaves_the_bytes_alone(store: LocalAssetStore) -> None:
    data = b"x" * 321
    digest = store.put(data)
    found = list(store.scan())

    assert [(o.digest, o.size) for o in found] == [(digest, 321)]
    assert found[0].modified_at.tzinfo is not None, "an age needs a timezone to be one"


def test_a_scan_ignores_a_write_still_in_progress(store: LocalAssetStore) -> None:
    """A `.partial` file is bytes arriving. Sweeping one would delete a live upload."""
    store.put(b"a photograph")
    partial = next(store.root.glob("*/"))
    (partial / f".{'b' * 64}.partial").write_bytes(b"half an image")

    assert len(list(store.scan())) == 1


def test_a_scan_of_an_empty_store_is_empty_rather_than_an_error(tmp_path: Path) -> None:
    """A worker sweeping before anything has been uploaded must not fall over."""
    assert list(LocalAssetStore(root=tmp_path / "never-used").scan()) == []


def test_an_unknown_id_is_not_ours_to_delete(store: LocalAssetStore) -> None:
    """Something else's file in the same bucket is left alone, not swept as an orphan."""
    store.put(b"a photograph")
    bucket = next(store.root.glob("*/"))
    (bucket / "notes.txt").write_text("someone else's")

    assert all(len(o.digest) == 64 for o in store.scan())
    assert (bucket / "notes.txt").exists()
