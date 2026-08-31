"""Persistence for EditGPT.

Assets are content-addressed blobs; everything else is a row. The contract layer
(`editgpt_core`) owns what a job *is*; this package only stores one.
"""

from editgpt_store.assets import (
    AssetNotFoundError,
    AssetStore,
    LocalAssetStore,
    S3AssetStore,
    StoredObject,
    digest_of,
)
from editgpt_store.engine import bootstrap, make_engine, make_session_factory
from editgpt_store.jobs import InMemoryJobStore, JobStore, SqlJobStore
from editgpt_store.lifecycle import Policy, SweepReport, sweep
from editgpt_store.models import ANONYMOUS_USER_ID, Artifact, Base, CostEntry, Image, User
from editgpt_store.progress import ProgressEvent, channel_for, last_event, publish, subscribe
from editgpt_store.records import (
    artifacts_for,
    record_artifact,
    record_cost,
    record_image,
    spend_since,
)

__all__ = [
    "ANONYMOUS_USER_ID",
    "Artifact",
    "AssetNotFoundError",
    "AssetStore",
    "Base",
    "CostEntry",
    "Image",
    "InMemoryJobStore",
    "JobStore",
    "LocalAssetStore",
    "Policy",
    "ProgressEvent",
    "S3AssetStore",
    "SqlJobStore",
    "StoredObject",
    "SweepReport",
    "User",
    "artifacts_for",
    "bootstrap",
    "channel_for",
    "digest_of",
    "last_event",
    "make_engine",
    "make_session_factory",
    "publish",
    "record_artifact",
    "record_cost",
    "record_image",
    "spend_since",
    "subscribe",
    "sweep",
]
