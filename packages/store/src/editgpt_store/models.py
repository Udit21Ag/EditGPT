"""The relational schema.

Six tables, and the reasons they are separate rather than columns on one another:

* ``users`` — an identity. Currently only ever the anonymous sentinel, because auth is
  not implemented; the column exists so adding it later is a backfill and not a
  migration of every foreign key in the system.
* ``images`` — an uploaded or produced blob's *metadata*. The bytes live in the asset
  store; this table is what lets a query ask "how many megapixels has this user sent"
  without touching object storage.
* ``jobs`` — one requested edit, holding its `EditSpec` verbatim as JSON. Storing the
  spec rather than exploding it into columns keeps the contract in one place: `EditSpec`
  validates it, and the database does not get a second, drifting opinion about what a
  legal edit is.
* ``job_steps`` — the progress trail. Append-only. This is what the SSE stream replays
  to a client that connected late, which is the common case on a mobile network.
* ``artifacts`` — outputs of a job. A job produces more than one (result, mask,
  intermediate), and they are addressed by digest.
* ``cost_ledger`` — what each provider call cost. Separate from jobs because a free-tier
  budget is consumed per account and has to be summable across jobs, including jobs that
  failed after spending.

Dialect note: `JSONB` and native UUIDs are Postgres features, declared with SQLite
variants so the test suite runs in memory without a container. Production is Postgres;
the variants exist to keep tests hermetic, not to promise portability.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

JSONColumn = postgresql.JSONB().with_variant(sa.JSON(), "sqlite")

AutoBigInt = sa.BigInteger().with_variant(sa.Integer(), "sqlite")
"""A 64-bit auto-incrementing primary key that also works on SQLite.

SQLite only gives rowid aliasing — and therefore autoincrement — to a column declared
exactly `INTEGER PRIMARY KEY`. A `BIGINT` one is a plain column and inserting without a
value violates NOT NULL. Postgres still gets a bigint.
"""

ANONYMOUS_USER_ID = UUID("00000000-0000-0000-0000-000000000001")
"""The stand-in owner until authentication exists.

A sentinel row rather than a nullable column, because idempotency is unique *per user*
and Postgres treats NULLs in a unique constraint as distinct — which would silently
disable deduplication for exactly the requests that have it today.
"""


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(sa.Uuid, primary_key=True, default=uuid4)
    external_id: Mapped[str | None] = mapped_column(sa.String(255), unique=True)
    """The identity provider's subject claim. Null for the anonymous sentinel."""

    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=_now)


class Image(Base):
    __tablename__ = "images"

    sha256: Mapped[str] = mapped_column(sa.String(64), primary_key=True)
    """The content digest *is* the key. Uploading the same file twice stores one row."""

    user_id: Mapped[UUID] = mapped_column(sa.ForeignKey("users.id"), index=True)
    width: Mapped[int] = mapped_column(sa.Integer)
    height: Mapped[int] = mapped_column(sa.Integer)
    content_type: Mapped[str] = mapped_column(sa.String(64))
    byte_size: Mapped[int] = mapped_column(sa.BigInteger)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=_now)

    __table_args__ = (sa.CheckConstraint("width > 0 AND height > 0", name="ck_images_positive"),)


class JobRow(Base):
    __tablename__ = "jobs"

    id: Mapped[UUID] = mapped_column(sa.Uuid, primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(sa.ForeignKey("users.id"), index=True)
    state: Mapped[str] = mapped_column(sa.String(16), index=True)
    spec: Mapped[dict[str, object]] = mapped_column(JSONColumn)
    idempotency_key: Mapped[str | None] = mapped_column(sa.String(255))
    result_sha256: Mapped[str | None] = mapped_column(sa.String(64))
    error: Mapped[str | None] = mapped_column(sa.Text)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), default=_now, onupdate=_now
    )

    steps: Mapped[list[JobStepRow]] = relationship(
        back_populates="job", cascade="all, delete-orphan", order_by="JobStepRow.id"
    )

    __table_args__ = (
        sa.UniqueConstraint("user_id", "idempotency_key", name="uq_jobs_user_idempotency"),
    )


class JobStepRow(Base):
    __tablename__ = "job_steps"

    id: Mapped[int] = mapped_column(AutoBigInt, primary_key=True, autoincrement=True)
    job_id: Mapped[UUID] = mapped_column(sa.ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    state: Mapped[str] = mapped_column(sa.String(16))
    detail: Mapped[str] = mapped_column(sa.Text, default="")
    progress: Mapped[float] = mapped_column(sa.Float, default=0.0)
    at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=_now)

    job: Mapped[JobRow] = relationship(back_populates="steps")


class Artifact(Base):
    __tablename__ = "artifacts"

    id: Mapped[UUID] = mapped_column(sa.Uuid, primary_key=True, default=uuid4)
    job_id: Mapped[UUID] = mapped_column(sa.ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    sha256: Mapped[str] = mapped_column(sa.String(64))
    kind: Mapped[str] = mapped_column(sa.String(32))
    """`result`, `mask` or `intermediate`. Kept as a string rather than an enum type so
    adding a kind is a code change and not a migration."""

    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=_now)


class CostEntry(Base):
    __tablename__ = "cost_ledger"

    id: Mapped[int] = mapped_column(AutoBigInt, primary_key=True, autoincrement=True)
    user_id: Mapped[UUID] = mapped_column(sa.ForeignKey("users.id"), index=True)
    job_id: Mapped[UUID | None] = mapped_column(sa.ForeignKey("jobs.id", ondelete="SET NULL"))
    provider: Mapped[str] = mapped_column(sa.String(64))
    operation: Mapped[str] = mapped_column(sa.String(32))
    units: Mapped[int] = mapped_column(sa.Integer, default=1)
    cents: Mapped[float] = mapped_column(sa.Float, default=0.0)
    """Cost in cents, zero on a free tier. Recorded even when free: the point of the
    ledger is knowing how close the free allowance is to running out."""

    at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=_now, index=True)
