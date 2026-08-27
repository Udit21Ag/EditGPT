"""The migration, applied to a real Postgres.

**Why this exists.** The initial migration was generated against SQLite, reviewed, merged
and never run. The first time anyone applied it to Postgres it failed: the anonymous-user
seed bound its id as an untyped parameter, Postgres inferred `character varying`, and
refused to put it in a `uuid` column. Every unit test in this package passed throughout,
because they all run on SQLite where that distinction does not exist.

So a migration is not verified by review. It is verified by applying it, on the dialect
it will actually meet, and then rolling it back.

Marked `service` and skipped when no Postgres is reachable, so a checkout with no containers
still runs green. CI provides one, which is where this genuinely runs on every push.

Each test builds a **throwaway database** rather than touching the developer's: a test
that drops every table is not something to point at a database somebody is using.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
import sqlalchemy as sa

pytestmark = [pytest.mark.service, pytest.mark.enable_socket]

STORE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = "postgresql+psycopg://editgpt:editgpt@localhost:5432/editgpt"

EXPECTED_TABLES = {"users", "images", "jobs", "job_steps", "artifacts", "cost_ledger"}
ANONYMOUS_USER_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")


def render(url: sa.URL) -> str:
    """A URL a driver can actually authenticate with.

    `str(URL)` renders the password as `***` — SQLAlchemy hides it by default, which is
    right for logs and silently wrong here: the connection fails authentication and looks
    exactly like "no database reachable". That cost twenty minutes once already.
    """
    return url.render_as_string(hide_password=False)


def admin_url() -> str:
    """The configured server, pointed at the default database so we can create another."""
    url = sa.make_url(os.environ.get("EDITGPT_DATABASE_URL", DEFAULT_URL))
    if not url.drivername.startswith("postgresql"):
        pytest.skip(f"{url.drivername} is not Postgres; this test is about dialect behaviour")
    return render(url.set(database="postgres"))


@pytest.fixture
def scratch_database() -> Iterator[str]:
    """A fresh, empty database, dropped afterwards whatever happens."""
    try:
        engine = sa.create_engine(admin_url(), isolation_level="AUTOCOMMIT")
        with engine.connect() as connection:
            connection.execute(sa.text("select 1"))
    except sa.exc.OperationalError as error:
        pytest.skip(f"no Postgres reachable ({type(error).__name__}); run `make compose-up`")

    name = f"editgpt_migration_{uuid.uuid4().hex[:12]}"
    with engine.connect() as connection:
        connection.execute(sa.text(f'CREATE DATABASE "{name}"'))
    try:
        yield render(sa.make_url(admin_url()).set(database=name))
    finally:
        with engine.connect() as connection:
            connection.execute(
                sa.text(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    "WHERE datname = :name AND pid <> pg_backend_pid()"
                ),
                {"name": name},
            )
            connection.execute(sa.text(f'DROP DATABASE IF EXISTS "{name}"'))
        engine.dispose()


def alembic_config(url: str) -> object:
    from alembic.config import Config

    config = Config(str(STORE_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(STORE_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", url)
    return config


def upgrade(url: str) -> None:
    from alembic import command

    # `env.py` reads the URL from the environment, so it is set rather than passed.
    previous = os.environ.get("EDITGPT_DATABASE_URL")
    os.environ["EDITGPT_DATABASE_URL"] = url
    try:
        command.upgrade(alembic_config(url), "head")  # type: ignore[arg-type]
    finally:
        if previous is None:
            del os.environ["EDITGPT_DATABASE_URL"]
        else:
            os.environ["EDITGPT_DATABASE_URL"] = previous


def downgrade(url: str) -> None:
    from alembic import command

    previous = os.environ.get("EDITGPT_DATABASE_URL")
    os.environ["EDITGPT_DATABASE_URL"] = url
    try:
        command.downgrade(alembic_config(url), "base")  # type: ignore[arg-type]
    finally:
        if previous is None:
            del os.environ["EDITGPT_DATABASE_URL"]
        else:
            os.environ["EDITGPT_DATABASE_URL"] = previous


def table_names(url: str) -> set[str]:
    engine = sa.create_engine(url)
    try:
        return set(sa.inspect(engine).get_table_names())
    finally:
        engine.dispose()


def test_the_migration_applies_to_postgres(scratch_database: str) -> None:
    """The test that would have caught the seed bug before it reached anyone."""
    upgrade(scratch_database)
    assert table_names(scratch_database) >= EXPECTED_TABLES


def test_the_anonymous_user_is_seeded_with_a_real_uuid(scratch_database: str) -> None:
    """The exact failure: a bound id arrived as `character varying` for a `uuid` column."""
    upgrade(scratch_database)
    engine = sa.create_engine(scratch_database)
    try:
        with engine.connect() as connection:
            rows = connection.execute(sa.text("SELECT id FROM users")).scalars().all()
    finally:
        engine.dispose()
    assert rows == [ANONYMOUS_USER_ID]


def test_postgres_gets_its_native_types_not_sqlite_stand_ins(scratch_database: str) -> None:
    """`Uuid`, `JSONB` and the bigint sequences are the whole reason for the variants.

    If a variant leaked the wrong way, every SQLite test would still pass while production
    stored UUIDs as text and specs as a string.
    """
    upgrade(scratch_database)
    engine = sa.create_engine(scratch_database)
    try:
        with engine.connect() as connection:
            types: dict[str, str] = {
                str(name): str(kind)
                for name, kind in connection.execute(
                    sa.text(
                        "SELECT column_name, data_type FROM information_schema.columns "
                        "WHERE table_name = 'jobs'"
                    )
                ).all()
            }
            step_default = connection.execute(
                sa.text(
                    "SELECT column_default FROM information_schema.columns "
                    "WHERE table_name = 'job_steps' AND column_name = 'id'"
                )
            ).scalar_one()
    finally:
        engine.dispose()

    assert types["id"] == "uuid"
    assert types["spec"] == "jsonb"
    assert types["created_at"] == "timestamp with time zone"
    assert "nextval" in step_default, "a bigint primary key must own a sequence"


def test_the_migration_rolls_back_completely(scratch_database: str) -> None:
    """A downgrade nobody has run is a rollback plan nobody has."""
    upgrade(scratch_database)
    downgrade(scratch_database)
    assert not (EXPECTED_TABLES & table_names(scratch_database))


def test_the_schema_the_migration_builds_matches_the_models(scratch_database: str) -> None:
    """The drift this catches: a model changed and nobody generated a migration.

    Compared by table and column name rather than by full type, because a type comparison
    would fail on details that are equivalent in practice. Names are where drift shows up.
    """
    from editgpt_store.models import Base

    upgrade(scratch_database)
    engine = sa.create_engine(scratch_database)
    try:
        inspector = sa.inspect(engine)
        actual = {
            name: {column["name"] for column in inspector.get_columns(name)}
            for name in inspector.get_table_names()
            if name != "alembic_version"
        }
    finally:
        engine.dispose()

    expected = {
        table.name: {column.name for column in table.columns}
        for table in Base.metadata.sorted_tables
    }
    assert actual == expected, "the migration and the models disagree; regenerate it"
