# packages/store

Persistence. Content-addressed blobs in an asset store, everything else in Postgres.

**The digest is the key.** Object names are SHA-256 of the bytes, never anything a user
supplied, so path traversal is impossible by construction rather than by filtering. Do
not add a "friendly name" key path; put the name in a column.

**`editgpt_core` owns what a job is.** This package round-trips `Job` through
`spec` as JSON and does not validate it a second time. Two validators drift, and the one
in the database is the harder to fix. If a rule needs enforcing, it belongs in
`editgpt_core.jobs`.

**The anonymous user is a row, not a NULL.** Idempotency is unique per user, and Postgres
treats NULLs in a unique constraint as distinct — a nullable `user_id` would silently
switch deduplication off for every request that has it today.

**Steps are replaced wholesale on save**, not diffed. They are append-only and there are
a handful per job; a diff would be a second place for the row and the in-memory job to
disagree.

**The remote asset store is not named after a vendor.** `S3AssetStore` takes an endpoint,
so MinIO in a container and any hosted provider are the same code path — that is what
lets `make compose-s3` verify it for real, with no account and no payment details. Do not
reintroduce a provider-specific adapter; the endpoint is configuration.

Two adapters exist for assets so tests and CI need no credentials: `LocalAssetStore` is
the default, `S3AssetStore` needs the `s3` extra. Same for jobs: `InMemoryJobStore` and
`SqlJobStore`. Tests run on SQLite in memory; the JSONB and UUID columns carry SQLite
variants for that reason alone. **Production is Postgres** — do not treat the variants as
a portability promise.

`bootstrap()` is for tests and a first local run. Schema changes go through Alembic in
`migrations/`; `create_all` cannot express a rename or a backfill. **A migration is not
verified until it has been applied to Postgres and rolled back** — that is what
`tests/test_migrations.py` is for, and why it exists at all.
