# apps/gateway

FastAPI. Upload, job intake, progress streaming, rate limits.

**No model ever loads in this process.** It is the web tier; models live in workers. If
you find yourself importing `editgpt_models` here, the work belongs in a task. The
gateway sends a Celery task **by name** for the same reason — importing the worker would
drag its dependencies in through the back door.

**This is the input boundary** (invariant 7). `uploads.py` validates once, and everything
downstream may then assume a real image of bounded size. The checks are not optional:
size before _and_ during the read, format from the magic bytes rather than the filename
or declared type, and pixel count from the header **before** decoding — a 200 KB PNG can
declare 60000×60000. Formats are an allowlist.

**`Depends` must not close over `create_app`.** This module uses
`from __future__ import annotations`, so FastAPI resolves each annotation as a string
against module globals; a closure name is not there, the marker fails to resolve, and the
parameter silently becomes a query field. `get_services` is module-level for that reason.

**Identity is decided in `auth.py` and nowhere else.** A route must take `PrincipalDep`
and pass it to the store rather than leaning on the store's default parameter — a route
that forgets behaves identically today and leaks every job. `tests/test_auth_boundary.py`
and `tests/test_clerk_auth.py` catch that, the latter by enumerating every `/v1` endpoint.

**It fails closed, and that is not negotiable.** An absent, expired or malformed token is
a 401. Never catch a verification failure and fall back to anonymous: that turns any
expired token into a way into the shared account. Only `session_token` is accepted —
Clerk also issues API keys and machine tokens, and accepting a token type you do not model
is how a credential meant for something else becomes a login.

Authentication is on when `EDITGPT_CLERK_SECRET_KEY` is set and off otherwise, which is
how tests and a fresh checkout run without a credential. `/ready` reports which, because
"authentication is off" must never be silent.

**`/v1/images/{digest}` is authenticated but not ownership-checked.** Storage is
content-addressed, so two people uploading the same picture share one row; an ownership
check would lock the second uploader out of their own upload. TD-019 has the fix.

**`EditSpec` does the rejecting, not the routes.** A handler translates its
`ValidationError` into a 422 with the reason intact. Re-implementing a rule here creates
a second opinion that drifts from the contract.

**The SSE stream replays persisted steps, then follows Redis.** Neither is complete
alone: the channel keeps no history and the table lags a write. A phone backgrounding the
browser is the normal case, not the edge case.

**Degrade loudly.** Missing Postgres or Redis falls back rather than refusing to boot, so
the frontend is still runnable locally — and `/ready` names every fallback in use. Never
make a fallback silent.

`/capabilities` advertises what this deployment can actually do rather than what the plan
hoped for. Settings come from `Settings` (env prefix `EDITGPT_`), read once — no
`os.environ` in handlers.
