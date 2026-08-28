# Runbook

## Setup, once

```bash
make setup        # uv sync --all-extras, then pnpm install
make models       # ~550 MB of weights into ~/.cache/editgpt/models
cp .env.example .env   # optional; nothing below is required to run locally
```

`pnpm` may live in a user-local prefix (`~/.npm-global/bin`). The Makefile resolves it
absolutely because `make` does not inherit an interactive shell's exports; add it to your
own PATH if you want to call `pnpm` directly.

## Credentials

**None of these are needed to run the stack.** Without them the gateway stores assets on
local disk and the generative lane is unavailable; `GET /ready` names every fallback it
is running in.

| Variable | Needed for | Where |
|---|---|---|
| `CLOUDFLARE_ACCOUNT_ID` | additions | dash.cloudflare.com → the 32-hex segment in the URL |
| `CLOUDFLARE_API_TOKEN` | additions | My Profile → API Tokens → **Workers AI** template |
| `GEMINI_API_KEY` | intent, critique | aistudio.google.com/apikey |
| `EDITGPT_S3_*` (four variables) | object storage instead of local disk | see **Object storage** below — no account needed to develop |
| `CLERK_SECRET_KEY` + `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` | sign-in | see **Authentication** below |

The Workers AI token needs **both `Workers AI — Edit` and `Workers AI — Read`**. Read
alone returns a 401 that reads exactly like a bad token. The template sets both.

## Authentication

Clerk, and its free tier asks for **no payment details**. Without the keys the gateway
runs unauthenticated — every request acts as one shared account, and `GET /ready` reports
`"auth": {"provider": "none"}` plus a line in `degraded`. That is fine locally and must
never be how it is deployed.

1. **dashboard.clerk.com** → create an application. Pick the sign-in methods you want;
   email + Google is a reasonable default.
2. **API Keys** → copy both into `.env`. Clerk's own names, unprefixed, because the
   Next.js SDK requires them verbatim and the gateway reads the same two — one variable
   each rather than the same secret in two places:
   ```
   NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_test_...
   CLERK_SECRET_KEY=sk_test_...
   ```
3. **API Keys → Show JWT public key → PEM.** Copy it into `CLERK_JWT_KEY`, quoted.
   Optional, and worth doing: with it, verifying a session is local arithmetic. Without
   it, the first request after every deploy waits on Clerk's JWKS endpoint.
4. Restart the gateway. `GET /ready` should now say `"provider": "clerk"`, and every
   `/v1` endpoint answers **401** without a session.

For production also set `EDITGPT_CLERK_AUTHORIZED_PARTIES` to your origin. It is the check
that stops a token minted for another application on the same Clerk instance from working
here; empty accepts any party.

**It fails closed.** An absent, expired or malformed token is a 401 — never a quiet fall
back to the shared account. Only `session_token` is accepted; Clerk's API keys and machine
tokens are refused, because this service models neither.

Users are provisioned on their first authenticated request: `users.external_id` holds
Clerk's subject, and every other table's `user_id` points at that row. There is no webhook
to miss.

## Object storage

**Nothing needs signing up for.** Assets default to local disk, and `make compose-s3`
starts MinIO — a real S3 server in a container, no account, no payment details. That is
also what the test suite and CI run against, so the object-storage path is genuinely
exercised rather than assumed.

```bash
make compose-s3     # MinIO console at http://localhost:9001 (editgpt / editgpt-dev-secret)
```

then in `.env`:

```
EDITGPT_S3_ENDPOINT_URL=http://localhost:9000
EDITGPT_S3_BUCKET=editgpt-assets
EDITGPT_S3_ACCESS_KEY_ID=editgpt
EDITGPT_S3_SECRET_ACCESS_KEY=editgpt-dev-secret
EDITGPT_S3_REGION=us-east-1
```

`GET /ready` will report `"storage": "s3"`. The bucket is created on startup if missing.

### Choosing a hosted provider, later

The adapter is **not** tied to a vendor: the endpoint is configuration, so any
S3-compatible service is four environment variables and no code change. This decision
belongs to deployment (Phase 10), not now.

| Provider | Free allowance | Notes |
|---|---|---|
| **Supabase Storage** | ~1 GB storage, 5 GB egress | **No payment details required**, commercial use allowed. Free projects pause after 7 days idle, which suits a demo and not a service. |
| Cloudflare R2 | 10 GB, no egress fees | The most generous, but **asks for a card** even on the free tier — which is why it is no longer the default. |
| Backblaze B2, Storj, AWS S3 | varies | All ask for payment details as far as we know. |

**Verify the signup requirements yourself before committing** — these change, and the
table above is what was true when it was written, not a guarantee. Nothing in the code
depends on which one you pick.

Storage switches over only when the endpoint, the bucket **and** an access key are all
present, so a half-filled `.env` fails at startup rather than writing a deployment's
artifacts to somebody's laptop. The `s3` extra must be installed: `uv sync --all-extras`
covers it.

## Daily

```bash
make compose-up   # redis + postgres
make migrate      # apply the schema (the gateway also creates it on first boot)
make worker       # the Celery worker — a separate terminal
make dev-lite     # gateway only — use this when a benchmark is also running
make dev          # redis + postgres + web
make check-fast   # inner loop
```

### Driving a job by hand

```bash
DIGEST=$(curl -sF file=@evals/photos/i1.jpg localhost:8000/v1/images | jq -r .sha256)
JOB=$(curl -s localhost:8000/v1/jobs -H 'content-type: application/json' \
        -H "Idempotency-Key: $(uuidgen)" \
        -d "{\"op\":\"remove\",\"image_sha256\":\"$DIGEST\",\"target\":\"the car\"}" \
      | jq -r .id)
curl -N localhost:8000/v1/jobs/$JOB/events        # server-sent progress
curl -s localhost:8000/v1/jobs/$JOB | jq .        # final state and result digest
```

The default editor is `noop`, which returns the image untouched — Phase 3 proves the pipe,
not the edit. Re-sending the same `Idempotency-Key` returns the original job with a 200
instead of creating a second one.

Use **OrbStack** or `colima`, not Docker Desktop: roughly 1.5 GB of RAM, which on an
8 GB machine is the difference between a pipeline fitting and not. Never run the compose
stack, the Next dev server and a model benchmark at the same time.

## When something is wrong

**A test hangs or fails on the network.** Sockets are disabled in tests. Something is
reaching out that should be stubbed.

**`ModuleNotFoundError: transformers`.** `uv sync --all-extras` applies to the root
project's extras, not a workspace member's. The root `pyproject.toml` names
`editgpt-models[text]` explicitly; if you removed it, that is why.

**Jobs are accepted but never finish.** `curl localhost:8000/ready`. If `degraded` names
"no queue", the gateway could not reach Redis and is recording tasks instead of sending
them. `make compose-up`, then restart the gateway.

**`/ready` says jobs are in memory.** Postgres was unreachable at boot, so the gateway
fell back to an in-memory store and every job dies with the process. Check the container,
then restart — the fallback is chosen once, at startup.

**A job sits in `running` forever.** The worker died mid-task. `task_reject_on_worker_lost`
is on, so the message is not redelivered; the job needs cancelling. The two Celery time
limits (soft 240 s, hard 300 s) exist so this is rare.

**Peak RSS over budget.** `make memory`. If it fails, something is holding two heavy
models: check `ModelSlot.resident` and that nothing imports a model at module scope.

**A mask looks torn rather than snapped to an object.** The MobileSAM encoder pads but
does not resize; the caller must resize the longest side to 1024 first. Skipping it
desynchronises the embedding from the point coordinates.

**Everything except the target got repainted.** MI-GAN's mask polarity is inverted:
255 means *keep*.

**An edit reports success but the object is still there.** Check the mask area. A
too-small mask used to pass every numeric check; `MIN_MASK_PX` now raises instead.

**Workers AI 401 with a token that verifies as valid.** The token lacks `Workers AI —
Edit`. `/user/tokens/verify` only says the token exists.

## Regenerating results

```bash
make eval                    # every runnable case
make eval-local              # skips the cases that spend provider quota
uv run python -m evals.run i1 i6c   # a subset
make eval-diff               # compare the last run against the baseline
make eval-baseline           # accept the last run as the new baseline
make memory                  # RSS tier, needs weights
```

## Browser tests

```bash
make compose-up && make dev-lite &   # Postgres, Redis, the gateway
make worker &                        # only needed for the signed-in flows
make e2e                             # Playwright starts Next itself
```

`make e2e` is not part of `make check`: it needs a gateway and a database, and the
signed-in half needs a Clerk user. That half **skips** unless `CLERK_E2E_EMAIL` and
`CLERK_E2E_PASSWORD` are set — create a user in the development instance under Users,
give it a password, and use it for nothing else. The smoke half needs none of that and
is what catches "the site does not build" and "the site does not run", both of which
happened.

Run against something already deployed with `EDITGPT_E2E_URL=https://… make e2e`, which
skips starting a local Next server.

`evals/out/report.json` is the machine-readable result; the PNG strips are
`original | mask | result`.

`evals/baseline.json` is what CI diffs against, and it holds the local cases only — the
generative ones return a different picture every call, so baselining them would record
noise and report it as change forever. Update it with `make eval-baseline` **after**
looking at the images: it asserts that the current output is acceptable.
