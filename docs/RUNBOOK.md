# Runbook

## Setup, once

```bash
make setup        # uv sync --all-extras, then pnpm install
make models       # ~285 MB of weights into ~/.cache/editgpt/models
cp .env.example .env   # then fill in the keys below
```

`pnpm` may live in a user-local prefix (`~/.npm-global/bin`). The Makefile resolves it
absolutely because `make` does not inherit an interactive shell's exports; add it to your
own PATH if you want to call `pnpm` directly.

## Credentials

| Variable | Needed for | Where |
|---|---|---|
| `CLOUDFLARE_ACCOUNT_ID` | additions | dash.cloudflare.com → the 32-hex segment in the URL |
| `CLOUDFLARE_API_TOKEN` | additions | My Profile → API Tokens → **Workers AI** template |
| `GEMINI_API_KEY` | intent, critique | aistudio.google.com/apikey |

The Workers AI token needs **both `Workers AI — Edit` and `Workers AI — Read`**. Read
alone returns a 401 that reads exactly like a bad token. The template sets both.

## Daily

```bash
make dev-lite     # gateway only — use this when a benchmark is also running
make dev          # redis + postgres + web
make check-fast   # inner loop
```

Use **OrbStack** or `colima`, not Docker Desktop: roughly 1.5 GB of RAM, which on an
8 GB machine is the difference between a pipeline fitting and not. Never run the compose
stack, the Next dev server and a model benchmark at the same time.

## When something is wrong

**A test hangs or fails on the network.** Sockets are disabled in tests. Something is
reaching out that should be stubbed.

**`ModuleNotFoundError: transformers`.** `uv sync --all-extras` applies to the root
project's extras, not a workspace member's. The root `pyproject.toml` names
`editgpt-models[text]` explicitly; if you removed it, that is why.

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
uv run python -m evals.run i1 i6c   # a subset
make memory                  # RSS tier, needs weights
```

`evals/out/report.json` is the machine-readable result; the PNG strips are
`original | mask | result`.
