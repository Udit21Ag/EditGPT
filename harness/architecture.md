# Architecture

**Read when:** you need to understand how a request flows, where a component belongs, or
whether a change crosses a boundary.
**Solves:** placing work correctly, and recognising an `ARCHITECTURAL` change before making it.
**Authority:** describes intent and invariants. Where it disagrees with the code, the code
wins and the disagreement is a defect — see `self_updation.md`.

## Overview

A pipeline that turns _(image, region, instruction)_ into an edited image. Grounding is
local and cheap; generation is remote and constrained; every result is scored before it
is returned.

The dominant force on this architecture is not taste — it is **8 GB of RAM**. One pipeline
resident measures 1.47–2.12 GB, which is why models cannot share a process with the API,
why only one heavy model is resident at a time, and why the heavy stages are destined for
a larger host.

## Structure by responsibility

| Component            | Owns                                                                   |
| -------------------- | ---------------------------------------------------------------------- |
| `packages/core`      | contracts every component speaks; quality scoring                      |
| `packages/models`    | model lifetime, grounding, erasure, compositing, **the edit dispatch** |
| `packages/providers` | remote generation, failover, circuit breaking                          |
| `packages/planner`   | instruction -> typed plan; rules first, the model for the rest         |
| `apps/mcp`           | the vision tools over MCP, as a client of the gateway                  |
| `apps/gateway`       | HTTP surface: health, capabilities, (later) upload and jobs            |
| `apps/web`           | browser UI: upload, region selection, progress                         |
| `evals`              | the golden set and its runner                                          |

## Diagram

```
image + (brush | box | text)
        │
        ▼
   INTENT ─────────► EditSpec                 packages/core/spec.py
        │            rejects unactionable work at construction
        ▼
   GROUNDING                                  packages/models/segment.py
        │  text ─► Grounding DINO ─► ranked boxes   (mIoU 0.469 answering top-1)
        │  narrow top-two margin ─► ASK: return candidates, do not edit
        │                           (0.832 hit / 0.731 mIoU if the user picks; ADR-0003)
        │  box | brush | seed ─► MobileSAM ─► mask + its own confidence
        │  below the gate the detector's box is kept as a filled rectangle
        │  CLIPSeg remains as a seed source for "stuff" nouns; see ADR-0002
        ▼
   MASK PREP        dilate by 5% of the object's longest side, never a pixel constant
        ▼
   ROUTER           by OPERATION, not by difficulty
        ├── REMOVE ──────► local erasers only, never remote
        ├── ADD/REPLACE ─► remote provider; mask must be clear, large and plausible
        ├── BACKGROUND ──► flood-fill the backdrop and recolour; no model call
        │                  falls back to a semantic mask when the border is not uniform
        └── UPSCALE ─────► tiled 2x enhancement; NOT an interactive operation
        ▼
   MULTI-PASS ERASE                           packages/models/pipeline.py
        │  pass 1  fast eraser
        │  pass 2  ALWAYS runs; kept only if it verifies better; rollback recorded
        │  pass 3  only while the result is still unacceptable
        ▼
   COMPOSITE        chroma-only match, feathered paste-back at full resolution
        ▼
   SCORE            fill cost + mask-growth penalty. Never cost alone.
                    Every threshold is read from `config.Thresholds`, never a literal.
```

## Implemented vs planned

Stating this honestly is the point of the section.

| Stage      | Today                                   | Planned                                     |
| ---------- | --------------------------------------- | ------------------------------------------- |
| Intent     | `EditSpec` constructed by the caller    | agent parses free text, asks when ambiguous |
| Router     | `execute.py`, one dispatch by operation | an agent behind A2A                         |
| Multi-pass | automatic, capped at 3                  | user-triggered "needs more work"            |
| Critic     | scoring inline                          | an agent with a retry budget                |
| Transport  | HTTP + Celery over Redis                | A2A JSON-RPC + SSE between processes        |
| Storage    | content-addressed, local disk or S3     | a hosted S3 endpoint, chosen at deploy time |
| Jobs       | queue, worker, SSE progress             | critic-driven retries within a job          |

**There is no agent mesh yet.** The pipeline is a library. Splitting it into processes
changes deployment, not shape — which is what `EditSpec` is for.

## Models

Keys are registry keys (`packages/models/src/editgpt_models/registry.py`), where the measured
peak resident set for each lives. `make models` fetches all of them (~552 MB).

| Key                           | Model            | Role                                           |    Disk |
| ----------------------------- | ---------------- | ---------------------------------------------- | ------: |
| `sam-encoder` / `sam-decoder` | MobileSAM        | box, point or seed to a precise mask           |   44 MB |
| `grounding-dino`              | G-DINO tiny int8 | phrase to candidate boxes — the text lane      |  204 MB |
| —                             | CLIPSeg-rd64     | fallback text seed for "stuff" (torch; TD-001) | ~150 MB |
| `migan`                       | MI-GAN           | primary eraser, fast path                      |   28 MB |
| `lama`                        | Big-LaMa         | escalation eraser                              |  208 MB |
| `esrgan-x2`                   | Real-ESRGAN x2   | tiled resolution enhancement                   |   67 MB |

| — | SD-1.5-inpainting | additions and replacements (remote) | — |

Two erasers ship because they fail differently: one erases a small object on a flat
background where the other leaves a ghost, and loses on large objects over texture. The
router picks by measurement rather than by preference.

## Grounding and the chooser

Turning a phrase into a region is a **separate call** from editing with it. `POST /v1/masks`
grounds and returns ranked candidates; the job then carries the chosen mask. Grounding is
cheap and reversible where an edit is neither, and separating them is what lets a user
approve a region before any model time is spent.

**Two prompts, one endpoint, different models.** `target` runs Grounding DINO and returns
ranked candidates; `points` runs SAM alone and returns the one region under the taps,
including negative points for what came along and should not have. They share a response
so the client has a single code path. Keeping them separate is not tidiness: a tap is
usually what a user reaches for _because_ words failed, and re-running the model that just
failed to understand them would cost 200 MB to give the same answer again. Measured on the
same picture: a phrase takes ~6.9 s, a tap 0.4-1.2 s.

**Ask only when the answer is shaky.** ADR-0003 measured 0.516 -> 0.832 on held-out
RefCOCOg from offering five candidates, and gates on the top-two score margin at 0.15.
Always-asking reaches 0.832 and was rejected: on 45% of phrases the detector scores 0.95
against 0.07, and a chooser there is friction for nothing. The client shows the region it
found and keeps the alternatives one tap away.

**The chosen mask travels with the job.** Re-grounding server-side would discard the
user's choice, pay for the expensive half of SAM twice, and can return a _different_ mask
the second time. It also means the worker skips grounding entirely — the detector and the
SAM encoder, about two seconds and 2 GB.

**Sizes.** Grounding runs at the worker's bounded working size, so a candidate's mask is
at most 2048 on its longest side and _not_ the upload's resolution. `EditSpec` therefore
accepts a mask at any size whose aspect matches the image and lets the worker scale it;
requiring an exact match only forced clients to upscale a mask the worker scales straight
back down. `decode_image` keeps full resolution for the _result_ (TD-021) — do not reuse
it for grounding, which regressed once exactly that way.

**Every mask on the wire is COCO RLE**: column-major, opening with a run of zeros, so odd
runs are the set ones. Interchangeable with the wider tooling ecosystem, and transposed if
you read it row-major.

## Serving an image

`GET /v1/images/{digest}` takes **either** a session or a signature in the query string.
The signature is what an `<img src>` can carry: HMAC-SHA256 over `digest:expires`, granting
one digest until one expiry and nothing else. Before it, every picture was fetched by
script and wrapped in an object URL — a copy of each image in the tab until something
revoked it, the whole download in front of the first paint, and the browser's own cache
made useless.

Expiry is checked separately from the comparison, because an expired signature is still
_arithmetically_ valid; the comparison is constant-time, because the obvious one leaks a
signature a byte at a time. `EDITGPT_URL_SIGNING_KEY` must be set wherever there is more
than one process or a restart policy — otherwise each process signs with its own key and
`/ready` says so.

## Reading the instruction

`POST /v1/plan` turns a sentence into an operation, or into a question. Rules first,
`editgpt_planner.rules`; the model — Gemini Flash, constrained to a schema derived from
`Intent` — sees only what they decline. Measured over 55 labelled instructions: rules
answer **45% at 0.007 ms and 1.000 accuracy**, the model answers the rest at 4.36 s median
and 201 tokens, and refusals are 1.000. `benchmarks/planner.py` is where those numbers come
from and where they are re-checked.

Planning is a **separate endpoint from job creation** on purpose: the client shows what was
understood before anything is spent, so "I meant the other car" costs a correction rather
than an edit. The plan is applied to the same chips and fields the flow already reads —
there is no second way to describe an edit.

Three rules that are load-bearing. The planner depends on `core` alone, so nothing that
plans has to carry an imaging stack. Which operations exist is _passed in_ (`app.OPERATIONS`)
rather than imported, because that is a fact about the models package. And planning keeps a
ten-second deadline of its own rather than the job's sixty: measured, the same model has
answered in 2.7 s and in 37.8 s, and past ten the rules-and-ask path is the better answer.

## The tool boundary

`apps/mcp` serves the vision capability as MCP tools, and it is **a client of the gateway,
not a second copy of it**: every tool is an HTTP call to the same API the web app uses, so
that process holds no models, no database and no credentials beyond a session token. That
is what makes "the orchestrator imports no model code" literally true, and it means an
agent gets the same auth, rate limits and `/ready` degradations a browser does.

Three tools are listed at start-up — `capabilities`, `plan_instruction`, `enable_toolset` —
and the rest appear only when asked for (`grounding`, `editing`). A manifest of everything
is tokens an agent pays for on every turn whether or not it edits a picture.

**Results carry references, never pixels.** A candidate comes back as score, box, area and
an id; the run-length encoding of a 15 MP selection is tens of thousands of integers, and
an agent that receives one pays for it on every subsequent turn while being no better able
to act. The A2A mesh this was once part of is cut — see `docs/PLAN.md` §5.

## Act, judge, decide

Two loops, stacked, and they must not become copies of each other.

**Inside one erase** (`editgpt_models.pipeline`): propose a pass, score it photometrically
against the same region, keep it or roll it back. It asks _did pass two beat pass one_, and
`Thresholds.escalate_cost` / `accept_cost` are its decision points.

**Around the whole edit** (`editgpt_models.critic` + `editgpt_core.review`): did the edit do
what was asked? Three checks — did anything change, does the fill agree with its
surroundings, and **is the target still detectable in the result**. The last is the only one
that knows what the user said, and the only one that can tell a beautiful fill of the wrong
region from a correct edit. It costs a model swap, so `editors._review` pays for it only
while a retry is still affordable: a check nobody can act on is seven seconds spent to feel
informed.

The outer loop's one lever is the **selection**, which every pass takes as given — and it is
only its lever when the _model_ chose the region. A brushed mask is not ours to widen;
`decide(..., can_widen=...)` is asked rather than inferred for exactly that reason. Out of
budget returns the best attempt with its verdict attached; out of levers hands the job back
with what is wrong and what the user can do about it.

`JobState.REVIEW` was a label the pipeline passed through on its way to `DONE`. This is what
it now describes.

## What the boundary removes

An upload is scrubbed of metadata before it is stored, in `apps/gateway/src/editgpt_gateway/scrub.py`, and the
digest is taken from the scrubbed bytes so what is stored and what is served are the same
thing. It used to be kept byte for byte: a phone photograph in this repository's own
fixtures carried make `iQOO`, model `iQOO Neo7` and the second it was taken, and one taken
outdoors would have carried the coordinates.

**Nothing is re-encoded that does not have to be.** An image with no metadata is returned
untouched; JPEG, PNG and WebP are edited at the segment or chunk level with the compressed
data never decoded. Measured on a 15.9 MP photograph: 19 EXIF tags removed, pixels
bit-identical, 640 KB smaller because the embedded thumbnail went with them. Only a
container this cannot edit in place _and_ that carries metadata is re-encoded, and it says
so in the log.

**ICC profiles and JFIF density are kept.** They describe how to interpret the pixels, not
who took them; dropping them changes what the picture looks like.

## Data stores

| Store             | Responsibility                                             | Status                                                          |
| ----------------- | ---------------------------------------------------------- | --------------------------------------------------------------- |
| Object storage    | uploaded images and artifacts, content-addressed by digest | live: local disk by default, any S3 endpoint via the `s3` extra |
| Postgres          | users, images, jobs, job steps, artifacts, cost ledger     | live; Alembic in `packages/store/migrations`                    |
| Redis             | Celery broker and the per-job progress channel             | live                                                            |
| Local model cache | ONNX weights, `~/.cache/editgpt/models` (~552 MB)          | live                                                            |
| `evals/photos`    | golden fixtures, committed                                 | live                                                            |

## What the store keeps, and for how long

Content-addressed bytes have no owner and no expiry of their own — the name is the
content, and nothing about a digest says whether anybody still wants it. Two rules,
`packages/store/src/editgpt_store/lifecycle.py`:

- **Orphans** — objects no `images` or `artifacts` row names — go once past
  `EDITGPT_ASSET_GRACE_HOURS`. They are invisible to every query by definition, so only a
  scan of the store finds them; that is what `AssetStore.scan` is for. The grace period
  exists because an upload whose row is still being committed is indistinguishable from
  an orphan for exactly as long as that takes.
- **Expiry** — `EDITGPT_ASSET_RETENTION_DAYS`, **off by default**. It deletes bytes, not
  rows: a job's history stays readable and fetching its result answers 404.

Fail closed: a sweep with no database raises rather than treating every object as an
orphan. `verdict()` is pure and both a dry run and a real run call it, so the report
cannot promise one thing and do another.

## External integrations

| Service                 | Role                          | Failure behaviour                          |
| ----------------------- | ----------------------------- | ------------------------------------------ |
| Cloudflare Workers AI   | generative fill for additions | circuit breaker, then next provider        |
| Google AI Studio (text) | intent parsing and critique   | planned                                    |
| Hugging Face Hub        | one-time model download       | fails loudly at setup, not at request time |

No credential values appear in this repository. Names of required variables are in
`.env.example`; how to obtain them is in `docs/RUNBOOK.md`.

### When the generative lane cannot be reached

`ADD`, `REPLACE`, `RESTYLE` and `RETOUCH` need a provider; `REMOVE`, `UPSCALE` and
`BACKGROUND` finish on this machine (`editors.LOCAL_OPS`). A job in the first group is
refused **before `region_for` runs** — grounding is seconds of CPU and most of the memory
budget, and discovering afterwards that there was nowhere to send the result wastes all of
it. `ProviderChain.availability()` answers without making a call.

The chain is `lru_cache`d per worker process, because a circuit breaker rebuilt per job is
not a breaker — three failures in a row only stop the fourth call if the count outlives the
job.

Two failures, deliberately worded differently. `ProviderUnavailableError` means nothing was
tried: the message names the variable to set or the seconds to wait, and reaches the user
as written. `ProviderExhaustedError` means everything was tried and failed: its message
quotes somebody else's HTTP replies, so `tasks._reason` replaces it with a sentence and
lets the log keep the detail.

## Deployment

- **Local:** services in Docker Compose (redis, postgres); application on the host, so a
  benchmark and the stack do not contend for the same 8 GB.
- **CI:** GitHub Actions — `check.yml` per PR, `memory.yml` nightly with real weights.
- **Target:** frontend on a static host, gateway on a small instance, heavy model stages
  on a larger-memory host. Not yet deployed. `docs/DEPLOYMENT.md` measures what each
  process needs against what the free tiers still offer, and the shape of the answer is
  that **only the worker is hard**: 86 MB for the gateway against a 2200 MB ceiling for the
  worker, whose ~13 s of work and long idle is a scale-to-zero profile, not a fixed-instance
  one. Read it before writing any deployment configuration; it is a survey, not a decision.

## Security architecture

- **Input boundary is the gateway.** Uploads are bounded by size and megapixels;
  everything downstream may assume validated input.
- **Secrets** come from the environment via a single settings object. No `os.environ`
  reads scattered through handlers. `.env` is gitignored and write-blocked.
- **Untrusted text** — user prompts, and any text derived from an image — must never be
  concatenated into a system instruction.
- **Object keys are content digests**, not user-supplied paths, so path traversal is
  structurally impossible rather than filtered.
- **Authentication is Clerk, and it fails closed.** `editgpt_gateway.auth` verifies the
  session token, provisions a `users` row keyed by Clerk's subject, and returns the owner
  every route passes to the store. An absent, expired or malformed token is a 401 — never
  a fall back to the shared account. Only `session_token` is accepted. It is off when no
  secret key is configured, which is how tests run; `/ready` reports which mode is live.
- **Authorization is a filter, not a check.** The store selects by owner, so an
  unauthorised read returns 404 rather than 403 — a 403 confirms the id exists.
- **Object storage is vendor-neutral.** The adapter speaks the S3 API and takes its
  endpoint from configuration, which is what lets MinIO in a container serve as the
  verified implementation. No provider is baked in.

## Architectural invariants

Violating one of these is an `ARCHITECTURAL` change and needs an ADR.

1. **Dependency direction is one-way:** `core` ← `models` ← `providers` ← apps. `core`
   imports nothing of ours.
2. **`core` stays light.** Its `__init__` exports contracts only, so a process that
   speaks `EditSpec` never loads an imaging stack.
3. **No model loads in the web process.** Models live in workers.
4. **Images travel by reference.** Components exchange asset references and encoded masks,
   never pixel arrays.
5. **One heavy model resident at a time**, mediated by the model slot.
6. **Removal never routes to a remote provider** (ADR-0001).
7. **The gateway owns the input boundary.** Validation happens once, there.
8. **`spike/` is frozen.** Nothing imports it.
9. **Every pipeline decision is logged.** A pass that runs without a log record is
   invisible in production whatever it returns; `harness/observability.md` has the fields.
