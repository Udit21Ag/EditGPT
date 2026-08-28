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
usually what a user reaches for *because* words failed, and re-running the model that just
failed to understand them would cost 200 MB to give the same answer again. Measured on the
same picture: a phrase takes ~6.9 s, a tap 0.4-1.2 s.

**Ask only when the answer is shaky.** ADR-0003 measured 0.516 -> 0.832 on held-out
RefCOCOg from offering five candidates, and gates on the top-two score margin at 0.15.
Always-asking reaches 0.832 and was rejected: on 45% of phrases the detector scores 0.95
against 0.07, and a chooser there is friction for nothing. The client shows the region it
found and keeps the alternatives one tap away.

**The chosen mask travels with the job.** Re-grounding server-side would discard the
user's choice, pay for the expensive half of SAM twice, and can return a *different* mask
the second time. It also means the worker skips grounding entirely — the detector and the
SAM encoder, about two seconds and 2 GB.

**Sizes.** Grounding runs at the worker's bounded working size, so a candidate's mask is
at most 2048 on its longest side and *not* the upload's resolution. `EditSpec` therefore
accepts a mask at any size whose aspect matches the image and lets the worker scale it;
requiring an exact match only forced clients to upscale a mask the worker scales straight
back down. `decode_image` keeps full resolution for the *result* (TD-021) — do not reuse
it for grounding, which regressed once exactly that way.

**Every mask on the wire is COCO RLE**: column-major, opening with a run of zeros, so odd
runs are the set ones. Interchangeable with the wider tooling ecosystem, and transposed if
you read it row-major.

## Data stores

| Store             | Responsibility                                             | Status                                                          |
| ----------------- | ---------------------------------------------------------- | --------------------------------------------------------------- |
| Object storage    | uploaded images and artifacts, content-addressed by digest | live: local disk by default, any S3 endpoint via the `s3` extra |
| Postgres          | users, images, jobs, job steps, artifacts, cost ledger     | live; Alembic in `packages/store/migrations`                    |
| Redis             | Celery broker and the per-job progress channel             | live                                                            |
| Local model cache | ONNX weights, `~/.cache/editgpt/models` (~552 MB)          | live                                                            |
| `evals/photos`    | golden fixtures, committed                                 | live                                                            |

## External integrations

| Service                 | Role                          | Failure behaviour                          |
| ----------------------- | ----------------------------- | ------------------------------------------ |
| Cloudflare Workers AI   | generative fill for additions | circuit breaker, then next provider        |
| Google AI Studio (text) | intent parsing and critique   | planned                                    |
| Hugging Face Hub        | one-time model download       | fails loudly at setup, not at request time |

No credential values appear in this repository. Names of required variables are in
`.env.example`; how to obtain them is in `docs/RUNBOOK.md`.

## Deployment

- **Local:** services in Docker Compose (redis, postgres); application on the host, so a
  benchmark and the stack do not contend for the same 8 GB.
- **CI:** GitHub Actions — `check.yml` per PR, `memory.yml` nightly with real weights.
- **Target:** frontend on a static host, gateway on a small instance, heavy model stages
  on a larger-memory host. Not yet deployed.

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
