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

| Component            | Owns                                                          |
| -------------------- | ------------------------------------------------------------- |
| `packages/core`      | contracts every component speaks; quality scoring             |
| `packages/models`    | model lifetime, grounding, erasure, compositing, pass routing |
| `packages/providers` | remote generation, failover, circuit breaking                 |
| `apps/gateway`       | HTTP surface: health, capabilities, (later) upload and jobs   |
| `apps/web`           | browser UI: upload, region selection, progress                |
| `evals`              | the golden set and its runner                                 |

## Diagram

```
image + (brush | box | text)
        │
        ▼
   INTENT ─────────► EditSpec                 packages/core/spec.py
        │            rejects unactionable work at construction
        ▼
   GROUNDING                                  packages/models/segment.py
        │  text ─► CLIPSeg heatmap ─► seed
        │  seed | box | brush ─► MobileSAM ─► mask + its own confidence
        │  refinement kept only if that confidence clears the gate
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
```

## Implemented vs planned

Stating this honestly is the point of the section.

| Stage      | Today                                | Planned                                     |
| ---------- | ------------------------------------ | ------------------------------------------- |
| Intent     | `EditSpec` constructed by the caller | agent parses free text, asks when ambiguous |
| Router     | a function in `pipeline.py`          | an agent behind A2A                         |
| Multi-pass | automatic, capped at 3               | user-triggered "needs more work"            |
| Critic     | scoring inline                       | an agent with a retry budget                |
| Transport  | direct function calls                | A2A JSON-RPC + SSE between processes        |
| Storage    | local files                          | object storage, content-addressed           |
| Jobs       | none                                 | queue + worker + progress stream            |

**There is no agent mesh yet.** The pipeline is a library. Splitting it into processes
changes deployment, not shape — which is what `EditSpec` is for.

## Models

Keys are registry keys (`packages/models/src/editgpt_models/registry.py`), where the measured
peak resident set for each lives. `make models` fetches all of them (~348 MB).

| Key                           | Model          | Role                                           |    Disk |
| ----------------------------- | -------------- | ---------------------------------------------- | ------: |
| `sam-encoder` / `sam-decoder` | MobileSAM      | box, point or seed to a precise mask           |   44 MB |
| —                             | CLIPSeg-rd64   | free text to a coarse seed (torch; see TD-001) | ~150 MB |
| `migan`                       | MI-GAN         | primary eraser, fast path                      |   28 MB |
| `lama`                        | Big-LaMa       | escalation eraser                              |  208 MB |
| `esrgan-x2`                   | Real-ESRGAN x2 | tiled resolution enhancement                   |   67 MB |

| — | SD-1.5-inpainting | additions and replacements (remote) | — |

Two erasers ship because they fail differently: one erases a small object on a flat
background where the other leaves a ghost, and loses on large objects over texture. The
router picks by measurement rather than by preference.

## Data stores

| Store             | Responsibility                                             | Status                        |
| ----------------- | ---------------------------------------------------------- | ----------------------------- |
| Object storage    | uploaded images and artifacts, content-addressed by digest | planned                       |
| Postgres          | users, images, jobs, job steps, artifacts, cost ledger     | container runs; no schema yet |
| Redis             | job queue and progress channel                             | container runs; unused        |
| Local model cache | ONNX weights, `~/.cache/editgpt/models` (~348 MB)          | live                          |
| `evals/photos`    | golden fixtures, committed                                 | live                          |

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
- Authentication and authorization are not implemented. Do not assume they are.

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
