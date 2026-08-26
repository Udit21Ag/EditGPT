# EditGPT — Master Plan

> Agentic, prompt-driven image editing. Upload an image, say _"remove the car"_ or _"add a moustache"_,
> optionally brush a region, get the edited image back — produced by a mesh of specialised agents
> talking over A2A, calling MCP tools, under an orchestrator, on an 8 GB laptop.

**Author:** f20230708@pilani.bits-pilani.ac.in · **Started:** 2026-08-25
**Hard constraint:** 8 GB unified memory (MacBook Air M1). Every architectural decision below is downstream of this.

---

## 0. The one decision that shapes everything

You cannot host a modern generative image model on 8 GB. SDXL-inpaint needs ~8 GB _by itself_; FLUX Fill needs 12–24 GB.
Trying to is the single fastest way to kill this project.

So EditGPT splits editing into two classes and routes between them:

| Class                            | What it does                                                            | Where it runs                                                       | Why                                                                 |
| -------------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------- |
| **Discriminative / restorative** | find the object, cut the mask, erase and fill plausibly, matte, upscale | **Local**, ONNX Runtime on CPU/CoreML, one model resident at a time | Small (25–250 MB), fast, free, no rate limit, no network            |
| **Generative**                   | _add_ a moustache, _replace_ the sky, restyle                           | **Remote free-tier API**                                            | Needs a billion-parameter model. The free options are weak but real |

This is not a compromise — it is the right industrial architecture, and Phase 0 proved it the hard way.
"Remove the car" is an **eraser** problem, not a diffusion problem: asked to erase the same car, the free
generative lane produced a stone slab, then a white car, then a boulder, while the local erasers removed it
cleanly in 0.3 s. Routing that decision is exactly what an agent should be doing.

### Local model roster — **measured, Phase 0**

> Superseded the original estimates. Every number below was measured on this machine;
> see [ADR-0001](adr/0001-model-routing.md) and `spike/out/report.md`.

| Model                        | Job                            | On disk |          Peak RSS |                      Warm p50 | Tier |
| ---------------------------- | ------------------------------ | ------: | ----------------: | ----------------------------: | ---- |
| **MobileSAM** (enc+dec ONNX) | click/box/brush → precise mask |   44 MB |            620 MB | 25 ms decoder, 0.41 s encoder | 1    |
| **CLIPSeg-rd64-refined**     | text → seed mask               | ~150 MB | 1188 MB _(torch)_ |                        0.22 s | 1    |
| **MI-GAN pipeline v2**       | **primary eraser**             |   28 MB | ~1150 MB @15.9 MP |                   0.29–0.75 s | 1    |
| **Big-LaMa** (Carve ONNX)    | **escalation eraser**          |  208 MB |            964 MB |                        2.82 s | 1    |
| BiRefNet-lite / RMBG-2.0     | background matting             | ~180 MB |          untested |                             — | 2    |
| Real-ESRGAN x2               | restore resolution             |   65 MB |          untested |                             — | 3    |

**Two erasers, not one, because they fail differently.** MI-GAN (Picsart, ICCV'23) erases a
bird against flat sky perfectly where LaMa leaves a white ghost; LaMa is clean on a mug where
MI-GAN produces a crumpled artifact. MI-GAN is 7× smaller, 40× faster to load, 3.6× faster at
full resolution, and its ONNX graph crops internally — so it is the default, and LaMa is the
escalation when the fill scores poorly.

**GroundingDINO-tiny was dropped:** CLIPSeg localised 10/11 unaided, so it earns nothing.

### Remote providers — **free tier, verified 2026-08-25**

| Priority | Provider                  | Model                                           | Free allowance              | Used for                              |
| -------- | ------------------------- | ----------------------------------------------- | --------------------------- | ------------------------------------- |
| 1        | **Cloudflare Workers AI** | `@cf/runwayml/stable-diffusion-v1-5-inpainting` | 10,000 neurons/day, no card | **Additions only** (`add`, `replace`) |
| 2        | **Google AI Studio**      | `gemini-3.6-flash` (text)                       | free tier active            | Intent parsing, critic reasoning      |
| —        | ~~Gemini image models~~   | ~~Nano Banana / 3.1 Flash Image~~               | **none — `limit: 0`**       | Withdrawn Dec 2025                    |

> **Removal never goes remote.** Cloudflare's SD-1.5-inpainting fills a masked hole with an
> object matching the prompt, every time: asked to erase a car it produced a stone slab, then
> a _white car_, then a boulder. LaMa and MI-GAN beat it outright, so there is nothing to
> escalate to. The router is by **operation**, not by difficulty.

> **Should you use ImageKit?** For _editing_, no — it does deterministic transforms
> (crop/resize/format/watermark), not semantic edits, and its AI add-ons are paid. Use
> **Cloudflare R2** (10 GB, S3 API, **zero egress fees**) as the object store and
> `sharp`/`pillow-simd` for deterministic ops. Optionally put **ImageKit free (20 GB
> bandwidth)** in front of R2 purely as a CDN + thumbnail layer. Storage layer, not editing layer.

### Deployment insight that buys you 16 GB for free

**Hugging Face Spaces free CPU tier gives 2 vCPU / 16 GB RAM, indefinitely.** Host the heavy
local agents (Perception + Erase) there as a Docker Space exposing an A2A endpoint. Phase 0
measured one pipeline at 1.47–1.85 GB resident, so on an 8 GB laptop this offload is
**load-bearing, not an optimisation**.

## 1. Architecture

### 1.1 Agent mesh (multi-level, multi-skilled)

```
                    ┌──────────────────────────────────────┐
   Next.js  ──SSE──▶│  API Gateway (FastAPI)               │
                    │  auth · quota · job intake · stream  │
                    └───────────────┬──────────────────────┘
                                    │ enqueue (Celery/Redis)
                    ┌───────────────▼──────────────────────┐
                    │  L0  ORCHESTRATOR  "Director"        │
                    │  plans a DAG, delegates over A2A,    │
                    │  owns budget/retries/state machine   │
                    └──┬─────┬─────┬─────┬─────┬─────┬─────┘
       A2A (JSON-RPC 2.0 + SSE, signed Agent Cards)
          ┌────────┘     │     │     │     │     └────────┐
   ┌──────▼─────┐ ┌──────▼───┐ ┌▼─────────┐ ┌▼──────────┐ ┌▼─────────┐
   │ L1 Intent  │ │L1 Percep-│ │L1 Edit   │ │L1 Composi-│ │L1 Critic │
   │ (LLM)      │ │tion      │ │(router)  │ │tor        │ │(QA loop) │
   └──────┬─────┘ └────┬─────┘ └────┬─────┘ └─────┬─────┘ └────┬─────┘
          │  MCP       │  MCP       │  MCP        │  MCP       │  MCP
     ┌────▼────┐  ┌────▼─────┐ ┌────▼─────┐  ┌────▼─────┐ ┌────▼─────┐
     │ prompt  │  │mobilesam │ │lama.erase│  │blend/    │ │clip_score│
     │ _parse  │  │clipseg   │ │gemini    │  │feather/  │ │artifact_ │
     │ _classify│ │gdino     │ │flux_fill │  │paste_back│ │detect/nsfw│
     └─────────┘  └──────────┘ └──────────┘  └──────────┘ └──────────┘
                                    │
                          ┌─────────▼──────────┐
                          │ L2 ProvenanceAgent │  audit log, cost ledger,
                          │  (cross-cutting)   │  C2PA-style edit manifest
                          └────────────────────┘
```

**Agent responsibilities**

| Agent               | Skills (A2A `skills[]`)                                                                   | Owns                                                                               |
| ------------------- | ----------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| **Orchestrator**    | `plan`, `execute_dag`, `replan`, `cancel`                                                 | Job state machine, budget (time/$/retries), agent discovery                        |
| **IntentAgent**     | `parse_edit`, `disambiguate`, `expand_prompt`                                             | Prompt → typed `EditSpec`. Asks the user a clarifying question when confidence < τ |
| **PerceptionAgent** | `mask_from_text`, `mask_from_points`, `mask_from_brush`, `refine_mask`, `describe_region` | All segmentation/grounding models. Never returns pixels — returns RLE masks        |
| **EditAgent**       | `erase`, `generative_edit`, `replace_bg`, `upscale`                                       | The **local-vs-remote routing decision**, provider failover, retry with backoff    |
| **CompositorAgent** | `feather`, `color_match`, `poisson_blend`, `paste_back`, `tile_plan`                      | Full-res reconstruction from a 1024px working canvas; the "invisible seam" work    |
| **CriticAgent**     | `score_edit`, `detect_artifacts`, `safety_check`, `suggest_retry`                         | Closes the loop. Returns `accept / retry(params) / escalate / reject`              |
| **ProvenanceAgent** | `record`, `export_manifest`                                                               | Append-only edit history, reproducibility, cost accounting                         |

**Why A2A and not just function calls:** each L1 agent is an independently deployable process with its own
Agent Card at `/.well-known/agent-card.json`, its own model memory budget, and its own scaling story
(Perception → HF Space 16 GB; Intent → serverless). The orchestrator discovers skills at runtime rather than
importing them. That is the industrial argument, and it is genuinely true here because Perception physically
cannot live in the same process as the API on an 8 GB box.

### 1.2 The core contract

```python
# packages/core/editgpt_core/spec.py
class EditOp(StrEnum):
    REMOVE = "remove"; ADD = "add"; REPLACE = "replace"
    RESTYLE = "restyle"; BACKGROUND = "background"
    RETOUCH = "retouch"; UPSCALE = "upscale"

class EditSpec(BaseModel):
    op: EditOp
    target: str | None            # "the red car"  (None when brush-masked)
    content: str | None           # "a moustache"  (for ADD/REPLACE)
    mask_source: Literal["text", "brush", "point", "auto", "whole"]
    mask_ref: MaskRef | None      # RLE-encoded, by reference — never inline pixels
    image_ref: AssetRef           # r2://bucket/sha256  — images ALWAYS by reference
    constraints: Constraints      # preserve_identity, max_seconds, max_cost_cents, nsfw_policy
    confidence: float
```

**Images never enter the agent conversation.** Agents exchange `AssetRef` (content-addressed R2 key) + RLE masks +
a 256px thumbnail when a model genuinely needs to look. This one rule cuts token cost ~100× and keeps peak RSS flat.

### 1.3 Pipelines per operation

```
REMOVE      intent → mask(text|brush) → refine(MobileSAM, gated on its own iou)
            → pass 1 MI-GAN → score → pass 2 ALWAYS (escalate | residual | cross)
            → keep only if it verifies better → pass 3 if still unacceptable → critic
BACKGROUND  flood-fill the backdrop from the border → recolour (no model call)
            falls back to matte + composite when the border is not uniform
ADD         intent → mask over EMPTY space → Cloudflare SD-1.5-inpaint → critic
REPLACE     intent → mask → Cloudflare SD-1.5-inpaint (masked) → blend → critic
RESTYLE     deferred to v2 — no free instruction-editing model exists
UPSCALE     Real-ESRGAN local, no agents beyond orchestrator
```

Critic → `retry` re-enters at the failing node with mutated params (dilate mask, lower guidance, switch provider),
capped at 2 retries by the orchestrator's budget.

---

## 2. Progressive disclosure & lazy loading — made concrete

This is the project's technical differentiator, so it gets a real design, at four layers:

**L1 — Skill disclosure (A2A).** Agent Cards advertise only `{id, name, one-line description}` per skill.
Full JSON Schema for inputs/outputs is fetched on demand from `/skills/{id}/manifest` and cached with an ETag.
An orchestrator prompt stays ~800 tokens instead of ~9 000.

**L2 — Tool disclosure (MCP).** Each MCP server exposes a small tier-1 toolset plus a meta-tool
`enable_toolset(name)`. Heavy toolsets (`grounding`, `matting`, `upscale`) materialise only after the agent asks —
the same pattern good agent tooling uses for deferred capabilities. Tool _results_ are also
progressive: return
`{mask_ref, area_px, bbox, preview_url}`, never the mask array.

**L3 — Weight lazy-loading (the memory-critical one).**

```python
class ModelSlot:
    """At most N_RESIDENT heavy models alive process-wide."""
    def __init__(self, max_resident=1, idle_ttl=90, rss_ceiling_mb=1400): ...

    async def acquire(self, key: str) -> InferenceSession:
        # 1. cache hit → bump LRU, return
        # 2. RSS watchdog: if psutil RSS > ceiling → evict LRU, gc, malloc_trim
        # 3. semaphore(max_resident) → ort.InferenceSession(path, providers=[...])
        # 4. schedule idle-unload timer
```

Plus: `ort.SessionOptions.enable_mem_pattern`, `arena_extend_strategy=kSameAsRequested`, int8 dynamic quantisation
at build time, `mmap` weight files, and a hard rule — **models load in Celery workers only, never in the web process.**

**L4 — Context disclosure.** The orchestrator LLM sees a rolling window: current node + last critic verdict +
compact DAG state. Completed node payloads are swapped to Redis and referenced by id.

**Acceptance test for this whole section:** a `pytest` that runs a 12-step mixed workload and asserts
`max(RSS) < 1.6 GB` and that model count resident never exceeds 1. If it fails, CI fails.

---

## 3. Tech stack

**Frontend** — Next.js 15 (App Router, TS strict) · Tailwind v4 · **shadcn/ui** · Zustand · TanStack Query ·
`react-konva` canvas for brush/eraser/lasso masking with touch + pinch-zoom · SSE for live job progress ·
`next-themes` · Framer Motion. Mobile-first responsive; the mask brush must work with a finger.

**Backend** — Python 3.12 + **uv** · FastAPI · **a2a-sdk** (JSON-RPC 2.0 + SSE) · **FastMCP** for tool servers ·
Pydantic v2 everywhere · **Celery + Redis** (`--concurrency=1`, separate `cpu_heavy` queue; ARQ is a lighter
drop-in if Celery's RSS annoys you) · SQLModel + Postgres.

**Infra (all free tier)** — Clerk (auth, 10k MAU) · Cloudflare R2 (storage) · Upstash Redis · Neon Postgres ·
Vercel (frontend) · **Hugging Face Spaces Docker** (heavy agents, 16 GB) · Fly.io or Render (gateway) ·
Sentry · **Langfuse** (agent tracing) · GitHub Actions.

**Quality** — ruff (lint+format) · mypy strict · pytest + pytest-asyncio + hypothesis + respx · coverage ≥80% gate ·
ESLint + Prettier + `tsc --noEmit` · Vitest + Testing Library · Playwright E2E · pre-commit · Renovate ·
commitlint + Conventional Commits · Docker multi-stage + compose · OpenTelemetry.

### Dev memory budget (8 GB, the thing that must not be violated)

> Revised against Phase 0 measurements. The original 1.2 GB "model slot" was too small by
> roughly half: one full pipeline resident measures **1.47–2.12 GB** across repeated runs.
> Peak RSS varies ±15% run to run, so the budget is set from the worst case observed.

| Process                                              | Budget      |
| ---------------------------------------------------- | ----------- |
| macOS + Chrome (3 tabs)                              | 3.50 GB     |
| Next.js dev server                                   | 0.70 GB     |
| FastAPI gateway                                      | 0.25 GB     |
| **Celery worker, one pipeline resident**             | **2.20 GB** |
| Redis + Postgres (native brew, _not_ Docker Desktop) | 0.25 GB     |
| Headroom                                             | 1.10 GB     |

`ModelSlot(max_resident=1)` means one **heavy** model at a time — MobileSAM's embedding is
computed once per image, cached, and released before an eraser loads. Measured worst case is 2116 MB
(native 15.9 MP), which is why the budget is 2.2 GB and why the Hugging Face Space offload is
load-bearing. The Phase 2 RSS regression test must assert against 2.2 GB **over repeated runs** —
a single sample would have set this limit 20% too low.

**The single largest saving available is exporting CLIPSeg to ONNX int8** (Phase 4): 1188 MB of
RSS for a 150 MB model is torch overhead, nothing more.

**Rules:** use **OrbStack** or `colima` instead of Docker Desktop — that alone is ~1.5 GB. Never
run the full compose stack, the Next.js dev server and a model benchmark at once; `make dev-lite`
exists for this. Docker is for CI and prod parity, not daily local dev.

## 4. Repository layout

```
EditGPT/
├── AGENTS.md                    # harness: repo-wide agent instructions, a map
├── Makefile                     # make check | dev | dev-lite | bench | eval
├── apps/
│   ├── web/                     # Next.js 15
│   └── gateway/                 # FastAPI: auth, upload, jobs, SSE
├── agents/                      # one A2A server per dir, own Dockerfile + agent-card.json
│   ├── orchestrator/ intent/ perception/ edit/ compositor/ critic/ provenance/
├── mcp/                         # FastMCP servers
│   ├── vision_tools/ edit_tools/ quality_tools/
├── packages/
│   ├── core/                    # EditSpec, AssetRef, MaskRef, errors, tracing
│   ├── models/                  # ModelSlot, ONNX wrappers, download+quantise scripts
│   └── storage/                 # R2 client, content-addressing, signed URLs
├── evals/                       # golden images, prompts, scorers, regression report
├── infra/                       # compose, Dockerfiles, fly/HF-Space configs
└── docs/                        # PLAN.md, ADRs, API.md, RUNBOOK.md
```

---

## 5. Phased plan

Sizing assumes solo work with heavy agent assistance. "Exit" = the objective gate that must pass before moving on.

### Phase 0 — Feasibility spike (3–4 days) · ✅ **COMPLETE** — see [ADR-0001](adr/0001-model-routing.md)

Prove the models work on **your** machine before designing around them.

1. `uv` scratch project; download Big-LaMa, MobileSAM, CLIPSeg; export/quantise to ONNX int8.
2. Benchmark each: cold-load time, warm latency @512/1024, **peak RSS** (`psutil` + `memory_profiler`), quality eyeball on 10 photos.
3. Get a Gemini AI Studio key; do a raw "add a moustache" instruction edit; measure latency and per-request quota burn.
4. Write `docs/adr/0001-model-routing.md` recording the numbers.

**Exit:** a table of real measured numbers; LaMa erase < 6 s @1024 on CPU under 1 GB RSS; Gemini edit round-trip < 15 s.
_If LaMa blows the budget, tile at 512 and re-measure before proceeding._

### Phase 1 — Foundation & setup (3–4 days) · ✅ **COMPLETE** — 26 Aug 2026

- Monorepo (pnpm workspaces + uv workspace), Python 3.12, Node 20.
- `ruff`, `mypy --strict`, `pytest`, ESLint, Prettier, `tsc`, Vitest, pre-commit, commitlint.
- One `Makefile` target that is the whole truth: **`make check`** = ruff + mypy + pytest + eslint + tsc + vitest.
- GitHub Actions: `check` on every PR, matrix (py3.12, node20), cached uv/pnpm, coverage upload, branch protection.
- `docker compose` skeleton: redis, postgres, gateway. Health endpoints from day one.
- `packages/core` with `EditSpec` + a passing schema round-trip test.

**Exit:** `make check` green locally and in CI on an empty-but-real skeleton.

### Phase 2 — 🔧 Harness engineering (3–4 days) · ✅ **COMPLETE** — 26 Aug 2026

Build the operating rules a coding agent works inside, before writing product code. The
harness is deliberately **tool-agnostic**: plain Markdown and a Python checker, so it
survives a change of editor or assistant.

**2.1 `AGENTS.md` — a map, not a manual.** Project identity, the source-of-truth
hierarchy, verified commands, operating rules, change classification, and routing into
`harness/`. Capped at 200 lines by the checker; detail belongs in `harness/`, loaded on
demand. Directory-scoped `AGENTS.md` files carry only what applies to that directory.

**2.2 Deterministic verification loop.** `make check` must be complete and fast (<90 s),
with `make check-fast` for the inner loop. Work that can self-verify does not arrive broken.

**2.3 The `harness/` documents.** Nine files, each declaring **when to read it**, **what
it solves** and **what authority it carries**, so only what a task needs gets loaded:
`architecture` · `code_generation` · `testing` · `looping` · `observability` ·
`docker_log_analysis` · `self_updation` · `human_in_the_loop` · `tech_debt_tracker`.

**2.4 Execution plans (`harness/exec-plans/`).** For work spanning components or sessions:
goal, constraints, tasks with a change class each, verification, decisions and progress.
Completed plans keep their dead ends, which is the most valuable thing in the directory.

**2.5 Technical debt register.** Not a changelog — deferred problems only, each with the
measurement behind it and a **trigger** for when it must be addressed. An item without a
trigger never gets paid down, and the checker enforces that.

**2.6 Harness integrity is enforced, not aspirational.** `scripts/check_harness.py` runs in
`make check` and CI, failing the build on broken links, missing paths, invented `make`
targets, credential-shaped strings, models in the registry but absent from the
architecture doc, operations the gateway advertises with no eval case, open debt with no
trigger, and stale execution plans. A rule nobody can check is a wish.

**2.7 The eval harness (`evals/`).** Golden (image, prompt, op) cases across the supported
operations, with scorers for mask agreement, fill quality and latency. `make eval` prints a
table and writes `evals/out/report.json`. This is what makes "did that change actually
improve things?" answerable instead of vibes.

**2.8 Held-out benchmarks (`benchmarks/`).** `evals/` is the dev set and its thresholds were
fitted there, so it cannot answer whether anything generalises. External datasets with their
own ground truth answer that instead.

**Exit:** a fresh session, given only `AGENTS.md`, can find its way to the right rules, make
a change, and verify it — with no human hand-holding.

### Phase 3 — Contracts & job pipeline (4–5 days)

Clerk auth on the gateway · R2 content-addressed upload with signed URLs + MIME/size/dimension validation ·
Postgres schema (`users, images, jobs, job_steps, artifacts, cost_ledger`) + Alembic ·
Celery job lifecycle `queued→planning→running→review→done|failed|cancelled` · SSE progress stream ·
idempotency keys · per-user rate limits and quota. **No AI yet** — a `noop_edit` task proves the whole pipe end-to-end.

**Exit:** upload → job → SSE progress → result URL, fully tested, with a stub editor.

### Phase 4 — Perception layer (5–6 days)

`ModelSlot` + RSS watchdog (with its memory test) · MobileSAM (brush/point/box → mask) · CLIPSeg (text → mask) ·
mask ops: dilate, feather, RLE encode/decode, bbox, area-sanity · `mask_from_text` ambiguity detection →
returns candidates for the UI to disambiguate · vision_tools MCP server with tiered disclosure.

**Exit:** `mask_from_text("the car")` beats 0.7 IoU on the eval subset; RSS test green.

### Phase 5 — Editing layer (5–6 days)

MI-GAN as the primary eraser (its ONNX graph crops internally — no tiling needed) · Big-LaMa as the
escalation path with 512-crop planning and full-res paste-back · `fill_metrics` scoring to choose between
them at runtime · Cloudflare Workers AI adapter for additions · **provider abstraction with circuit breaker,
exponential backoff, quota tracking and deterministic fallback order** (prototyped in Phase 0 as
`spike/bench/providers.py`) · cost ledger writes · compositor (feather, **Lab chroma-only** colour match,
object-relative dilation at 5% of the longest side).

Carry the two Phase 0 fixes over verbatim — they are what made i8 work: match chroma but **not** luminance,
and scale dilation with the object rather than using a pixel constant.

**Exit:** "remove the car" and "add a moustache" both work end-to-end from a CLI script.

### Phase 6 — Agent mesh (6–7 days)

Wrap each capability as an **A2A server**: Agent Card, executor, task lifecycle, SSE streaming, health.
Implement the progressive-disclosure skill manifests. Agent registry + discovery with card caching.
Auth between agents (shared-secret JWT, and validate cards). Contract tests per agent; `a2a-conformance` subagent in CI.

**Exit:** every capability reachable only via A2A; the orchestrator imports no agent code.

### Phase 7 — Orchestrator & critic loop (5–6 days)

LLM planner producing a typed DAG (constrained decoding to a Pydantic schema, not free-form JSON) ·
deterministic fast-paths for the 7 known ops (LLM only plans the ambiguous ones — cheaper and far more reliable) ·
budget enforcement (wall-clock, retries, cents) · CriticAgent scoring + bounded replan · cancellation · Langfuse tracing.

**Exit:** critic-triggered retry demonstrably fixes a seeded failure case in the eval set.

### Phase 8 — Frontend (7–9 days)

Landing · Clerk auth · upload (drag/drop, paste, camera on mobile) · **canvas editor**: pan/zoom, brush/eraser with
size slider, lasso, "magic select" (tap → MobileSAM), mask overlay opacity, undo/redo · prompt bar with op chips and
example prompts · live progress with per-agent step timeline (this is your demo money-shot) · before/after slider ·
version history & branch-from-version · download/share · quota display · full a11y pass · mobile layout with
bottom-sheet tools. Playwright E2E for the two headline flows.

**Exit:** Lighthouse ≥90 across the board on mobile; both flows pass E2E.

### Phase 9 — Hardening & observability (4–5 days)

Prompt-injection defence on image-derived text · NSFW/safety gate in the critic · PII/EXIF stripping ·
signed URL expiry + object lifecycle (auto-delete after N days) · OTel spans across agent hops with a shared trace id ·
structured JSON logs · Sentry · load test with `locust` (concurrency 1 worker — find the queue depth that breaks) ·
graceful degradation when every provider is quota-exhausted · **runbook**.

### Phase 10 — Ship (3–4 days)

Vercel deploy · agents to HF Spaces (Docker) + gateway to Fly.io · Upstash/Neon/R2 prod config · secrets via GH
environments · staging environment · README with architecture diagram + GIF demo · ADR index · API docs from OpenAPI ·
CONTRIBUTING · `make eval` baseline committed · v1.0.0 tag.

**Total: ~9–11 weeks solo part-time; ~5 weeks focused.**

---

## 6. Testing strategy

| Layer        | Tooling                 | What's actually asserted                                                                         |
| ------------ | ----------------------- | ------------------------------------------------------------------------------------------------ |
| Unit         | pytest, hypothesis      | mask RLE round-trips for any shape; tile-plan covers the image with no gaps; EditSpec validation |
| **Memory**   | pytest + psutil         | peak RSS < 1.6 GB over a 12-step workload; ≤1 resident model; no leak over 50 iterations         |
| Provider     | respx / VCR cassettes   | zero real API calls in CI; failover order; circuit breaker opens and recovers                    |
| A2A contract | a2a-sdk test client     | every Agent Card validates; every advertised skill is invocable; SSE task lifecycle correct      |
| MCP          | FastMCP in-proc client  | tier-1 tool list is small; `enable_toolset` reveals tier-2; results carry refs not pixels        |
| Integration  | docker compose + pytest | upload → job → SSE → artifact, with fake providers                                               |
| **Eval**     | `make eval`             | quality/latency/cost regression vs. `main`, commented on the PR                                  |
| Frontend     | Vitest + RTL            | mask store reducers, undo/redo, prompt parsing UI                                                |
| E2E          | Playwright              | remove-object and add-object flows, desktop + mobile viewport                                    |

Determinism: pin seeds, freeze time, snapshot masks as PNG hashes, and never let a test touch the network
(`pytest-socket --disable-socket`, with an explicit `@pytest.mark.live` opt-in tier run nightly).

---

## 7. Risks

| Risk                                           | Mitigation                                                                                                                                           |
| ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| **8 GB is exceeded anyway**                    | Phase 0 measures before designing; RSS test in CI; heavy agents relocate to HF Space 16 GB; tile at 512                                              |
| Free tier changes or dies                      | Provider abstraction + 3-deep failover from day one; quota tracked in the cost ledger; local LaMa always works offline                               |
| Text→mask picks the wrong object               | Confidence threshold → return candidates → user taps the right one. The brush is always the escape hatch                                             |
| Free generative quality is poor                | Measured and accepted: SD-1.5 handles additions weakly and cannot remove at all. Removal is local; enabling Gemini billing lifts the cap in one line |
| Cast shadows survive removal                   | **Known v1 limitation.** ObjectClear/OmniEraser solve it but are SDXL/FLUX-based and GPU-only                                                        |
| Agent mesh becomes over-engineered latency tax | Deterministic fast-paths for the 7 known ops; LLM planning only on ambiguity; every hop traced and budgeted                                          |
| Celery RSS overhead                            | `--concurrency=1 --max-tasks-per-child=10`; ARQ as the documented escape hatch                                                                       |
| Scope creep                                    | Phase exits are objective gates; anything not in the 7 ops is a v2 issue                                                                             |

---

## 8. Immediate next actions

1. `git init` + push an empty repo with branch protection.
2. **Phase 0 spike** — the benchmark table is the input to every later decision.
3. Get the Gemini AI Studio key and a Cloudflare R2 bucket now (both free, both 5 minutes).
4. Then Phase 1, then the harness in Phase 2 — and let the harness build the rest.
