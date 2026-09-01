# EditGPT — Master Plan

> Agentic, prompt-driven image editing. Upload an image, say _"remove the car"_ or _"add a moustache"_,
> optionally brush a region, get the edited image back — planned by an orchestrator, executed
> through MCP tools, checked by a critic that can replan, on an 8 GB laptop.

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
    image_ref: AssetRef           # asset://bucket/sha256 — images ALWAYS by reference
    constraints: Constraints      # preserve_identity, max_seconds, max_cost_cents, nsfw_policy
    confidence: float
```

**Images never enter the agent conversation.** Agents exchange `AssetRef` (a content digest) + RLE masks +
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
├── apps/worker/                 # Celery: the job lifecycle and the edit step
├── mcp/                         # FastMCP: vision_tools (planned, Phase 6)
│                                # ~~agents/ — the A2A mesh, cut 1 Sep 2026~~
├── packages/
│   ├── core/                    # EditSpec, AssetRef, MaskRef, errors, tracing
│   ├── models/                  # ModelSlot, ONNX wrappers, download+quantise scripts
│   └── storage/                 # S3 client, content-addressing, signed URLs
├── evals/                       # golden images, prompts, scorers, regression report
├── infra/                       # compose, Dockerfiles, fly/HF-Space configs
└── docs/                        # PLAN.md, ADRs, API.md, RUNBOOK.md
```

---

## 5. Phased plan

Sizing assumes solo work with heavy agent assistance. "Exit" = the objective gate that must pass before moving on.

### Scope, revised 1 Sep 2026

**What this project is for:** demonstrating ML, agentic and distributed-systems engineering
under a hard device constraint, to people reading a CV — interviewers at Amazon, Flipkart,
Google and similar, for data science, SDE, distributed systems and AI/ML roles. It will
never serve a customer. Nobody will run it; someone may read it, and someone will ask about
it for forty minutes.

That changes what "finished" means. **A decision that was measured and written down is worth
more than a feature that works**, because the first can be defended in an interview and the
second cannot be seen. Two consequences, applied to everything below:

- **Operations stop earning.** Deploying, staging, secrets rotation, uptime, accessibility
  audits and Lighthouse scores produce nothing a reader can evaluate. The deployment *audit*
  ([DEPLOYMENT.md](DEPLOYMENT.md)) is kept and the deployment itself is cut: "here is what
  hosting this costs and why" is a better systems answer than a URL that resolves.
- **The agentic layer is the gap.** What exists today is an excellent job pipeline. The claim
  on the front of this document — orchestrator, planner, critic — is the part that is not yet
  built, and it is the part the target roles are hiring for.

Phases 6–10 are rewritten below against that. Phases 0–5 and 9 stand as delivered.

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

### Phase 3 — Contracts & job pipeline (4–5 days) · ✅ **COMPLETE** — 27 Aug 2026

Content-addressed upload with MIME/size/dimension validation at the gateway (the only input
boundary) · `packages/store`: asset store behind a protocol with local and S3 adapters ·
Postgres schema (`users, images, jobs, job_steps, artifacts, cost_ledger`) + Alembic ·
`apps/worker`: Celery lifecycle `queued→planning→running→review→done|failed|cancelled` ·
SSE progress stream that replays persisted steps then follows Redis · idempotency keys ·
per-client rate limits · cost ledger that records free calls too. **No AI** — a `noop`
editor proves the pipe end-to-end.

**Exit met:** upload → job → SSE progress → result digest, with a stub editor, under test.

**Two departures from the plan as written, both deliberate.**

*Object storage is not R2.* R2 asks for payment details even on its free tier. The adapter
now speaks the S3 API with the endpoint as configuration, so no vendor is baked in — and
**MinIO in a container is the verified implementation**, which needs no account at all and
is what the test suite and CI run against. Picking a hosted provider is a Phase 10
decision; `docs/RUNBOOK.md` lists which ones do not ask for a card.

*Clerk arrived after the fact, and cheaply.* The authorization half was built first —
identity decided in one place, every route passing it, the owner travelling in the queue
message — so wiring the provider touched **one function and no routes**. It fails closed,
provisions users on first request, and is off when no key is configured so tests and a
fresh checkout need no credential. See TD-018.

### Phase 4 — Perception layer (5–6 days) · partly delivered early

`ModelSlot` + RSS watchdog, MobileSAM, mask ops and the multi-pass policy all shipped in
Phase 0/1. Text grounding moved to Grounding DINO on 27 Aug 2026 —
[ADR-0002](adr/0002-text-grounding.md) — taking held-out RefCOCOg mIoU from 0.389 to 0.469
and removing torch from the shipping path.

What is left is the part that actually matters now: **ambiguity detection returning
candidates for the UI to disambiguate**. TD-015 measured that 36% of held-out phrases fail
below IoU 0.1 and that the figure is identical for both grounding models, because both
ground the noun and then pick an arbitrary instance. `detect()` already ranks every box;
surfacing the top few turns a wrong edit into one extra click. Plus the vision_tools MCP
server with tiered disclosure.

**Exit met, 27 Aug 2026.** `POST /v1/masks` returns ranked candidates and flags ambiguity;
verified live on six RefCOCOg samples with recorded narrow margins, five of which correctly
asked. `make memory` green with the detector resident, including a test that the detector
and an eraser are never resident together.

The vision_tools MCP server moves to Phase 6, where the agent mesh that needs it is built —
building a disclosure mechanism before anything discloses through it is guesswork.

### Phase 5 — Editing layer (5–6 days)

MI-GAN as the primary eraser (its ONNX graph crops internally — no tiling needed) · Big-LaMa as the
escalation path with 512-crop planning and full-res paste-back · `fill_metrics` scoring to choose between
them at runtime · Cloudflare Workers AI adapter for additions · **provider abstraction with circuit breaker,
exponential backoff, quota tracking and deterministic fallback order** (prototyped in Phase 0 as
`spike/bench/providers.py`) · cost ledger writes · compositor (feather, **Lab chroma-only** colour match,
object-relative dilation at 5% of the longest side).

Carry the two Phase 0 fixes over verbatim — they are what made i8 work: match chroma but **not** luminance,
and scale dilation with the object rather than using a pixel constant.

**Exit met, 27 Aug 2026 — and through the real queue, not a CLI script.** `editgpt_models.execute`
holds the one edit dispatch; the worker and the golden set both call it. Verified live:
upload → ground → job → SSE → an image with the car gone, 12.8 s end to end, 7.8% of pixels
changed. Cost-ledger writes land on every job.

The extraction was checked against the golden set with the *same* grounding on both sides:
all 11 removals identical on cost and pass sequence. Only `bbox_iou` moved, deliberately —
it now measures the raw mask rather than the dilated one, so it scores grounding rather than
the eraser's footprint.

### Phase 6 — Agent mesh · **cut to one MCP server (1 day)**

~~Wrap each capability as an A2A server: Agent Card, executor, task lifecycle, SSE streaming, health.
Agent registry + discovery with card caching. Auth between agents (shared-secret JWT).
Contract tests per agent; `a2a-conformance` subagent in CI.~~

Cut. Six agent servers, a registry, card caching and inter-agent JWT is a week of plumbing
whose entire output is one sentence — *"capabilities are reached across a boundary, not
imported"* — and that sentence is true of a single tool server too.

**What is kept:** one **MCP server over the vision tools** — ground, segment, erase — with
progressive disclosure, results carrying refs rather than pixels, and the orchestrator
importing no model code. That is the same architectural claim at a tenth of the cost, in a
protocol a reader recognises.

**Exit:** the orchestrator reaches perception through the MCP server only; the tier-1 tool
list stays small and `enable_toolset` reveals the rest.

### Phase 7 — Orchestrator & critic loop (3 days) · **next, and the priority**

The one part of the original design that has no substitute. Everything under it already
works; this is what makes it agentic rather than scripted.

**Intent → typed plan.** A free-text instruction becomes an `EditSpec` by **constrained
decoding into the Pydantic schema**, not by parsing free-form JSON out of a model's prose.
The schema is the contract that already exists; the planner fills it or fails validation.

**Deterministic fast paths.** The seven known operations never reach the LLM. "remove the
car" is decidable by rule, and routing around a model wherever the task is decidable is the
answer to *how do you make an LLM reliable and cheap* — the planner is for the ambiguous
remainder, and every fast-path hit is a request that cannot hallucinate, cost anything or
time out.

**Budgets, enforced.** Wall-clock, retry count and cents, checked before each hop and on
exhaustion — the ledger and `Constraints` already carry the fields.

**The critic.** Score the result with the metrics that already exist — `fill_metrics`, mask
coverage, the blank-frame check — and on a bad score **replan within a bound**: escalate
MI-GAN → LaMa, widen dilation, or fall back to returning candidates for disambiguation
rather than a confident wrong edit.

**Exit:** seeded failures in the eval set, and a number — *critic-triggered retry fixed k of
N* — plus the cost and latency the loop added. A critic that cannot be shown to fix
something is decoration.

**Delivered 1 Sep 2026.** The planner: rules answer in **0.007 ms median** over 1200 calls
against **2.7 s / 5.3 s / 37.8 s** for three live model calls, and the model is constrained
by a schema derived from `Intent` rather than asked for JSON. That 37.8 s is why planning
keeps its own ten-second deadline rather than the job's sixty.

The critic: three checks — did anything change, does the fill agree with its surroundings,
**is the target still detectable in the result** — and four actions, `accept · widen · ask ·
stop`. The semantic check is the one no photometric score can make, and it is paid for only
while a retry is still affordable. The outer loop's single lever is the selection, which
every pass takes as given, and it is only a lever when the *model* chose the region: an
existing test caught the first version letting the detector second-guess a hand-drawn mask.

**Wired and measured, 1 Sep 2026.** `POST /v1/plan` and an instruction box in the editor:
a sentence fills the chips and fields the flow already reads, and the page says *who*
answered — "matched a rule, no model was asked" is the fast-path claim made checkable by
the person using it. `benchmarks/planner.py` scores 55 labelled instructions across two
splits and found four rule defects in its first run; rule accuracy went 0.840 → 1.000, and
the run then exhausted the free tier's quota and scored eight unreachable rows as failures,
which is now excluded rather than counted.

**Exit not yet met:** the k-of-N number needs one run with real weights. The loop itself is
verified end to end against staged failures, and mutation-checked — forcing `can_widen` to
false fails two tests.

### Phase 8 — Frontend (7–9 days) · **in progress**

Landing · Clerk auth · upload (drag/drop, paste, camera on mobile) · **canvas editor**: pan/zoom, brush/eraser with
size slider, lasso, "magic select" (tap → MobileSAM), mask overlay opacity, undo/redo · prompt bar with op chips and
example prompts · live progress with per-agent step timeline (this is your demo money-shot) · before/after slider ·
version history & branch-from-version · download/share · quota display · full a11y pass · mobile layout with
bottom-sheet tools. Playwright E2E for the two headline flows.

**Delivered, 28–29 Aug 2026.** Landing and Clerk auth; upload; op chips; the **candidate
picker** ADR-0003 measured and nothing could reach; live SSE progress; result and download;
**brush/eraser** with size, undo/redo and clear, on pointer events so a finger and a mouse
take one code path. Two things had to be fixed first: the site had never built (Clerk Core
3 replaced the control components with stubs that throw at render) and had never run (Next
reads `.env` from `apps/web`, not the repository root).

**Verified live** against a real gateway, worker, Postgres and Redis. "the minaret" on the
Taj Mahal grounds ambiguously at margin 0.089; the detector's top pick is the right-hand
minaret, and choosing the second candidate erased the left-hand one — 96.0% of the chosen
region changed against 0.0% of the top-ranked one. A painted stroke erased 87.6% of itself
and 0.8% outside it.

**Delivered 29 Aug 2026**, continuing: **magic select** (tap → MobileSAM, with negative
points; ~0.5 s against ~6.9 s for a phrase, and it seeds the brush rather than replacing
it), drop/paste/camera upload, the **before/after wipe**, **version history with
branch-from-version**, the per-step timeline the worker was already publishing, and
example prompts. **Playwright, both headline flows passing** against a real browser, a real Clerk session,
the gateway, the worker and the models — 20.6 s for describe-confirm-edit and 17.2 s for
tap-and-erase. That is Phase 8's E2E exit criterion met.

It earned its cost on the first run that got far enough to try: **the gateway had no CORS
middleware at all**, so a browser's preflight was answered 405 by the router and every
cross-origin call was blocked before it was sent. The frontend could never have worked.
Nothing else could have caught it — curl does not preflight, and the component tests
replace `fetch`.

**Cut, 1 Sep 2026:** pan/zoom, lasso, mask-overlay opacity, quota display, the a11y contrast
and focus-order audit, the bottom-sheet mobile layout, and Lighthouse ≥90. The frontend's job
in this project is to make a ninety-second recording legible, and it already does — the
candidate picker, the brush, magic select, the step timeline and the before/after wipe are
what the recording shows.

**Exit (met):** both headline flows pass E2E against a real browser, a real Clerk session,
the gateway, the worker and the models.

### Phase 9 — Hardening & observability (4–5 days) · **in progress**

**Delivered 29 Aug 2026.** **EXIF/PII stripping** at the upload boundary, losslessly —
verified on a real 15.9 MP phone photograph: 19 tags gone, pixels bit-identical, 640 KB
smaller. **CORS**, which did not exist at all and which the browser suite found: an
explicit origin allowlist, credentials off, and `/ready` now reports a deployment that
kept the development defaults.

**Also delivered 30 Aug 2026.** **Signed image links** — `GET /v1/images/{digest}` takes a
signature instead of a session, so an `<img src>` loads a picture directly; every image
used to be fetched by script and wrapped in an object URL. **Structured JSON logs** with
correlation: thirty-one `extra={...}` call sites were rendering as bare messages because
nothing configured logging, and Celery was replacing the handlers on top of that. Every
gateway line now carries a request id, every worker line the job id, and credentials are
removed by the formatter rather than by remembering.

**Settled, not built:** [ADR-0004](adr/0004-content-safety.md). No content classifier —
the generative lane's provider already enforces a policy we cannot override, and the local
lane generates nothing to classify. Prompt-injection defence is not applicable until
something reads text out of an image.

**Also delivered 31 Aug 2026.** **Graceful degradation on the generative lane.** The
failover chain and circuit breaker written in Phase 5 were never called — the worker
constructed a bare provider, so both existed only in their own tests. They are wired in
now, and a job that needs a provider is refused *before* grounding rather than after:
`ProviderChain.availability()` answers without a call, so a missing key costs a
millisecond instead of a full detector-and-segmenter run. A user sees "set
CLOUDFLARE_ACCOUNT_ID…" or "retry in about 47s"; the raw HTTP reply stays in the log.

**Also delivered 31 Aug 2026.** **Object lifecycle.** Signed links expired; the objects
behind them did not. A sweep deletes orphans — objects no row names, which is every
upload that never became a job and which no query can find — and, where a deployment sets
a retention, the bytes of old referenced objects while their rows survive. Verified live:
two objects stored, one recorded; a dry run reported one orphan and deleted nothing, then
`APPLY=1` deleted exactly it. Fail-closed with no database, because without references
everything looks orphaned. **The locust load test**: 44 uploads and 11 jobs in 30 s
against a gateway with no worker behind it, upload median 3 ms and 79 ms at p90, and every
job correctly reported as never finishing — which is the queue-depth failure it exists to
show.

**Settled, not built:** OTel spans, [TD-025](../harness/tech_debt_tracker.md). A request
id is minted at the gateway, echoed on `X-Request-Id`, carried across the broker in the
task message and re-bound in the worker, so every line of both processes joins on it.
Spans buy the shape of a fan-out, and there is no fan-out until Phase 6's agent mesh.
Sentry is skipped by choice.

**Phase 9 exit met.**


Prompt-injection defence on image-derived text · NSFW/safety gate in the critic · PII/EXIF stripping ·
signed URL expiry + object lifecycle (auto-delete after N days) · OTel spans across agent hops with a shared trace id ·
structured JSON logs · Sentry · load test with `locust` (concurrency 1 worker — find the queue depth that breaks) ·
graceful degradation when every provider is quota-exhausted · **runbook**.

### Phase 10 — Ship · **redefined: the write-up, not a deployment (2 days)**

~~Vercel deploy · agents to HF Spaces (Docker) + gateway to Fly.io · staging environment ·
secrets via GH environments · CONTRIBUTING · v1.0.0 tag.~~ Cut. Nobody clicks the link.

**The audit is kept as the artifact.** [DEPLOYMENT.md](DEPLOYMENT.md), 1 Sep 2026, measured
what each process needs against what the free tiers still give: the gateway is 86 MB idle and
126 MB under uploads, while the worker holds a 2200 MB ceiling for thirteen seconds and then
nothing — a scale-to-zero profile against a market where every remaining free fixed instance
is 512 MB. Two of the three hosting choices this plan was written around had quietly stopped
being true. *That* is the deployment answer worth having in an interview; a running URL is
not, and would cost €5 a month to keep true.

**What is built instead:**

- **README** — the constraint in the first line, an architecture diagram, the results table,
  a ninety-second demo recording, and four links into the ADRs of the form *problem →
  measurement → decision*.
- **A one-page results summary** — collecting numbers that are currently scattered
  across this plan, the ADRs and the debt register: RefCOCOg mIoU 0.389 → 0.469 on the model
  switch, ambiguity margin 0.089 with 96.0% against 0.0% changed-region on the picker, the
  detector's 1372 MB peak against a 2200 MB ceiling, 12.8 s end to end, upload p90 79 ms.
- **A recording**, which replaces the deployment.

**Exit:** a reader who spends five minutes on the repository can state the central constraint,
the routing decision that follows from it, and one thing that was measured and then *not*
shipped because the measurement said not to.

### What was cut, and why

| Cut                                                      | Why it earns nothing here                                       |
| -------------------------------------------------------- | ---------------------------------------------------------------- |
| Deploying to Cloud Run / Fly / Oracle, staging, secrets  | Nobody runs it; the costed audit is the better answer            |
| A2A mesh: registry, discovery, card caching, agent JWT   | One MCP server makes the same claim in a day                     |
| OTel spans, Sentry (TD-025)                              | Correlated JSON logs already answer the question at this size    |
| Frontend polish, a11y audit, Lighthouse                  | No signal for the roles this is aimed at                         |
| OpenAPI docs site, CONTRIBUTING, v1.0.0 tag              | Ceremony                                                         |
| Further hardening                                        | Auth, CORS, scrubbing, signed URLs and lifecycle are done. Stop. |

**Remaining: ~6 days.** Orchestrator and critic (3) · MCP vision server (1) · README, RESULTS and the recording (2). Everything else is delivered or cut.

---

## 6. Testing strategy

| Layer        | Tooling                 | What's actually asserted                                                                         |
| ------------ | ----------------------- | ------------------------------------------------------------------------------------------------ |
| Unit         | pytest, hypothesis      | mask RLE round-trips for any shape; tile-plan covers the image with no gaps; EditSpec validation |
| **Memory**   | pytest + psutil         | peak RSS < 1.6 GB over a 12-step workload; ≤1 resident model; no leak over 50 iterations         |
| Provider     | respx / VCR cassettes   | zero real API calls in CI; failover order; circuit breaker opens and recovers                    |
| MCP          | FastMCP in-proc client  | tier-1 tool list is small; `enable_toolset` reveals tier-2; results carry refs not pixels        |
| Planner      | pytest + recorded LLM   | a malformed completion fails validation rather than reaching the pipeline; fast paths never call the model |
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
| **8 GB is exceeded anyway**                    | Phase 0 measures before designing; RSS test in CI; tile at 512. ~~Relocate to a 16 GB HF Space~~ — free Docker Spaces ended, see [DEPLOYMENT.md](DEPLOYMENT.md)  |
| Free tier changes or dies                      | Provider abstraction + 3-deep failover from day one; quota tracked in the cost ledger; local LaMa always works offline                               |
| Text→mask picks the wrong object               | Confidence threshold → return candidates → user taps the right one. The brush is always the escape hatch                                             |
| Free generative quality is poor                | Measured and accepted: SD-1.5 handles additions weakly and cannot remove at all. Removal is local; enabling Gemini billing lifts the cap in one line |
| Cast shadows survive removal                   | **Known v1 limitation.** ObjectClear/OmniEraser solve it but are SDXL/FLUX-based and GPU-only                                                        |
| Agent mesh becomes over-engineered latency tax | Deterministic fast-paths for the 7 known ops; LLM planning only on ambiguity; every hop traced and budgeted                                          |
| Celery RSS overhead                            | `--concurrency=1 --max-tasks-per-child=10`; ARQ as the documented escape hatch                                                                       |
| Scope creep                                    | Phase exits are objective gates; anything not in the 7 ops is a v2 issue                                                                             |

---

## 8. Immediate next actions

1. **Orchestrator and critic** — constrained decoding into `EditSpec`, deterministic fast
   paths for the seven ops, enforced budgets, and a critic that replans within a bound.
2. **Seed failures into the eval set** and report what the critic fixed, with the latency and
   cost it added. The number is the deliverable, not the loop.
3. **One MCP server** over ground / segment / erase, with the orchestrator importing no model
   code.
4. **README, a one-page results summary and a ninety-second recording** — the part a reader actually sees.

Not next, and deliberately: deploying anything, the agent mesh, spans, frontend polish.
