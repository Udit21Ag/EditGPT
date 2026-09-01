# EditGPT

Prompt-driven image editing that plans, edits, checks its own work, and asks when it is not
sure — **on an 8 GB laptop, CPU only, with no paid service anywhere in the stack.**

Upload a photograph, say *"remove the car on the left"*, and get it back without the car.
Or brush the region yourself, or tap it once and let the segmenter find its boundary.

```
┌──────────┐   instruction   ┌──────────────┐   rules first, model  ┌──────────┐
│  Next.js │ ──────────────► │   planner    │ ────────────────────► │  Gemini  │
│  editor  │                 │  typed plan  │   only for the rest   │  Flash   │
└────┬─────┘                 └──────┬───────┘                       └──────────┘
     │ upload, mask, job            │ EditSpec
     ▼                              ▼
┌──────────────────────────┐   Celery    ┌───────────────────────────────────────┐
│  FastAPI gateway         │ ──────────► │  worker                               │
│  auth · scrub · sign     │   Redis     │  ┌─────────────────────────────────┐  │
│  rate limit · SSE        │ ◄────────── │  │ ground → erase → composite      │  │
└──────────┬───────────────┘  progress   │  │ ModelSlot: one model resident   │  │
           │                             │  └────────────┬────────────────────┘  │
           ▼                             │               ▼ act, judge, decide    │
   Postgres · S3/R2                      │  ┌─────────────────────────────────┐  │
                                         │  │ critic: changed? plausible?     │  │
   ┌─────────────────┐                   │  │ still there? → widen or ask     │  │
   │ MCP vision tools│ ──── HTTP ────────┘  └─────────────────────────────────┘  │
   │ (a client, not  │      to the gateway  └───────────────────────────────────┘
   │  a second copy) │
   └─────────────────┘
```

---

## The one decision everything else follows from

You cannot host a modern generative image model on 8 GB. SDXL-inpaint needs ~8 GB by
itself. So the system **splits editing into two classes and routes between them**:

| | What it does | Where | Why |
|---|---|---|---|
| **Discriminative / restorative** | find the object, cut the mask, erase, matte, upscale | local ONNX, **one model resident at a time** | 25–250 MB each, fast, free, offline |
| **Generative** | add a moustache, replace the sky | free-tier remote API | needs a billion parameters |

This was not a guess. Phase 0 measured it: asked to erase the same car, the free generative
lane returned a stone slab, then a different car, then a boulder — while the local erasers
removed it cleanly in 0.3 s. **"Remove the car" is an eraser problem, not a diffusion
problem**, and routing that decision is exactly what an agent should be doing.
→ [ADR-0001](docs/adr/0001-model-routing.md)

---

## What is measured

Every number here was produced on the machine described above and is reproducible with a
`make` target. Nothing is quoted from a paper.

**Grounding — a phrase to a mask.** Swapping CLIPSeg for int8 Grounding DINO on held-out
RefCOCOg, with SAM refining the box:

| | CLIPSeg | Grounding DINO |
|---|---:|---:|
| mIoU, 250 held-out samples | 0.389 | **0.469** |
| torch on the shipping path | yes | **no** |

→ [ADR-0002](docs/adr/0002-text-grounding.md) · `make bench-grounding`

**Ambiguity — when to ask instead of answering.** 36% of held-out phrases ground below
IoU 0.1 because a detector grounds the *noun* and then picks an instance arbitrarily. So
the system offers candidates when the top two scores are close:

| | hit rate | mIoU |
|---|---:|---:|
| answering top-1 *(the obvious design)* | 0.516 | 0.469 |
| letting the user pick from five | **0.832** | **0.731** |

Live, on the Taj Mahal: *"the minaret"* grounds at margin 0.089 — the detector's own top
pick was the right-hand minaret, and choosing the second candidate changed **96.0%** of the
chosen region against **0.0%** of the top-ranked one. The gate turns a wrong edit into one
extra tap. → [ADR-0003](docs/adr/0003-ask-when-unsure.md) · `make bench-ambiguity`

**Planning — the model asked as little as possible.** 55 labelled instructions, two splits:

| lane | share answered | accuracy | latency | cost |
|---|---:|---:|---:|---:|
| rules | 45% | **1.000** | **0.007 ms** | 0 tokens |
| model | 18% | **1.000** | 4.36 s median | 201 tokens |
| refusals *(not an edit, or unimplemented)* | 36% | **1.000** | — | — |

`make bench-planner` — its first run found four rule defects that thirty unit tests had
missed, and then found a defect in itself: thirty calls in ninety seconds exhausted the
free tier and it scored eight unreachable rows as failures. Unmeasured is not wrong.

**Memory — the binding constraint.** `ModelSlot` holds one heavy model at a time under a
2200 MB ceiling, with an RSS watchdog and idle eviction:

| model | on disk | peak RSS | warm latency |
|---|---:|---:|---:|
| Grounding DINO (int8) | 194 MB | 1372 MB | 6.9 s |
| MI-GAN — primary eraser | 27 MB | ~1150 MB | 0.29 s |
| Big-LaMa — escalation | 198 MB | 964 MB | 2.8 s |
| MobileSAM enc + dec | 43 MB | 620 MB | 0.41 s enc, **24 ms dec** |
| Real-ESRGAN x2 | 64 MB | ~430 MB | 14.8 s |

`make memory` asserts the ceiling with real weights, including that the detector and an
eraser are never resident together.

**End to end.** Upload → ground → job → SSE → an image with the car gone: **12.8 s**, 7.8%
of pixels changed. A 15.9 MP phone photograph goes in and comes back at 3456×4608 in 12.3 s
with every untouched pixel bit-identical. The gateway itself runs in **86 MB idle, 126 MB
under upload**, and answers uploads with a 3 ms median.

---

## The part most projects skip: what was measured and *not* shipped

Three features were built, measured, and left switched off. The measurement is the asset.

- **Occluder shielding** (TD-004) — stop an erase mask swallowing the object standing in
  front of the target. It works: the jumper's shoe goes from 10.3% erased to 0.5%. It also
  makes the erase *worse overall*, because dilation had been quietly compensating for the
  segmenter under-cutting an object's base. One clear win, one clear regression, the rest
  unchanged. Parked behind `Thresholds.shield=False`, with the numbers in the docstring.
- **A feature-space quality score** (ReMOVE, TD-013) — a reference-free replacement for the
  photometric cost. Scored against paired ground truth on two benchmarks; it did not earn
  the router seat. Kept, disabled, tested.
- **A content classifier** — not built. The local lane generates nothing (erasing a car
  continues the background), so a classifier there would inspect private photographs to
  decide whether their owner may edit them; the generative lane already goes through a
  provider whose safety checker is not ours to switch off. → [ADR-0004](docs/adr/0004-content-safety.md)

And one that is still unresolved and stays visible: **two held-out benchmarks disagree on
the *sign* of the correlation between the fill-cost proxy and visual quality** (TD-017).
A photometric score has picked the visually worse image three times in this project.

---

## Harness engineering

This repository is built to be worked on by an agent, and that is a first-class part of the
design rather than a note at the end.

`AGENTS.md` is a **map, not a manual**: it routes to one file in `harness/` per task, so a
change to logging loads the observability rules and nothing else.

| File | Binding on |
|---|---|
| [`harness/architecture.md`](harness/architecture.md) | what the system is, and the invariants that hold it together |
| [`harness/code_generation.md`](harness/code_generation.md) | how a change is made — minimal *coherent* unit, constants carry their provenance |
| [`harness/testing.md`](harness/testing.md) | the tiers, the markers, and why `service` earns its exception to "no network" |
| [`harness/observability.md`](harness/observability.md) | what may be logged; credentials are removed by the formatter, not by remembering |
| [`harness/tech_debt_tracker.md`](harness/tech_debt_tracker.md) | 25 entries, each with a **trigger** — an item without one never gets paid down |
| [`harness/self_updation.md`](harness/self_updation.md) | when the harness itself must change |

Four things make it more than documentation:

1. **`make check` is the contract**, and CI invokes the same targets, so the two cannot
   drift. 760 Python tests, 162 web tests, mypy strict over 106 source files, 91.15% coverage,
   `next build` — one command.
2. **`make harness` validates the harness**: every link, path and command named in any of
   these documents is checked to exist. 87 paths, 110 commands, on every run.
3. **Documentation that contradicts executable behaviour is a defect to report**, not a
   question to resolve silently. The weight figures said ~285 MB in one place and ~552 MB
   in another against a measured 527 MB; that is a bug, and it was filed and fixed like one.
4. **Decisions are recorded where they will be found again.** Four ADRs, a debt register,
   and a plan that is rewritten when the goal changes rather than left to rot.

---

## Running it

```bash
make setup                 # uv + pnpm, everything
make models                # ~527 MB of ONNX weights
make compose-up            # redis + postgres (OrbStack or colima, not Docker Desktop)
make dev-lite & make worker & make dev
make check                 # the gate: everything CI runs
```

No credentials are needed to run the whole local stack. Adding them turns on more:
`GEMINI_API_KEY` for the planner's model lane, `CLOUDFLARE_*` for generative edits,
`CLERK_*` for real sessions. Everything degrades to a working state without them and
`/ready` says exactly what is missing. Details in [`docs/RUNBOOK.md`](docs/RUNBOOK.md).

---

## Where it stands

**Working:** upload · text, tap and brush selection · candidate disambiguation · erase ·
background · upscale · generative add/replace · SSE progress · version history ·
before/after · planning from a sentence · the critic's retry loop · MCP tools ·
auth, CORS, EXIF scrubbing, signed URLs, structured logs, an asset lifecycle sweep.

**Deliberately not built:** a deployment (the [costed audit](docs/DEPLOYMENT.md) is the
better answer — the worker wants 2200 MB for thirteen seconds and every remaining free
fixed instance is 512 MB), the A2A agent mesh (one MCP server makes the same claim),
OTel spans (TD-025), and frontend polish.

**Honest gaps:** cast shadows survive removal (TD-002); two of seven operations are
unimplemented and say so rather than failing (TD-006); grounding generalisation is still
the P1 (TD-012).

---

*Built solo, 26 Aug – 1 Sep 2026. ~11.8k lines of source, ~11.3k lines of tests, and a
harness that checks its own documentation on every run.*
