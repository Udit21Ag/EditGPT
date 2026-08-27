# 002 — Make the product work: real editing, and asking when unsure

## Goal

Two things are true when this is done:

1. **The worker edits images.** `POST /v1/jobs` with `"remove the car"` returns an image
   with the car gone, produced by the real pipeline, through the real queue. Not `noop`.
2. **A wrong guess costs a click, not an edit.** When grounding cannot tell which
   instance a phrase means, the API returns ranked candidates instead of confidently
   erasing the wrong one.

## Context

Phase 4 and Phase 5 are mostly already delivered — `ModelSlot`, MobileSAM, the erasers,
the compositor, the multi-pass policy and the provider chain all shipped in Phase 0/1,
and text grounding moved to Grounding DINO in ADR-0002. What is left of each is small,
and what is left of both is the same thing: **the product does not actually edit
anything yet.** The worker runs `noop`.

So this plan merges the remainder of both phases rather than following them in order.
The two open items that matter:

- **TD-015.** 36% of held-out phrases score below IoU 0.1, and the figure is _identical_
  for CLIPSeg and Grounding DINO. Two unrelated models failing on the same third is not
  coincidence: those are relational expressions ("the zebra on the left"), and both
  models ground the noun then pick an arbitrary instance. No better detector fixes it;
  every model that resolves relations is GPU-class. The tractable answer is to stop
  pretending we know.
- **Phase 5's exit criterion** — "remove the car" and "add a moustache" end to end — is
  unmet only because nothing connects the pipeline to the worker.

## Current state

- `apps/worker` ships one editor, `noop`, which returns the image untouched. It does not
  depend on `editgpt-models` at all.
- **`evals/run.py` holds the only implementation of "given an op, an image and a mask,
  produce the edited image".** It is ~120 lines of dispatch covering REMOVE, UPSCALE,
  BACKGROUND, ADD and REPLACE, including the two Phase 0 fixes that made i8 work.
- `detect()` already returns every candidate box ranked by score. `mask_from_phrase`
  throws all but the first away.
- `EditSpec.confidence` exists on the contract and nothing sets it.
- Provider calls write nothing to `cost_ledger`.

## Proposed approach

### A — Extract the editor before wiring it

Wiring the worker by copying `evals/run.py`'s dispatch would leave two implementations of
the same policy, and this repository has already grown near-duplicate mask utilities
twice. So the dispatch moves into `packages/models` as one entry point, and **both** the
worker and the golden set call it.

That makes the golden set the regression test for the extraction: if `make eval` still
produces the same table, the move was faithful. Doing it in this order is the whole
point — extract against a working reference, then wire.

The new dependency edge is `apps/worker → editgpt-models`, which is the architecture's
intent ("models live in workers") and the reason the gateway sends a task _by name_.

### B — Measure the disambiguation before building it

The feature is only worth building if a correct candidate is usually _there_ to be
picked. That is `recall@K`, and it is unmeasured. So before any UI contract:

- record the top-K detections per RefCOCOg sample and each candidate's mask IoU;
- report `hit@1` against `hit@3` and `hit@5` — the ceiling disambiguation can reach;
- test whether the **top1−top2 score margin** predicts the top-1 being wrong, which is
  the signal an "ask the user" rule would fire on.

If `recall@5` is barely above `hit@1`, the candidates are not there and the feature is
theatre; that outcome gets recorded rather than built around. **The threshold is fitted
on a split, not chosen** — the same discipline that made `tune.py` refuse to write twice.

### C — Ask only when it helps

An always-ask flow puts a modal in front of every edit, which is worse than being wrong
occasionally. So ambiguity is a _gate_: answer confidently when the margin is wide,
return candidates when it is not. Where that gate sits comes from B.

## Constraints

- 8 GB RAM. The worker will hold Grounding DINO (measured 1372 MB peak), MobileSAM,
  MI-GAN and Big-LaMa. `ModelSlot` keeps one heavy model resident; `make memory` must
  stay green with the detector in the slot, and that is a real risk, not a formality.
- `make check` stays hermetic: no test may load a model or reach a network.
- The two Phase 0 fixes are carried over verbatim and must not be re-derived: match
  chroma but **not** luminance, and scale dilation with the object rather than a pixel
  constant.
- Removal never routes to a remote provider (ADR-0001, invariant 6).

## Tasks

- [x] `benchmarks/ambiguity.py` — recall@K and the margin signal — LOCAL
- [x] `editgpt_models.execute` — the single edit dispatch — ARCHITECTURAL — `make eval`
      reproduces the current table
- [x] `evals/run.py` calls `execute` — CROSS_COMPONENT — same table, fewer lines
- [x] Worker depends on `editgpt-models`; real editors replace `noop` — ARCHITECTURAL —
      integration test, then a live run
- [x] Models in the worker through `ModelSlot` — CROSS_COMPONENT — `make memory`
- [x] Cost ledger writes on provider calls — CROSS_COMPONENT — asserted in a test
- [x] `MaskCandidate` on the contract — ARCHITECTURAL
- [x] `segment.candidates_from_phrase` — CROSS_COMPONENT
- [x] Ambiguity gate, threshold fitted on a split — DATA_AFFECTING
- [x] `POST /v1/masks` returning candidates — CROSS_COMPONENT + SECURITY_SENSITIVE
- [x] ADR for the extraction and the ask-when-unsure rule
- [x] Harness: architecture, debt items, plan

## Verification

| Claim                          | How it is checked                                    |
| ------------------------------ | ---------------------------------------------------- |
| the extraction was faithful    | `make eval` — same cases, same outcomes              |
| the worker really edits        | live: upload → job → an image with the object gone   |
| disambiguation is worth having | `recall@5` against `hit@1`, measured before building |
| the gate is not invented       | fitted on one split, reported on another             |
| memory still fits              | `make memory` with the detector resident             |
| nothing else broke             | `make check`                                         |

## Risks

| Risk                                             | Early signal                            |
| ------------------------------------------------ | --------------------------------------- |
| the worker breaches the RSS ceiling              | `make memory` before wiring the gateway |
| extraction changes behaviour subtly              | `make eval` diff against the last table |
| candidates are not there to pick (recall@5 ≈ @1) | measured in B, before any contract      |
| a real edit is too slow for the queue's timeout  | time one removal end to end             |

## Decisions

**2026-08-27 — the edit dispatch moves to `packages/models`, not into the worker.**
Copying `evals/run.py`'s dispatch into the worker would leave two implementations of the
same policy. Extracting it makes the golden set the regression test for the extraction,
and gives the worker a dependency on `editgpt-models` — which is the architecture's
intent, and why the gateway sends a task by _name_.

**2026-08-27 — `MIN_MASK_PX` applies to the selected region, not the dilated one.**
Found by calibrating a fixture against real numbers instead of guessing: dilation has an
8 px floor, so a _single_ pixel grows to exactly 64 and clears the threshold. A
post-dilation check could never fire for any non-empty mask — a guard that looked like
one and was not.

**2026-08-27 — `BACKGROUND` may be requested with no region at all.** Flood-filling the
backdrop needs none; the mask is only a fallback for when the border is not uniform.
Requiring one would have refused the request the operation is best at.

**2026-08-27 — the encoder runs once per image, not once per box.** `mask_from_box`
re-encoded for every prompt, so a five-candidate request cost five encoder passes. That
is the difference between a 40-minute benchmark run and a two-hour one, and it would
have been paid on every disambiguation request. `masks_from_boxes` encodes once and
decodes K times; `mask_from_box` is now a one-element call to it.

## Progress

**2026-08-27** — `editgpt_models.execute` extracted and tested (13 tests, models
injected). `evals/run.py` now calls it. `apps/worker` gained `editgpt-models` and a real
`editors.py`; `EDITORS` ships `noop` alongside `default`, and the gateway's default
editor is now the real one.

**2026-08-27, a hermeticity break of my own making.** Changing the default editor made
the integration tests load 550 MB of weights — they kept passing, 84 seconds slower,
silently. Every job in `tests/` now names `noop` explicitly (it proves the pipe, not the
edit), and the root `conftest.py` refuses the real editor for any unmarked test. The
fast tier is back to ~4 s. This is the same rule as `--disable-socket`, for the other
expensive resource.

**2026-08-27, the first eval comparison was worthless and I nearly acted on it.** The
`report.json` on disk predated the CLIPSeg→Grounding DINO swap, because that change
never triggered a re-run. Diffing against it conflated the extraction with the grounding
change — 16 of 18 cases "changed". The real baseline is HEAD's `evals/run.py` with the
current grounding, which is what the comparison now uses.

**2026-08-27, two audit-trail regressions caught by that diff.** The extraction had
flattened `cloudflare` to `remote fill` and dropped the dimensions from the upscale
summary. Which provider served a generative result is the first question asked about a
bad one; the sizes are the entire point of an upscale. Both restored.

## Outcome

Both goals met.

**1. The worker edits images.** Verified live end to end: upload → ground → job → SSE →
an image with the car gone. 12.8 s, 7.8% of pixels changed, through the real queue with a
real Clerk session. `make memory` green with the detector resident.

**2. A wrong guess costs a click.** `POST /v1/masks` returns ranked candidates and flags
ambiguity. Measured ceiling: hit 0.516 → **0.832**, mIoU 0.469 → **0.731** — inside the
band published _trained_ RES models occupy, and which two debt items had recorded as
unreachable without a GPU. **The gap was never a model gap.**

The extraction was verified against the golden set with the same grounding on both sides:
all 11 removals identical on cost and pass sequence.

## What was learned that outlived the work

- **A guard that cannot fire is not a guard.** `MIN_MASK_PX` was checked after dilation,
  and dilation has an 8 px floor — a single pixel grows to exactly 64 and clears it. Found
  by calibrating a fixture against real numbers instead of guessing at one.
- **The first comparison I ran was worthless and I nearly reported it.** The baseline
  predated a grounding change that never triggered a re-run, so the diff conflated two
  causes and showed 16 of 18 cases "changed". A comparison is only evidence if exactly one
  thing differs.
- **I broke hermeticity myself, and it passed.** Flipping the default editor made the
  integration tests load 550 MB — still green, 84 s slower, silently. There is now a
  `conftest` guard, the same rule as `--disable-socket` for the other expensive resource.
- **Two silent format bugs cost more than a loud one would have.** AVIF was missing from
  the worker's map, so an upload was re-encoded as PNG _and_ recorded as AVIF: 54 KB in,
  476 KB out, served under a type the bytes were not.
- **An item can disappear from the debt register while still looking present.** TD-015 and
  TD-016 had table rows and no sections, and the checker only inspected sections that
  existed. It now cross-checks both directions.

## Deferred work

| ID         | What                                                   | Why                                                                                                                                   |
| ---------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| **TD-021** | Edits come back at 2048 px, not the uploaded size      | Bounds edit _time_; the machinery to fix it (crop, edit, paste back) already exists in the compositor. P1 because every user sees it. |
| **TD-020** | Background colour is always green                      | Parsing a colour from text belongs to the intent agent, which does not exist yet.                                                     |
| TD-017     | The two paired benchmarks disagree                     | Needs a third dataset.                                                                                                                |
| TD-019     | Any signed-in user can fetch any image by digest       | Needs a join table; content addressing makes an ownership check wrong.                                                                |
| —          | **The 0.832 ceiling assumes the user picks correctly** | Unmeasurable without shipping it. The gate asking on 46% of phrases is a friction judgement, not a measurement.                       |
| —          | vision_tools MCP server                                | Moved to Phase 6, where the mesh that would disclose through it is built.                                                             |
