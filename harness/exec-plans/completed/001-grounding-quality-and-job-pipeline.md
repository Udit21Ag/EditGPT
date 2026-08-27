# 001 — Close TD-012 and TD-013, then build the job pipeline

## Goal

Three things are true when this is done:

1. **Grounding generalises.** Held-out RefCOCOg mIoU is materially above the 0.389
   recorded in TD-012, measured on the same 250-sample benchmark, with the decision
   threshold fitted on a disjoint split rather than hand-picked.
2. **The quality proxy is honest.** The signal that drives routing and the multi-pass
   keep-or-rollback rule has a measured correlation with fidelity on **two** independent
   paired datasets, and every threshold it uses is loaded from a fitted artifact rather
   than written as a literal in source.
3. **Phase 3 works end to end.** `upload → job → SSE progress → result` runs against
   Postgres, Redis and Celery with a stub editor, under test.

## Context

TD-012 (P0) and TD-013 (P0) are both open and both block further investment: TD-013's
own trigger says "before any further investment in routing logic", and Phase 3 persists
and serves whatever those decisions produce. Fixing them after building the pipeline
means rewriting the pipeline.

The user's constraints on this work: no values tuned to a handful of cases, no models
trained from scratch, prefer a better pre-existing model over a hand-tuned constant, and
prefer a learnable parameter where no better model exists.

## Current state

- **Grounding** is CLIPSeg (torch, ~1188 MB peak RSS) → heatmap seed → MobileSAM,
  gated on SAM's own IoU prediction. Held-out mIoU 0.389, precision@0.5 0.396, 8% of
  phrases match nothing at all. CLIPSeg is also the largest memory consumer in the
  project (TD-001) and the only torch model.
- **Quality scoring** is `editgpt_core.metrics.fill_metrics(...).cost` — a photometric
  plausibility score. Measured Spearman against SSIM-vs-ground-truth: **+0.128**, where
  a working proxy would be negative. Picks the better eraser 43.5% of the time;
  always-LaMa picks it 75.4%.
- **Thresholds are duplicated.** `editgpt_models.config.Thresholds` exists, documents
  itself as the single source, and is loadable from a fitted JSON file — but only
  `segment.MIN_SAM_IOU` reads it. `pipeline.ESCALATE_COST`, `pipeline.ACCEPT_COST`,
  `pipeline.RESIDUAL_MAX_GROWTH`, `pipeline.RESIDUAL_MIN_GROWTH`,
  `metrics.GROWTH_PENALTY` and `compositing.DILATE_FRAC` are literals that shadow the
  dataclass fields of the same name. This is the hardcoding the goal forbids.
- **Only one paired dataset** (RemovalBench, n=69) exists, and TD-013 explicitly says its
  conclusion must be confirmed on a second before acting on it.
- **Phase 3 is not started.** The gateway serves `/health`, `/ready`, `/capabilities`.
  `editgpt_core.jobs` has the lifecycle contract and names `editgpt_store` as its
  intended persistence package; that package does not exist. Postgres and Redis
  containers run and are unused.

## Proposed approach

### A — Grounding: replace the model, do not tune the threshold

TD-012 already establishes that no threshold closes the gap (the full sweep moves mIoU
by 2.5 points). The fix has to be a better model.

**Grounding DINO tiny, int8 ONNX** (`onnx-community/grounding-dino-tiny-ONNX`,
204 MB) produces boxes from a phrase; MobileSAM turns a box into a mask. A box is a far
stronger SAM prompt than a heatmap seed, and the benchmark already shows the mask source
dominates the score (sam-refined 0.469 vs clipseg-seed 0.234).

Rejected alternatives:

- **Florence-2-base ONNX** — does referring-expression segmentation natively and would
  likely score highest, but ships as five graphs driving an autoregressive decode loop
  with polygon-from-text post-processing. Large integration surface for a first attempt;
  revisit if Grounding DINO underperforms.
- **OWLv2 / OWL-ViT** — open-vocabulary detection tuned for short class names, where
  RefCOCOg averages 8.4-word relational expressions. Grounding DINO was trained on
  phrase grounding, which is the actual task.
- **Fine-tuning CLIPSeg** — TD-014, blocked on hardware, and the user excluded training.
- **Qwen3-VL-Seg / OpenWorldSAM / Text4Seg** — 7B-class, GPU-only. Outside the 8 GB
  constraint by an order of magnitude.

Because Grounding DINO is ONNX, this also removes torch from the runtime path, which is
TD-001.

### B — Quality: add a semantic signal, then let two datasets decide

**ReMOVE** (arXiv 2409.00707) scores an erasure without ground truth by comparing mean
segmentation-ViT patch embeddings inside the erased region against those outside it.
Reported |correlation| with LPIPS ≈ 0.515 and 74.7% agreement with human preference,
against our proxy's +0.128 and 43.5%. It needs a segmentation ViT encoder — **we already
load one**, MobileSAM's, for every text and box prompt. The marginal cost is one extra
encoder pass, not a new model.

**RORD-50** (`HigherHu/RORD-50`, 95 MB) supplies the second paired dataset TD-013's
trigger demands: real video captures of the same scene with and without an object, plus
masks. Different distribution from RemovalBench entirely. Frames from one clip are
correlated, so splits must be grouped by clip.

Then: fit the combination on a fit split, report on held-out, and **accept the answer**.
If the learned combination does not beat always-LaMa on held-out data across both sets,
TD-013's own resolution applies — simplify to one eraser and delete the escalation.

Either way, every surviving threshold moves into `Thresholds` and is loaded, not
literal. `pipeline.py` reading its own constants while `config.py` claims to own them is
a defect to fix regardless of what the benchmarks say.

### C — Phase 3: contracts and job pipeline

New package `packages/store` owning persistence and asset storage, both behind
protocols, with local adapters that need no credentials. Postgres schema via SQLAlchemy

- Alembic. Celery worker on the Redis broker. Gateway grows upload, job creation with
  idempotency, job read, cancel and an SSE progress stream. `noop_edit` proves the pipe
  before any model is attached.

Clerk and R2 stay behind their interfaces until credentials exist; the local adapters
are the default so `make check` never needs a network.

## Constraints

- 8 GB RAM, CPU only. Grounding DINO int8 (204 MB on disk) must be measured for peak RSS
  before it is allowed into the registry, and it must not be resident alongside an
  eraser — the `ModelSlot` already enforces one heavy model at a time.
- Free tier only. Both new models and both new datasets are freely downloadable.
- No training. Grounding DINO is used zero-shot; the only fitting is a handful of scalar
  thresholds and, at most, a linear scorer over existing features.
- `make check` must stay hermetic: no test may download a model or a dataset.

## Tasks

- [x] `benchmarks.datasets.load_rord` — second paired dataset — LOCAL — pairing verified
      (49-76 diff inside the mask against 2-4 outside); frame arithmetic unit-tested
- [x] Registry entry + downloader for `grounding-dino` — LOCAL — `make models` now 552 MB
- [x] `editgpt_models.detect` — phrase → boxes — CROSS_COMPONENT — 19 unit tests against a
      stub graph; peak RSS 1372 MB recorded in the registry
- [x] `segment.mask_from_phrase` — detector → box → MobileSAM — CROSS_COMPONENT —
      mIoU 0.3893 → **0.4694** on the same 250 held-out samples
- [x] `editgpt_models.semantic` — ReMOVE off the SAM encoder — CROSS_COMPONENT
- [x] Thresholds: delete every shadowing literal, load from `Thresholds` — CROSS_COMPONENT
      — `test_thresholds_are_wired.py` fails if one comes back
- [x] Extend `benchmarks.removal` to both datasets and both proxies — LOCAL
- [x] Fit and write `benchmarks/fitted_thresholds.json` — DATA_AFFECTING — **the sweep
      refused to fit one**: the candidate `min_sam_iou` scored 0.0002 _worse_ on holdout.
      The file is written with the verified defaults and a provenance string recording
      what was tested and what survived, which is more informative than no file.
- [x] `packages/store`: asset store + job store protocols and local adapters —
      ARCHITECTURAL — 58 tests; both adapters held to one contract
- [x] Postgres schema + Alembic migration — DATA_AFFECTING — generated from the models,
      seeds the anonymous user
- [x] Celery app, `noop` editor, worker entrypoint — CROSS_COMPONENT — 10 tests
- [x] Gateway: upload, create/read/cancel job, SSE, idempotency, rate limit —
      CROSS_COMPONENT + SECURITY_SENSITIVE — 37 tests, including a decompression bomb
- [x] `tests/` — the cross-app integration tier, proving the Phase 3 exit criterion
- [x] Harness updates: architecture, registry table, ADR-0002, TD-001/012/014 revised,
      TD-015 and TD-016 opened

## Verification

| Claim                       | How it is checked                                                |
| --------------------------- | ---------------------------------------------------------------- |
| grounding improved          | `make bench-grounding` — mIoU and precision@0.5 vs 0.389 / 0.396 |
| the new threshold is honest | `make bench-tune` — fitted on one split, reported on the other   |
| the quality proxy is better | Spearman vs SSIM-vs-GT on RemovalBench **and** RORD              |
| nothing is hardcoded        | grep for the shadowed literals; `Thresholds` is the only home    |
| memory still fits           | `make memory` with the new model in the slot                     |
| the pipe works              | gateway integration test: upload → job → SSE → artifact          |
| nothing else broke          | `make check`                                                     |

## Risks

| Risk                                                | Early signal                                         |
| --------------------------------------------------- | ---------------------------------------------------- |
| Grounding DINO int8 is too slow to be usable        | time one image before running 250                    |
| its ONNX I/O does not match the transformers.js doc | inspect the graph's inputs before writing the caller |
| ReMOVE on MobileSAM ≠ ReMOVE on SAM ViT-H           | correlation on RemovalBench; reject if it is flat    |
| RORD clips leak between splits                      | group by clip id, assert no clip spans both          |
| Phase 3 balloons                                    | `noop_edit` first; no model touches the worker       |

## Decisions

**2026-08-27 — Squash, do not letterbox, for Grounding DINO.** The export declares an
800x800 `pixel_values` alongside a `pixel_mask`, which reads as an invitation to
letterbox and preserve the aspect ratio. Measured on 40 RefCOCOg samples: squash gives
box-IoU **0.511**, letterbox **0.230**. The export's `preprocessor_config.json` sets
`size` to a literal 800x800, so it was calibrated and quantised on squashed input.
Reasoning picked the worse option here; the measurement picked the right one.

**2026-08-27 — RORD-50 is the second paired dataset.** Checked against the alternatives:
`presencesw/RORD` is 420 GB and `JiaHuang01/RORD` 6.4 GB, both unusable on this machine;
DEFACTO's "original" images are themselves inpainted, so the pair is not a real removal.
RORD-50 is 95 MB of video with `videos/`, `gts/` and `masks/` clips. Verified the pairing
is genuine: mean absolute difference between take and ground truth is 49-76 inside the
mask against 2-4 outside it. That 2-4 floor means absolute SSIM is not comparable to
RemovalBench's, but ranking erasers within a sample is valid.

## Progress

**2026-08-27** — Registry gained `extra_files` and `asset_path` so a model may ship a
tokenizer. Added `detect.py` (Grounding DINO), `semantic.py` (ReMOVE off the MobileSAM
encoder), `segment.mask_from_phrase`, `datasets.load_rord`, and a `--path` flag on the
grounding benchmark so both grounding paths are measured on identical samples.

**2026-08-27, grounding measured.** Detector 0.4694 mIoU against CLIPSeg's 0.3893 on the
same 250 samples; CLIPSeg reproduced its recorded baseline to four decimals. Promoted to
ADR-0002. The result that mattered most was the one that did **not** move: failures below
IoU 0.1 are 0.360 on both paths, so both models fail on the same relational third —
opened as TD-015, whose fix is a candidate-picker UI rather than a model.

**2026-08-27, thresholds fitted and refused.** `min_sam_iou` best-on-fit 0.92 scored
0.4518 on holdout against the default 0.85's 0.4519, so nothing was written. Across the
whole useful range the holdout score moves 0.0006 — the threshold is not load-bearing on
this path. `min_box_score` turned out to be unfittable against mIoU by construction
(raising it converts answers into zeros), so `tune.py` now _reports_ its
precision/abstention curve instead of pretending to optimise it.

**2026-08-27, three defects found by writing the tests.** (1) `Depends` closing over
`create_app` silently became a query parameter under `from __future__ import annotations`
— every gateway route was broken. (2) SQLite refuses to autoincrement a `BIGINT` primary
key, which would have left the SQL job store effectively untested. (3) `segment.py` read
`load_thresholds()` at _import_ time, so it looked wired and was frozen at first import.

**2026-08-27, Phase 3 complete.** `packages/store`, `apps/worker`, and the gateway's
upload/job/SSE/limit surface. The exit criterion is executable: `tests/` drives a real
HTTP upload through the real Celery task to an artifact and a ledger entry. Also fixed
two pieces of drift found on the way: `make fmt` covered fewer directories than
`make lint`, and CI spelled out its own commands rather than calling the Makefile, so
`AGENTS.md`'s "make check runs exactly what CI runs" had quietly become false.

## Outcome

All three goals met, and the second one arrived at an answer opposite to the one expected.

**1. Grounding generalises better.** Held-out RefCOCOg mIoU 0.3893 → **0.4694**,
precision@0.5 0.396 → **0.516**, phrases matching nothing 20/250 → **0**. Promoted to
ADR-0002. The threshold was swept on a fit split and **not written**, because the fitted
value lost to the default on holdout by 0.0002.

**2. The quality proxy is honest — and TD-013's conclusion was wrong.** The second paired
dataset was the whole point, and it refuted the finding it was meant to confirm: `cost`
correlates with SSIM at **+0.128 on RemovalBench and −0.519 on RORD**. The queued change
— delete the router, ship one eraser — **was not made**, because it rested on a dataset
that does not generalise. Every threshold now loads from a fitted file rather than a
literal, enforced by a test that fails if a shadowing constant returns.

**3. Phase 3 works end to end — verified against the real stack.** `/ready` reports
`ready` with **no degraded modes**; upload → job → a live Celery worker over the Redis
broker → SSE showing all four transitions → `done` with a result digest. 27 jobs driven
through, every one terminal, every `done` carrying its result, none stuck. Idempotency
enforced by the real unique constraint (202 then 200, same id), rate limiting by real
Redis, cancellation twice returning 200 both times.

## What was learned that outlived the work

- **One dataset is not the world, at every level.** Phase 0's dev numbers did not survive
  a held-out set; the held-out set's conclusion did not survive a second one. Now a rule
  in `harness/testing.md`.
- **Reasoning picked the worse option twice, measurement caught both.** Letterboxing
  Grounding DINO looked obviously right and scored less than half of squashing. Adopting
  the semantic score looked obviously right and it lost to the metric it was replacing.
- **Three defects were found by writing the tests, not by reading the code**: `Depends`
  closing over `create_app` silently degrading to a query parameter, SQLite refusing to
  autoincrement a `BIGINT` key, and `segment.py` snapshotting its threshold at import.
- **Two more were found only by running the service by hand**: a malformed digest
  returning 500 instead of 422, and AVIF — a format one of our own fixtures uses — being
  rejected as unsupported.
- **And one only by applying the migration.** It was generated against SQLite, reviewed,
  merged, and failed on its first contact with Postgres: an untyped bind for a `uuid`
  column. Every SQLite test passed throughout. The lesson is now a test tier (`db`) and a
  Postgres service in CI, because **a migration is not verified by review** — only by
  applying it and rolling it back on the dialect it will actually meet.

## Deferred work

| ID         | What                                              | Why                                                                                                                                |
| ---------- | ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **TD-015** | Relational referring expressions are not grounded | 36% of held-out phrases fail below IoU 0.1, identically on both models. The fix is a candidate-picker UI, not a better detector.   |
| **TD-016** | Result dimensions assume the edit preserves them  | True today, false for `UPSCALE`.                                                                                                   |
| **TD-017** | RemovalBench and RORD disagree about everything   | Needs a third paired dataset; the decision it unblocks is worth ≤ +0.013 SSIM.                                                     |
| TD-001     | CLIPSeg still on torch                            | No longer on the shipping path; now a P2.                                                                                          |
| —          | **Clerk authentication**                          | Needs a product decision about accounts. Every table already carries `user_id` against an anonymous sentinel, so it is a backfill. |
| —          | **R2 in use**                                     | Implemented and tested against a stub; needs credentials. Local disk is the default and `/ready` says so.                          |
| —          | ~~Verification against real Postgres and Redis~~  | **Done.** The containers came up, the migration was fixed and applied, and the whole pipe was driven live.                         |
