# ADR-0002 — Grounding DINO replaces CLIPSeg as the text lane

- **Status:** accepted
- **Date:** 2026-08-27
- **Context:** TD-012. Text grounding scored mIoU 0.389 on held-out RefCOCOg against
  ~91% on our own 18 hand-built cases, so every downstream decision had been made on a
  figure roughly double the reality.
- **Amends:** ADR-0001's model roster, entry 2. Everything else in ADR-0001 stands.

## Decision

**A phrase is grounded by Grounding DINO tiny (int8 ONNX), whose best box prompts
MobileSAM.** CLIPSeg stays installed behind the optional `text` extra and is tried only
when the detector abstains.

## The measurement

Both paths, 250 RefCOCOg validation samples, identical samples in identical order,
ground truth being the dataset's own segmentation masks rather than a box we drew.

| | CLIPSeg → SAM | Grounding DINO → SAM |
| --- | ---: | ---: |
| mIoU | 0.3893 | **0.4694** |
| median IoU | 0.3348 | **0.5411** |
| precision@0.5 | 0.396 | **0.516** |
| precision@0.7 | 0.292 | **0.440** |
| failures below IoU 0.1 | 0.360 | 0.360 |
| phrases matching nothing | 20 / 250 (8%) | **0** |
| median seconds | 0.89 | 3.67 |

CLIPSeg reproduced TD-012's recorded 0.3893 to four decimals, so the comparison is
against a verified baseline and not a remembered one.

Two things in that table are worth more than the headline. The median moves further than
the mean (+0.21 against +0.08), which says the detector is much better when it is right
and no better when it is wrong. And **the failure rate below IoU 0.1 is identical at
0.360** — the two models fail on the same third of RefCOCOg, which is the third whose
expressions are relational ("the zebra on the left in the right-hand picture"). Neither
model reasons about relations; the detector simply grounds the noun better.

## Why this model

- **It is the task.** Grounding DINO was trained on phrase grounding. CLIPSeg was trained
  on single-concept segmentation and was being asked to resolve 8.4-word expressions.
- **A box is a better SAM prompt than a heatmap.** The old benchmark already showed the
  mask source dominating: 0.469 where SAM refined, 0.234 where it did not.
- **It removes torch.** CLIPSeg was the only torch model and the largest single memory
  consumer at ~1188 MB peak RSS. Grounding DINO int8 measures 1372 MB peak alone but is
  ONNX, so it lives in the same `ModelSlot` regime as everything else instead of beside
  it. This is the largest part of TD-001.
- **It fits.** 204 MB on disk, ~2.5 s per image on this CPU. Slower than CLIPSeg's 0.9 s,
  which is affordable precisely because Phase 3 put edits on a job queue.

## Alternatives rejected

| Considered | Why not |
| --- | --- |
| **Florence-2-base ONNX** | Does referring-expression segmentation natively and would likely score highest. Ships as five graphs driving an autoregressive decode loop with polygon-from-text post-processing — a large integration surface. Reconsider if relational expressions become the dominant failure. |
| **OWLv2 / OWL-ViT** | Open-vocabulary detection tuned for short class names, not for the relational phrases this benchmark is made of. |
| **Fine-tuning CLIPSeg on RefCOCOg** | TD-014. No GPU, and nothing in this project trains. |
| **Qwen3-VL-Seg, OpenWorldSAM, Text4Seg** | 7B-class, GPU-only. Outside the 8 GB constraint by an order of magnitude. |
| **Tuning the existing gate harder** | Already refuted in TD-012: the full sweep moved mIoU by 2.5 points. Confirmed again here — see below. |

## Two thresholds, and why only one was fitted

Both were swept on a 122-sample fit split and reported on a disjoint 128-sample holdout.

**`min_sam_iou`** — chooses between SAM's traced mask and the detector's rectangle. Best
on fit was 0.92 → **holdout 0.4518**, against the inherited default 0.85 → **holdout
0.4519**. The fitted value is worse by 0.0002, so it was **not written**. Across the
entire sweep the holdout score moves by 0.0006 except at the degenerate endpoint where
everything falls back to boxes (0.3178). On this path the threshold is not load-bearing,
and that is now a measured fact rather than an assumption.

**`min_box_score`** — chooses between answering and abstaining, and **mIoU cannot fit
it**: raising it converts answers into empty masks scoring IoU 0, so the objective is
monotonically decreasing and its "optimum" is always zero. That is a property of the
objective, not a gap in the sweep — mIoU has no way to say that a confidently wrong mask
erases the wrong object while an abstention merely shows the user a brush. So the curve
is **reported rather than optimised**:

| gate | abstained | mIoU (answered) | precision@0.5 (answered) |
| ---: | ---: | ---: | ---: |
| 0.25 (default) | 0% | 0.4518 | 0.500 |
| 0.40 | 14.8% | 0.4512 | 0.505 |
| 0.50 | 43.8% | 0.5038 | 0.556 |
| 0.60 | 53.9% | 0.5621 | 0.627 |
| 0.70 | 71.9% | 0.5921 | 0.667 |

The score is genuinely informative about correctness — precision on what remains climbs
monotonically. The default stays at 0.25, which never abstains, because the brush is
always available and a wrong mask a user can see and correct beats a refusal they cannot.
**Raising it is a product decision, and the data for it is now on the table.**

## Consequences

- Text grounding is ~2.8 s slower per request. Acceptable on a job queue; it would not
  have been on a request path.
- CLIPSeg is no longer on the default path, so the `text` extra becomes genuinely
  optional. TD-001 is substantially addressed but not closed — CLIPSeg is still the
  fallback for "stuff" nouns, which a detector grounds poorly.
- **Relational expressions remain unsolved.** 36% of RefCOCOg still fails below IoU 0.1
  and this ADR does not change that. The tractable mitigation is returning *candidates*
  and letting the user pick, which is already the planned disambiguation UI, not a better
  threshold.
- `evals/run.py` now routes text through the detector first, so the golden set exercises
  the shipping path.
