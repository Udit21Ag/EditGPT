# ADR-0003 — Offer candidates instead of guessing

- **Status:** accepted
- **Date:** 2026-08-27
- **Context:** TD-015. 36% of held-out phrases score below IoU 0.1, and the figure is
  *identical* for CLIPSeg and Grounding DINO.
- **Relates to:** [ADR-0002](0002-text-grounding.md), which improved grounding without
  touching this third at all.

## Decision

**When grounding cannot tell which instance a phrase means, return the ranked candidates
and let the user pick.** The gate is the score margin between the best and second-best
detection; above it we answer, below it we ask.

## Why a UI change and not a better model

Two unrelated models failing on the same third of RefCOCOg is not coincidence. Those are
the relational expressions — "the zebra on the left in the right-hand picture" — and
neither model reasons about relations. Both ground the noun and pick an arbitrary
instance. ADR-0002 raised mIoU from 0.389 to 0.469 and moved `failures@0.1` by **zero**.

Every model that does resolve relations is 7B-class and GPU-only, which the 8 GB CPU
constraint rules out. So the question is not how to guess better. It is whether the right
answer is *available* and simply not being offered.

## The measurement

250 held-out RefCOCOg samples, top-5 detections each, every candidate segmented and
scored against ground truth (`benchmarks/ambiguity.py`, `make bench-ambiguity`):

| K | hit@K | mIoU@K |
| ---: | ---: | ---: |
| 1 *(answering, today)* | 0.516 | 0.4688 |
| 2 | 0.692 | 0.6220 |
| 3 | 0.776 | 0.6863 |
| 4 | 0.816 | 0.7121 |
| 5 | **0.832** | **0.7309** |

**mIoU 0.731 is inside the 0.65–0.75 band that published *trained* referring-segmentation
models occupy** — the band TD-012 and TD-014 recorded as out of reach without hardware we
do not have. It is reachable with one extra click, on the model already shipping.

The gap was never a model gap. It was a UI gap.

## The gate

The margin between the top two detection scores separates cleanly. Top-1 accuracy on each
side of a cut:

| margin < | share asked | hit \| narrow | hit \| wide |
| ---: | ---: | ---: | ---: |
| 0.05 | 20.4% | 0.372 | 0.553 |
| 0.10 | 35.2% | 0.420 | 0.568 |
| 0.20 | 57.2% | 0.427 | 0.635 |
| 0.40 | 78.4% | 0.439 | 0.796 |

What the gate delivers overall, assuming the user picks correctly:

| gate | asked | hit | mIoU |
| ---: | ---: | ---: | ---: |
| 0.00 *(never ask)* | 0% | 0.516 | 0.4688 |
| 0.05 | 20.4% | 0.596 | 0.5338 |
| **0.15** | 46.0% | **0.704** | 0.6240 |
| 0.40 | 78.4% | 0.800 | 0.7080 |
| 1.01 *(always ask)* | 100% | 0.832 | 0.7309 |

**0.15 ships as the default**, as the point returning the most accuracy per unit of
friction. Two honest caveats: the table is a *ceiling* — it assumes the user always picks
right — and 0.15 was chosen against the whole set. This project has twice shipped nothing
because a value fitted on its own reporting data lost on holdout, so the number is
recorded here as a starting point and `benchmarks.tune` refits it on a split.

Asking on 46% of phrases is a lot. That is a product judgement about friction, not a
measurement, and the curve above is the data for revisiting it.

## Alternatives rejected

| Considered | Why not |
| --- | --- |
| A better detector | ADR-0002 already did that. It moved the failing third by zero. |
| A relational model (Qwen3-VL-Seg, OpenWorldSAM) | 7B-class, GPU-only. Outside the constraint by an order of magnitude. |
| Always ask | Reaches 0.832 but puts a chooser in front of every edit, including the 45% where the detector scores 0.95 against 0.07. |
| Never ask, show the mask and let the user undo | An erase is destructive and the multi-pass loop is seconds of model time. Asking first is cheaper than undoing. |
| Ask when the *top* score is low | Rejected on the data: absolute score is a much weaker separator than the margin, because a confidently-detected wrong instance scores high. |

## Consequences

- `POST /v1/masks` grounds without editing. Dispatched to a worker and waited on: models
  live in workers (invariant 3), and the gateway must not grow a 2 GB dependency.
- One encoder pass serves all K candidates. `mask_from_box` used to re-encode per box, so
  five candidates cost five encoder passes — this would have been paid on every
  disambiguation request.
- Each candidate carries its mask. Sending a box back for re-segmentation would pay for
  the expensive half of SAM twice and could return a *different* mask the second time.
- **The 0.832 ceiling assumes a user who picks correctly.** What they actually do is
  unmeasured, and cannot be measured without shipping it.
