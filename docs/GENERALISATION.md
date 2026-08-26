# Generalisation study

**Date:** 2026-08-26 · **Question:** do the numbers from our 18 hand-built cases hold on
data we did not curate and did not tune against?

**Answer: no, and not by a small margin.** The study also found that the metric driving
several pipeline decisions does not measure what we assumed.

---

## Method

| Set | Role | n | Ground truth |
|---|---|---:|---|
| `evals/` | **dev** — thresholds were fitted here | 18 | boxes drawn by hand |
| RefCOCOg val | **held out** — grounding | 250 | real segmentation masks |
| RemovalBench | **held out** — removal | 69 | paired scene with and without the object |

Nothing was tuned on the held-out sets. All commands are reproducible:
`make bench-grounding`, `make bench-removal`, `make bench-tune`, `make bench-classifier`.

**RORD was rejected**, despite being the obvious choice. The only available copies are a
6.4 GB monolithic archive and a 420 GB one, neither subsettable; a third dataset sharing
the acronym turned out to be driving footage with depth maps. RemovalBench provides the
same capability — paired ground truth plus masks — at 502 MB. **OBER-Wild** is not
published in a fetchable form, so shadow-specific held-out testing remains open.

---

## 1. Grounding does not generalise

| | dev set | held-out RefCOCOg |
|---|---|---|
| Success | **10/11 ≈ 91%** | **precision@0.5 = 39.6%** |
| | | mIoU **0.389**, median 0.335 |
| | | **36% score below IoU 0.1** |

Breakdown:

| Mask source | n | mIoU |
|---|---:|---:|
| SAM-refined | 185 | 0.469 |
| CLIPSeg seed kept (SAM rejected) | 45 | 0.234 |
| no match at all | 20 | 0.000 |

| Phrase length | n | mIoU |
|---|---:|---:|
| ≤ 5 words | 55 | 0.469 |
| > 5 words | 195 | 0.367 |

CLIPSeg finds **nothing at all** for 8% of phrases. Published referring-expression
segmentation models reach mIoU 0.65–0.75 on this benchmark; zero-shot CLIPSeg plus SAM is
simply a weaker approach, and that is the ceiling we are working under. (TD-012)

---

## 2. Threshold fitting: a negative result, correctly refused

The confidence gate `min_sam_iou` was hand-picked at 0.85 from a single Phase 0 case.
`benchmarks/tune.py` records the outcome under **both** branches for every sample, splits
deterministically by id hash, sweeps on the fit half and reports on holdout.

| gate | fit mIoU | holdout mIoU |
|---:|---:|---:|
| 0.00 | 0.4356 | 0.3976 |
| 0.68 *(best on fit)* | **0.4357** | 0.3977 |
| 0.85 *(current)* | 0.4314 | **0.4006** |
| 1.01 | 0.3917 | 0.3754 |

The fitted optimum is **−0.0029 worse on held-out data** than the value it would have
replaced. The tuner declined to write it.

Two things follow. The hand-picked 0.85 was fine. And the gate barely matters: 2.5 points
of mIoU across its entire range, against a 26-point gap to published models. **No tuning
available to us closes this.**

---

## 3. Removal, against real paired ground truth

SSIM measured **inside the edited region** — scoring whole frames would report ~0.99 for
every method, since the edit is a few percent of the pixels.

| Method | mean SSIM |
|---|---:|
| MI-GAN alone | 0.5440 |
| **LaMa alone** | **0.5791** |
| our router (shipped) | 0.5630 |
| oracle (perfect choice) | 0.5836 |
| OmniEraser baseline *(GPU diffusion)* | 0.5808 |

Two results worth separating.

**LaMa alone essentially matches a GPU diffusion baseline** — 0.5791 against 0.5808. A
208 MB CPU model is competitive with OmniEraser on this benchmark.

**Our router is worse than doing nothing.** It scores 0.5630 against 0.5791 for always
choosing LaMa, and picks the better eraser **30.4%** of the time where always-LaMa
achieves 75.4%.

---

## 4. The finding underneath: plausibility is not fidelity

The router chooses by `fill_metrics.cost`. Correlating that against SSIM-vs-ground-truth
over 138 individual fills:

```
Pearson  +0.145        cost is lower-is-better
Spearman +0.128        SSIM is higher-is-better
                       →  a proxy that worked would be NEGATIVE
```

**The metric is not wrong — it measures what it claims.** `cost` measures how well a fill
agrees with the pixels around it: *plausibility*. It was being used to stand in for how
close the fill is to what was really behind the object: *fidelity*. Nobody checked that
one implied the other, and it does not.

This matters beyond the router: the same score drives the multi-pass keep-or-rollback
decision and the `cost` column in every eval table this project has produced. (TD-013)

---

## 5. The learned eraser chooser: built, and rejected

Nine features computable before any eraser runs — mask coverage, compactness, solidity,
aspect ratio, ring edge density, ring chroma and luma spread, border contact, relative
scale. Logistic regression, numpy only, 5-fold stratified cross-validation, scaler fitted
per fold so a held-out sample cannot see its own statistics.

| Chooser | accuracy | mean SSIM |
|---|---:|---:|
| majority (always LaMa) | 0.754 | **0.5791** |
| current router | 0.304 | 0.5630 |
| **learned, cross-validated** | **0.812** | 0.5789 |
| oracle | 1.000 | 0.5836 |

It beats the majority baseline on **accuracy** (+0.058) and not on **SSIM** (−0.0001): it
wins precisely the cases where the two erasers barely differ.

**The whole choice is worth at most +0.0045 SSIM** — the oracle's margin over always-LaMa.
That ceiling bounds any router, learned or hand-written. The adoption rule was originally
written against accuracy, which would have accepted this model; it now requires the
outcome metric to move, and correctly rejects it.

The strongest learned weight is `ring_luma_std` (+1.55, favouring LaMa) — LaMa is
preferred where the surroundings vary in brightness. That is a real signal; it is just
not worth acting on.

---

## 6. Fine-tuning: what would help, and why we cannot

**Nothing in this project trains.** Every model is pretrained and frozen; there is no
optimiser or loss anywhere in the repository. Adding data measures the system; it does not
change it.

The one place training would be honest is **grounding**, which is the dominant bottleneck:

| | mIoU |
|---|---:|
| ours (zero-shot CLIPSeg + SAM) | 0.389 |
| published trained RES models | 0.65–0.75 |

Fine-tuning CLIPSeg's decoder on RefCOCOg train (42,226 expressions, already downloaded)
would be worth roughly twice any pipeline change available to us. **It requires a GPU**,
which this project does not have, so it is recorded as TD-014 rather than attempted —
adding a training path on an 8 GB CPU machine would be theatre.

Fine-tuning the *erasers* is not indicated: LaMa already matches a GPU diffusion baseline.

---

## 7. What to do

| Priority | Action | Evidence |
|---|---|---|
| **P0** | Stop reporting dev-set grounding numbers as if they generalise | §1 |
| **P0** | Decide the router question: simplify to one eraser, or find a fidelity-correlated signal | §3, §4 |
| **P0** | Re-examine the multi-pass keep/rollback rule — it uses the same discredited score | §4 |
| P1 | Fine-tune or replace CLIPSeg when a GPU is available | §6, TD-014 |
| P1 | Second paired dataset before acting on §3 — this is one benchmark, and our dev set disagrees | §3 |
| P2 | Find a fetchable shadow benchmark to replace OBER-Wild | Method |

**Caveat that limits all of the above:** RemovalBench is 69 samples from one source, and
its conclusion (LaMa dominant) contradicts what our dev set suggested (complementary
erasers). One of the two is unrepresentative and this study does not establish which.
That is the next measurement, not a conclusion to act on yet.
