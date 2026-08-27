# Generalisation study

**Date:** 2026-08-26, revised 2026-08-27 · **Question:** do the numbers from our 18
hand-built cases hold on data we did not curate and did not tune against?

**Answer: no, and not by a small margin.** The study also found that the metric driving
several pipeline decisions does not measure what we assumed.

> **Revision, 2026-08-27.** Sections 1 and 2 acted on: grounding moved to Grounding DINO
> ([ADR-0002](adr/0002-text-grounding.md)), lifting held-out mIoU from 0.389 to **0.469**.
> A second paired dataset — **RORD-50** — was found in a usable form and section 3 now
> reports both. Section 4's finding about the quality proxy is the one that drove the
> most change, and is revised in place. Original numbers are kept alongside the new ones,
> because a study that quietly overwrites what it used to say is not evidence.

---

## Method

| Set | Role | n | Ground truth |
|---|---|---:|---|
| `evals/` | **dev** — thresholds were fitted here | 18 | boxes drawn by hand |
| RefCOCOg val | **held out** — grounding | 250 | real segmentation masks |
| RemovalBench | **held out** — removal | 69 | paired scene with and without the object |

Nothing was tuned on the held-out sets. All commands are reproducible:
`make bench-grounding`, `make bench-removal`, `make bench-tune`, `make bench-classifier`.

| RORD-50 | **held out** — removal, second distribution | 69 | paired video takes, with and without the object |

Nothing was tuned on the held-out sets. Splits are deterministic by id hash, and RORD
frames are grouped by clip so two frames of one scene cannot straddle a split.

**RORD full was rejected and a subset found instead.** The complete dataset exists only as
a 6.4 GB monolithic archive and a 420 GB one, neither subsettable, and a third dataset
sharing the acronym is driving footage with depth maps. `HigherHu/RORD-50` is 95 MB of 50
clips with `videos/`, `gts/` and `masks/`, from which frames are sampled. The pairing was
verified rather than trusted: mean absolute difference between take and ground truth is
**49–76 inside the mask against 2–4 outside it**. That 2–4 floor is real camera drift and
compression, so absolute SSIM is not comparable across the two datasets — but it applies
equally to both erasers on a given sample, so *ranking* them is valid, which is the only
thing section 4 asks of it.

**OBER-Wild** is still not published in a fetchable form, so shadow-specific held-out
testing remains open.

---

## 1. Grounding did not generalise — and the fix was a model, not a threshold

The original finding, on 250 held-out RefCOCOg samples:

| | dev set | held-out RefCOCOg |
|---|---|---|
| Success | **10/11 ≈ 91%** | **precision@0.5 = 39.6%** |
| | | mIoU **0.389**, median 0.335 |
| | | **36% score below IoU 0.1** |

The dev-set figure was roughly double the reality, and it was the one being quoted.

**Acted on, 2026-08-27.** Grounding DINO tiny (int8 ONNX) replaced CLIPSeg as the text
lane, its best box prompting MobileSAM. Same 250 samples, same order:

| | CLIPSeg → SAM | Grounding DINO → SAM |
| --- | ---: | ---: |
| mIoU | 0.3893 | **0.4694** |
| median IoU | 0.3348 | **0.5411** |
| precision@0.5 | 0.396 | **0.516** |
| precision@0.7 | 0.292 | **0.440** |
| **failures below IoU 0.1** | **0.360** | **0.360** |
| phrases matching nothing | 20 / 250 | **0** |
| median seconds | 0.89 | 3.67 |

CLIPSeg reproduced its recorded 0.3893 to four decimals, so this is a comparison against a
verified baseline rather than a remembered one.

**The most informative row is the one that did not move.** The failure rate below IoU 0.1
is identical. Two unrelated models failing on the same third of the benchmark is not
coincidence: those are the relational expressions — "the zebra on the left in the
right-hand picture" — and neither model reasons about relations. They ground the noun and
pick an arbitrary instance. The median moving twice as far as the mean says the same
thing from the other side: the detector is much better when it is right, and no better
when it is wrong.

So the gain is real and the ceiling has moved, but the *shape* of the remaining failure is
unchanged, and no better detector fixes it. Split out as **TD-015**, whose resolution is a
UI change rather than a model: `detect()` already ranks every candidate box, so surfacing
the top few turns a wrong edit into one extra click. (TD-012, TD-015)

---

## 2. Threshold fitting: a negative result, correctly refused — twice

`benchmarks/tune.py` records the outcome under **both** branches for every sample, splits
deterministically by id hash, sweeps on the fit half and reports on holdout. It has now
refused to write a fitted value on two different grounding paths.

**On the CLIPSeg path** (original run): best on fit 0.68 → holdout 0.3977, against the
hand-picked 0.85 → holdout **0.4006**. The fitted optimum was 0.0029 *worse*.

**On the Grounding DINO path** (2026-08-27, 122 fit / 128 holdout):

| `min_sam_iou` | fit mIoU | holdout mIoU |
|---:|---:|---:|
| 0.00 | 0.4871 | 0.4513 |
| 0.85 *(current)* | 0.4877 | **0.4519** |
| 0.92 *(best on fit)* | **0.4895** | 0.4518 |
| 1.01 | 0.3415 | 0.3178 |

Worse by 0.0002. Not written. Across the whole useful range the holdout score moves by
**0.0006** — on this path the threshold is simply not load-bearing, which is now measured
rather than assumed.

### The threshold that cannot be fitted this way, and why saying so matters

`min_box_score` chooses between answering and abstaining. Raising it turns answers into
empty masks scoring IoU 0, so mIoU is monotonically non-increasing in it and its
"optimum" is always zero. That is a property of the objective: **mIoU has no way to
express that a confidently wrong mask erases the wrong object while an abstention merely
shows the user a brush.** Fitting it against mIoU anyway would produce a number that looks
measured and means nothing.

So the curve is reported instead, on holdout:

| gate | abstained | mIoU (answered) | precision@0.5 (answered) |
| ---: | ---: | ---: | ---: |
| 0.25 *(default)* | 0% | 0.4518 | 0.500 |
| 0.40 | 14.8% | 0.4512 | 0.505 |
| 0.50 | 43.8% | 0.5038 | 0.556 |
| 0.60 | 53.9% | 0.5621 | **0.627** |
| 0.70 | 71.9% | 0.5921 | 0.667 |

The score *is* informative — precision on what remains climbs monotonically. The default
stays at 0.25, which never abstains, because the brush is always there and a wrong mask a
user can see and fix beats a refusal they cannot. Raising it is a **product decision**,
and this table is the data for it.

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

**Our router is worse than doing nothing** — on this dataset. It scores 0.5630 against
0.5791 for always choosing LaMa, and picks the better eraser **30.4%** of the time where
always-LaMa achieves 75.4%.

### Then the second dataset arrived, and disagreed about everything

| | RemovalBench (n=69) | RORD-50 (n=69) |
|---|---:|---:|
| MI-GAN alone | 0.5440 | 0.8244 |
| LaMa alone | **0.5791** | **0.8303** |
| our router | 0.5630 | 0.8289 |
| oracle | 0.5836 | 0.8372 |
| **MI-GAN wins / LaMa wins** | **17 / 52** | **35 / 34** |
| router picked the winner | 30.4% | **60.9%** |
| always-majority picks it | 75.4% (LaMa) | 50.7% (MI-GAN) |

RemovalBench says LaMa dominates three to one. RORD says the two erasers are level. They
cannot both describe the images a user will send. That is **TD-017**, and it is now the
open question rather than a footnote.

---

## 4. The finding underneath: plausibility is not fidelity — on one dataset

The router chooses by `fill_metrics.cost`. Correlating it against SSIM-vs-ground-truth
over 138 individual fills, on RemovalBench:

```
Pearson  +0.145        cost is lower-is-better
Spearman +0.128        SSIM is higher-is-better
                       →  a proxy that worked would be NEGATIVE
```

The conclusion recorded at the time was that cost is a plausibility score being misused as
a fidelity score, and the resolution was to **confirm it on a second paired dataset before
acting**. That was the right instinct, and the confirmation **refuted it**:

| | RemovalBench | RORD-50 |
|---|---:|---:|
| Spearman `cost` vs SSIM *(negative is correct)* | **+0.128** | **−0.519** |
| Spearman `semantic` vs SSIM *(positive is correct)* | −0.146 | **+0.415** |
| `cost` picks the better eraser | 43.5% | **68.1%** |
| `semantic` picks it | 52.2% | 44.9% |

On RORD, cost correlates at **−0.519** — the right sign, and the same magnitude the ReMOVE
paper reports for its own metric against LPIPS (−0.515). It beats always-MI-GAN by 17
points. **The metric works there.**

**So the router was not deleted.** The change queued behind the original finding — ship
one eraser, drop the escalation — rested entirely on a dataset that does not generalise,
which is the same mistake the finding itself was written to warn about, made one level up.

**Both proxies fail together and succeed together.** Cost and the semantic score flip sign
on the same dataset boundary. Two independent metrics agreeing with truth on RORD and
disagreeing on RemovalBench points at the dataset rather than at the metrics. The leading
hypothesis, recorded as a hypothesis in TD-017: RemovalBench is a curated *hard* set —
mean SSIM of the better eraser is 0.584 there against 0.837 on RORD — and when every fill
is poor, no reference-free score can rank fills that are all wrong.

### The candidate replacement, measured and rejected

`editgpt_models.semantic` implements ReMOVE (arXiv:2409.00707) off the MobileSAM encoder
the pipeline already loads, so the marginal cost is one forward pass. It does not earn a
place: it flips sign across the same boundary as cost, and where the signal is strongest
it is beaten by the metric that costs nothing extra (44.9% against 68.1%). It stays as a
**benchmark instrument** — `make bench-removal` reports it every run — so a third dataset
can revisit the question without anyone re-implementing the paper.

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

The original list, and what became of each item on 2026-08-27:

| Was | Action | Outcome |
|---|---|---|
| **P0** | Stop reporting dev-set grounding numbers as if they generalise | **done** — §1 now reports held-out first, and `EVALUATION.md` refuses the dev-set figure on its own |
| **P0** | Decide the router question: simplify, or find a fidelity-correlated signal | **not done, deliberately** — the evidence for simplifying did not replicate (§4). Deleting the router would have been acting on an artefact |
| **P0** | Re-examine the multi-pass keep/rollback rule — same discredited score | **withdrawn** — the score is not discredited; it correlates at −0.519 on the second dataset |
| P1 | Fine-tune or replace CLIPSeg when a GPU is available | **replaced, no GPU needed** — Grounding DINO, [ADR-0002](adr/0002-text-grounding.md) |
| P1 | Second paired dataset before acting on §3 | **done, and it changed the answer** — §4 |
| P2 | Find a fetchable shadow benchmark to replace OBER-Wild | still open |

What is now on the list instead:

| Priority | Action | Evidence |
|---|---|---|
| **P1** | A third paired dataset, to break the RemovalBench/RORD tie | §3, §4, TD-017 |
| **P1** | Return grounding *candidates* instead of one confident answer | §1, TD-015 |
| P2 | Stratify all paired sets by mask coverage — test whether proxy quality tracks difficulty rather than dataset identity | TD-017 |
| P2 | A fetchable shadow benchmark | Method |

**The lesson this study keeps teaching, now at two levels.** Phase 0's dev-set numbers did
not survive a held-out set. Then the held-out set's own conclusion did not survive a second
held-out set. Both times the error was the same shape: **one dataset, treated as the
world.** The rule that follows is in `harness/testing.md` — a metric is not validated
until it holds on a second set, and a fitted value that loses to the default does not ship.
