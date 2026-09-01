# Evaluation

What counts as evidence that a change helped. Screenshots do not.

## The rule that keeps being learned the hard way

**Fill cost compares fills within a fixed mask. It must never be used to compare masks.**

A larger flat erase is photometrically consistent with itself, so it scores better while
looking worse. During Phase 0 a photometric score picked the visually worse image three
separate times:

1. i6's residual pass scored **best of every variant (−18.8)** while flattening the desk
   to a bluish smear. It had erased 78% more area.
2. A prompt that over-segmented the laptop to 45% of the frame scored the **best fill
   cost of five variants (7.5)** while destroying the desk.
3. Comparing variants over *different* eval regions made the shadow pass look like a
   failure when the image plainly showed it working.

The mitigations are in `editgpt_core.metrics`: `compare()` scores both candidates over
one fixed region and charges a growth penalty per unit of area erased beyond the base
mask. Use `compare()`, not raw `.cost`, whenever masks differ.

The penalty is a **parameter**, not a constant: `editgpt_models.config.Thresholds` owns
the fitted value and passes it in. `core` cannot read that file — it imports nothing of
ours — so the module-level number is a documented default and nothing more.

## Layers of measurement

### 1. Localisation — is the mask on the right thing?

`box_metrics(mask, reference_box, shape)` → `inside_frac`, `box_recall`, `bbox_iou`, `hit`.

Gated on `inside_frac` **with** `bbox_iou` — box against box, the only like-for-like
comparison available without painted ground truth.

- Mask-against-box is invalid: a correct mask of a non-convex object (a lattice tower)
  covers ~37% of its own bounding box and would score as a failure.
- Precision alone is invalid: a six-pixel mask in the right place scores `inside_frac`
  1.000 and is useless. That passed a rescore while the object was still visibly present.

Current on the dev set: **10/11**. Treat that number with suspicion — the held-out
figure is roughly half it, and the gap is the entire subject of
[GENERALISATION.md](GENERALISATION.md). Report the held-out mIoU alongside it or not at
all.

### 2. Fill quality — does the result agree with its surroundings?

`fill_metrics(result, source, mask)` → `chroma_delta`, `lum_delta`, `edge_ratio`.

- `chroma_delta` ~25 is a visible pale ghost. This is the defect LaMa actually has:
  luminance correct to 3/255, chroma drifting blue.
- `edge_ratio` ≪ 1 is a smear; ≫ 1 is invented detail. 1.0 matches the surroundings.

### 3. Prompt alignment — for additions, did we get what was asked?

CLIP similarity between the edited region and the request. Validated blind in Phase 0:
the lowest-scoring moustache variant (28.54) was exactly the one that rendered no
moustache at all. The scorer caught it before a human looked.

### 4. End to end

`make eval` runs the golden set and prints status, seconds, cost, IoU and the pass chain
per case. `evals/out/report.json` is the machine-readable form. Compare against `main`
before claiming an improvement.

## Plausibility is not fidelity

**`fill_metrics.cost` measures how well a fill agrees with its surroundings. It does not
measure how close the fill is to what was really behind the object.** Measured on 138
fills with paired ground truth, the Spearman correlation between cost and
SSIM-against-truth is **+0.128** — and because cost is lower-is-better while SSIM is
higher-is-better, a proxy that worked would correlate *negatively*.

So cost is valid for what it claims: comparing two fills of the same mask for local
consistency. It is **not** valid as a stand-in for quality, and using it to choose between
erasers performs worse than choosing one and never changing (43.5% against 75.4%). See
TD-013.

### The candidate replacement, and how to judge it

`editgpt_models.semantic.consistency` implements ReMOVE (arXiv:2409.00707): mean
segmentation-ViT patch embeddings inside the erased region against those outside it,
compared by cosine similarity. **Higher is better**, the opposite direction to `cost`.

It costs one extra forward pass of an encoder the pipeline already loads for every box or
text prompt, so it is close to free — but *cheap is not the same as valid*. Two
substitutions from the paper are unproven here: MobileSAM's TinyViT is not SAM's ViT-H,
and the region crop is ours. Whether it survives them is answered by
`make bench-removal`, which reports the Spearman correlation of **both** proxies against
SSIM-vs-ground-truth on **two** independent paired datasets.

**Do not adopt a proxy on one dataset.** That is precisely how `cost` came to drive the
router. A correlation that appears on RemovalBench and not on RORD is a property of
RemovalBench.

## A number cannot tell you the image is missing

`i3` once came back with a **black rectangle stamped on a face**, and the eval printed
`ok`. The fill cost rose from 75 to 196 — a signal, if anyone had been reading it as one —
and nothing else objected, because none of these metrics has an opinion about whether an
image exists. It was caught by looking at the picture.

That is the third time here a metric has agreed with something a human saw was wrong. The
lesson is not "add a blackness metric": the provider now refuses a blank return outright
(TD-022), which is a *guard*, not a score. **Guards belong where the failure happens;
metrics rank things that are all basically valid.** Confusing the two produces a scoreboard
that grades a missing image.

The same session, `i8` and `i9` were flagged by eye for erasing a shoe and a hand along
with the target. Every number for both cases looked healthy. See TD-004.

## What we can and cannot measure

- **Our own fixtures have no paired ground truth.** For removal there is no "correct"
  output image for them, so reference metrics are unavailable *there*. That is a property
  of our fixtures, not of the problem: `benchmarks/` uses RemovalBench, which ships the
  scene with and without the object, and SSIM and PSNR are valid on it.
- Do not report FID on 18 images.
- **Shadows.** No metric detects a surviving cast shadow. It is caught by eye.
- **Neither does any metric detect the mask swallowing a bystander.** TD-004 was found by
  looking at two strips, not by a column moving.
- **Identity drift** on additions is observed, not measured.
- **Human evaluation** has not been run. When it is: blind A/B, ≥5 raters, a fixed
  criterion per question, and report inter-rater agreement or do not report at all.

## Unmeasured is not wrong

A measurement that scores its own failures as the system's is worse than no measurement.
`benchmarks/planner.py` learned this on its first run: thirty model calls in ninety seconds
exhausted the free tier's quota, eight instructions came back as questions because the model
was never reached, and the summary counted all eight as planner errors — reporting 0.855
where the answer over rows that were actually answered was 1.000.

Two rules follow, and they apply to anything measured through a metered API:

1. **Pace the run.** `--rpm` exists so a benchmark is not measuring a rate limit. A number
   produced faster than the provider will serve it is a number about the provider.
2. **Exclude what was not answered, and say how many.** Rows the model never saw are
   counted separately, never scored. The subtler half: a row that happens to be *correct*
   because nothing was reached — a refusal that is right for no reason — inflates the score
   in the direction you want, which is exactly why it has to go.

## Ablations worth running

Each names the hypothesis it tests. Anything that does not is noise.

| Ablation | Hypothesis |
|---|---|
| box-only vs box+text mask | text grounding improves the mask enough to justify CLIPSeg's 1188 MB |
| seed vs SAM-refined | refinement helps weak seeds and hurts strong ones (Phase 0: it does) |
| single vs multi-pass | the mandatory second pass earns its latency |
| dilation fraction sweep | 5% is the knee, not a guess |
| MI-GAN vs LaMa per case | the two erasers are complementary rather than redundant |
| crop vs whole-image inpaint | crop wins on small objects, ties on large |

## Adding a case

Use `/eval-case` or the `add-eval-case` skill. Every case carries a `note` explaining
what it tests — a case nobody can interpret when it regresses is worse than no case.
