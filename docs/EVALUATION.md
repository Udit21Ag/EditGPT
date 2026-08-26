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
one fixed region and charges `GROWTH_PENALTY` per unit of area erased beyond the base
mask. Use `compare()`, not raw `.cost`, whenever masks differ.

## Layers of measurement

### 1. Localisation — is the mask on the right thing?

`box_metrics(mask, reference_box, shape)` → `inside_frac`, `box_recall`, `bbox_iou`, `hit`.

Gated on `inside_frac` **with** `bbox_iou` — box against box, the only like-for-like
comparison available without painted ground truth.

- Mask-against-box is invalid: a correct mask of a non-convex object (a lattice tower)
  covers ~37% of its own bounding box and would score as a failure.
- Precision alone is invalid: a six-pixel mask in the right place scores `inside_frac`
  1.000 and is useless. That passed a rescore while the object was still visibly present.

Current: CLIPSeg localises **10/11**; after MobileSAM refinement, also 10/11.

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

## What we can and cannot measure

- **Our own fixtures have no paired ground truth.** For removal there is no "correct"
  output image for them, so reference metrics are unavailable *there*. That is a property
  of our fixtures, not of the problem: `benchmarks/` uses RemovalBench, which ships the
  scene with and without the object, and SSIM and PSNR are valid on it.
- Do not report FID on 18 images.
- **Shadows.** No metric detects a surviving cast shadow. It is caught by eye.
- **Identity drift** on additions is observed, not measured.
- **Human evaluation** has not been run. When it is: blind A/B, ≥5 raters, a fixed
  criterion per question, and report inter-rater agreement or do not report at all.

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
