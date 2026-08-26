# ADR-0001 — Model selection and local/remote routing

- **Status:** accepted (extended after Phase 1)
- **Date:** 2026-08-25
- **Context:** Phase 0 feasibility spike. MacBook Air M1, 8 GB unified memory, free-tier only.
- **Supersedes:** the provider and model tables in `docs/PLAN.md` §1 and §4 as originally written.

> **Case numbering changed after this ADR was written.** The photo set was renumbered on
> 2026-08-25: old `i12` (framed picture) is now **i10**, old `i13` (bedroom air conditioner)
> is now **i11**, and the mug (old `i11`) and fruit-bowl (old `i10`) photos were deleted.
> Every case id below refers to the **original** numbering. `spike/assets/cases.json` carries
> the current one, with the mapping in each case's `note`.

## Decision

**Proceed to Phase 1, with three scope changes.** Removal works and ships. Additions ship
with a stated quality cap. Shadow removal is explicitly out of v1.

### Final model roster

| #   | Model                                          |    Size | Role                             | Where                          |
| --- | ---------------------------------------------- | ------: | -------------------------------- | ------------------------------ |
| 1   | **MobileSAM** (Acly ONNX export)               |   44 MB | box/point/brush → precise mask   | local, ONNX CPU                |
| 2   | **CLIPSeg-rd64-refined**                       | ~150 MB | text → seed mask                 | local, torch (ONNX in Phase 4) |
| 3   | **MI-GAN pipeline v2** (Picsart, ICCV'23)      |   28 MB | **primary eraser**               | local, ONNX CPU                |
| 4   | **Big-LaMa** (Carve ONNX)                      |  208 MB | **escalation eraser**            | local, ONNX CPU                |
| 5   | **SD-1.5-inpainting** on Cloudflare Workers AI |       — | additions only                   | remote, free 10k neurons/day   |
| 6   | **Gemini 3.6 Flash** (text)                    |       — | intent parsing, critic reasoning | remote, free tier              |

### Routing rule

```
op == remove          -> local only, never remote
                         MI-GAN first (0.3-0.75 s)
                         score the fill with fill_metrics
                         escalate to Big-LaMa if the score is poor
op == add | replace   -> Cloudflare, mask constrained to empty regions
intent / critique     -> Gemini 3.6 Flash (pinned id, not the -latest alias)
```

Removal never escalates to the remote lane. That is not a preference — the free remote
lane is measurably incapable of removal (see below), so there is nothing to escalate to.

## Measurements

### Models in isolation

| run                   | peak RSS (MB) | cold load (s) |                     warm p50 (s) |
| --------------------- | ------------: | ------------: | -------------------------------: |
| MobileSAM (enc + dec) |           620 |          0.34 | 0.025 (decoder) / 0.41 (encoder) |
| Big-LaMa erase        |           964 |          3.31 |                             2.82 |
| MobileSAM + MI-GAN    |          1470 |          0.08 |                             0.36 |
| CLIPSeg text-to-mask  |          1188 |          4.16 |                             0.22 |

### Pipelines

Peak RSS varies **±15% run to run** — ONNX arena allocation is not deterministic — so the
range across repeated runs is given, and the budget below is set from the maximum observed,
not from one sample.

| run                                   | peak RSS (MB) | warm p50 (s) |
| ------------------------------------- | ------------: | -----------: |
| box → MobileSAM → LaMa @1024          |     1556–1687 |      4.1–4.2 |
| text → CLIPSeg → MobileSAM → LaMa     |     1852–2031 |      3.9–4.7 |
| native 15.9 MP, crop vs whole         |     1666–2116 |      5.2–5.6 |
| LaMa vs MI-GAN head-to-head           |     1586–1809 |         0.38 |
| Cloudflare Workers AI (network-bound) |           792 |     6.1–11.0 |

### MI-GAN resolution scaling

| input   | seconds | RSS delta (MB) |
| ------- | ------: | -------------: |
| 0.8 MP  |    0.37 |            600 |
| 3.1 MP  |    0.52 |           1150 |
| 7.1 MP  |    0.48 |            951 |
| 15.9 MP |    0.75 |           1150 |

Memory plateaus rather than scaling with input; the ONNX graph crops internally.
LaMa needs 2.8 s and a hand-built crop/paste-back to do the same job.

## What the evidence says

### Text-to-mask is good enough to be the default

CLIPSeg localises **10/11**; MobileSAM refinement also **10/11**. Users do not have to
brush. Scored as predicted-mask bounding box vs. a hand-drawn reference box plus a
precision check — mask-vs-box scoring is invalid because a correct mask of a non-convex
object (i8's lattice) covers only ~37% of its own bounding box.

Refinement helps where the seed is weak and hurts where it is strong: i8 0.677→0.854,
i12 0.762→0.907, i9 0.573→0.671, but i1/i4/i11 each lose 0.03–0.05 and i13 collapses
0.230→0.050. **Refinement must be conditional.** The SAM decoder already returns
`iou_predictions`, which the spike discards; that is the runtime signal to gate on.

### The two erasers are complementary, so ship both

Composite photometric cost gives LaMa 7 / MI-GAN 4, but the metric is confounded where
the reference ring contains a cast shadow, so the images decide. MI-GAN erases i7's bird
perfectly where LaMa leaves a white ghost; LaMa is clean on i11's mug where MI-GAN
produces a crumpled-napkin artifact. MI-GAN is 7× smaller, 40× faster to load and 3.6×
faster at full resolution, so it is the default and LaMa is the escalation.

### LaMa's failures were fixable; two fixes did it

1. **Chroma, not luminance.** LaMa reproduces luminance to 3/255 and drifts chroma ~25
   toward blue — that is the pale-rectangle ghost. A Lab **chroma-only** transfer against
   a ring of real pixels fixes it. Matching luminance too re-stamps the object wherever
   the ring spans two surfaces (i11: table plus cast shadow).
2. **Dilation must scale with the object.** 12 px was ample at 1024 px and left a
   rectangular outline at 15.9 MP. Residual edge energy on i12 reaches the blank-wall
   floor (1.36 vs. 1.52) at ~5% of the object's longest side and buys nothing beyond.

Together these fixed **i8**, which had been logged as an unresolvable failure. Both
erasers now remove the Eiffel Tower cleanly.

### Crop vs. whole crosses over

i12 (1.9% of frame): crop clean, whole unusable. i11 (19.8%): comparable. Route on object
size relative to frame — and `whole` should never run above a few megapixels.

### The free remote lane cannot remove objects

Gemini's image API returns `limit: 0` on all six image models the key can list; Google
withdrew the image free tier in December 2025. Text is unaffected.

Cloudflare's SD-1.5-inpainting fills the masked hole with an object matching the prompt,
every time. On i1: "empty cobblestone plaza" → a stone slab; "background" → **a white
car**; the same prompt at guidance 3.0 → a boulder. The API rejects an empty prompt
(`length must be >= 1`). i11 became a different cup of coffee; i13 became a wall niche.
This is the model's nature, not a prompting failure.

Additions do work: i3's moustache is convincing, though identity drifts — the plan's
`preserve_identity` constraint is real and unhonoured. i10's banana corrupts the apple
when the mask overlaps it and produces only a weak shape over clear space, so **add-masks
must be constrained to empty regions**.

## Rejected options, with reasons

| Option                                     | Why not                                                                                                                                                   |
| ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Gemini image API as the generative lane    | Free tier withdrawn Dec 2025; `limit: 0` on every image model                                                                                             |
| SD-1.5-inpainting for removal              | Inserts an object every time; robust to prompt and guidance changes                                                                                       |
| **ObjectClear** (CVPR'26) / **OmniEraser** | The correct answer for shadows — removes objects _with_ their effects — but built on SDXL-Inpainting and FLUX respectively. GPU-only, no ONNX or CPU path |
| **MAT**, ZITS, FcF, LDM                    | IOPaint recommends LaMa/MAT/MI-GAN; MI-GAN plus LaMa already covers the range, and ZITS's wireframe module is documented as very slow on CPU              |
| GroundingDINO-tiny                         | Unnecessary — CLIPSeg reached 10/11 unaided. Removed from the roster                                                                                      |
| Heuristic shadow absorption                | Built and tested: partial improvement on i1, none on i11, and it induced a hallucinated box. Does not work                                                |
| Adopting **IOPaint** wholesale             | It is a server plus UI with its own architecture; taking it would replace the project rather than build it. We took its model choices instead             |
| Adopting **SegMedit**                      | Same SAM + LaMa pipeline this spike already built and measured, without the agent layer                                                                   |

## Consequences

1. **`docs/PLAN.md` §1 and §4 are superseded** by the roster and memory table here.
2. **The 1200 MB "model slot" was too small by roughly half.** One pipeline resident measures
   1470–2116 MB across repeated runs. The worker budget becomes **2.2 GB**, set from the worst
   observed case rather than an average, and `ModelSlot(max_resident=1)` means one _heavy_
   model at a time with MobileSAM's embedding cached and released. The RSS regression test in
   Phase 2 must assert against 2.2 GB, and must run repeatedly — a single sample would have
   set this limit 20% too low.
3. **Offloading the heavy agents to a Hugging Face Space is load-bearing, not an optimisation.**
4. **CLIPSeg must be exported to ONNX int8 in Phase 4.** 1188 MB of RSS for a 150 MB model
   is torch overhead, and it is the single largest saving available.
5. **`fill_metrics` becomes the CriticAgent's scorer** in Phase 7 — it already routes
   MI-GAN vs LaMa here. Known confound: it misreads a fill whose surrounding ring spans
   more than one surface.

## End-to-end verification

`spike/bench/run_final.py` implements the routing rule above and runs all 11 removal cases
prompt-only, no brush. Result: **5/11 completed on the MI-GAN fast path at ~1.2 s**, 6/11
escalated to LaMa at ~3.9 s. The SAM confidence gate fired on exactly one case — i13, at
iou 0.83 — which is precisely the case where refinement had collapsed the mask (0.230 →
0.050). The gate was derived from that evidence and, run blind, caught it.

## Follow-up: what the generative lane can and cannot place

Three additional `add` probes after the main spike sharpened the rule beyond "mask empty regions":

| probe                          | mask                                                            | result                                                                                                                                            |
| ------------------------------ | --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| television on the bedroom wall | clear wall between the AC and the table                         | **Works.** Correctly mounted, black bezel, plausible shadow. Ignored "screen switched off" and rendered content — prompt adherence, not placement |
| chair, generous box            | overlaps the desk's right panel, cup and mouse                  | **Good chair, wrecked scene.** The object is convincing; everything that was inside the mask is gone                                              |
| chair, clear-space box         | the only unoccupied region, floating white space above the desk | **Preserved the scene, incoherent object.** Produced an abstract mesh fragment, not a chair                                                       |

So an `add` mask needs three things, not one: **clear of existing content**, **large enough**,
and **semantically plausible for the object**. A wall is where a television goes, so the first
probe worked. Floating white space above a desk gives the model nothing to ground a chair on,
so the third failed even though the mask was clean. A user's brush stroke satisfies the first
two; only the CriticAgent can catch the third, and it should reject implausible placements
before spending a remote call.

## Prompt sweeps: what description actually helps

`spike/bench/sweep.py` runs five phrasings per case and scores them. Removal and addition
are different problems: the erasers never see text, so there a prompt only shapes the
**mask**; for additions the prompt reaches SD-1.5 verbatim and is scored by CLIP similarity
between the edited region and the request.

| case                    | best phrasing                                           | effect                                                                                                                               |
| ----------------------- | ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **i11** air conditioner | "the wall mounted white air conditioner with vents"     | bbox_iou **0.29 → 0.865**, fill cost **103 → 23**, and it drops off the LaMa escalation onto the MI-GAN fast path: **6.0 s → 1.4 s** |
| **i6** laptop           | "the laptop and its screen and keyboard"                | bbox_iou 0.874 → **0.888**, fill cost 31.3 → **22.1**, also onto the fast path: **5.9 s → 1.2 s**                                    |
| **i1** car              | no improvement available                                | All five phrasings collapse to the identical mask (bbox_iou 0.776) — SAM normalises them                                             |
| **i3** moustache        | "a realistic moustache on the upper lip" (the simplest) | CLIP 31.40. **Elaborate prompt engineering made it worse**, down to 28.54                                                            |
| **i6c** chair           | none                                                    | Every variant scored **negative** against the untouched region. Not a prompt problem                                                 |

Three things this established beyond the individual wins:

1. **CLIP similarity works as a blind critic signal.** On i3 the lowest-scoring variant
   (28.54) is precisely the one that rendered no moustache at all — a pale smudge. The
   scorer caught the failure without a human looking. That is `CriticAgent.score_edit`.
2. **Shadows are not reachable through the prompt.** "the car and its shadow on the ground"
   returned an _empty_ CLIPSeg seed — nothing above threshold. Defect 1 stands.
3. **Fill cost alone can be gamed, so the critic must not use it alone.** On i6,
   "the laptop computer resting on the wooden desk" over-segmented to 45% of the frame and
   scored the _best_ fill cost (7.5) while destroying the desk — a large smooth region is
   photometrically consistent with itself. Mask-area sanity has to gate the cost.

Also fixed here: an empty CLIPSeg seed crashed `refine()` on a `zero-size array` reduction.
A prompt matching nothing is a legitimate outcome, so it now returns an empty mask and zero
confidence and the caller falls through to asking the user.

## Second-pass feedback: does re-running the models help?

`spike/bench/refeed.py` tests three ways of feeding a result back, all scored over one
fixed region so a pass that erases more area cannot flatter itself:

| strategy                                                   | result                                                  |
| ---------------------------------------------------------- | ------------------------------------------------------- |
| **again** — same mask, second pass                         | No value anywhere: +0.6 to +3.5 cost on all three cases |
| **cross** — LaMa then MI-GAN                               | Inconsistent: −5.7 on i2, +6.6 on i1, +18.6 on i6       |
| **residual** — detect what is still wrong and erase _that_ | **The one that works**, when gated                      |

The residual pass matters because it rescues the shadow defect. The same dark-region
detector **failed on the original image** — the reference median was polluted by the object
and its own shadow — but run _after_ the object is gone it has a clean surface to compare
against, and it finds the shadow. On i1 the car's cast shadow drops from a dark blob to a
faint smudge.

It must be gated on how much it grows the mask, and `fill_metrics` **cannot make that call**:

| case        | mask growth | outcome                                                                                                                        |
| ----------- | ----------: | ------------------------------------------------------------------------------------------------------------------------------ |
| i5 painting |        +27% | applied, clean                                                                                                                 |
| i1 car      |        +35% | applied — **shadow substantially reduced**                                                                                     |
| i6 laptop   |        +78% | **blocked.** Ungated it flattens the desk to a bluish smear — and scores the _best_ cost of any variant (−18.8) while doing so |
| i2 woman    |       +119% | blocked                                                                                                                        |
| i9 bat      |       +124% | blocked                                                                                                                        |

`RESIDUAL_MAX_GROWTH = 0.50` separates these cleanly. In the final run the pass fires on
3/14 cases and is correctly withheld from every case it would have damaged.

**This is the third time a photometric score has picked the visually worse image**, and the
pattern is always the same: a larger flat erase is photometrically self-consistent. The
CriticAgent must combine fill cost with a mask-growth penalty, never use cost alone. Cost is
valid for comparing fills _within a fixed mask_ and invalid for comparing masks.

Latency: the shadow pass costs an extra LaMa call, ~2.5-3.5 s. On i11 it fired for only a
10% growth and tripled the wall time for a marginal gain, so Phase 5 should also require a
minimum residual area, not just a maximum.

## Phase 1 addendum: the multi-pass policy

Every image now goes through the erasers **at least twice**, with a third pass when the
second leaves the fill above `ACCEPT_COST`. Because Phase 0 measured that a *naive*
second pass makes things slightly worse (+0.6 to +3.5 cost on every case tried), the
second pass always runs but its output is kept only if it scores better. Mandatory work,
verified result — a rolled-back pass is recorded rather than hidden, because a strategy
that was tried and rejected is a finding the critic loop needs.

Measured across the golden set:

| behaviour | cases |
|---|---|
| escalation kept | i1, i4, i5, i8, i11 |
| residual (shadow) pass kept | i1 |
| second pass attempted and rolled back | i2, i4, i5, i6, i7, i8, i9, i10 |
| stopped at two passes because the fill was already acceptable | i7, i9, i10 |

Peak RSS for the whole run: **1072 MB**, comfortably inside the 2.2 GB worker budget.

## Phase 1 addendum: flat backgrounds are compositing, not generation

The `i6c` case — recolour a product shot's white backdrop to green — exposed that
semantic segmentation is the wrong instrument for a flat backdrop. Asked for "the wooden
laptop desk", CLIPSeg plus MobileSAM returned the tabletop and **dropped the thin legs**,
which were then painted over, and its boundary sat inside the object, leaving a fringe of
the old backdrop.

Flooding inward from the image border has neither failure: thin structures survive
because they are simply not background, and the boundary lands exactly where the colour
changes. `flat_background_mask` returns `None` when the border is not uniform enough to
be a backdrop, so a real scene falls back to the semantic mask. No generative call is
involved — asking a diffusion model to paint a flat colour costs a round trip and returns
something less exact.

## Known defects carried into Phase 1

| #   | Defect                                                                                                                                         | Status                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Cast shadows survive removal                                                                                                                   | **Partially mitigated.** A gated residual second pass reduces i1's car shadow from a dark blob to a faint smudge. Not eliminated — full removal still needs an ObjectClear-class model and a GPU                                                                                                                                                                                                                                                                                         |
| 2   | i6: both erasers smear the desk's geometric structure                                                                                          | Worst remaining case; retest after Phase 5 compositing                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| 3   | i13: CLIPSeg finds only the vent slats, SAM narrows further                                                                                    | **Downgraded — this is a prompt problem, not a model problem.** "the white air conditioner unit mounted on the wall" gives bbox_iou 0.865 (from 0.050) and SAM confidence 1.00 (from 0.83), and the erase is clean. IntentAgent's `expand_prompt` skill covers it. Caveat: expansion must be _specific_ — "the white rectangular box on the wall" grabbed half the wall, so the SAM-confidence gate and a mask-area sanity check must reject bad expansions. Brush stays as the fallback |
| 4   | Additions capped by SD-1.5 quality; identity drifts; prompt adherence is loose ("screen switched off" ignored)                                 | Lifted the day billing is enabled on Gemini                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| 6   | An `add` mask must be clear, large, **and** semantically plausible. Nothing currently checks the third                                         | Phase 7: CriticAgent rejects implausible placements before the remote call                                                                                                                                                                                                                                                                                                                                                                                                               |
| 5   | **Dilation bleeds into occluding foreground.** On i8 the 5%-of-object dilation ate part of the jumper's shoe, which sits in front of the tower | Found in the final end-to-end run. Phase 5: suppress dilation across boundaries SAM assigns to another instance — an occlusion edge is not a background edge                                                                                                                                                                                                                                                                                                                             |
