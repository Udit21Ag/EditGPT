# Model and feature review

**Date:** 2026-08-26 · **Basis:** 18 golden cases, `evals/out/report.json`, measured on an
M1 MacBook Air, 8 GB, CPU only.

A senior review of what each model does, how well, and what would actually improve it.
Numbers are from this machine in this session, not quoted from documents.

---

## 1. The model roster

| Model | Role | Disk | Peak RSS | Latency | Verdict |
|---|---|---:|---:|---:|---|
| MobileSAM enc+dec | box/seed → precise mask | 44 MB | 620 MB | 0.41 s enc, **24 ms dec** | Keep. Excellent value. |
| CLIPSeg-rd64 | text → coarse seed | ~150 MB | **1188 MB** | 0.22 s | Keep, but export to ONNX. |
| MI-GAN pipeline v2 | primary eraser | 28 MB | ~1150 MB | **0.29 s** | Keep. Best ratio in the stack. |
| Big-LaMa | escalation eraser | 208 MB | 964 MB | 2.8 s | Keep. Earns its place 6/11. |
| Real-ESRGAN x2 | tiled 2× enhancement | 67 MB | ~430 MB | 14.8 s @0.26 MP | Keep, but async only. |
| SD-1.5-inpaint (remote) | add / replace | — | — | 4–10 s | Keep. No free alternative. |

### MobileSAM — the strongest component

24 ms on the decoder makes tap-to-select genuinely interactive; the encoder runs once per
image and is cached. The confidence gate (`iou >= 0.85`) is doing real work: it fired on
exactly the case where refinement collapsed a mask, without being told which one.

**Improve:** the decoder returns three mask candidates and we use one. Offering the user
the alternates when confidence is marginal is nearly free and would address the ambiguity
failures directly.

### CLIPSeg — correct, and disproportionately expensive

Localises 10/11. The cost is the problem: **1188 MB of resident memory for 150 MB of
weights** is torch overhead, and it is the single largest consumer in the stack.

**Improve:** export to ONNX int8 (TD-001). This is the largest saving available anywhere
in the system.

### MI-GAN — the best decision in the project

7× smaller than LaMa, 40× faster to load, 3.6× faster at full resolution, and its graph
crops internally so it needs none of the hand-built paste-back machinery. It resolves
5 of 11 removals alone.

**Improve:** it is used only as a first pass. On the cases it wins it wins outright, so a
cheap predictor of "MI-GAN will be sufficient" would avoid loading LaMa at all.

### Big-LaMa — justified, but only just

Kept in 6 of 11 removals, so it is not redundant. But it costs 2.8 s and 208 MB, and its
fixed 512 input forces the crop-and-paste-back path that produces the boundary artifact in
TD-009.

**Improve:** it is the natural thing to replace if a structure-aware inpainter appears
that fits the budget. Not before.

### Real-ESRGAN x2 — works, but not interactively

Adds **3.0× the detail of bicubic** by gradient magnitude, and the tiled blend leaves no
visible lattice (seam energy 1.02× surrounding texture). But a 1024² input takes **84 s**.

**Improve:** it belongs behind the job queue. Tile 192 was measured 2× faster than 256
because the fixed overlap makes larger tiles reprocess more area — worth revisiting if
overlap becomes proportional.

### SD-1.5-inpainting — the weakest link, and unavoidable

Cannot remove objects at all: it fills a masked hole with an object matching the prompt.
Additions work unevenly, and identity drifts visibly on faces.

**Improve:** nothing available under the free-tier constraint. Enabling billing on a
current image model is the single change that lifts the ceiling on every generative
operation at once.

---

## 2. Features, in depth

### REMOVE — 11 cases, median 4.07 s, median cost 22.1

| | |
|---|---|
| Best | i10 framed picture, cost 1.1, IoU 0.907 |
| Worst | i8 Eiffel Tower, cost 56.0 — succeeds visually despite the score |
| Fast path only | 5/11 resolve on MI-GAN alone, ~1.5 s |
| Escalation kept | 6/11 |
| Residual (shadow) pass kept | 2/11 |

**What works:** small and mid-size objects on flat or repeating backgrounds are
essentially solved. Thin objects held by a person (i9) come out clean without eating the
hands.

**What does not:**
- **Cast shadows.** Quantified this session: identical masks score IoU 0.776 against an
  object-only box and **0.500** against a shadow-inclusive one. That 0.28 gap is the
  shadow, expressed as a number for the first time.
- **Geometric structure** (i6's desk) smears in both erasers.
- **Boundary seam** (TD-009): measured at columns 367/371/373 against a mask spanning
  73–366 — ~1.5× surrounding texture energy, immediately outside the mask.

### ADD — 3 cases, median 4.59 s

Works when the region is **clear, large enough, and semantically plausible**. All three
conditions are necessary: a clean, large mask floating in white space above a desk still
produced an incoherent blob, while the same model put a convincing television on a wall.

Identity drift on faces is visible and unmeasured.

### REPLACE — 1 case, 4.51 s

Substituted a horse with a sheep at correct scale and lighting. The strongest generative
result in the set — replacement is easier than insertion because the mask already
describes an object-shaped region.

### BACKGROUND — 2 cases

A flat product backdrop is **solved and free**: flood-fill from the border, recolour, no
model call, 0.02 s, thin table legs preserved. A real scene is **not**: flood-fill
correctly declines, the semantic fallback leaves original foliage and rough hair edges.

### UPSCALE — 1 case

Measurably sharper than bicubic, no tile lattice, bounded memory. Too slow for a request
path.

---

## 3. What the measurements taught us

**Prompt expansion is not monotonic.** Expanding a target phrase helped enormously in one
case and catastrophically hurt in another:

| Case | Bare noun | Expanded | Effect |
|---|---|---|---|
| i11 air conditioner | IoU 0.29 | "the wall mounted white air conditioner with vents" → **0.865** | +0.58 |
| i2 woman | IoU 0.876 | "the woman in the white dress" → **0.319** | −0.56 |

The difference is *what the added words name*. "With vents" describes the whole object;
"in the white dress" names a **part**, and CLIPSeg segments the part. An automated prompt
expander that does not know this will make things worse about as often as better.

**Reference boxes are part of the measurement, and can be wrong.** i7 scored IoU 0.533 on
a mask that was visually perfect; the box had been drawn loose. Tightening it gave 0.647
with the mask unchanged. A metric disagreeing with the eye is not automatically the eye's
fault — but it is not automatically the metric's either.

**A reporting defect corrupts every decision made from the report.** `summary()` labelled
every non-kept pass "rolled back", including strategies that never ran. **Nine of twelve
removals were mislabelled.** Fixing it also exposed a test that passed for the wrong
reason. Reporting deserves the same scrutiny as behaviour.

---

## 4. Ranked improvements

| # | Change | Why | Cost | Evidence it worked |
|---|---|---|---|---|
| 1 | Export CLIPSeg to ONNX int8 | Largest memory saving available; ~1 GB | Medium | Peak RSS on `make memory` |
| 2 | Diff `make eval` against `main` in CI | Quality regressions are currently invisible | Low | A seeded regression is caught |
| 3 | Poisson blend instead of alpha feather | Removes the boundary seam on every large removal | Medium | Edge energy at the mask boundary falls to ~1.0× |
| 4 | Suppress dilation across instance boundaries | Stops silent damage to occluding foreground | Medium | The jumper's shoe survives i8 |
| 5 | Offer SAM's alternate masks when confidence is marginal | Directly addresses ambiguity, nearly free | Low | i11-class cases resolve without prompt surgery |
| 6 | Matting model for real-scene backgrounds | Unblocks BACKGROUND beyond product shots | Medium | i3bg hair edges |
| 7 | Async job for upscale | Makes a working feature usable | Medium | Wall-clock on the request path |
| 8 | Critic rejects implausible add placements | Saves quota and prevents bad output | Medium | i6b-class rejected before the call |

**Not recommended:** replacing either eraser, adding GroundingDINO (CLIPSeg reaches 10/11
unaided), or any model requiring a GPU. Each has been measured or reasoned about and none
earns its cost under the current constraints.

---

## 5. Honest assessment

**Genuinely good:** the routing decision is evidence-based rather than assumed; the
verified multi-pass loop with rollback is unusual and correct; the memory discipline is
real and enforced in CI; the failure modes are characterised rather than hidden.

**Weakest areas:** the generative lane is capped by what is free; shadows remain the most
visible defect; there is no paired ground truth, so all quality metrics are reference-free
and none of them substitutes for looking.

**Biggest risk:** the golden set is 18 cases on one machine. It is enough to catch
regressions and not enough to claim generalisation. Growing it is worth more than tuning
any single model.
