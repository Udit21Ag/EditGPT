# Technical debt

**Read when:** you are about to defer a problem, or choosing what to improve next.
**Solves:** keeping known compromises visible instead of rediscovering them.
**Authority:** the register below is the record. Add to it; do not curate it away.

**This is not a changelog.** Completed work does not belong here. Only deferred
engineering problems: temporary workarounds, known duplication, skipped optimisations,
missing coverage, architectural compromises, and dependency, compute or environment
limitations.

## Recording an item

Append an entry with all fields. An item without a **trigger** never gets paid down.

```
### TD-00N — Title
Status: open | in progress | resolved | accepted
Priority: P0 critical | P1 high | P2 useful | P3 optional
Identified: YYYY-MM-DD
Area: package or component
Problem: what is wrong
Why it matters: the concrete cost, ideally measured
Workaround: what we do instead today
Why deferred: what made fixing it now the wrong call
Trigger: the condition under which this must be addressed
Resolution: the approach, if known
```

`accepted` means a deliberate permanent decision — the cost is understood and will not be
paid. It still stays visible.

## Register

| ID     | Title                                                        | Status   | Priority | Area      |
| ------ | ------------------------------------------------------------ | -------- | -------- | --------- |
| TD-001 | CLIPSeg runs on torch, costing ~1 GB of overhead             | open     | P1       | models    |
| TD-002 | Cast shadows survive object removal                          | accepted | P1       | models    |
| TD-003 | Both erasers smear geometric structure                       | open     | P2       | models    |
| TD-004 | Mask dilation bleeds into occluding foreground               | open     | P2       | models    |
| TD-005 | Background op only handles flat backdrops                    | open     | P2       | models    |
| TD-006 | Two of seven planned operations unimplemented                | accepted | P2       | models    |
| TD-007 | Eval quality is not diffed against main in CI                | open     | P1       | evals     |
| TD-008 | Add-mask plausibility is unchecked                           | open     | P2       | providers |
| TD-009 | Visible transition artifact at the mask boundary             | open     | P2       | models    |
| TD-010 | Upscaling is too slow to be interactive                      | accepted | P2       | models    |
| TD-011 | Pass reporting conflated "not applicable" with "rolled back" | resolved | P1       | models    |
| TD-012 | Grounding does not generalise beyond the dev set             | open     | P0       | models    |
| TD-013 | Fill cost is a poor proxy for real quality                   | open     | P0       | core      |
| TD-014 | Fine-tuning CLIPSeg is blocked on hardware                   | accepted | P1       | models    |

---

### TD-001 — CLIPSeg runs on torch, costing ~1 GB of overhead

Status: open · Priority: P1 · Identified: 2026-08-25 · Area: `packages/models`
**Problem:** text grounding uses a torch model where every other model is ONNX. Measured
peak resident set is ~1188 MB for weights of ~150 MB.
**Why it matters:** it is the single largest memory consumer on an 8 GB machine and the
main reason the full pipeline approaches its ceiling.
**Workaround:** the `text` extra is optional; the slot evicts aggressively.
**Why deferred:** Phase 0 measured the easy path first, deliberately. Exporting is real
work and the pipeline fit without it.
**Trigger:** the worker budget is breached, or a second torch model is proposed.
**Resolution:** export to ONNX int8 and re-measure. Expected to be the largest single
saving available.

### TD-002 — Cast shadows survive object removal

Status: accepted · Priority: P1 · Identified: 2026-08-25 · Area: `packages/models`
**Problem:** removing an object leaves its cast shadow.
**Why it matters:** the most visible remaining quality defect on outdoor photographs.
**Workaround:** a gated residual second pass reduces it substantially — a dark blob
becomes a faint smudge — but does not eliminate it.
**Why deferred:** the models that solve it properly are diffusion-based and require a
GPU. A heuristic detector was built and tested; it did not work reliably.
**Trigger:** a GPU host becomes available, or a lightweight shadow model appears.
**Resolution:** an ObjectClear-class model on suitable hardware. Accepted as a v1
limitation until then.

### TD-003 — Both erasers smear geometric structure

Status: open · Priority: P2 · Identified: 2026-08-25 · Area: `packages/models`
**Problem:** removing an object that occludes hard geometry leaves a smear where the
structure should continue.
**Why it matters:** the worst remaining case in the golden set; product shots and
interiors are full of straight edges.
**Workaround:** none. The multi-pass loop tries and correctly rolls back.
**Why deferred:** neither available eraser reconstructs geometry; this is a model
limitation, not a bug.
**Trigger:** a structure-aware inpainter that fits the memory budget.
**Resolution:** unclear. Possibly line-segment detection to condition the fill.

### TD-004 — Mask dilation bleeds into occluding foreground

Status: open · Priority: P2 · Identified: 2026-08-25 · Area: `packages/models`
**Problem:** dilation treats every mask boundary as a background boundary, so it eats
into objects standing in front of the target.
**Why it matters:** silently damages content the user did not ask to change — the worst
class of defect, because it is not where they are looking.
**Trigger:** any report of unintended damage near an edit.
**Workaround:** none.
**Why deferred:** found late in Phase 0; the fix needs instance information the mask
does not currently carry.

**Resolution:** suppress dilation across boundaries the segmenter assigns to a different
instance. An occlusion edge is not a background edge.

### TD-005 — Background op only handles flat backdrops

Status: open · Priority: P2 · Identified: 2026-08-26 · Area: `packages/models`
**Problem:** background replacement floods inward from the image border, which works for
a product shot and not for a real scene.
**Why it matters:** limits the operation to one photo genre.
**Workaround:** the detector returns nothing when the border is not uniform, so the
caller falls back to semantic segmentation — which drops thin structures.
**Why deferred:** proper matting needs a model never benchmarked under the memory budget.
**Trigger:** a background case on a real scene enters the golden set.
**Resolution:** benchmark a lightweight matting model; keep flood-fill as the fast path.

### TD-006 — Two of seven planned operations unimplemented

Status: accepted · Priority: P2 · Identified: 2026-08-25 · Updated: 2026-08-26 · Area: `packages/models`
**Problem:** `RESTYLE` and `RETOUCH` are declared in the contract and not implemented.
`UPSCALE` shipped on 2026-08-26 (Real-ESRGAN x2, tiled) and is now advertised.
**Why it matters:** the contract promises more than the system does.
**Workaround:** the gateway's capabilities endpoint advertises only what works, so
clients are not misled.
**Why deferred:** `RESTYLE` has no free instruction-editing model at all; the other two
are additive and were out of scope.
**Trigger:** a user-facing commitment to any of them.
**Resolution:** `RETOUCH` is a straightforward addition. `RESTYLE` blocks on provider
availability and may never be free.

### TD-007 — Eval quality is not diffed against main in CI

Status: open · Priority: P1 · Identified: 2026-08-26 · Area: `evals`
**Problem:** `make eval` runs locally and by hand. Nothing compares a branch's quality
against `main` automatically.
**Why it matters:** quality regressions are invisible until someone thinks to look. This
undercuts the whole point of having a golden set.
**Workaround:** manual comparison against `evals/out/report.json`.
**Why deferred:** it consumes real free-tier provider quota per run, so it cannot go on
every push without a budget decision.
**Trigger:** any quality regression that reaches `main`.
**Resolution:** run the local-only subset per PR and comment the diff; keep provider
cases nightly.

### TD-008 — Add-mask plausibility is unchecked

Status: open · Priority: P2 · Identified: 2026-08-26 · Area: `packages/providers`
**Problem:** an addition mask is validated for size and emptiness but not for whether the
requested object plausibly belongs there.
**Why it matters:** measured — a clean, large mask in a semantically implausible location
produced incoherent output while consuming provider quota.
**Workaround:** none; the request is sent and the result is poor.
**Why deferred:** needs the critic, which is a later phase.
**Trigger:** the critic component being built.
**Resolution:** reject implausible placements before the remote call, not after.

### TD-009 — Visible transition artifact at the mask boundary

Status: open · Priority: P2 · Identified: 2026-08-26 · Area: `packages/models`
**Problem:** on large masks over textured backgrounds, the join between filled and
original pixels is visible as a soft vertical seam.
**Why it matters:** it is the artifact a viewer notices first on an otherwise good
result, because the eye is drawn to a straight edge in an organic scene.
**Measured:** on the horse case, the strongest vertical edges in the output sit at
columns 367/371/373 — immediately outside a mask spanning 73–366 — at ~1.5x the energy of
surrounding grass texture. Two hypotheses were tested and refuted first: alpha bleeding at
the crop-window border (measured 0.000 at the border on every case) and the crop window
itself (the seam is at the mask edge, not the window edge).
**Workaround:** the Gaussian feather hides it on small and mid-size masks.
**Why deferred:** the fix is gradient-domain blending, which is a compositing change
worth doing deliberately rather than folding into a review.
**Trigger:** any case where the seam is the dominant remaining artifact.
**Resolution:** Poisson blend the patch instead of alpha-feathering it, so the boundary
gradient is matched rather than averaged. Already anticipated in the plan's compositor.

### TD-010 — Upscaling is too slow to be interactive

Status: accepted · Priority: P2 · Identified: 2026-08-26 · Area: `packages/models`
**Problem:** 2x enhancement of a 1024x1024 image takes ~84 s on CPU.
**Why it matters:** it cannot sit on a request path; a user waiting 84 s assumes failure.
**Measured:** median of three on a 512x512 input — tile 192 -> 14.8 s / 336 MB,
256 -> 28.0 s / 349 MB, 384 -> 28.6 s / 630 MB. Larger tiles are slower because the fixed
overlap makes them reprocess proportionally more area, not because the model is slower.
**Workaround:** the eval drives it at a reduced size, so the suite measures behaviour
rather than patience.
**Why deferred:** inherent to a super-resolution network on CPU, not a defect. The job
queue that makes it acceptable is a later phase.
**Trigger:** upscaling being exposed to users.
**Resolution:** run it as an asynchronous job with progress. A GPU host removes it entirely.

### TD-011 — Pass reporting conflated "not applicable" with "rolled back"

Status: resolved · Priority: P1 · Identified: 2026-08-26 · Area: `packages/models`
**Problem:** `EraseOutcome.summary()` rendered every non-kept pass as "(rolled back)", so
the eval table reported strategies as tried-and-rejected when no model had run.
**Why it matters:** it misrepresented what the pipeline did in the project's primary
quality report. Nine of twelve removals were mislabelled.
**Resolution:** `PassRecord.attempted` now distinguishes them, `summary()` renders "(n/a)",
and `EraseOutcome.rolled_back` counts only real rejections. Fixing it also revealed a test
passing for the wrong reason — it asserted "nothing was kept" on a case where no strategy
ever applied. Kept here as a record: reporting defects are worth the same scrutiny as
behavioural ones, because they corrupt every decision made from the report.

### TD-012 — Grounding does not generalise beyond the dev set

Status: open · Priority: P0 · Identified: 2026-08-26 · Area: `packages/models`
**Problem:** on our 18 hand-built cases, text-to-mask succeeds 10/11 (~91%). On 250
held-out RefCOCOg samples it reaches **mIoU 0.389, precision@0.5 = 0.396**, with 36% of
predictions below IoU 0.1.
**Why it matters:** the dev-set number is the one quoted everywhere, and it is roughly
double the held-out reality. Every downstream decision was made on the optimistic figure.
**Measured:** `benchmarks/out/grounding.json`. By mask source: sam-refined 0.469 (n=185),
clipseg-seed 0.234 (n=45), no match at all 0.0 (n=20, 8% of phrases). By phrase length:
≤5 words 0.469, >5 words 0.367.
**Workaround:** the brush is always available, and the product's own prompts are simpler
than RefCOCOg's relational expressions.
**Why deferred:** the fix is a better grounding model, not a threshold. Sweeping the
confidence gate over its whole range moves mIoU by 2.5 points (0.375 to 0.401), so no
tuning available to us closes this.
**Trigger:** any claim about grounding accuracy made outside the dev set.
**Resolution:** see TD-014. A trained referring-expression segmentation model reaches
mIoU 0.65-0.75 on this benchmark; zero-shot CLIPSeg plus SAM is simply a weaker approach.

### TD-013 — Fill cost is a poor proxy for real quality

Status: open · Priority: P0 · Identified: 2026-08-26 · Area: `packages/core`
**Problem:** `fill_metrics(...).cost` measures **plausibility** — how well a fill agrees
with the pixels around it — and we have been using it as a stand-in for **fidelity**, how
close the fill is to what was actually behind the object. Those are different properties,
and on paired data the assumption that one implies the other fails.

**Measured, n=138 individual fills with paired ground truth:** Spearman correlation
between cost and SSIM-against-truth is **+0.128**. Cost is lower-is-better and SSIM is
higher-is-better, so a proxy that worked would correlate _negatively_. Choosing by cost
picks the better eraser in 43.5% of cases; always choosing LaMa picks it in 75.4%.
**Why it matters:** this metric drives the router, the multi-pass keep-or-rollback
decision, and the `cost` column in the eval table. The routing built on it is worse than
doing nothing — routed SSIM 0.5630 against always-LaMa 0.5791 — and the same doubt now
attaches to every decision made from it.

The metric is not _wrong_; it measures what it says it measures. The error was ours, in
treating a plausibility score as a quality score without ever checking the correlation.
**Measured:** `benchmarks/out/removal.json`, n=69 RemovalBench samples with paired GT.
**Workaround:** none in place.
**Why deferred:** the replacement is not obvious. A learned chooser was built and
**rejected** — it beat the majority baseline on accuracy (+0.058) and not on SSIM
(-0.0001), because the entire choice is worth at most +0.0045 SSIM.
**Trigger:** before any further investment in routing logic.
**Resolution:** the evidence points at _simplifying_: use one eraser for removal and
delete the cost-based escalation. Confirm on a second paired dataset first — this is one
benchmark, and our own dev set disagrees with it.

### TD-014 — Fine-tuning CLIPSeg is blocked on hardware

Status: accepted · Priority: P1 · Identified: 2026-08-26 · Area: `packages/models`
**Problem:** grounding is the largest quality bottleneck (TD-012) and the obvious remedy
is fine-tuning CLIPSeg on RefCOCOg, or replacing it with a trained referring-expression
segmentation model.
**Why it matters:** it is worth roughly twice the quality of any pipeline change
available to us — published RES models reach mIoU 0.65-0.75 where we measure 0.389.
**Workaround:** confidence-gated SAM refinement recovers some of the gap (0.234 to 0.469
where it applies).
**Why deferred:** **no training is possible on this hardware.** Fine-tuning CLIPSeg needs
a GPU; nothing in this project trains, and adding a training path on an 8 GB CPU machine
would be theatre. The data is free and ready (RefCOCOg train: 42,226 expressions).
**Trigger:** access to a GPU, or a hosted fine-tuning budget.
**Resolution:** fine-tune CLIPSeg's decoder on RefCOCOg train, hold out val, report mIoU
against the 0.389 baseline recorded here. Everything needed except the hardware exists.
