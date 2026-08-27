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

| ID     | Title                                                        | Status   | Priority | Area       |
| ------ | ------------------------------------------------------------ | -------- | -------- | ---------- |
| TD-001 | CLIPSeg runs on torch, costing ~1 GB of overhead             | open     | P2       | models     |
| TD-002 | Cast shadows survive object removal                          | accepted | P1       | models     |
| TD-003 | Both erasers smear geometric structure                       | open     | P2       | models     |
| TD-004 | The mask swallows objects that occlude or touch the target   | open     | P1       | models     |
| TD-005 | Background op only handles flat backdrops                    | open     | P2       | models     |
| TD-006 | Two of seven planned operations unimplemented                | accepted | P2       | models     |
| TD-007 | Eval quality is not diffed against main in CI                | resolved | P1       | evals      |
| TD-008 | Add-mask plausibility is unchecked                           | open     | P2       | providers  |
| TD-009 | Visible transition artifact at the mask boundary             | open     | P2       | models     |
| TD-010 | Upscaling is too slow to be interactive                      | accepted | P2       | models     |
| TD-011 | Pass reporting conflated "not applicable" with "rolled back" | resolved | P1       | models     |
| TD-012 | Grounding does not generalise beyond the dev set             | open     | P1       | models     |
| TD-013 | Fill cost's failure did not replicate on a second dataset    | open     | P2       | core       |
| TD-017 | RemovalBench and RORD disagree about which eraser is better  | open     | P1       | benchmarks |
| TD-018 | Authentication is unimplemented pending a product decision   | resolved | P1       | gateway    |
| TD-019 | Any signed-in user can fetch any image by its digest         | open     | P2       | gateway    |
| TD-014 | Fine-tuning CLIPSeg is blocked on hardware                   | accepted | P2       | models     |
| TD-015 | Relational referring expressions are not grounded at all     | resolved | P1       | models     |
| TD-016 | Recorded result dimensions assume the edit preserves them    | resolved | P2       | store      |
| TD-020 | The background colour is fixed green, not requested          | resolved | P2       | models     |
| TD-021 | Edits come back at 2048 px, not the uploaded resolution      | resolved | P1       | worker     |
| TD-022 | A provider's blank frame was composited as a result          | resolved | P1       | providers  |
| TD-023 | A phrase matching nothing still returns a confident region  | open     | P2       | models     |

---

### TD-001 — CLIPSeg runs on torch, costing ~1 GB of overhead

Status: open · Priority: P2 · Identified: 2026-08-25 · Updated: 2026-08-27 · Area: `packages/models`
**Problem:** text grounding used a torch model where every other model is ONNX. Measured
peak resident set ~1188 MB for weights of ~150 MB.
**Largely addressed on 2026-08-27.** ADR-0002 moved the default text lane to Grounding
DINO, which is ONNX, so **nothing on the shipping path loads torch**. CLIPSeg remains
behind the optional `text` extra as the fallback for "stuff" nouns — sky, grass, a wall —
which an object detector grounds poorly, and `evals/run.py` still reaches it when the
detector abstains.
**Why it still matters:** a deployment that wants that fallback still pays ~1188 MB for
it, and the `text` extra is still installed by default in the root project.
**Workaround:** the detector answers first and abstains rarely (0 of 250 held-out
phrases at the default gate), so the fallback is seldom reached in practice.
**Why deferred:** it is no longer the binding memory constraint, and exporting CLIPSeg is
real work with a smaller payoff than it had.
**Trigger:** the worker budget is breached, or a "stuff" noun case enters the golden set
and makes the fallback load-bearing.
**Resolution:** export CLIPSeg to ONNX int8, or replace the fallback with a lightweight
semantic segmenter and drop torch entirely.

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

### TD-004 — The mask swallows objects that occlude or touch the target

Status: open · Priority: **P1** (was P2) · Identified: 2026-08-25 · Updated: 2026-08-27 · Area: `packages/models`
**Problem:** the mask covers things the user did not name. Two mechanisms compound:
SAM returns a *solid silhouette* for a box prompt, so anything overlapping the target is
inside it; and dilation then treats every boundary as a background boundary and widens
further into whatever stands in front.
**Why it matters:** it silently damages content nobody asked to change — the worst class
of defect, because it is not where the user is looking.
**Observed on the golden set, 2026-08-27**, and reported by eye rather than by any metric:

- **i8** "remove the Eiffel Tower" — the mask is a solid tower silhouette, and the jumping
  figure's shoe sits inside it. The shoe is erased with the tower.
- **i9** "remove the cricket bat" — the mask traces the bat *and* the glove gripping it.
  The batsman's hand is erased.

**Raised to P1** because it is now the most visible remaining defect in the product, and
because it is the failure a user notices immediately.
**Made more visible by ADR-0002, not caused by it.** A tighter detector box gives SAM a
stronger prompt, so the silhouette it fills is more complete — including the occluders.
The previous heatmap seed produced a patchier mask that happened to spare them. Better
grounding made a latent defect legible, which is what better grounding is for.
**Trigger:** already fired.
**Workaround:** none. Every quality number in the eval table looks fine for these cases;
only the images show it.

**Attempted 2026-08-28 and not adopted — `segment.occluder_shield`, off behind
`Thresholds.shield`.** The experiment above was run. It works as designed and still makes
the product worse, which is worth recording in full because the next person will have the
same idea.

The shield probes SAM with point prompts across the band dilation would reach, every
probe *outside* the target, and withholds any returned region that is coherent, bounded,
and mostly not the target. Measured on the golden set:

- `i8`: the jumper's shoe goes from 10.3% erased to 0.5%. The shoe is visibly intact.
- `i4`: unchanged — the horse keeps all of itself.
- `i6`: cost 28.5 -> 15.0, a clear improvement.
- `i8` **overall: worse.** A pale ghost of the tower's base is left standing, and it is
  far more visible than the smeared shoe it prevents.

**What that revealed is the real finding.** Dilation had been silently compensating for
SAM *under*-segmenting an object where it meets clutter. Withhold the ring where the
tower meets the tree line and the compensation goes with it. The bleed and the
under-segmentation are the same 5% of slack doing two jobs, and removing the slack from
one job removes it from both. Occluder detection is not the missing piece; a mask that is
right at the boundary is.

**Two things measured along the way, both worth not re-deriving:**

- *SAM is part-aware.* A probe near a horse's edge returns **the leg** — small, confident,
  low IoU against the whole horse. Filtering on size or IoU shields a quarter of the
  animal from its own erasure. Containment is the test that separates a part from an
  occluder, at 0.30; at the 0.80 first tried, the horse regression returns.
- *Probing is affordable.* Skipping probe points already covered by an earlier probe's
  region turns an 8 px grid into ~30 decoder calls and 0.8 s, because one probe on the
  sky explains all of the sky.

**Still unfixed and out of reach here:** an occluder *inside* the silhouette — the glove
gripping the bat in `i9`, which SAM folds into the bat. Outward probing cannot see it,
and probing inward is what caused the horse regression. Separating the two needs the mask
hierarchy of MobileSAM's **multi-mask decoder**; the export in the registry is
single-mask, so this is a model-swap experiment, not a threshold.
**Resolution:** fix the under-segmentation at the object boundary — a mask that does not
need 5% of slack to look right is the thing that makes shielding safe. Then re-enable
`Thresholds.shield`, which is tested and waiting.

### TD-005 — Background op only handles flat backdrops

Status: open · Priority: P2 · Identified: 2026-08-26 · Area: `packages/models`
**Problem:** background replacement floods inward from the image border, which works for
a product shot and not for a real scene.
**Why it matters:** limits the operation to one photo genre.
**Workaround:** the detector returns nothing when the border is not uniform, so the
caller falls back to semantic segmentation — which drops thin structures.
**Seen live on 2026-08-28**, now that the browser can ask for this. A background change on
a real scene (the Taj Mahal photograph) refuses outright with "the border is not a uniform
backdrop, so the subject must be selected" until the subject is named; naming it works and
leaves a pale fringe around the subject where hair meets the new colour. Both are this
item: the refusal is the flood fill declining a scene, and the fringe is the semantic mask
standing in for a matte.
**Why deferred:** proper matting needs a model never benchmarked under the memory budget.
**Trigger:** a background case on a real scene enters the golden set. **Fired** — the
operation is reachable from the UI and both symptoms are one click away.
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

Status: resolved · Priority: P1 · Identified: 2026-08-26 · Resolved: 2026-08-28 · Area: `evals`
**Problem:** `make eval` runs locally and by hand. Nothing compares a branch's quality
against `main` automatically.
**Why it matters:** quality regressions are invisible until someone thinks to look. This
undercuts the whole point of having a golden set.
**Workaround:** manual comparison against `evals/out/report.json`.
**Why deferred:** it consumes real free-tier provider quota per run, so it cannot go on
every push without a budget decision.
**Trigger:** any quality regression that reaches `main`.
**Resolved 2026-08-28.** `evals/diff.py` compares a run against a committed
`evals/baseline.json`; the `eval` job in `check.yml` runs `make eval-local` on every pull
request, comments the diff, and uploads the images as an artifact. `make eval-baseline`
records a new baseline once a human has looked at the pictures.

**What blocks and what only reports, and why the split is not arbitrary.** A case that
stopped working, a case that vanished from the run, and a drop in grounding IoU all
block: each has a right answer. A cost move only reports. That is this project's own
measurement talking — TD-017 has two paired benchmarks disagreeing on the *sign* of
cost's correlation with visual quality, and TD-004 supplied the sharpest example yet: the
occluder shield moved `i8` by +1.8 cost while visibly ruining it and moved `i6` by -13.5
while improving it. Gating on that number would be a coin toss wearing a lab coat.

**Cost alone would not have caught the thing that motivated this.** The `i8` regression
moved it 3.2%, under any tolerance wide enough to be usable. So the baseline also carries
a 32x32 thumbnail of each result pane, and the diff reports when the picture moves: `i8`
scored 0.400 against 0.000 between two runs of unchanged code.

**That thumbnail is only usable because the local half of the golden set is bit-exact.**
Measured, not assumed — two runs of identical code gave the same cost to the decimal for
all eleven removals and pixel-identical images. There is no reproducibility noise to
leave headroom for, which is why the cost tolerance came down from a guessed 15% to 5%.
The remaining unknown is the *cross-machine* floor: a CI worker has a different CPU and
ONNX Runtime build, and that has not been measured yet. Both the image and cost checks
therefore only report; a tolerance that turns out too tight costs a line in a comment
rather than a red build. Tighten them once CI has run enough times to show its floor.

**This does not make looking at the images unnecessary.** It routes attention to them.
**The generative cases stay out of CI.** They spend a daily free-tier allowance shared
with development; per-push spend of it is what would make the golden set unrunnable.

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

Status: open · Priority: P1 · Identified: 2026-08-26 · Updated: 2026-08-27 · Area: `packages/models`
**Problem:** on our 18 hand-built cases, text-to-mask succeeded 10/11 (~91%). On 250
held-out RefCOCOg samples it reached **mIoU 0.389, precision@0.5 = 0.396**, with 36% of
predictions below IoU 0.1 — roughly half the dev-set figure that was being quoted.
**Improved on 2026-08-27, not closed.** ADR-0002 replaced CLIPSeg with Grounding DINO on
the same 250 samples:

|                          | CLIPSeg | Grounding DINO |
| ------------------------ | ------: | -------------: |
| mIoU                     |  0.3893 |     **0.4694** |
| median IoU               |  0.3348 |     **0.5411** |
| precision@0.5            |   0.396 |      **0.516** |
| phrases matching nothing |      20 |          **0** |

**Why it still matters:** 0.469 is better but it is not good. Published trained RES models
reach 0.65-0.75 on this benchmark, and **the failure rate below IoU 0.1 did not move at
all** (0.360 both ways) — the gain is entirely in cases that were already partly right.
**Measured:** `benchmarks/out/grounding-detector.json` and `-clipseg.json`.
**Workaround:** the brush is always available, and the product's own prompts are simpler
than RefCOCOg's relational expressions.
**Why deferred:** what remains is not a threshold problem. Both gates were swept on a fit
split and reported on a holdout (ADR-0002): `min_sam_iou` moves the holdout score by
0.0006 across its whole useful range, so the fitted value was **not written**.
**Trigger:** any claim about grounding accuracy made outside the dev set.
**Resolution:** the residual failure is relational reasoning, split out as TD-015.

### TD-013 — Fill cost's failure did not replicate on a second dataset

Status: open · Priority: P2 (was P0) · Identified: 2026-08-26 · Updated: 2026-08-27 · Area: `packages/core`

**The original finding.** `fill_metrics(...).cost` measures **plausibility** — how well a
fill agrees with the pixels around it — and it was being used as a stand-in for
**fidelity**, how close the fill is to what was actually behind the object. On
RemovalBench (n=69, 138 fills) the Spearman correlation between cost and
SSIM-against-truth was **+0.128**; cost is lower-is-better and SSIM higher-is-better, so a
working proxy correlates _negatively_. Choosing by cost picked the better eraser 43.5% of
the time against always-LaMa's 75.4%. The recorded conclusion was that the routing built
on it is worse than doing nothing, and the recorded resolution was to **confirm on a
second paired dataset before acting**.

**2026-08-27: the second dataset was run, and it refuted the conclusion.**

|                                | RemovalBench (n=69) | RORD-50 (n=69) |
| ------------------------------ | ------------------: | -------------: |
| Spearman `cost` vs SSIM        |          **+0.128** |     **−0.519** |
| Spearman `semantic` vs SSIM    |              −0.146 |     **+0.415** |
| `cost` picks the better eraser |               43.5% |      **68.1%** |
| always-majority                |        75.4% (LaMa) | 50.7% (MI-GAN) |
| MI-GAN wins / LaMa wins        |             17 / 52 |        35 / 34 |
| mean SSIM of the better eraser |               0.584 |          0.837 |

On RORD, cost correlates at **−0.519** — the correct sign, and the same magnitude the
ReMOVE paper reports for its own metric against LPIPS (−0.515). It beats always-MI-GAN by
17 points at picking the winner. **The metric works there.**

**Why it is now a P2 and not a P0.** The claim "cost is a poor proxy" rests entirely on
RemovalBench, and it does not generalise — which is the exact mistake the original entry
was written to warn about, made one level up. `compare()` and the multi-pass rollback
rule are not known to be broken, so the change that was queued behind this — deleting the
router and shipping one eraser — is **not justified** and has not been made.

**What is actually established:** the two datasets disagree, both about the metric and
about which eraser is better at all. Split out as **TD-017**, because "which of our two
benchmarks is unrepresentative" is a different question from "is this metric valid".

**Workaround:** none needed. The router is unchanged and every threshold now loads from
`benchmarks/fitted_thresholds.json` rather than a literal.
**Trigger:** a third paired dataset, or any decision that would rest on cost alone
without checking `benchmarks/out/removal.json` first.
**Resolution:** a third dataset breaks the tie. Until then, treat cost as valid for
comparing fills of the same mask — which is all it ever claimed — and not as a quality
score.

**A candidate replacement was measured and rejected.** `editgpt_models.semantic`
implements ReMOVE off the MobileSAM encoder we already load. It **flips sign between the
same two datasets** (−0.146 / +0.415), picks the winner 52.2% / 44.9%, and loses to cost
on RORD where the signal is strongest — while costing a full encoder pass per candidate.
It is not wired into the router. The module is kept because `benchmarks/removal.py` still
reports it, so the next person can re-measure instead of re-implementing.

### TD-017 — RemovalBench and RORD disagree about which eraser is better

Status: open · Priority: P1 · Identified: 2026-08-27 · Area: `benchmarks`
**Problem:** our two paired datasets do not agree on anything that matters. RemovalBench
says LaMa wins 52 of 69; RORD says the two are level at 35/34. Both reference-free proxies
correlate with ground truth on RORD and anti-correlate on RemovalBench.
**Why it matters:** every routing decision, and the honesty of TD-013, depends on which
one describes the images users will send. Right now we do not know, and a conclusion drawn
from either alone has a 50% chance of being an artefact — which has already happened once.
**Measured:** `benchmarks/out/removal.json`. Note the absolute scale as well: mean SSIM of
the better eraser is 0.584 on RemovalBench against 0.837 on RORD.
**Leading hypothesis, not established:** RemovalBench is a curated _hard_ set — large
objects, difficult backgrounds — where every fill is poor, so no reference-free score can
rank fills that are all wrong, and the "better" eraser is nearly a constant. RORD's
removals are mostly small handheld-video objects where fills are genuinely close and the
ordering carries signal. If that is right, RORD is the better guide for ordinary photos
and RemovalBench for the hard tail, and both should be reported rather than averaged.
**Workaround:** `make bench-removal` runs both and prints them side by side, so no
conclusion can be drawn from one without seeing the other.
**Why deferred:** resolving it needs a third dataset, and the decision it would unblock is
worth at most +0.013 SSIM (the oracle's gain over always-picking-one).
**Trigger:** any further investment in routing, or a third paired dataset becoming
available.
**Resolution:** add a third paired set, and stratify all three by mask coverage — if the
hypothesis holds, the correlation should track difficulty rather than dataset identity.

### TD-018 — Authentication is unimplemented pending a product decision

Status: resolved · Priority: P1 · Identified: 2026-08-27 · Resolved: 2026-08-27 · Area: `apps/gateway`
**Problem:** every request resolved to a single shared anonymous user.
**Resolution:** v1 has accounts, so Clerk was integrated. `auth.current_identity` verifies
the session token through `clerk-backend-api`, provisions a `users` row on first sight
keyed by Clerk's subject, and returns the owner every route passes to the store. It
**fails closed** — an absent, expired or malformed token is a 401, never a fall back — and
accepts only `session_token`. `EDITGPT_CLERK_JWT_KEY` makes verification networkless.
Authentication is on when the secret key is present and off otherwise, so tests and a
fresh checkout need no credential; `/ready` reports which mode is live.
**Kept here as a record:** the _authorization_ half was built a step earlier, before any
provider was chosen — one place deciding identity, every route passing it explicitly, the
owner travelling in the queue message. That is why adding the provider touched one
function and no routes. Doing it in that order is the reusable part.

### TD-019 — Any signed-in user can fetch any image by its digest

Status: open · Priority: P2 · Identified: 2026-08-27 · Area: `apps/gateway`
**Problem:** `GET /v1/images/{digest}` requires a session but does not check ownership, so
any signed-in user holding a digest can fetch that image.
**Why it matters:** a digest is a 256-bit unguessable name, so this is not open to the
internet — but digests do leak, into logs, screenshots and shared links, and "unguessable"
is a weaker promise than "not yours".
**Why it is not simply an ownership check:** storage is content-addressed. Two users
uploading the same photograph share one digest and one `images` row, whose `user_id` is
whoever arrived first. An ownership check would lock the second uploader out of their own
upload, which is worse than the exposure.
**Workaround:** authentication is required, so the audience is signed-in users rather than
anyone.
**Why deferred:** the correct fix is a schema change, and the exposure needs both an
account and a leaked digest.
**Trigger:** the service being used by people who do not know each other.
**Resolution:** a `user_images` join table so many owners can share one blob, then check
membership. Signed, expiring URLs (already planned for Phase 9) are the complement — they
also let `<img src>` work without a header, which the frontend currently works around by
fetching to a blob.

### TD-014 — Fine-tuning CLIPSeg is blocked on hardware

Status: accepted · Priority: P2 · Identified: 2026-08-26 · Area: `packages/models`
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

### TD-015 — Relational referring expressions are not grounded at all

Status: resolved · Priority: P1 · Identified: 2026-08-27 · Resolved: 2026-08-27 · Area: `packages/models`
**Problem:** 36% of held-out predictions fell below IoU 0.1, and the figure was
**identical** for CLIPSeg and Grounding DINO. Those are the expressions naming an instance
by its relation to something else; neither model reasons about relations, so both ground
the noun and pick an arbitrary instance.
**Resolution: stop guessing, offer the alternatives.** On 250 held-out samples, picking
from five candidates takes the hit rate from **0.516 to 0.832** and mIoU from **0.469 to
0.731** — inside the 0.65-0.75 band published _trained_ RES models occupy, which
TD-012/TD-014 recorded as unreachable without a GPU. **The gap was never a model gap; it
was a UI gap.** ADR-0003 has the ceiling curve and the gate.
**What remains genuinely open:** 0.832 assumes the user picks correctly, which is
unmeasurable without shipping it; and the default gate asks on 46% of phrases, a product
judgement about friction rather than a measurement.

### TD-016 — Recorded result dimensions assume the edit preserves them

Status: resolved · Priority: P2 · Identified: 2026-08-27 · Resolved: 2026-08-27 · Area: `packages/store`
**Problem:** `run_job` recorded the result using the _source_ dimensions from the spec —
true for every operation shipped then, false for `UPSCALE`.
**Resolution:** editors return a `Produced` carrying the bytes with their real content type
and size, and the lifecycle records that. **A worse instance surfaced on the way:** AVIF
was missing from the worker's format map, so an AVIF upload was silently re-encoded as PNG
while the row still said `image/avif` — a 54 KB photograph came back as 476 KB of bytes
served under a type they were not. Fixed by covering everything
`uploads.ALLOWED_FORMATS` accepts, with a test asserting the two cannot drift. After:
476 KB → 111 KB, genuinely AVIF.

### TD-020 — The background colour is fixed green, not requested

Status: resolved · Priority: P2 · Identified: 2026-08-27 · Resolved: 2026-08-28 · Area: `packages/models`
**Problem:** `BACKGROUND` always paints the same green, so "change the background to blue"
produces green.
**Why it matters:** the operation advertises a capability it does not have, and it fails
_silently_ — the edit succeeds and returns the wrong colour.
**Workaround:** none. The golden set only ever asks for green, which is how this survived.
**Was deferred because** parsing a colour from free text belongs to the intent agent,
which does not exist yet, and a colour table in `models` would be the wrong home for it.
**Trigger:** any request for a background colour other than green. **Fired** the moment
the browser grew a Background button with a field for exactly that.

**Resolved 2026-08-28**, and the deferral turned out to be reasoning about the wrong
problem. `EditSpec.colour` carries `#rrggbb`, the gateway validates it at the boundary,
and `rgb_colour` converts it next to the pattern that validates it so two places cannot
disagree about which end is red. No parsing was needed: **the client asks for a colour
with a colour input.** This operation composites a flat backdrop rather than generating
one (TD-005), so a hex value loses nothing prose would have carried — the free-text field
was promising something the operation could not do even once someone read the answer.

`content` still accepts free text, for the day an intent agent can turn "a sunset" into
something paintable, and `background` now requires one or the other rather than requiring
a description nothing read.

Verified live: asked for `#3366ff` and got `#3366ff` across 92% of the frame; asked for
`#cc00aa` and got `#cb00aa`, one level off through JPEG.

### TD-021 — Edits come back at 2048 px, not the uploaded resolution

Status: resolved · Priority: P1 · Identified: 2026-08-27 · Resolved: 2026-08-28 · Area: `apps/worker`
**Problem:** the worker downscales to 2048 px on the longest side before editing, so a
15.9 MP upload returns at ~3 MP.
**Why it matters:** a visible product decision wearing an implementation detail's clothes.
Nobody asked for their photograph to be shrunk, and the API does not say it happened.
**Why it exists:** the gateway's 40 MP cap bounds _memory_; this bounds _time_. A 15.9 MP
erase is minutes of model work, and the multi-pass loop runs an eraser up to three times.
**Workaround:** none.
**Trigger:** any user who downloads a result.
**Resolved 2026-08-28** by `compositing.reproject`, and not by editing at full
resolution. Two knobs had been collapsed into one: `MAX_SIDE` now bounds the *work* and
no longer bounds the *answer*. The edit still runs at 2048 px; the finished result is
blended onto the full-resolution original, where the alpha is zero outside the mask so
the untouched pixels are the original's own rather than a downscale-and-upscale of them.

**Resampling the edit up loses nothing**, which is what makes this cheap rather than a
trade. Every fill in the pipeline is generated at `CROP_SIZE` (512) and enlarged to the
crop window already, so the model's output carries the same detail either way. What the
bounded return had actually been discarding was the ~90% of the photograph that was never
edited.

**Editing at full resolution was the obvious fix and is the wrong one:** `fill_metrics`
runs two full-frame colour conversions and a 60 px dilation *per pass*, so it would have
put the multi-pass loop on the megapixel count for no visible gain.
**`UPSCALE` stays bounded on both sides** — its purpose is to enlarge, so a 15.9 MP input
would ask Real-ESRGAN for a 63 MP output and meet the RSS ceiling first.

### TD-022 — A provider's blank frame was composited as a result

Status: resolved · Priority: P1 · Identified: 2026-08-27 · Resolved: 2026-08-27 · Area: `packages/providers`
**Problem:** Stable Diffusion returns an all-black frame when its safety checker fires —
a 200 response carrying no generation. The compositor pasted it, and the job reported
`ok`. On the golden set's `i3` this stamped a **black rectangle on a face** and recorded a
completed edit.
**Why it mattered:** the project's rule is that returning the input unchanged looks like
success. This was worse: it looked like a deliberate edit. Nothing in the pipeline
disagreed — the fill cost rose from 75 to 196 and the case still printed `ok`, because
cost has no opinion about whether an image exists.
**Found by looking at the output image.** No number caught it, which is the third time in
this project a metric has agreed with something a human immediately saw was wrong.
**Resolution:** the provider now refuses a return that is both near-black and near-flat
inside the mask, raising `ProviderError` so the chain fails over and the job reports
honestly. Thresholds calibrated on real returns: a correct fill measured mean 122.5 with
per-channel spread 61.7; the failure measured 11.3 and 6.7 after compositing. Both
conditions are required, because darkness alone is not a fault — `i4r` measured mean 37.2
and was correct.
**Confirmed transient:** the identical call minutes later produced a correct moustache.

### TD-023 — A phrase matching nothing still returns a confident region

Status: open · Priority: P2 · Identified: 2026-08-28 · Area: `packages/models`
**Problem:** grounding "a unicorn" against a photograph of the Taj Mahal returns one
candidate at margin 1.000, which the ambiguity gate reads as *certainty* and answers
without asking. Measured live on 2026-08-28 while verifying the picker.
**Why it matters:** the empty-candidates branch — "nothing here matches those words,
try describing it differently" — is written on both sides of the wire and effectively
never fires. A user who mistypes or describes something absent gets a confident wrong
region rather than being told.
**Why it exists:** the same property ADR-0002 was pleased about. Grounding DINO abstained
on 0 of 250 held-out phrases where CLIPSeg abstained on 20, which is what made it the
better lane; always answering is the flip side of always finding something. The margin
gate then compounds it, because a single candidate has nothing to be close to and so
scores maximally unambiguous.
**Workaround, and it is a real one:** the region is drawn on the picture before anything
is edited, and "Not what you meant?" opens the alternatives. The user sees the wrong
region rather than discovering it in the result.
**Why deferred:** ADR-0003 measured absolute detection score as a much weaker separator
than the margin and rejected it as the *gate*. Using it as an abstention signal is a
different question and needs its own measurement rather than a guessed threshold.
**Trigger:** a user reports an edit to something they did not name.
**Resolution:** measure whether the top score separates present from absent phrases on a
held-out set with negatives — `benchmarks/ambiguity.py` has the harness — and if it does,
abstain below it rather than answering.
