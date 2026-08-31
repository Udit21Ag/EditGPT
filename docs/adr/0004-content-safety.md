# ADR-0004 — No content classifier; the generative provider is the control

- **Status:** accepted
- **Date:** 2026-08-30
- **Context:** the Phase 9 plan lists "NSFW/safety gate in the critic" and
  "prompt-injection defence on image-derived text". Both were written before the system
  existed; this records what they turned out to mean once it did.
- **Relates to:** [ADR-0001](0001-model-routing.md), which split the local and remote lanes
  and is why the answer differs between them.

## Decision

**Do not build a content classifier.** For the only operations that can generate an image,
the provider already enforces a policy we cannot override. For the operations that cannot,
there is nothing being generated to classify.

## Why the two lanes differ

`REMOVE`, `UPSCALE` and `BACKGROUND` run locally and generate nothing. Erasing a car
continues the background around it; upscaling interpolates; a background change composites
a flat colour. The output is the user's own photograph with something taken out of it. A
classifier on that lane would be inspecting private pictures to decide whether their owner
may edit them, which is a surveillance feature wearing a safety label.

`ADD` and `REPLACE` reach Cloudflare Workers AI. Its terms and its Stable Diffusion safety
checker apply to every call, and we already know what that looks like from the other side:
the checker returns a black frame on a 200, which cost this project an afternoon and is
recorded as TD-022. **The control exists, it is enforced upstream, and it is not ours to
switch off.**

## Why not add one anyway

- **It would not fit.** A classifier is another model inside a 2200 MB resident ceiling
  that already holds a detector, a segmenter and two erasers.
- **It would duplicate a control we do not own.** The lane that can generate is already
  governed; the lane that cannot has nothing to govern.
- **A small classifier is not the thing that would matter.** The genuinely serious cases —
  sexual imagery involving minors, and intimate imagery of real people made without their
  consent — are not what an off-the-shelf NSFW score detects. Those are hard lines
  regardless of jurisdiction, and the honest position is that a threshold on a general
  classifier is not what enforces them. Provider policy on the generative lane is the
  control that exists today; anything stronger is a moderation system, not a filter, and
  it is not something to half-build.

**Adult content between consenting adults is legal in most jurisdictions and is a product
decision, not a legal one.** This ADR does not make that decision. It records that the
system does not currently have a mechanism to make it either way, and that adding one to
the local lane would mean inspecting private photographs.

## Prompt injection

**Not applicable yet, and worth stating so it is not mistaken for done.** The defence
concerns text *extracted from an image* — OCR, a caption — being fed to a model that acts
on instructions. Nothing in this system reads text out of an image, so there is no path
today.

The rule for when there is one: image-derived text is **data, never instruction**. It is
quoted into a prompt as content to be described, never concatenated where a directive
would be read, and never allowed to select an operation. The moment a captioner or an OCR
step lands — Phase 7's orchestrator is the likely place — this becomes real and needs its
own ADR.

## Consequences

- No classifier, no extra model, no extra latency.
- The generative lane's behaviour under refusal is already handled: `_reject_blank` turns a
  safety-filtered black frame into an error a user can read rather than compositing it.
- If a deployment needs its own policy, it belongs in front of the gateway or in the
  provider account, not in the pipeline.
