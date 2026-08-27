# packages/providers

Remote generative providers. A provider is a fill: `(rgb, mask, prompt) -> rgb`.

**Removal never routes here.** Measured in Phase 0: asked to erase a car, this lane
produced a stone slab, then a different car, then a boulder. It fills a masked hole with
an object matching the prompt rather than continuing the background, and no prompt or
guidance setting changes that. See ADR-0001.

Everything around the fill — crop window, chroma match, feather — is shared with the
local erasers, so comparing lanes compares models rather than plumbing. Keep it that way.

## Cloudflare Workers AI

- The token needs **`Workers AI — Edit` and `Workers AI — Read`**. Read alone returns a
  401 indistinguishable from a bad token.
- The mask goes in a field named `mask`, as a list of PNG bytes. An error naming
  `mask_image` means no mask was recognised at all, not that the field is misnamed.
- The API rejects an empty prompt, which is precisely why this lane cannot do removal.
- SD-1.5 is mask+prompt, not instruction-following: the prompt must describe what _should
  be there_. Translating an instruction into that is the IntentAgent's job.

**A 200 is not a result.** Stable Diffusion returns an all-black frame when its safety
checker fires, and the compositor will happily paste it onto a user's photograph and
report the job done. Every provider must refuse a return that carries no image; the
thresholds in `cloudflare.py` are calibrated on real ones, and darkness alone is not the
test — a legitimate fill of a shadow measured mean 37.

Add a provider by implementing the `Provider` protocol and putting it in the chain. The
circuit breaker treats quota exhaustion and transient faults alike — they return the same
HTTP shape, so back off and probe again rather than trying to tell them apart.
