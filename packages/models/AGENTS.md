# packages/models

Local model execution. **Importing this package must not load a model.**

- `slot.py` — `ModelSlot`. One heavy model resident; LRU, idle TTL, RSS ceiling.
- `registry.py` — what exists, where from, and its _measured_ peak RSS.
- `segment.py` — CLIPSeg seed → MobileSAM refinement.
- `erase.py` — MI-GAN and LaMa, residual detection, backdrop detection.
- `compositing.py` — crop window, chroma match, feathered paste-back.
- `pipeline.py` — the multi-pass policy.

## Traps that have already cost time

- **MI-GAN's mask is inverted**: 255 means _keep_. Getting it backwards produces a
  plausible image with everything except the target repainted — easy to miss in review.
- **The MobileSAM encoder pads but does not resize.** Resize the longest side to 1024
  first or the embedding desynchronises from the point coordinates and the mask tears
  along the prompt box.
- **LaMa's input is fixed at 512×512**, so the caller owns the crop and paste-back.
  MI-GAN's graph crops internally and takes any resolution.
- **Match chroma, never luminance.** Matching luminance re-stamps the object wherever the
  reference ring spans two surfaces, such as a table plus the object's own shadow.
- **Dilation scales with the object** (5% of its longest side). A pixel constant that
  works at 1024 px leaves a visible rectangular outline at 15.9 MP.

## The multi-pass policy

Pass 2 always runs; its output is kept only if it verifies better, because a naive second
pass measurably makes things slightly worse. A rolled-back pass is **recorded, not
hidden** — a rejected strategy is a finding the critic loop needs.

Never gate the residual pass on fill cost alone. It prefers the version that flattens the
scene. `RESIDUAL_MAX_GROWTH` is structural for that reason.
