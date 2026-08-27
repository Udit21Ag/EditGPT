# packages/models

Local model execution. **Importing this package must not load a model.**

- `slot.py` — `ModelSlot`. One heavy model resident; LRU, idle TTL, RSS ceiling.
- `registry.py` — what exists, where from, and its _measured_ peak RSS.
- `config.py` — **every tunable in the pipeline**, loadable from a fitted file.
- `detect.py` — Grounding DINO: a phrase → candidate boxes. The text lane.
- `segment.py` — box, brush or seed → MobileSAM → a precise mask.
- `semantic.py` — the ReMOVE score, off the encoder that is already loaded.
- `erase.py` — MI-GAN and LaMa, residual detection, backdrop detection.
- `compositing.py` — crop window, chroma match, feathered paste-back.
- `pipeline.py` — the multi-pass policy.

## Traps that have already cost time

- **MI-GAN's mask is inverted**: 255 means _keep_. Getting it backwards produces a
  plausible image with everything except the target repainted — easy to miss in review.
- **The MobileSAM encoder pads but does not resize.** Resize the longest side to 1024
  first or the embedding desynchronises from the point coordinates and the mask tears
  along the prompt box.
- **Grounding DINO wants the image squashed to 800×800, not letterboxed.** Its export
  declares a `pixel_mask`, which reads as an invitation to pad. Measured on 40 RefCOCOg
  samples: squash 0.511 box-IoU, letterbox 0.230. Its prompt must also be lowercase and
  end with a period.
- **LaMa's input is fixed at 512×512**, so the caller owns the crop and paste-back.
  MI-GAN's graph crops internally and takes any resolution.
- **Match chroma, never luminance.** Matching luminance re-stamps the object wherever the
  reference ring spans two surfaces, such as a table plus the object's own shadow.
- **Dilation scales with the object**, never in pixels. A constant that works at 1024 px
  leaves a visible rectangular outline at 15.9 MP.

## Thresholds live in `config.py`, and only there

Every decision point reads `load_thresholds()`. Do **not** add a module constant named
after a `Thresholds` field: that exact mistake shipped once — `pipeline.py`,
`compositing.py` and `metrics.py` each carried a literal shadowing the field it was
supposed to read, so editing the fitted file changed nothing.
`tests/test_thresholds_are_wired.py` fails if it comes back.

Read the value **inside the function**, not at import. A module-level
`X = load_thresholds().x` looks wired and is not — it freezes at first import.

## The multi-pass policy

Pass 2 always runs; its output is kept only if it verifies better, because a naive second
pass measurably makes things slightly worse. A rolled-back pass is **recorded, not
hidden** — a rejected strategy is a finding the critic loop needs.

Never gate the residual pass on fill cost alone. It prefers the version that flattens the
scene. `residual_max_growth` is structural for that reason.

**`fill_metrics(...).cost` measures plausibility, not fidelity** — see TD-013. Do not
add a decision that trusts it without checking `benchmarks/out/removal.json` first.
