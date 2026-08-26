# Phase 0 — Feasibility spike

> **✅ COMPLETE — 25 Aug 2026.** Decision and full results in
> [`docs/adr/0001-model-routing.md`](../docs/adr/0001-model-routing.md);
> generated tables in [`out/report.md`](out/report.md).
>
> **Outcome:** proceed to Phase 1. Removal ships. Additions ship with a stated quality
> cap. Shadow removal is out of v1.

## Reproducing it

```bash
make setup-all     # uv sync with the torch and remote extras (they are NOT additive
                   # across separate `uv sync` calls — syncing one removes the other)
make models        # ~285 MB: Big-LaMa, MobileSAM enc/dec, MI-GAN
make verify        # every bench except bench-remote, then the report
make cf-check      # verify Cloudflare credentials before spending neurons
make bench-remote  # spends the free daily neuron budget, so it is not in `verify`
```

| Target            | Question it answers                                                              |
| ----------------- | -------------------------------------------------------------------------------- |
| `bench-sam`       | Is the MobileSAM decoder fast enough for interactive magic-select? (24 ms — yes) |
| `bench-lama`      | Does Big-LaMa fit the memory budget, and is crop better than whole?              |
| `bench-migan`     | What does the MobileSAM + MI-GAN path cost on its own?                           |
| `bench-erasers`   | LaMa vs MI-GAN, same masks, scored by photometric agreement                      |
| `bench-pipeline`  | Box → mask → erase, end to end at 1024 px                                        |
| `bench-clipseg`   | Can text alone find the object? (10/11)                                          |
| `bench-text2edit` | Fully text-driven removal, no brush, scored on the final mask                    |
| `bench-fullres`   | Crop vs whole at native 15.9 MP, where they finally diverge                      |
| `cf-check`        | Are the Cloudflare credentials good? (needs Workers AI **Edit + Read**)          |
| `bench-remote`    | Can the free generative lane add a moustache, and can it remove anything?        |

---

**Question this phase answers:** on _this_ MacBook Air, does the local half of EditGPT fit inside
the 1.2 GB model slot from `docs/PLAN.md` §4, fast enough to be usable — and is Gemini good enough
to be the generative half?

The original day-by-day runbook follows, kept because it records the order the questions
were asked in and the gates each day had to clear.

---

## Day 1 — Environment and models

```bash
cd spike
make setup          # uv sync — onnxruntime, opencv, pillow, psutil (no torch yet)
```

Put **10 real photos** in `spike/assets/photos/`. Not stock renders — the cases you'll actually be
judged on. At minimum:

| #   | Photo                               | Exercises                                         |
| --- | ----------------------------------- | ------------------------------------------------- |
| 1–2 | a car parked on a street            | `remove` on a large rigid object with reflections |
| 3–4 | a portrait, face clearly visible    | `add a moustache`, identity preservation          |
| 5   | a person against a busy background  | mask precision at hair edges                      |
| 6   | a landscape with a signpost or logo | small-object removal, text                        |
| 7   | a group shot                        | can CLIPSeg pick _one_ person?                    |
| 8   | something with hard shadows         | does LaMa leave a ghost shadow?                   |
| 9   | a repeating texture (brick, grass)  | the case LaMa is best at                          |
| 10  | a low-light / noisy phone shot      | the case it's worst at                            |

```bash
make models         # ~250 MB: Big-LaMa ONNX (208 MB) + MobileSAM encoder/decoder
```

---

## Day 2 — The erase path (the load-bearing one)

```bash
make bench-lama
open out/lama_crop_*.png out/lama_whole_*.png
```

The Carve LaMa export has a **fixed 512×512 input**, so there are only two strategies and the
benchmark runs both:

- `whole` — downscale the entire image to 512, inpaint, upscale back. Cheap, soft.
- `crop` — crop a 1.6× context window around the mask, inpaint at 512, feather back in at full
  resolution. Sharp. **This is the strategy `CompositorAgent` inherits in Phase 5.**

**Judge, in this order:** (1) is `crop` visibly sharper than `whole`? (2) is the feathered seam
invisible? (3) peak RSS versus the 1.2 GB slot. (4) p50 latency versus the 6 s target.

Optional, and worth 10 minutes — the M1's Neural Engine:

```bash
EDITGPT_PROVIDERS=CoreMLExecutionProvider,CPUExecutionProvider make bench-lama
```

Record both rows. CoreML is sometimes slower on first run and much faster after; sometimes it
silently falls back. Trust the measurement, not the marketing.

---

## Day 3 — The mask path

```bash
make bench-sam        # MobileSAM: box prompt -> mask
open out/mobilesam_overlay.png

make bench-clipseg    # installs torch (~1.2 GB, one time), then text -> mask
open out/clipseg_*.png
```

Two independent gates:

- **MobileSAM decoder p50 must be under ~100 ms.** The encoder runs once per image and is cached;
  the decoder runs on every brush stroke. If the decoder is slow, the "tap to select" UX in Phase 8
  is dead and you fall back to brush-only.
- **CLIPSeg hit rate.** For each of the 4 prompts across 10 photos, mark hit / partial / miss.
  Below roughly 7/10 hits, text-to-mask can't be the default path — it becomes a _suggestion_ the
  user confirms, and the brush is the primary input. That's a product decision, so decide it here.

---

## Day 4 — The remote lane

```bash
# https://aistudio.google.com/apikey  → free, no card
echo 'GEMINI_API_KEY=...' > .env
make bench-gemini
open out/gemini_0.png out/gemini_1.png
```

`out/gemini_0.png` is "add a moustache" — the operation you _cannot_ do locally.
`out/gemini_1.png` is "remove the car" — the operation you _can_.

**Put `gemini_1.png` next to `lama_crop_*.png` and look at them side by side.** If LaMa wins, or
even ties, the whole local/remote router in the plan is justified by evidence rather than by
argument. If Gemini clearly wins on removal too, that's a real finding — simplify the plan, drop
LaMa to a fallback, and note it in the ADR.

The script also reports which model id actually answered (ids move) and how many free-tier
requests the run consumed.

---

## Day 5 — Decide

```bash
make report
```

Prints the table and drafts `docs/adr/0001-model-routing.md`. Fill in the quality notes by hand —
the scorers are your eyes at this stage — then pick one of the four decisions in the template.

### Go / no-go gates

| Gate                    | Threshold   | If it fails                                                                |
| ----------------------- | ----------- | -------------------------------------------------------------------------- |
| LaMa peak RSS           | ≤ 1.2 GB    | Crop-only mode; if still over, LaMa moves to the HF Space                  |
| LaMa p50                | < 6 s @1024 | Acceptable up to ~10 s with good progress UI; past that, remote-only erase |
| MobileSAM decoder p50   | < 100 ms    | Brush-only UX, drop magic-select from Phase 8                              |
| CLIPSeg hit rate        | ≥ 7/10      | Text-to-mask becomes suggest-and-confirm, not automatic                    |
| Gemini round-trip       | < 15 s      | Keep, but the job must be async with SSE — which it already is             |
| Any two models resident | never       | Enforce `ModelSlot(max_resident=1)` — already the plan                     |

---

## How the measurement works

`bench/harness.py` runs each model in a **subprocess** and samples its RSS (plus any children) at
20 Hz from the parent, so the number isn't polluted by the parent interpreter. The child reports
its own timings back on stdout as a single `##BENCH##{json}` line. Results land in `out/*.json`.

That subprocess-plus-sampler design is not throwaway: it becomes the `make bench` target and the
CI memory test in Phase 2.
