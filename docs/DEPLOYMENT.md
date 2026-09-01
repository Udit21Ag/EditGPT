# Deployment audit — 1 September 2026

**What this is:** what each process actually needs, what the free tiers actually give
today, and where the two do not meet. Written before any deployment configuration,
because the plan's hosting assumptions were made in Phase 0 and three of them have since
stopped being true.

**Headline:** everything in this system has a free home except the worker. The gateway,
the web app, Postgres, Redis, object storage and auth all fit inside free tiers with room
to spare. The worker needs ~2 GB of RAM for a few seconds at a time, and the free tiers
that used to offer that are gone.

---

## 1. What has to run

Measured on this machine today unless marked otherwise.

| Process      | Memory                                                     | Disk                          | Shape                                |
| ------------ | ---------------------------------------------------------- | ----------------------------- | ------------------------------------ |
| **Gateway**  | **86 MB idle → 126 MB** after 8 uploads of a 3.9 MP photo   | none                          | HTTP, bursty, scale-to-zero friendly |
| **Worker**   | **2200 MB ceiling**; peaks 1372 MB (detector), 1150 MB (MI-GAN) | **527 MB** of weights     | one edit at a time, ~13 s each       |
| **Web**      | build-time only                                            | ~30 MB of static output       | static + a thin server component     |
| **Postgres** | rows only — no pixels                                      | megabytes                     | idle almost always                   |
| **Redis**    | progress channels + rate-limit counters                    | kilobytes                     | idle almost always                   |
| **Objects**  | —                                                          | ~4–8 MB per edit (in + out)   | write once, read a few times         |

The worker's numbers are not estimates: `DEFAULT_RSS_CEILING_MB = 2200` is enforced in
`packages/models/src/editgpt_models/slot.py`, and the two peaks are what
`packages/models/tests/test_memory.py` asserts against with real weights. **1372 MB is one
model.** The ceiling is why concurrency is 1 and why the slot evicts between operations.

The weights are 527 MB across six ONNX files, measured in the cache today. Both figures in
the repository were wrong — `make models` said ~285 MB and `AGENTS.md` said ~552 MB — and
are corrected in this change.

**Worker image size is an estimate, not a measurement.** No Docker build has been done.
The runtime closure without the `text` extra is onnxruntime, opencv-headless, numpy,
pillow, tokenizers, celery, redis, sqlalchemy, boto3 — call it 350–450 MB of site-packages,
plus 527 MB of weights, plus a slim Python base: **~1.0–1.3 GB**. Torch is *not* in that
closure (it is only for the CLIPSeg fallback, `editgpt-models[text]`), and keeping it out
is worth 500 MB of image. Build it and measure before trusting the number.

---

## 2. What the free tiers give, today

| Service                | Free allowance (checked 1 Sep 2026)                                     | Fits us? |
| ---------------------- | ----------------------------------------------------------------------- | -------- |
| **Vercel** Hobby       | 100 GB transfer, 1M edge requests, 6,000 build-min. **Non-commercial**   | ✅ web    |
| **Neon** Postgres      | 0.5 GB storage, 100 compute-hours/mo, autosuspends after 5 min idle      | ✅        |
| **Upstash** Redis      | 500k commands/mo, 256 MB, 10 GB bandwidth                               | ✅        |
| **Cloudflare R2**      | 10 GB-month, 1M class A, 10M class B ops, **zero egress**               | ✅        |
| **Clerk**              | 50,000 monthly retained users (raised from 10k in Feb 2026)             | ✅        |
| **Cloudflare Workers AI** | 10,000 neurons/day, shared across models                             | ⚠️ §4    |
| **Google Cloud Run**   | 180k vCPU-s, 360k GiB-s, 2M requests/mo; **1 GiB free egress**          | ⚠️ §3    |
| **Render**             | 512 MB, 0.1 CPU, spins down after 15 min, ~1 min cold start             | gateway only |
| **Hugging Face Spaces**| **CPU Basic is 2 vCPU / 16 GB — but creating a Docker or Gradio Space now requires a paid plan.** Static Spaces only on free. | ❌ |
| **Fly.io**             | **No free tier for new accounts** since Oct 2024. ~$1.94/mo for 256 MB  | ❌ free   |
| **Koyeb**              | **Closed to new signups** after the Mistral acquisition (Feb 2026)      | ❌        |
| **Oracle Always Free** | ARM Ampere, **halved to 2 OCPU / 12 GB on 15 Jun 2026**; capacity varies by region | ✅ if you can get one |

### Three plan assumptions that are no longer true

1. **"Agents to HF Spaces (Docker)"** — Docker Spaces are no longer free to create. The
   hardware is right (16 GB would hold the worker four times over); the plan is not.
2. **"Gateway to Fly.io"** — Fly has no free tier for new accounts. It is still *cheap*
   (~$2/mo for the gateway's size), but it is not free.
3. **"Free tier everywhere"** as a whole — Koyeb closed, Oracle halved, Render's spin-down
   shortened. The pattern since this project started is one-way.

---

## 3. The worker is the whole problem

It needs ~2 GB for ~13 seconds per edit and near-zero the rest of the time. That profile is
ideal for scale-to-zero and terrible for a fixed free instance, and every remaining free
fixed instance is 512 MB.

**Option A — Oracle Always Free ARM VM (2 OCPU / 12 GB).** Free, holds the worker, Redis
and Postgres together if you want them there. Costs: it is a lottery — "out of host
capacity" is normal in busy regions — you operate the box yourself, and it is ARM64, so
every wheel (onnxruntime, opencv-headless, tokenizers) must have an aarch64 build. They do,
but that has not been verified for this project and should be before committing.

**Option B — the cheapest reliable box.** Hetzner CAX11 (ARM, 4 GB) or CX22 (x86, 4 GB),
roughly €4.50–6/month with 20 TB of traffic. This is the honest answer if you want the
demo to be up when someone clicks the link. One box runs worker, Redis and Postgres, and
the free tiers stop mattering.

**Option C — Cloud Run at 2 GiB, scale to zero.** The free allowance is 360,000 GiB-s and
180,000 vCPU-s a month. At 2 GiB and ~13 s per edit that is ~25 GiB-s per edit → about
**14,000 edits a month inside the free tier**, and the vCPU side lands in the same place.
That is far more than a portfolio demo will ever serve. Two costs: the worker must stop
being a Celery consumer and become HTTP-triggered (the queue abstraction in
`apps/gateway/src/editgpt_gateway/deps.py` is already the seam — `Queue.send` is one
method), and a cold start has to pull a ~1.2 GB image, so weights must be baked into the
image rather than downloaded at boot.

**Option D — keep the worker local.** The gateway and web deploy; the worker runs on your
laptop against the hosted broker. Free and honest for a demo you drive yourself; it is not
a deployment, and the link is dead when the laptop is.

**Recommendation: C if you want strictly free and are willing to make the worker
HTTP-triggered; B if you would rather spend €5 than change the architecture.** A is
tempting and free, but "provisioning is a lottery" is a bad property for the thing you
show people. Do not pick until the image is built and measured — that number decides
whether C's cold start is 10 seconds or 40.

---

## 4. Things this audit could not settle

**How many edits a day the free generative lane buys.** Workers AI gives 10,000 neurons a
day; nothing in this repository records what one inpainting call costs in neurons. The
cost ledger writes `units=1, cents=0.0` for every job — true for the local lane, useless
for the remote one. Record the neuron cost from the response and the answer follows.

**Whether image bytes should leave through the gateway.** Signed links are verified by the
gateway, which then streams the bytes, so every image download is gateway egress — 1 GiB
free on Cloud Run is about 340 downloads of a 3 MB photo. R2 charges nothing for egress,
so presigning R2 URLs directly would move that traffic off the priced path entirely. The
signing seam already exists (`apps/gateway/src/editgpt_gateway/signing.py`); it would sign
an R2 URL instead of a gateway route. Worth doing before, not after, the first bill.

**Whether the ARM wheels work.** Options A and B (CAX11) are ARM64. Verified only that the
wheels exist, not that the pipeline runs on them.

**Prices.** Vendor pages were read where they exist (Hugging Face hardware and plan rules,
Neon, Cloudflare R2, Cloud Run); the VPS and Fly figures come from third-party trackers and
should be confirmed on the vendor's own page before any card is entered.

---

## 5. What this changes

- Storage stays inside R2's 10 GB only if bytes are deleted. `EDITGPT_ASSET_RETENTION_DAYS`
  exists for exactly this and is off by default; a deployment should set it (14–30 days),
  and `make beat` must actually run somewhere or nothing sweeps.
- Neon autosuspends after 5 minutes. The first request after an idle period pays a wake-up;
  the gateway already treats a missing database as a degradation rather than a crash, so
  this shows up as latency, not as an outage.
- Vercel Hobby is non-commercial. Fine for a portfolio; a decision if this ever earns money.
- The `/ready` degradation list is the deployment checklist. If it is not empty in
  production, something in this document was skipped.
