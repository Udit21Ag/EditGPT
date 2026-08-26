"""Remote generative providers, behind one interface.

A provider is just a fill: `(rgb512, mask512) -> rgb512`. Everything around it —
the crop window, the colour match, the feather — is shared with the local LaMa
lane in run_lama.erase_crop_with, so comparing the two lanes compares the models
and nothing else.

Gemini's image API lost its free tier in December 2025 (every image model returns
`limit: 0`), so Cloudflare Workers AI is the free lane. Its SD-1.5-inpainting is
mask+prompt, not instruction-following, which is itself a finding: the prompt has
to describe what SHOULD be there, which is work the IntentAgent now has to do.
"""

from __future__ import annotations

import base64
import os
from io import BytesIO

import numpy as np
from PIL import Image

CF_MODEL = "@cf/runwayml/stable-diffusion-v1-5-inpainting"
CF_URL = "https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{model}"
TIMEOUT = 120


def _png_bytes(arr: np.ndarray, mode: str) -> bytes:
    buf = BytesIO()
    Image.fromarray(arr, mode=mode).save(buf, format="PNG")
    return buf.getvalue()


def cloudflare_fill(
    prompt: str,
    *,
    steps: int = 20,
    guidance: float = 7.5,
    negative: str = "blurry, distorted, watermark, text",
):
    """Build a fill backed by Workers AI. Raises with instructions if unconfigured."""
    import httpx

    account = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    if not (account and token):
        raise SystemExit(
            "Cloudflare Workers AI is not configured. In spike/.env set:\n"
            "  CLOUDFLARE_ACCOUNT_ID=...   (dash.cloudflare.com -> Workers & Pages,\n"
            "                               the Account ID in the right sidebar)\n"
            "  CLOUDFLARE_API_TOKEN=...    (My Profile -> API Tokens -> Create Token,\n"
            "                               template 'Workers AI'. If building a custom\n"
            "                               token, the REST API needs BOTH Account ->\n"
            "                               Workers AI -> Edit AND -> Read; Read alone 401s)\n"
            "Free tier is 10,000 neurons/day, no card required."
        )

    url = CF_URL.format(account=account, model=CF_MODEL)
    headers = {"Authorization": f"Bearer {token}"}

    def fill(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        payload = {
            "prompt": prompt,
            "negative_prompt": negative,
            "image_b64": base64.b64encode(_png_bytes(rgb, "RGB")).decode(),
            "mask": list(_png_bytes((mask > 0).astype(np.uint8) * 255, "L")),
            "num_steps": steps,
            "guidance": guidance,
            "width": w,
            "height": h,
        }
        r = httpx.post(url, headers=headers, json=payload, timeout=TIMEOUT)
        if r.status_code != 200 or r.headers.get("content-type", "").startswith("application/json"):
            raise RuntimeError(f"Workers AI {r.status_code}: {r.text[:400]}")
        out = Image.open(BytesIO(r.content)).convert("RGB")
        if out.size != (w, h):
            out = out.resize((w, h), Image.LANCZOS)
        return np.array(out)

    return fill


def gemini_fill(instruction: str, model: str = "gemini-3.1-flash-image"):
    """Instruction editing, no mask. Requires billing on the AI Studio project as of
    December 2025 — kept so the swap is one line if that changes or you enable it."""
    from google import genai

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("Set GEMINI_API_KEY in spike/.env")
    client = genai.Client(api_key=key)

    def fill(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        resp = client.models.generate_content(
            model=model, contents=[Image.fromarray(rgb), instruction]
        )
        for cand in resp.candidates or []:
            for part in cand.content.parts or []:
                if getattr(part, "inline_data", None) and part.inline_data.data:
                    out = Image.open(BytesIO(part.inline_data.data)).convert("RGB")
                    return np.array(out.resize(rgb.shape[1::-1], Image.LANCZOS))
        raise RuntimeError("Gemini returned no image part")

    return fill


PROVIDERS = {"cloudflare": cloudflare_fill, "gemini": gemini_fill}
