"""Cloudflare Workers AI: the free generative lane.

Free tier is 10,000 neurons a day with no card. The model is mask+prompt, not
instruction-following, so the caller must say what *should* be there rather than what
to do — translating an instruction into a scene description is the IntentAgent's job.

Two API details that cost time to discover:

* the REST endpoint needs an API token with **both** ``Workers AI - Read`` and
  ``Workers AI - Edit``; Read alone returns a 401 that reads like a bad token;
* the mask goes in a field named ``mask`` as a list of PNG bytes. An error naming
  ``mask_image`` means no mask was recognised at all, not that the field is misnamed.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from io import BytesIO

import httpx
import numpy as np
from editgpt_core.errors import ProviderError
from PIL import Image

from editgpt_providers.base import RGB, Mask

MODEL = "@cf/runwayml/stable-diffusion-v1-5-inpainting"
ENDPOINT = "https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{model}"
DEFAULT_NEGATIVE = "blurry, distorted, watermark, text"

BLANK_MEAN = 16.0
BLANK_SPREAD = 8.0
"""When a 200 response is not a generation.

Stable Diffusion returns a **black frame** when its safety checker fires — a successful
HTTP call carrying nothing. Composited, that is a black rectangle stamped on a user's
photograph, reported as a completed edit. It happened once on the golden set's `i3`, on a
prompt asking for a moustache, and the identical call a few minutes later produced a
correct result. So it is transient, silent and destructive: exactly the combination that
has to be caught rather than watched for.

Calibrated on real returns rather than chosen. A correct fill measured mean 122.5 with a
per-channel spread of 61.7 inside the mask; the failure measured 11.3 and 6.7 after
compositing, and would be nearer zero before it. Both conditions must hold, because
darkness alone is not a fault — a legitimate fill of a shadow measured mean 37.2, and a
generation of *any* kind carries far more texture than this.
"""


def _png(array: np.ndarray, mode: str) -> bytes:
    buffer = BytesIO()
    Image.fromarray(array, mode=mode).save(buffer, format="PNG")
    return buffer.getvalue()


@dataclass
class CloudflareWorkersAI:
    name: str = "cloudflare"
    steps: int = 20
    guidance: float = 7.5
    timeout_s: float = 120.0
    account_id: str | None = None
    api_token: str | None = None

    def __post_init__(self) -> None:
        self.account_id = self.account_id or os.environ.get("CLOUDFLARE_ACCOUNT_ID")
        self.api_token = self.api_token or os.environ.get("CLOUDFLARE_API_TOKEN")

    def is_configured(self) -> bool:
        return bool(self.account_id and self.api_token)

    def fill(self, rgb: RGB, mask: Mask, prompt: str) -> RGB:
        if not self.is_configured():
            raise ProviderError(
                "Cloudflare Workers AI is not configured. Set CLOUDFLARE_ACCOUNT_ID and "
                "CLOUDFLARE_API_TOKEN; the token needs Workers AI Edit *and* Read."
            )
        if not prompt.strip():
            # The API rejects an empty prompt outright, so there is no way to ask for
            # "nothing" — which is why this lane cannot be used for removal.
            raise ProviderError("this model requires a non-empty prompt")

        height, width = rgb.shape[:2]
        payload = {
            "prompt": prompt,
            "negative_prompt": DEFAULT_NEGATIVE,
            "image_b64": base64.b64encode(_png(rgb, "RGB")).decode(),
            "mask": list(_png((mask > 0).astype(np.uint8) * 255, "L")),
            "num_steps": self.steps,
            "guidance": self.guidance,
            "width": width,
            "height": height,
        }
        url = ENDPOINT.format(account=self.account_id, model=MODEL)
        try:
            response = httpx.post(
                url,
                headers={"Authorization": f"Bearer {self.api_token}"},
                json=payload,
                timeout=self.timeout_s,
            )
        except httpx.HTTPError as exc:
            raise ProviderError(f"transport error: {exc}") from exc

        content_type = response.headers.get("content-type", "")
        if response.status_code != 200 or content_type.startswith("application/json"):
            raise ProviderError(f"Workers AI {response.status_code}: {response.text[:300]}")

        out = Image.open(BytesIO(response.content)).convert("RGB")
        if out.size != (width, height):
            out = out.resize((width, height), Image.Resampling.LANCZOS)
        filled = np.array(out, dtype=np.uint8)

        _reject_blank(filled, mask)
        return filled


def _reject_blank(filled: RGB, mask: Mask) -> None:
    """Refuse a 200 that carries no image.

    Raising is the whole point. Returning this would composite a black rectangle onto a
    photograph and report the job as done — the failure mode the project's own rule about
    silence exists for, in its worst form: it does not merely look like success, it looks
    like a deliberate edit.
    """
    inside = mask > 0
    region = filled[inside] if inside.any() else filled.reshape(-1, 3)
    if region.size and region.mean() < BLANK_MEAN and region.std(axis=0).max() < BLANK_SPREAD:
        raise ProviderError(
            f"the model returned a blank fill (mean {region.mean():.1f}, spread "
            f"{region.std(axis=0).max():.1f}), which is what Stable Diffusion sends when "
            "its safety checker fires. Nothing was generated; retry or rephrase."
        )
