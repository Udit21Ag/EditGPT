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
        return np.array(out, dtype=np.uint8)
