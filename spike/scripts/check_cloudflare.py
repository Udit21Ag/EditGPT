"""Verify Workers AI credentials and tell you exactly which part is wrong.

Diagnosing this after the fact costs more than checking it up front: a bad
account id, a token without Workers AI permission, and an exhausted daily
neuron budget all surface as unhelpful HTTP errors mid-benchmark.
"""

from __future__ import annotations

import base64
import os
import sys
from io import BytesIO
from pathlib import Path

import httpx
import numpy as np
from dotenv import load_dotenv
from PIL import Image

MODEL = "@cf/runwayml/stable-diffusion-v1-5-inpainting"


def main() -> int:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    account = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
    token = os.environ.get("CLOUDFLARE_API_TOKEN", "").strip()

    if not token:
        print("✗ CLOUDFLARE_API_TOKEN missing from spike/.env")
        return 1
    r = httpx.get(
        "https://api.cloudflare.com/client/v4/user/tokens/verify",
        headers={"Authorization": f"Bearer {token}"},
        timeout=20,
    )
    body = r.json()
    if not body.get("success"):
        print(f"✗ token rejected: {body.get('errors')}")
        return 1
    print(f"✓ token valid and active (token id {body['result']['id']})")

    if not account:
        print(
            "✗ CLOUDFLARE_ACCOUNT_ID missing from spike/.env\n"
            "  dash.cloudflare.com -> the 32-hex segment in the URL, or\n"
            "  Workers & Pages -> right sidebar -> Account details -> Account ID.\n"
            "  NOTE: that is NOT the token id printed above."
        )
        return 1
    if len(account) != 32 or not all(c in "0123456789abcdef" for c in account.lower()):
        print(f"✗ CLOUDFLARE_ACCOUNT_ID={account!r} is not a 32-character hex id")
        return 1

    # One real inference. The inpainting model rejects a bare prompt — it requires an
    # image and a mask — so the smoke test has to send both or it fails for the wrong
    # reason and reads as a credential problem.
    def _png(arr, mode):
        buf = BytesIO()
        Image.fromarray(arr, mode=mode).save(buf, format="PNG")
        return buf.getvalue()

    img = np.full((256, 256, 3), 200, dtype=np.uint8)
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[96:160, 96:160] = 255

    r = httpx.post(
        f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{MODEL}",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "prompt": "a plain grey wall",
            "image_b64": base64.b64encode(_png(img, "RGB")).decode(),
            "mask": list(_png(mask, "L")),
            "num_steps": 4,
            "width": 256,
            "height": 256,
        },
        timeout=120,
    )
    if r.status_code == 200 and not r.headers.get("content-type", "").startswith(
        "application/json"
    ):
        print(f"✓ Workers AI reachable — {MODEL} returned {len(r.content)} bytes")
        print("\nReady. Run: make bench-remote")
        return 0

    detail = r.text[:300]
    if r.status_code == 404:
        print(f"✗ 404 — the account id is probably wrong (or has no Workers AI): {detail}")
    elif r.status_code in (401, 403):
        print(
            f"✗ {r.status_code} — token lacks Workers AI permission. Recreate it with the\n"
            f"  'Workers AI' template, or add Account -> Workers AI -> Read: {detail}"
        )
    elif r.status_code == 429:
        print(f"✗ 429 — daily neuron budget exhausted, resets 00:00 UTC: {detail}")
    else:
        print(f"✗ {r.status_code}: {detail}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
