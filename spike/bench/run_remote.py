"""Day 4: the remote generative lane, whichever provider is configured.

Three questions, in the order they matter to the plan:

1. Can it do what LaMa structurally cannot — i3's moustache, i10's banana?
2. Can it remove a cast shadow with its object, on i1 and i11? That is the one
   defect local compositing could not fix.
3. Can it clear i13's air conditioner, where the whole local path failed?

Remote fills go through the SAME crop, colour match and feather as LaMa
(run_lama.erase_crop_with), so a side-by-side compares models, not plumbing.
"""

from __future__ import annotations

import os
import time

import cv2
import numpy as np
from dotenv import load_dotenv

from bench import cases
from bench.common import MODELS, OUT, emit, load, session
from bench.providers import PROVIDERS
from bench.run_lama import erase_crop_with
from bench.run_pipeline import segment

PROBES = ["i3", "i10", "i1", "i11", "i13"]


def main() -> None:
    load_dotenv(os.path.join(os.path.dirname(__file__), os.pardir, ".env"))
    name = os.environ.get("REMOTE_PROVIDER", "cloudflare")
    if name not in PROVIDERS:
        raise SystemExit(f"REMOTE_PROVIDER={name!r}; choose one of {list(PROVIDERS)}")
    make_fill = PROVIDERS[name]

    enc = session(MODELS / "mobilesam-encoder.onnx")
    dec = session(MODELS / "mobilesam-decoder.onnx")
    by_id = {c.id: c for c in cases.load()}
    OUT.mkdir(exist_ok=True)

    results = {}
    for cid in PROBES:
        case = by_id[cid]
        image = load(case.path)
        rgb = np.array(image)

        # Removals reuse the exact mask the local lane got, so the comparison is
        # of the fill alone. Additions have no object to segment, so the reference
        # box is the mask — that is what a user's brush would give us anyway.
        if case.op == "remove":
            mask = segment(enc, dec, rgb, case.box)
        else:
            h, w = rgb.shape[:2]
            x0, y0, x1, y1 = case.box
            mask = np.zeros((h, w), np.uint8)
            mask[round(y0 * h) : round(y1 * h), round(x0 * w) : round(x1 * w)] = 255

        started = time.perf_counter()
        try:
            out = erase_crop_with(make_fill(case.fill), image, mask)
        except Exception as exc:  # noqa: BLE001 — the reason is the finding
            results[cid] = {
                "prompt": case.prompt,
                "error": f"{type(exc).__name__}: {str(exc)[:300]}",
            }
            continue
        elapsed = round(time.perf_counter() - started, 2)

        overlay = rgb.copy()
        overlay[mask > 0] = (0.55 * overlay[mask > 0] + 0.45 * np.array([255, 40, 40])).astype(
            np.uint8
        )
        cv2.imwrite(
            str(OUT / f"remote_{name}_{cid}.png"),
            cv2.cvtColor(np.concatenate([rgb, overlay, np.array(out)], axis=1), cv2.COLOR_RGB2BGR),
        )

        results[cid] = {"prompt": case.prompt, "fill": case.fill, "op": case.op, "seconds": elapsed}

    ok = [r for r in results.values() if "error" not in r]
    emit(
        {
            "provider": name,
            "warm_p50_s": ok[0]["seconds"] if ok else None,
            "succeeded": f"{len(ok)}/{len(PROBES)}",
            "cases": results,
            "note": "compare remote_*_i1 and _i11 against the local lane — shadows are the test",
        }
    )


if __name__ == "__main__":
    main()
