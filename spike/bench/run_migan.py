"""MI-GAN alone: the number that decides the memory plan.

If MI-GAN is the primary eraser and LaMa only loads on escalation, the common
path's resident set is MobileSAM + MI-GAN rather than MobileSAM + LaMa.
"""

from __future__ import annotations

import time

import numpy as np

from bench import cases
from bench.common import MODELS, OUT, describe_io, emit, load, session, timed
from bench.run_erasers import migan_erase
from bench.run_pipeline import segment


def main() -> None:
    enc = session(MODELS / "mobilesam-encoder.onnx")
    dec = session(MODELS / "mobilesam-decoder.onnx")

    t0 = time.perf_counter()
    migan = session(MODELS / "migan-pipeline.onnx")
    cold = round(time.perf_counter() - t0, 3)

    todo = cases.load(ops={"remove"}, need_box=True)
    img = load(todo[0].path)
    mask = segment(enc, dec, np.array(img), todo[0].box)
    stats = timed(lambda: migan_erase(migan, img, mask), repeats=5)

    OUT.mkdir(exist_ok=True)
    per_case = {}
    for case in todo:
        image = load(case.path)
        src = np.array(image)
        m = segment(enc, dec, src, case.box)
        started = time.perf_counter()
        out = np.array(migan_erase(migan, image, m))
        per_case[case.id] = {
            "target": case.target,
            "seconds": round(time.perf_counter() - started, 3),
            **cases.fill_metrics(out, src, m),
        }

    emit(
        {
            "cold_load_s": cold,
            "warm_p50_s": stats["warm_p50_s"],
            "detail": stats,
            "io": describe_io(migan),
            "cases": per_case,
            "note": "MobileSAM + MI-GAN only — LaMa deliberately not loaded",
        }
    )


if __name__ == "__main__":
    main()
