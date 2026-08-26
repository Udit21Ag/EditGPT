"""Head-to-head: LaMa vs MI-GAN on every removal case, each at its best setting.

MI-GAN (Picsart, ICCV 2023) is 28 MB against LaMa's 208 MB, and its ONNX pipeline
does the crop, resize and blend inside the graph — the machinery this spike built
by hand for LaMa. If it matches LaMa's quality it is strictly the better choice
under an 8 GB ceiling; if it beats LaMa on i8 it also closes the one case the
local lane could not do at all.

Both get the identical MobileSAM mask, so this compares inpainters and nothing else.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
from PIL import Image

from bench import cases
from bench.common import MODELS, OUT, emit, load, session, timed
from bench.run_lama import erase_crop
from bench.run_pipeline import segment


def migan_erase(sess, img: Image.Image, mask: np.ndarray) -> Image.Image:
    """MI-GAN's pipeline export takes the whole image at arbitrary resolution.

    Note the inverted mask convention: 255 means KEEP, 0 means inpaint — the
    opposite of LaMa. Getting this backwards produces a plausible-looking image
    with everything except the object repainted, which is easy to miss.
    """
    rgb = np.array(img).transpose(2, 0, 1)[None, ...]
    keep = np.where(mask > 0, 0, 255).astype(np.uint8)[None, None, ...]
    out = sess.run(None, {"image": rgb.astype(np.uint8), "mask": keep})[0]
    return Image.fromarray(out[0].transpose(1, 2, 0))


def main() -> None:
    enc = session(MODELS / "mobilesam-encoder.onnx")
    dec = session(MODELS / "mobilesam-decoder.onnx")

    t0 = time.perf_counter()
    lama = session(MODELS / "lama.onnx")
    lama_cold = round(time.perf_counter() - t0, 3)
    t0 = time.perf_counter()
    migan = session(MODELS / "migan-pipeline.onnx")
    migan_cold = round(time.perf_counter() - t0, 3)

    todo = cases.load(ops={"remove"}, need_box=True)
    OUT.mkdir(exist_ok=True)

    first = load(todo[0].path)
    m0 = segment(enc, dec, np.array(first), todo[0].box)
    lama_t = timed(lambda: erase_crop(lama, first, m0), repeats=3)
    migan_t = timed(lambda: migan_erase(migan, first, m0), repeats=3)

    scored, wins = {}, {"lama": 0, "migan": 0}
    for case in todo:
        image = load(case.path)
        src = np.array(image)
        mask = segment(enc, dec, src, case.box)

        a = np.array(erase_crop(lama, image, mask))
        b = np.array(migan_erase(migan, image, mask))
        ma = cases.fill_metrics(a, src, mask)
        mb = cases.fill_metrics(b, src, mask)

        # One number per fill: photometric error plus how far its detail level
        # strays from the surrounding texture, in either direction.
        def cost(m):
            return m["chroma_delta"] + m["lum_delta"] + 12 * abs(np.log(max(m["edge_ratio"], 1e-3)))

        winner = "lama" if cost(ma) < cost(mb) else "migan"
        wins[winner] += 1

        overlay = src.copy()
        overlay[mask > 0] = (0.55 * overlay[mask > 0] + 0.45 * np.array([255, 40, 40])).astype(
            np.uint8
        )
        cv2.imwrite(
            str(OUT / f"erasers_{case.id}.png"),
            cv2.cvtColor(np.concatenate([src, overlay, a, b], axis=1), cv2.COLOR_RGB2BGR),
        )

        scored[case.id] = {
            "target": case.target,
            "winner": winner,
            "lama": ma,
            "migan": mb,
            "lama_cost": round(cost(ma), 1),
            "migan_cost": round(cost(mb), 1),
        }

    emit(
        {
            "cold_load_s": {"lama": lama_cold, "migan": migan_cold},
            "warm_p50_s": migan_t["warm_p50_s"],
            "latency": {"lama_crop": lama_t, "migan_pipeline": migan_t},
            "wins": wins,
            "cases": scored,
            "note": "panels are original | mask | LaMa | MI-GAN",
        }
    )


if __name__ == "__main__":
    main()
