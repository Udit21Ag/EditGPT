"""MobileSAM: box/point prompt -> mask. Encoder and decoder are timed separately,
because production caches the embedding per image and re-runs only the decoder
on every brush stroke. If the decoder isn't interactive-fast (<100 ms), the
'magic select' UX in Phase 8 doesn't work.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from bench.common import MODELS, OUT, describe_io, emit, load, photos, session, timed

ENC = 1024
SAM_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
SAM_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def preprocess(rgb: np.ndarray, sess) -> tuple[np.ndarray, float]:
    """Shape the encoder input to whatever this export actually asks for.

    Acly's export wraps SAM's own preprocessing: it takes bare HWC float RGB at the
    original size and does the resize/pad/normalise internally. Other exports want a
    pre-normalised NCHW 1024x1024 tensor. We branch on the declared input shape rather
    than assuming, and return the 1024-space scale either way, because the decoder's
    point_coords must be in that space regardless.
    """
    h, w = rgb.shape[:2]
    scale = ENC / max(h, w)
    shape = sess.get_inputs()[0].shape

    if len(shape) == 3:
        # Acly's export normalises and PADS to 1024, but does not resize. SAM's
        # ResizeLongestSide is the caller's job, and skipping it silently desyncs
        # the embedding from the point coords — the mask then tears along the box
        # instead of snapping to an object.
        resized = cv2.resize(
            rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA
        )
        return resized.astype(np.float32), scale

    resized = cv2.resize(rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((ENC, ENC, 3), dtype=np.uint8)
    canvas[: resized.shape[0], : resized.shape[1]] = resized
    x = (canvas.astype(np.float32) - SAM_MEAN) / SAM_STD
    return x.transpose(2, 0, 1)[None, ...], scale


def main() -> None:
    t0 = time.perf_counter()
    enc = session(MODELS / "mobilesam-encoder.onnx")
    dec = session(MODELS / "mobilesam-decoder.onnx")
    cold = round(time.perf_counter() - t0, 3)

    enc_in = enc.get_inputs()[0]

    files = photos()
    img = load(files[0])
    rgb = np.array(img)
    x, scale = preprocess(rgb, enc)

    enc_stats = timed(lambda: enc.run(None, {enc_in.name: x}), repeats=3)
    embedding = enc.run(None, {enc_in.name: x})[0]

    # A box prompt over the middle of the frame: SAM encodes a box as two
    # corner points labelled 2 (top-left) and 3 (bottom-right).
    h, w = rgb.shape[:2]
    corners = np.array([[[w * 0.35, h * 0.45], [w * 0.65, h * 0.85]]], dtype=np.float32) * scale
    labels = np.array([[2, 3]], dtype=np.float32)

    feed_all = {
        "image_embeddings": embedding,
        "point_coords": corners,
        "point_labels": labels,
        "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
        "has_mask_input": np.zeros(1, dtype=np.float32),
        "orig_im_size": np.array([h, w], dtype=np.float32),
    }
    names = {i.name for i in dec.get_inputs()}
    missing = names - feed_all.keys()
    if missing:
        raise SystemExit(
            f"Decoder wants unmapped inputs {missing}; see io dump: {describe_io(dec)}"
        )
    feed = {k: v for k, v in feed_all.items() if k in names}

    dec_stats = timed(lambda: dec.run(None, feed))
    masks = dec.run(None, feed)[0]

    OUT.mkdir(exist_ok=True)
    mask = (masks[0, 0] > 0).astype(np.uint8) * 255
    if mask.shape != (h, w):  # decoder returned 256/1024-space logits
        mask = cv2.resize(mask, (ENC, ENC), interpolation=cv2.INTER_NEAREST)
        mask = mask[: round(h * scale), : round(w * scale)]
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    overlay = rgb.copy()
    overlay[mask > 0] = (0.5 * overlay[mask > 0] + 0.5 * np.array([255, 40, 40])).astype(np.uint8)
    cv2.imwrite(str(OUT / "mobilesam_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    emit(
        {
            "cold_load_s": cold,
            "warm_p50_s": dec_stats["warm_p50_s"],
            "encoder": enc_stats,
            "decoder": dec_stats,
            "encoder_input": {"name": enc_in.name, "shape": enc_in.shape, "type": enc_in.type},
            "io": {"encoder": describe_io(enc), "decoder": describe_io(dec)},
            "note": "decoder p50 must be <0.1s for interactive magic-select",
        }
    )


if __name__ == "__main__":
    main()
