"""A reference-free quality score for an erasure, in feature space rather than pixels.

`editgpt_core.metrics.fill_metrics` asks a photometric question — does this fill have the
same colour and detail level as the ring of pixels around it? On RemovalBench that turned
out to be a poor stand-in for the question we actually care about, which is whether the
object is really gone and the background really continued: its correlation with
SSIM-against-truth was **+0.128**, where a working proxy would be negative. This module
was written as the candidate replacement (TD-013).

This module implements the ReMOVE score (arXiv:2409.00707): take a segmentation ViT's
patch embeddings, average them inside the erased region and outside it, and report the
cosine similarity. The published version reports |correlation| ~0.515 against LPIPS and
74.7% agreement with human preference — where LPIPS itself, with access to the ground
truth, scored 71.9%.

The reason it belongs here rather than in a paper's repository is that **we already load
the encoder it needs**. MobileSAM's image encoder runs for every box and text prompt and
emits a 64x64 grid of 256-dimensional patch embeddings, which is exactly the input the
score is defined over. The marginal cost is one forward pass, not a new model.

Two deviations from the paper, both forced: the encoder is MobileSAM's TinyViT rather
than SAM's ViT-H, and the region crop below is our `crop_window`, not theirs.

**Measured, and not adopted.** On 2026-08-27 this was scored against paired ground truth
on both benchmarks, and it does not earn a place in the router:

| | RemovalBench | RORD-50 |
| --- | ---: | ---: |
| Spearman vs SSIM (positive is correct) | **-0.146** | **+0.415** |
| picks the better eraser | 52.2% | 44.9% |
| the same figures for `fill_metrics.cost` | 43.5% | **68.1%** |

It flips sign between datasets exactly as the photometric score does, and where the signal
is strongest it is beaten by the metric that costs nothing extra. So the router still uses
`cost`, and this stays as a **benchmark instrument**: `benchmarks/removal.py` reports it
on every run, so a third dataset can revisit the question without anyone re-implementing
the paper. See TD-013 and TD-017.

Do not wire it into a decision without re-reading `benchmarks/out/removal.json`.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from editgpt_models.compositing import RGB, Mask, crop_window

MIN_PATCHES = 4
"""Fewest patches a region may occupy and still have a meaningful mean.

The embedding grid is 64x64 for a 1024px input, so one patch is ~1.6% of the longest
side. Below four the mean is dominated by whichever single patch straddles the boundary.
"""


def embed(encoder: Any, rgb: RGB) -> tuple[np.ndarray, tuple[int, int]]:
    """Patch embeddings for an image, plus the grid extent its real pixels occupy.

    The encoder pads to a square, so the bottom and right of the grid are embeddings of
    black padding. Returning the valid extent keeps callers from averaging it in.
    """
    from editgpt_models.segment import ENCODER_SIZE, preprocess_for_encoder

    prepared, scale = preprocess_for_encoder(rgb, encoder)
    embedding = encoder.run(None, {encoder.get_inputs()[0].name: prepared})[0]

    grid = int(embedding.shape[-1])
    stride = ENCODER_SIZE / grid  # 16 for MobileSAM: 1024 px in, a 64x64 grid out
    height, width = rgb.shape[:2]
    # `scale` is how much the image was shrunk to fit the padded square; dividing by the
    # patch stride converts those pixels into grid cells.
    rows = min(max(round(height * scale / stride), 1), grid)
    cols = min(max(round(width * scale / stride), 1), grid)
    return embedding[0], (rows, cols)


def consistency(encoder: Any, rgb: RGB, mask: Mask) -> float:
    """How well the erased region agrees, semantically, with what surrounds it.

    Returns cosine similarity in [-1, 1]; **higher is better**, unlike `FillMetrics.cost`.
    A region still containing the object embeds differently from its background and
    scores low; a region that has genuinely become background scores high.

    Returns 0.0 when the mask is empty or so large that nothing is left to compare
    against — an honest "cannot tell", not a passing grade.
    """
    binary = (mask > 0).astype(np.uint8)
    if not binary.any() or binary.all():
        return 0.0

    # The paper crops so the masked and unmasked patch counts are comparable: on a full
    # frame a small object occupies one or two patches and its mean is noise. Our
    # existing crop window already does this, and reusing it keeps the score measuring
    # the same neighbourhood the compositor works in.
    rows, cols, _ = crop_window(binary, binary.shape)
    crop_rgb = np.ascontiguousarray(rgb[rows, cols])
    crop_mask = np.ascontiguousarray(binary[rows, cols])
    if not crop_mask.any() or crop_mask.all():
        return 0.0

    features, (valid_rows, valid_cols) = embed(encoder, crop_rgb)
    grid = features.shape[-1]
    small = cv2.resize(crop_mask, (grid, grid), interpolation=cv2.INTER_AREA)

    valid = np.zeros((grid, grid), dtype=bool)
    valid[:valid_rows, :valid_cols] = True
    inside = (small > 0) & valid
    outside = (small == 0) & valid
    if int(inside.sum()) < MIN_PATCHES or int(outside.sum()) < MIN_PATCHES:
        return 0.0

    flat = features.reshape(features.shape[0], -1)
    inner = flat[:, inside.reshape(-1)].mean(axis=1)
    outer = flat[:, outside.reshape(-1)].mean(axis=1)

    denominator = float(np.linalg.norm(inner) * np.linalg.norm(outer))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(inner, outer) / denominator)
