"""Run-length codec for binary masks.

Column-major, starting with a run of zeros, so `counts[1::2]` are the set runs. This
matches the COCO convention, which means masks are interchangeable with the wider
tooling ecosystem without a translation layer.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from editgpt_core.spec import MaskRef


def encode(mask: npt.NDArray[np.uint8]) -> MaskRef:
    """Binary mask (H, W), non-zero meaning set, to a `MaskRef`."""
    if mask.ndim != 2:
        raise ValueError(f"expected a 2-D mask, got shape {mask.shape}")
    height, width = mask.shape
    flat = (mask.reshape(-1, order="F") > 0).astype(np.uint8)

    # Boundaries between runs, plus the implicit ones at each end.
    changes = np.flatnonzero(np.diff(flat))
    edges = np.concatenate(([-1], changes, [flat.size - 1]))
    counts = np.diff(edges).astype(int).tolist()
    if flat.size and flat[0] == 1:
        counts.insert(0, 0)  # the encoding always opens with a zero run
    return MaskRef(width=width, height=height, counts=counts)


def decode(ref: MaskRef) -> npt.NDArray[np.uint8]:
    """`MaskRef` back to a binary mask (H, W) of 0 and 1."""
    flat = np.zeros(ref.width * ref.height, dtype=np.uint8)
    pos = 0
    for i, run in enumerate(ref.counts):
        if i % 2:
            flat[pos : pos + run] = 1
        pos += run
    return flat.reshape((ref.height, ref.width), order="F")
