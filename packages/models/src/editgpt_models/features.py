"""Features describing an editing task, computable before any eraser runs.

Everything here is derived from the image and the mask alone — no ground truth, no model
output. That constraint is what makes these usable at inference time to *predict* which
eraser will do better, rather than only to explain it afterwards.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import cv2
import numpy as np

from editgpt_models.compositing import RGB, RING_PX, Mask


@dataclass(frozen=True, slots=True)
class TaskFeatures:
    """A compact description of "what kind of erase is this?"."""

    mask_coverage: float
    """Masked pixels as a fraction of the frame. The single most predictive quantity in
    Phase 0's manual analysis: the two erasers swapped places as objects grew."""

    mask_compactness: float
    """4*pi*area / perimeter^2. 1.0 is a circle; a thin or ragged shape tends to 0."""

    mask_solidity: float
    """Area over the area of its bounding box. Separates a lattice from a slab."""

    aspect_ratio: float
    """Longer side over shorter side of the mask's bounding box."""

    ring_edge_density: float
    """Mean gradient magnitude just outside the mask — how textured the surroundings are."""

    ring_chroma_std: float
    """Colour spread just outside the mask. High means the fill must match more than one
    surface, which is where a single-colour match fails."""

    ring_luma_std: float

    border_contact: float
    """Fraction of the mask's outline that lies on the image edge, where there is context
    on one side only."""

    relative_scale: float
    """Mask's longest side over the image's longest side."""

    def as_vector(self) -> np.ndarray:
        return np.array([getattr(self, f.name) for f in fields(self)], dtype=np.float64)

    @staticmethod
    def names() -> list[str]:
        return [f.name for f in fields(TaskFeatures)]

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def extract(image: RGB, mask: Mask) -> TaskFeatures:
    """Describe an erase task from its inputs alone."""
    binary = (mask > 0).astype(np.uint8)
    height, width = binary.shape
    area = float(binary.sum())
    if area == 0:
        return TaskFeatures(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = sum(cv2.arcLength(c, True) for c in contours) or 1.0
    compactness = min(4.0 * np.pi * area / (perimeter**2), 1.0)

    ys, xs = np.nonzero(binary)
    box_h = float(ys.max() - ys.min() + 1)
    box_w = float(xs.max() - xs.min() + 1)
    solidity = area / (box_h * box_w)
    aspect = max(box_h, box_w) / max(min(box_h, box_w), 1.0)

    ring = (cv2.dilate(binary, np.ones((RING_PX, RING_PX), np.uint8)) > 0) & (binary == 0)
    if int(ring.sum()) < 50:
        ring = binary == 0
    if int(ring.sum()) == 0:
        # A mask covering the whole frame leaves nothing outside it. Statistics over the
        # image itself are the honest answer; an empty slice yields NaN, which would
        # poison any model trained on these features without ever failing loudly.
        ring = np.ones_like(binary, dtype=bool)

    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    edges = np.abs(cv2.Laplacian(grey, cv2.CV_32F))
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)

    edge_on_border = (
        binary[0, :].sum() + binary[-1, :].sum() + binary[:, 0].sum() + binary[:, -1].sum()
    )

    return TaskFeatures(
        mask_coverage=float(area / (height * width)),
        mask_compactness=float(compactness),
        mask_solidity=float(solidity),
        aspect_ratio=float(aspect),
        ring_edge_density=float(edges[ring].mean()),
        ring_chroma_std=float(np.hypot(lab[ring, 1].std(), lab[ring, 2].std())),
        ring_luma_std=float(lab[ring, 0].std()),
        border_contact=min(float(edge_on_border) / max(perimeter, 1.0), 1.0),
        relative_scale=float(max(box_h, box_w) / max(height, width)),
    )
