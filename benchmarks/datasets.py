"""Fetching and iterating the external benchmark sets.

Each loader yields a small, uniform record so the benchmark runners do not care which
dataset they are reading. Everything is cached under a single directory and nothing is
committed — these are large and freely re-downloadable.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

RGB = npt.NDArray[np.uint8]
Mask = npt.NDArray[np.uint8]

# One parquet per split, images embedded. The `jxu124/refcocog` annotations-only mirror
# is smaller but references COCO train2014 (13 GB), so this self-contained file is the
# cheaper and more reproducible choice.
REFCOCO_REPO = "jxu124/refcoco-benchmark"
REFCOCOG_VAL = "data/refcocog_umd_val-00000-of-00001-23c691a8c4bd006d.parquet"
REMOVALBENCH = "BaiLing/RemovalBench"


def cache_dir() -> Path:
    """Where benchmark data lives. Override with EDITGPT_BENCH_DIR."""
    root = os.environ.get("EDITGPT_BENCH_DIR")
    return Path(root) if root else Path.home() / ".cache" / "editgpt" / "benchmarks"


@dataclass(frozen=True, slots=True)
class GroundingSample:
    """An image, a phrase, and the mask a human said the phrase refers to."""

    id: str
    image: RGB
    phrase: str
    mask: Mask
    """Ground truth, from the dataset. Not a box we drew."""


@dataclass(frozen=True, slots=True)
class RemovalSample:
    """An image with an object, the same scene without it, and the object's mask."""

    id: str
    image: RGB
    ground_truth: RGB
    """What the scene should look like once the object is gone."""
    mask: Mask
    baseline: RGB | None = None
    """A reference system's output, where the dataset ships one."""


def _to_rgb(image: Any) -> RGB:
    from PIL import Image

    if not isinstance(image, Image.Image):
        image = Image.open(image)
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _polygons_to_mask(segmentation: Any, height: int, width: int) -> Mask:
    """Rasterise COCO polygon segmentation.

    COCO stores a mask as a list of flattened [x, y, x, y, ...] polygons; a single object
    may have several when it is split by occlusion, so all of them are filled.
    """
    import cv2

    mask = np.zeros((height, width), dtype=np.uint8)
    polygons = segmentation if isinstance(segmentation, list) else [segmentation]
    for polygon in polygons:
        if not isinstance(polygon, list) or len(polygon) < 6:
            continue
        points = np.asarray(polygon, dtype=np.float64).reshape(-1, 2).round().astype(np.int32)
        cv2.fillPoly(mask, [points], 255)
    return mask


def _sentences(ref: dict[str, Any]) -> list[str]:
    """Pull referring phrases out of a ref record, whatever shape it arrived in."""
    info = ref.get("ref_info") or {}
    found: list[str] = []
    for value in (info.get("sentences"), info.get("captions"), info.get("raw_sentences")):
        if isinstance(value, str):
            found.append(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    found.append(item)
                elif isinstance(item, dict):
                    text = item.get("raw") or item.get("sent")
                    if isinstance(text, str):
                        found.append(text)
    return [f.strip() for f in found if f and f.strip()]


def load_grounding(limit: int = 300, *, per_image: int = 1) -> Iterator[GroundingSample]:
    """RefCOCOg validation: referring expressions with ground-truth segmentation masks.

    Expressions average 8.4 words, so this exercises the same free-text grounding the
    product does rather than single nouns — and the mask is the dataset's, not a box we
    drew, so IoU here is the field's standard metric.

    `per_image` caps how many expressions are taken from one image, so a crowded photo
    cannot dominate the sample.
    """
    import io

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from PIL import Image

    path = hf_hub_download(
        repo_id=REFCOCO_REPO,
        repo_type="dataset",
        filename=REFCOCOG_VAL,
        cache_dir=str(cache_dir()),
    )

    yielded = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
        rows = batch.to_pylist()
        for row in rows:
            if yielded >= limit:
                return
            try:
                image = np.asarray(
                    Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB"), dtype=np.uint8
                )
            except (KeyError, TypeError, OSError):
                continue

            height, width = image.shape[:2]
            for ref in (row.get("ref_list") or [])[:per_image]:
                if yielded >= limit:
                    return
                phrases = _sentences(ref)
                segmentation = (ref.get("ann_info") or {}).get("segmentation")
                if not phrases or segmentation is None:
                    continue
                mask = _polygons_to_mask(segmentation, height, width)
                if int(mask.sum()) == 0:
                    continue  # a sample we cannot score is not a sample

                yield GroundingSample(
                    id=str((ref.get("ann_info") or {}).get("id", yielded)),
                    image=image,
                    phrase=phrases[0],
                    mask=mask,
                )
                yielded += 1


def load_removal(limit: int = 100) -> Iterator[RemovalSample]:
    """RemovalBench: paired before/after images with object masks.

    Because the "after" image is real rather than generated, reference metrics such as
    SSIM and PSNR are valid here — which they are not on our own fixtures.
    """
    from huggingface_hub import snapshot_download

    root = Path(
        snapshot_download(
            repo_id=REMOVALBENCH,
            repo_type="dataset",
            cache_dir=str(cache_dir()),
            allow_patterns=["images/*", "gt/*", "masks/*", "results_omnieraser_base/*"],
        )
    )
    names = sorted(
        (p.name for p in (root / "images").glob("*.png")),
        key=lambda n: (len(n), n),
    )
    for name in names[:limit]:
        gt_path, mask_path = root / "gt" / name, root / "masks" / name
        if not (gt_path.exists() and mask_path.exists()):
            continue
        baseline_path = root / "results_omnieraser_base" / name
        mask = _to_rgb(mask_path)[:, :, 0]
        yield RemovalSample(
            id=Path(name).stem,
            image=_to_rgb(root / "images" / name),
            ground_truth=_to_rgb(gt_path),
            mask=np.asarray((mask > 127) * 255, dtype=np.uint8),
            baseline=_to_rgb(baseline_path) if baseline_path.exists() else None,
        )
