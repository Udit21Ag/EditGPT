"""What models exist, where they come from, and what they cost.

The RSS figures are measured on an M1 with 8 GB, not vendor claims. They are what the
`ModelSlot` budget and the CI memory ceiling are derived from.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ModelSpec:
    key: str
    repo_id: str
    filename: str
    role: str
    peak_rss_mb: int
    """Measured peak resident set, alone, on the reference machine."""
    note: str = ""


REGISTRY: dict[str, ModelSpec] = {
    "sam-encoder": ModelSpec(
        key="sam-encoder",
        repo_id="Acly/MobileSAM",
        filename="mobile_sam_image_encoder.onnx",
        role="image embedding for mask refinement",
        peak_rss_mb=620,
        note="Normalises and PADS to 1024; the caller must resize the longest side first.",
    ),
    "sam-decoder": ModelSpec(
        key="sam-decoder",
        repo_id="Acly/MobileSAM",
        filename="sam_mask_decoder_single.onnx",
        role="prompt to mask",
        peak_rss_mb=620,
        note="Returns iou_predictions; that confidence gates whether we trust the refinement.",
    ),
    "migan": ModelSpec(
        key="migan",
        repo_id="andraniksargsyan/migan",
        filename="migan_pipeline_v2.onnx",
        role="primary eraser",
        peak_rss_mb=1150,
        note="Mask polarity is INVERTED: 255 means keep. Crops internally, any resolution.",
    ),
    "esrgan-x2": ModelSpec(
        key="esrgan-x2",
        repo_id="SceneWorks/real-esrgan-onnx",
        filename="real_esrgan_x2.onnx",
        role="2x resolution enhancement",
        peak_rss_mb=760,
        note="Dynamic input size, but activations scale with area — always drive it tiled.",
    ),
    "lama": ModelSpec(
        key="lama",
        repo_id="Carve/LaMa-ONNX",
        filename="lama_fp32.onnx",
        role="escalation eraser",
        peak_rss_mb=964,
        note="Fixed 512x512 input, so the caller owns the crop and paste-back.",
    ),
}


def models_dir() -> Path:
    """Where weights live. Override with EDITGPT_MODELS_DIR."""
    return Path(os.environ.get("EDITGPT_MODELS_DIR", Path.home() / ".cache" / "editgpt" / "models"))


def model_path(key: str, *, download: bool = True) -> Path:
    """Absolute path to a model's weights, fetching them on first use."""
    spec = registry(key)
    dest = models_dir() / f"{key}.onnx"
    if dest.exists():
        return dest
    if not download:
        raise FileNotFoundError(f"{key} not present at {dest} and download=False")

    from huggingface_hub import hf_hub_download

    dest.parent.mkdir(parents=True, exist_ok=True)
    src = hf_hub_download(repo_id=spec.repo_id, filename=spec.filename)
    dest.write_bytes(Path(src).read_bytes())
    return dest


def registry(key: str) -> ModelSpec:
    try:
        return REGISTRY[key]
    except KeyError:
        raise KeyError(f"unknown model {key!r}; known: {sorted(REGISTRY)}") from None
