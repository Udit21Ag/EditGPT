"""Download the Phase 0 model set into spike/models/. Idempotent."""

from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
from rich.console import Console
from rich.table import Table

MODELS = Path(__file__).resolve().parents[1] / "models"

# (label, repo_id, filename, note)
FILES = [
    ("lama", "Carve/LaMa-ONNX", "lama_fp32.onnx", "fixed 512x512, opset 17"),
    ("mobilesam-encoder", "Acly/MobileSAM", "mobile_sam_image_encoder.onnx", "ViT-T image encoder"),
    ("mobilesam-decoder", "Acly/MobileSAM", "sam_mask_decoder_single.onnx", "single-mask decoder"),
    (
        "migan-pipeline",
        "andraniksargsyan/migan",
        "migan_pipeline_v2.onnx",
        "primary eraser; crops internally",
    ),
]


def main() -> int:
    MODELS.mkdir(parents=True, exist_ok=True)
    console = Console()
    table = Table(title="Phase 0 model set", header_style="bold")
    for col in ("label", "file", "MB", "note"):
        table.add_column(col)

    for label, repo, fname, note in FILES:
        dest = MODELS / f"{label}.onnx"
        if not dest.exists():
            console.print(f"[dim]downloading {repo}/{fname} …[/dim]")
            src = hf_hub_download(repo_id=repo, filename=fname)
            dest.write_bytes(Path(src).read_bytes())
        table.add_row(label, fname, f"{dest.stat().st_size / 1e6:.1f}", note)

    console.print(table)
    console.print(
        "\n[bold]CLIPSeg[/bold] is pulled by transformers at first use "
        "(CIDAS/clipseg-rd64-refined, ~150 MB) — no action needed here."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
