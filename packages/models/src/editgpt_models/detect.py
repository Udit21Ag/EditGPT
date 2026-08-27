"""Turning a phrase into candidate boxes, with Grounding DINO.

This replaces the CLIPSeg heatmap as the entry point to text grounding. The reason is
measured rather than architectural: on held-out RefCOCOg the mask source dominated the
score (SAM-refined 0.469 against a bare CLIPSeg seed 0.234), and a *box* is a far
stronger SAM prompt than a heatmap's bounding rectangle. Grounding DINO was trained on
phrase grounding, which is the task; CLIPSeg was trained on single-concept segmentation
and was being asked to resolve eight-word relational expressions.

Two traps, both learned from the export rather than the paper:

* The graph declares a **fixed 800x800** `pixel_values` with a companion `pixel_mask`,
  which reads like an invitation to letterbox. It is not. This export's
  `preprocessor_config.json` sets `size` to a literal 800x800, so it was calibrated and
  quantised on a squashed image, and it expects one. Measured on 40 RefCOCOg samples:
  squash gives box-IoU **0.511**, aspect-preserving letterbox with a `pixel_mask`
  gives **0.230**. Distorting the aspect ratio is the correct thing to do here.
* Text must be **lowercase and terminated with a period**. Grounding DINO splits its
  prompt into phrases on the period, so an unterminated prompt is one token short of
  the phrase the model expects and scores lower for no visible reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from editgpt_models.compositing import RGB
from editgpt_models.config import load_thresholds
from editgpt_models.registry import asset_path, model_path

DETECTOR_KEY = "grounding-dino"
INPUT_SIDE = 800

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True, slots=True)
class Detection:
    """One candidate region for a phrase, in fractions of the image."""

    box: tuple[float, float, float, float]
    """(x0, y0, x1, y1), each in [0, 1]. Fractions, not pixels, so a detection survives
    the resizing every stage of this pipeline does."""

    score: float
    """Max token-level probability over the phrase. Well separated in practice: a
    confident match scores ~0.95 where the runner-up scores ~0.07."""

    @property
    def area(self) -> float:
        return max(self.box[2] - self.box[0], 0.0) * max(self.box[3] - self.box[1], 0.0)


@dataclass(frozen=True, slots=True)
class Detector:
    """A loaded detector: the ONNX graph plus the tokenizer its text input needs."""

    session: Any
    tokenizer: Any


def load_detector(threads: int = 4) -> Detector:
    """Load the graph and its tokenizer, fetching both on first use."""
    from tokenizers import Tokenizer

    from editgpt_models.erase import make_session

    return Detector(
        session=make_session(model_path(DETECTOR_KEY), threads=threads),
        tokenizer=Tokenizer.from_file(str(asset_path(DETECTOR_KEY, "tokenizer.json"))),
    )


def normalise_phrase(phrase: str) -> str:
    """Lowercase, trimmed, and period-terminated — the form the model was trained on."""
    text = phrase.strip().lower().rstrip(".").strip()
    return f"{text}." if text else ""


def _preprocess(rgb: RGB) -> tuple[np.ndarray, np.ndarray]:
    """Squash the image to the declared 800x800 input and normalise it.

    The aspect ratio is deliberately not preserved — see the module docstring. Because
    nothing is padded, `pixel_mask` is all ones and predicted boxes are already
    fractions of the original image.
    """
    resized = cv2.resize(rgb, (INPUT_SIDE, INPUT_SIDE), interpolation=cv2.INTER_LINEAR)
    tensor = ((resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(
        2, 0, 1
    )
    return tensor[None, ...], np.ones((1, INPUT_SIDE, INPUT_SIDE), dtype=np.int64)


def detect(
    detector: Detector,
    rgb: RGB,
    phrase: str,
    *,
    min_score: float | None = None,
    top_k: int = 5,
) -> list[Detection]:
    """Candidate regions for `phrase`, best first.

    An empty list means the phrase matched nothing above the gate. That is a real
    outcome — the user asked for something that is not in the picture — so the caller
    decides whether to ask them, not this function.
    """
    text = normalise_phrase(phrase)
    if not text:
        return []

    gate = load_thresholds().min_box_score if min_score is None else min_score
    ids = np.array([detector.tokenizer.encode(text).ids], dtype=np.int64)
    if ids.shape[1] <= 2:  # [CLS] and [SEP] only; nothing was tokenised
        return []

    pixel_values, pixel_mask = _preprocess(rgb)
    logits, boxes = detector.session.run(
        None,
        {
            "pixel_values": pixel_values,
            "input_ids": ids,
            "token_type_ids": np.zeros_like(ids),
            "attention_mask": np.ones_like(ids),
            "pixel_mask": pixel_mask,
        },
    )

    # (queries, text_positions) -> one score per query. The special tokens are dropped:
    # [CLS] and [SEP] carry sentence-level activation that is high for every query and
    # would flatten the ranking the caller depends on.
    probabilities = 1.0 / (1.0 + np.exp(-logits[0]))
    scores = probabilities[:, 1 : ids.shape[1] - 1].max(axis=1)

    found: list[Detection] = []
    for index in np.argsort(-scores)[: max(top_k, 1)]:
        score = float(scores[index])
        if score < gate:
            break  # sorted, so everything after this is worse too
        cx, cy, w, h = (float(v) for v in boxes[0, index])
        # Already fractions of the image, since nothing was padded. Clipped because the
        # model happily predicts a box a percent or two outside the frame.
        box = (
            float(np.clip(cx - w / 2, 0.0, 1.0)),
            float(np.clip(cy - h / 2, 0.0, 1.0)),
            float(np.clip(cx + w / 2, 0.0, 1.0)),
            float(np.clip(cy + h / 2, 0.0, 1.0)),
        )
        if box[2] > box[0] and box[3] > box[1]:
            found.append(Detection(box=box, score=round(score, 4)))
    return found
