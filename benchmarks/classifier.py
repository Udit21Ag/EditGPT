"""Learn which eraser to use, instead of guessing with a threshold.

The shipped router picks by photometric cost after running the fast eraser. On held-out
paired data that choice is often wrong, and the gap to a perfect chooser is large. This
fits a small model on the *outcome* of both erasers — every benchmark run produces
labelled examples for free, because it scores both.

Deliberately numpy-only: nine features and a binary label do not justify a new
dependency, and an explicit implementation makes the leakage protections visible rather
than assumed.

Honest baselines are the point. A classifier must beat:
  * **majority** — always pick whichever eraser wins more often overall;
  * **the current router** — what ships today.
Beating neither means the features carry no signal, which is a real result.

    uv run python -m benchmarks.classifier --folds 5
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from editgpt_models.compositing import grow
from editgpt_models.features import TaskFeatures, extract

from benchmarks.datasets import load_removal
from benchmarks.removal import WORKING_SIDE, _fit, _fit_mask

OUT_DIR = Path(__file__).resolve().parent / "out"
MODEL_PATH = Path(__file__).resolve().parent / "eraser_choice.json"


@dataclass
class LogisticModel:
    """L2-regularised logistic regression with standardised inputs.

    The scaler's mean and scale are fitted on the training fold only and carried with the
    weights, so scoring a held-out sample cannot see statistics from its own fold.
    """

    weights: np.ndarray
    bias: float
    mean: np.ndarray
    scale: np.ndarray
    feature_names: list[str]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        z = ((x - self.mean) / self.scale) @ self.weights + self.bias
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x) >= 0.5).astype(int)

    def to_dict(self) -> dict[str, object]:
        return {
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "feature_names": self.feature_names,
        }


def fit_logistic(
    x: np.ndarray, y: np.ndarray, names: list[str], *, l2: float = 1.0, steps: int = 4000
) -> LogisticModel:
    mean = x.mean(axis=0)
    scale = np.where(x.std(axis=0) < 1e-9, 1.0, x.std(axis=0))
    z = (x - mean) / scale

    weights = np.zeros(z.shape[1])
    bias = 0.0
    n = len(y)
    for step in range(steps):
        p = 1.0 / (1.0 + np.exp(-np.clip(z @ weights + bias, -30, 30)))
        error = p - y
        grad_w = z.T @ error / n + l2 * weights / n
        grad_b = float(error.mean())
        # Decaying step size: large enough to move early, small enough to settle.
        lr = 0.5 / (1.0 + step / 500)
        weights -= lr * grad_w
        bias -= lr * grad_b
    return LogisticModel(weights, bias, mean, scale, names)


def stratified_folds(y: np.ndarray, folds: int, seed: int = 0) -> list[np.ndarray]:
    """Indices per fold, keeping the class balance of the whole set in each."""
    rng = np.random.default_rng(seed)
    buckets: list[list[int]] = [[] for _ in range(folds)]
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        for position, sample in enumerate(idx):
            buckets[position % folds].append(int(sample))
    return [np.array(sorted(b)) for b in buckets]


def build_dataset(limit: int) -> tuple[np.ndarray, np.ndarray, list[str], list[dict[str, object]]]:
    """Features from the images, labels from a previous `benchmarks.removal` run."""
    scored = json.loads((OUT_DIR / "removal.json").read_text())["results"]
    by_id = {r["id"]: r for r in scored}

    rows, labels, meta = [], [], []
    for sample in load_removal(limit=limit):
        record = by_id.get(sample.id)
        if record is None:
            continue
        image = _fit(sample.image, WORKING_SIDE)
        mask = grow(_fit_mask(sample.mask, WORKING_SIDE))
        if int(mask.sum()) == 0:
            continue
        rows.append(extract(image, mask).as_vector())
        labels.append(1 if record["winner"] == "lama" else 0)
        meta.append(record)
    return np.array(rows), np.array(labels), TaskFeatures.names(), meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--limit", type=int, default=69)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    if not (OUT_DIR / "removal.json").exists():
        print("run `python -m benchmarks.removal` first — it produces the labels", file=sys.stderr)
        return 2

    x, y, names, meta = build_dataset(args.limit)
    if len(y) < 20:
        print(f"only {len(y)} labelled samples; too few to fit", file=sys.stderr)
        return 2

    majority = int(np.bincount(y).argmax())
    majority_acc = float((y == majority).mean())
    router_acc = float(np.mean([m["routed_matched_winner"] for m in meta]))

    folds = stratified_folds(y, args.folds)
    correct, predictions = 0, np.zeros(len(y), dtype=int)
    for fold in folds:
        train = np.setdiff1d(np.arange(len(y)), fold)
        if len(np.unique(y[train])) < 2:
            predictions[fold] = majority  # a degenerate fold cannot teach anything
            continue
        model = fit_logistic(x[train], y[train], names)
        predictions[fold] = model.predict(x[fold])
    correct = int((predictions == y).sum())
    cv_acc = correct / len(y)

    # What the choice is worth, in the metric that matters.
    ssim_if = lambda pick: float(  # noqa: E731
        np.mean([m["ssim_lama"] if p else m["ssim_migan"] for p, m in zip(pick, meta, strict=True)])
    )
    print(f"\nEraser choice, {len(y)} labelled samples, {args.folds}-fold cross-validation")
    print(f"  class balance            lama {int(y.sum())} / migan {int((1 - y).sum())}")
    print(f"\n  {'chooser':26} {'accuracy':>9} {'mean SSIM':>10}")
    print(
        f"  {'majority (always ' + ('lama' if majority else 'migan') + ')':26} "
        f"{majority_acc:9.3f} {ssim_if(np.full(len(y), majority)):10.4f}"
    )
    print(
        f"  {'current router':26} {router_acc:9.3f} "
        f"{float(np.mean([m['ssim_routed'] for m in meta])):10.4f}"
    )
    print(f"  {'learned (cross-validated)':26} {cv_acc:9.3f} {ssim_if(predictions):10.4f}")
    print(f"  {'oracle':26} {1.0:9.3f} {ssim_if(y):10.4f}")

    full = fit_logistic(x, y, names)
    order = np.argsort(-np.abs(full.weights))
    print("\n  strongest features (standardised weight, + favours LaMa):")
    for i in order[:5]:
        print(f"    {names[i]:22} {full.weights[i]:+.3f}")

    # Accuracy is a proxy; SSIM is the outcome. A chooser can win on accuracy by getting
    # right the cases where the two erasers barely differ, and be worth nothing. Adoption
    # requires the outcome to move, not the proxy.
    ssim_majority = ssim_if(np.full(len(y), majority))
    ssim_learned = ssim_if(predictions)
    ssim_gain = ssim_learned - ssim_majority
    accuracy_gain = cv_acc - majority_acc
    ceiling = ssim_if(y) - ssim_majority

    if ssim_gain > 0.005:
        verdict = f"adopt — SSIM {ssim_gain:+.4f} over the majority baseline"
    elif accuracy_gain > 0.05:
        verdict = (
            f"reject — accuracy {accuracy_gain:+.3f} but SSIM only {ssim_gain:+.4f}; "
            "it wins the cases where the choice barely matters"
        )
    else:
        verdict = "reject — no better than the majority baseline"

    print(f"\n  verdict: {verdict}")
    print(
        f"  the entire choice is worth at most {ceiling:+.4f} SSIM (oracle over majority), "
        "which bounds what any router can achieve here."
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "classifier.json").write_text(
        json.dumps(
            {
                "n": len(y),
                "cv_accuracy": round(cv_acc, 4),
                "majority_accuracy": round(majority_acc, 4),
                "router_accuracy": round(router_acc, 4),
                "ssim_majority": round(ssim_if(np.full(len(y), majority)), 4),
                "ssim_learned": round(ssim_if(predictions), 4),
                "ssim_oracle": round(ssim_if(y), 4),
                "weights": dict(zip(names, full.weights.round(4).tolist(), strict=True)),
                "ssim_gain_over_majority": round(ssim_gain, 4),
                "accuracy_gain_over_majority": round(accuracy_gain, 4),
                "oracle_ceiling_over_majority": round(ceiling, 4),
                "verdict": verdict,
            },
            indent=2,
        )
    )
    if args.write and verdict.startswith("adopt"):
        MODEL_PATH.write_text(json.dumps(full.to_dict(), indent=2))
        print(f"  wrote {MODEL_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
