"""Every tunable must actually come from `Thresholds`, not from a literal beside it.

This file exists because of a real defect: `config.Thresholds` documented itself as the
single home for these values and shipped a loader for a fitted file, while `pipeline.py`,
`compositing.py` and `metrics.py` each carried a constant of the same name that the
pipeline actually read. Editing the fitted file changed nothing.

A test that a value *is* 0.05 would not have caught that — both places said 0.05. So each
test below writes a **deliberately absurd** value and asserts the behaviour moves. If
someone reintroduces a shadowing literal, the behaviour stops moving and this fails.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from editgpt_models.compositing import DILATE_MAX, dilate_px
from editgpt_models.config import Thresholds, load_thresholds
from editgpt_models.pipeline import Erasers, erase


@pytest.fixture
def fitted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the loader at a file this test writes."""
    path = tmp_path / "fitted_thresholds.json"
    monkeypatch.setenv("EDITGPT_THRESHOLDS", str(path))
    return path


def write(path: Path, **overrides: object) -> Thresholds:
    values = {**json.loads(Thresholds().to_json()), **overrides}
    path.write_text(json.dumps(values))
    return Thresholds.from_mapping(values)


def square(size: int = 200, box: int = 100) -> np.ndarray:
    mask = np.zeros((size, size), np.uint8)
    mask[50 : 50 + box, 50 : 50 + box] = 255
    return mask


def test_the_loader_reads_the_file_the_environment_points_at(fitted: Path) -> None:
    write(fitted, min_sam_iou=0.11, provenance="written by a test")
    loaded = load_thresholds()
    assert loaded.min_sam_iou == pytest.approx(0.11)
    assert loaded.provenance == "written by a test"


def test_a_missing_file_falls_back_to_documented_defaults(fitted: Path) -> None:
    """Running without fitted values is normal; a typo in them is not — see below."""
    assert not fitted.exists()
    assert load_thresholds() == Thresholds()


def test_a_malformed_file_fails_loudly_rather_than_being_ignored(fitted: Path) -> None:
    fitted.write_text(json.dumps({"min_sam_ioU": 0.5}))
    with pytest.raises(ValueError, match="unknown threshold"):
        load_thresholds()


def test_dilation_follows_the_fitted_fraction(fitted: Path) -> None:
    """Object-relative dilation is what fixed the rectangular outline at 15.9 MP."""
    mask = square(size=400, box=200)
    write(fitted, dilate_frac=0.05)
    modest = dilate_px(mask)

    write(fitted, dilate_frac=0.30)
    generous = dilate_px(mask)

    assert generous > modest, "the fitted fraction must reach the pipeline"
    assert generous == min(round(0.30 * 199), DILATE_MAX)


class RecordingErasers(Erasers):
    """Erasers that record which ones ran, and return a flat grey fill."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        super().__init__(migan=self._migan, lama=self._lama)

    def _fill(self, name: str, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        self.calls.append(name)
        out = image.copy()
        out[mask > 0] = 128
        return out

    def _migan(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return self._fill("migan", image, mask)

    def _lama(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return self._fill("lama", image, mask)


def textured_image(size: int = 200) -> np.ndarray:
    generator = np.random.default_rng(seed=7)
    return generator.integers(0, 255, (size, size, 3), dtype=np.uint8)


def test_the_accept_cost_decides_whether_a_third_pass_runs(fitted: Path) -> None:
    """A ceiling of 0 can never be satisfied, so the loop must keep going; a huge one stops it."""
    image, mask = textured_image(), square()

    write(fitted, accept_cost=0.0)
    greedy = erase(RecordingErasers(), image, mask, min_passes=2, max_passes=3)

    write(fitted, accept_cost=10_000.0)
    contented = erase(RecordingErasers(), image, mask, min_passes=2, max_passes=3)

    # Counted in *passes*, not eraser calls: a third pass that finds no applicable
    # strategy still runs and is recorded, but never invokes a model.
    assert len(greedy.passes) == 3
    assert len(contented.passes) == 2, (
        "accept_cost must reach the pass loop; an equal count means a literal shadows it"
    )


def test_the_growth_penalty_reaches_the_keep_or_rollback_decision(fitted: Path) -> None:
    """The penalty is what stops "erase more of the scene" from scoring as an improvement."""
    image, mask = textured_image(), square()

    write(fitted, growth_penalty=0.0, residual_min_growth=0.0, residual_max_growth=10.0)
    permissive = erase(RecordingErasers(), image, mask, min_passes=2, max_passes=3)

    write(fitted, growth_penalty=1e6, residual_min_growth=0.0, residual_max_growth=10.0)
    strict = erase(RecordingErasers(), image, mask, min_passes=2, max_passes=3)

    assert strict.kept_passes <= permissive.kept_passes
    assert int((strict.mask > 0).sum()) <= int((permissive.mask > 0).sum())


def test_an_explicit_thresholds_argument_beats_the_file(fitted: Path) -> None:
    """A benchmark sweeping values must not have to write a file per candidate."""
    write(fitted, accept_cost=10_000.0)
    image, mask = textured_image(), square()

    from_file = erase(RecordingErasers(), image, mask, min_passes=2, max_passes=3)
    explicit = erase(
        RecordingErasers(),
        image,
        mask,
        min_passes=2,
        max_passes=3,
        thresholds=Thresholds(accept_cost=0.0),
    )
    assert len(from_file.passes) == 2
    assert len(explicit.passes) == 3


def test_no_module_reintroduces_a_shadowing_literal() -> None:
    """The specific regression: a constant named like a `Thresholds` field, beside it.

    Grep-based rather than behavioural on purpose. The behavioural tests above catch a
    shadow that changes an outcome; this catches one added but not yet wired, which is how
    the original defect started.
    """
    root = Path(__file__).resolve().parents[3]
    watched = {
        root / "packages/models/src/editgpt_models/pipeline.py",
        root / "packages/models/src/editgpt_models/compositing.py",
        root / "packages/models/src/editgpt_models/segment.py",
    }
    prefixes = tuple(
        f"{name.upper()} = " for name in Thresholds.__dataclass_fields__ if name != "provenance"
    )
    offenders = [
        f"{path.name}: {line.strip()}"
        for path in watched
        for line in path.read_text().splitlines()
        if line.startswith(prefixes)
    ]
    assert not offenders, f"these shadow a Thresholds field: {offenders}"
