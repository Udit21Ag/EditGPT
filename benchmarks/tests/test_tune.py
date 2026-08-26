"""Threshold fitting: the split must be honest and the sweep must be arithmetic."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from benchmarks.tune import Observation, score, sweep


def obs(ident: str, confidence: float, refined: float, seed: float) -> Observation:
    return Observation(id=ident, words=5, confidence=confidence, iou_refined=refined, iou_seed=seed)


def test_the_split_is_deterministic_across_runs() -> None:
    """A split that reshuffles would let a value be fitted and reported on the same data."""
    first = [obs(str(i), 0.5, 0.5, 0.5).split for i in range(200)]
    second = [obs(str(i), 0.9, 0.1, 0.2).split for i in range(200)]
    assert first == second, "the split must depend on the id alone"


def test_the_split_depends_on_the_id_not_the_order() -> None:
    a = obs("sample-a", 0.5, 0.5, 0.5).split
    assert obs("sample-a", 0.1, 0.9, 0.2).split == a


def test_both_splits_are_populated() -> None:
    splits = [obs(str(i), 0.5, 0.5, 0.5).split for i in range(400)]
    assert 100 < splits.count("fit") < 300
    assert 100 < splits.count("holdout") < 300


def test_a_gate_of_zero_always_takes_the_refinement() -> None:
    data = [obs("a", 0.1, 1.0, 0.0), obs("b", 0.9, 1.0, 0.0)]
    assert score(data, 0.0) == pytest.approx(1.0)


def test_a_gate_above_one_never_takes_the_refinement() -> None:
    data = [obs("a", 0.1, 1.0, 0.0), obs("b", 0.9, 1.0, 0.0)]
    assert score(data, 1.01) == pytest.approx(0.0)


def test_the_gate_selects_per_sample_by_confidence() -> None:
    data = [obs("a", 0.4, 0.0, 1.0), obs("b", 0.8, 1.0, 0.0)]
    # At 0.6, "a" falls back to its seed (1.0) and "b" keeps its refinement (1.0).
    assert score(data, 0.6) == pytest.approx(1.0)


def test_an_empty_set_scores_zero_rather_than_raising() -> None:
    assert score([], 0.5) == 0.0


def test_the_sweep_covers_every_candidate() -> None:
    data = [obs(str(i), i / 10, 1.0, 0.0) for i in range(10)]
    candidates = np.round(np.arange(0.0, 1.01, 0.1), 2)
    curve = sweep(data, candidates)
    assert len(curve) == len(candidates)
    assert curve[0][1] >= curve[-1][1], "raising the gate can only reject refinements here"


def test_the_sweep_is_monotonic_when_refinement_always_wins() -> None:
    """A sanity check on the sweep itself: if refinement is always better, a higher gate
    can never score higher."""
    data = [obs(str(i), i / 20, 1.0, 0.0) for i in range(20)]
    scores = [s for _, s in sweep(data, np.round(np.arange(0, 1.01, 0.05), 2))]
    assert all(a >= b - 1e-9 for a, b in pairwise(scores))
