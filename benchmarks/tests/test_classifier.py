"""The eraser-choice model: it must learn, and it must not cheat."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.classifier import fit_logistic, stratified_folds

NAMES = ["a", "b"]


def separable(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two clearly separated clouds — if the fitter cannot learn this, it is broken."""
    rng = np.random.default_rng(seed)
    x = np.vstack([rng.normal(-2, 0.5, (n // 2, 2)), rng.normal(2, 0.5, (n // 2, 2))])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    return x, y


def test_it_learns_a_separable_problem() -> None:
    x, y = separable()
    model = fit_logistic(x, y, NAMES)
    assert float((model.predict(x) == y).mean()) > 0.95


def test_it_does_not_learn_pure_noise() -> None:
    """Guards against a fitter that memorises: with random labels, training accuracy
    should stay near chance rather than approaching 1.0."""
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, (200, 2))
    y = rng.integers(0, 2, 200)
    accuracy = float((fit_logistic(x, y, NAMES).predict(x) == y).mean())
    assert accuracy < 0.75


def test_probabilities_stay_in_range() -> None:
    x, y = separable()
    probabilities = fit_logistic(x, y, NAMES).predict_proba(x)
    assert probabilities.min() >= 0.0
    assert probabilities.max() <= 1.0


def test_it_survives_a_constant_feature() -> None:
    """A zero-variance column would divide by zero in the scaler."""
    x, y = separable()
    x = np.hstack([x, np.ones((len(x), 1))])
    model = fit_logistic(x, y, [*NAMES, "constant"])
    assert np.all(np.isfinite(model.weights))
    assert np.all(np.isfinite(model.predict_proba(x)))


def test_the_scaler_is_carried_with_the_weights() -> None:
    """Scoring must not recompute statistics from the data being scored — that is how
    a held-out sample leaks information about itself."""
    x, y = separable()
    model = fit_logistic(x, y, NAMES)
    single = model.predict_proba(x[:1])
    batch = model.predict_proba(x)[:1]
    assert single == pytest.approx(batch)


def test_folds_partition_the_data_exactly_once() -> None:
    y = np.array([0] * 30 + [1] * 20)
    folds = stratified_folds(y, 5)
    combined = np.concatenate(folds)
    assert len(combined) == len(y)
    assert len(np.unique(combined)) == len(y)


def test_folds_preserve_class_balance() -> None:
    y = np.array([0] * 40 + [1] * 20)
    for fold in stratified_folds(y, 4):
        share = float(y[fold].mean())
        assert share == pytest.approx(1 / 3, abs=0.2)


def test_folds_are_deterministic_for_a_seed() -> None:
    y = np.array([0] * 30 + [1] * 30)
    first = [f.tolist() for f in stratified_folds(y, 5, seed=7)]
    second = [f.tolist() for f in stratified_folds(y, 5, seed=7)]
    assert first == second


def test_cross_validated_accuracy_is_lower_than_training_accuracy_on_noise() -> None:
    """The point of cross-validation: on noise, held-out accuracy must not look good."""
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, (120, 4))
    y = rng.integers(0, 2, 120)

    predictions = np.zeros(len(y), dtype=int)
    for fold in stratified_folds(y, 4):
        train = np.setdiff1d(np.arange(len(y)), fold)
        predictions[fold] = fit_logistic(x[train], y[train], ["a", "b", "c", "d"]).predict(x[fold])
    assert float((predictions == y).mean()) < 0.7
