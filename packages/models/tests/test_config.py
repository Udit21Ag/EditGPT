"""Thresholds are data, not literals — so loading them must be strict where it matters."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from editgpt_models.config import Thresholds, load_thresholds


def test_defaults_are_self_describing() -> None:
    """A reader must be able to tell an inherited value from a measured one."""
    assert "not fitted" in Thresholds().provenance


def test_a_missing_file_falls_back_to_defaults(tmp_path: Path) -> None:
    """Running without fitted values is normal and must not be an error."""
    assert load_thresholds(tmp_path / "absent.json") == Thresholds()


def test_a_fitted_file_is_used(tmp_path: Path) -> None:
    path = tmp_path / "fitted.json"
    path.write_text(json.dumps({"min_sam_iou": 0.42, "provenance": "fitted on X"}))
    loaded = load_thresholds(path)
    assert loaded.min_sam_iou == pytest.approx(0.42)
    assert loaded.provenance == "fitted on X"
    assert loaded.escalate_cost == Thresholds().escalate_cost, "unset values keep defaults"


def test_an_unknown_threshold_is_rejected_loudly(tmp_path: Path) -> None:
    """A typo in a fitted file must fail, not be silently ignored — the whole point is
    that these values are auditable."""
    path = tmp_path / "typo.json"
    path.write_text(json.dumps({"min_sam_ioU": 0.42}))
    with pytest.raises(ValueError, match="unknown threshold"):
        load_thresholds(path)


def test_thresholds_are_immutable() -> None:
    with pytest.raises(AttributeError):
        Thresholds().min_sam_iou = 0.1  # type: ignore[misc]


def test_round_trips_through_json() -> None:
    original = Thresholds(min_sam_iou=0.7, provenance="test")
    assert Thresholds.from_mapping(json.loads(original.to_json())) == original
