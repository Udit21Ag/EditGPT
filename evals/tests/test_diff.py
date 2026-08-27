"""Comparing a golden-set report against its baseline.

The judgement encoded here is which differences stop a merge and which only ask for a
human. It follows this project's own measurement of `cost`: TD-017 has two paired
benchmarks disagreeing on the *sign* of its correlation with visual quality, so a cost
move is a prompt to look at the picture, never a verdict on it. Grounding IoU and a case
that stopped working are different — those have right answers, and they block.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evals.diff import REMOTE_OPS, compare, load, main, render, signature


def row(case: str, **over: Any) -> dict[str, Any]:
    base = {
        "id": case,
        "prompt": f"remove the {case}",
        "op": "remove",
        "status": "ok",
        "seconds": 10.0,
        "passes": "migan -> residual",
        "kept_passes": 2,
        "cost": 20.0,
        "bbox_iou": 0.800,
        "mask_source": "sam-box (0.98)",
        "detail": "",
    }
    return base | over


def index(*rows: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {r["id"]: r for r in rows}


def kinds(findings: list[Any]) -> set[str]:
    return {f.kind for f in findings}


def test_an_identical_report_produces_nothing() -> None:
    same = index(row("i1"), row("i2"))
    assert compare(same, same) == []


def test_a_case_that_stopped_working_blocks() -> None:
    was, now = index(row("i1")), index(row("i1", status="failed"))
    (finding,) = compare(was, now)
    assert finding.blocking
    assert "ok -> failed" in finding.detail


def test_a_case_that_started_working_does_not_block() -> None:
    """A fix is welcome news, not a merge gate."""
    was, now = index(row("i1", status="failed")), index(row("i1"))
    (finding,) = compare(was, now)
    assert not finding.blocking


def test_a_dropped_case_blocks() -> None:
    """A run that silently covers less than the baseline is the failure mode a green
    check would otherwise hide."""
    was, now = index(row("i1"), row("i2")), index(row("i1"))
    (finding,) = compare(was, now)
    assert finding.kind == "missing"
    assert finding.blocking


def test_grounding_that_stopped_finding_the_object_blocks() -> None:
    """Unlike cost, IoU measures something with a right answer."""
    was, now = index(row("i1")), index(row("i1", bbox_iou=0.500))
    findings = compare(was, now)
    grounding = [f for f in findings if f.kind == "grounding"]
    assert grounding, "a 0.30 IoU drop went unreported"
    assert grounding[0].blocking


def test_a_cost_rise_is_reported_but_does_not_block() -> None:
    """The measured position on `cost`: it moves for reasons that are not quality.

    The occluder shield moved `i8` by +1.8 while visibly ruining the image, and moved
    `i6` by -13.5 while improving it. A gate on this number would be a coin toss.
    """
    was, now = index(row("i1")), index(row("i1", cost=40.0))
    findings = compare(was, now)
    cost = [f for f in findings if f.kind == "cost"]
    assert cost, "a doubling of cost should still be reported"
    assert not cost[0].blocking


def test_run_to_run_noise_is_not_reported() -> None:
    """Identical code differs a few percent because the multi-pass loop takes a different
    route near `accept_cost`. A check that fires on that gets ignored."""
    was, now = index(row("i1")), index(row("i1", cost=20.4))
    assert not [f for f in compare(was, now) if f.kind == "cost"]


def test_a_changed_route_is_reported() -> None:
    was, now = index(row("i1")), index(row("i1", passes="migan -> escalate"))
    assert "route" in kinds(compare(was, now))


def test_blocking_findings_are_listed_first() -> None:
    was = index(row("i1"), row("i2"))
    now = index(row("i1", cost=40.0), row("i2", status="failed"))
    findings = compare(was, now)
    assert findings[0].blocking
    assert not findings[-1].blocking


def test_the_rendering_is_a_markdown_table_for_a_pull_request_comment() -> None:
    was, now = index(row("i1")), index(row("i1", status="failed"))
    out = render(compare(was, now))
    assert out.startswith("| case |")
    assert "`i1`" in out


def test_no_change_renders_as_a_sentence_not_an_empty_table() -> None:
    assert render([]) == "No change against the baseline."


# ------------------------------------------------------------------ the command


def write(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text(json.dumps(rows))
    return path


def test_the_command_fails_on_a_blocking_change(tmp_path: Path, monkeypatch: Any) -> None:
    baseline = write(tmp_path / "baseline.json", [row("i1")])
    report = write(tmp_path / "report.json", [row("i1", status="failed")])
    monkeypatch.setattr("sys.argv", ["diff", "--report", str(report), "--baseline", str(baseline)])
    assert main() == 1


def test_the_command_passes_on_a_cost_move(tmp_path: Path, monkeypatch: Any) -> None:
    baseline = write(tmp_path / "baseline.json", [row("i1")])
    report = write(tmp_path / "report.json", [row("i1", cost=40.0)])
    monkeypatch.setattr("sys.argv", ["diff", "--report", str(report), "--baseline", str(baseline)])
    assert main() == 0


def test_a_missing_baseline_is_explained_rather_than_crashing(
    tmp_path: Path, monkeypatch: Any
) -> None:
    report = write(tmp_path / "report.json", [row("i1")])
    monkeypatch.setattr(
        "sys.argv",
        ["diff", "--report", str(report), "--baseline", str(tmp_path / "nothing.json")],
    )
    assert main() == 2


def test_update_records_the_current_report(tmp_path: Path, monkeypatch: Any) -> None:
    baseline = tmp_path / "baseline.json"
    report = write(tmp_path / "report.json", [row("i1", cost=99.0)])
    monkeypatch.setattr(
        "sys.argv",
        ["diff", "--report", str(report), "--baseline", str(baseline), "--update"],
    )
    assert main() == 0
    assert load(baseline)["i1"]["cost"] == 99.0


# ------------------------------------------------------------------ the baseline


def test_a_remote_case_in_the_report_is_not_reported_as_new(tmp_path: Path) -> None:
    """They are deliberately absent from the baseline, so flagging them would fire on
    every run — and a report that always says something says nothing."""
    baseline = write(tmp_path / "baseline.json", [row("i1")])
    report = write(tmp_path / "report.json", [row("i1"), row("i3", op="add", cost=196.1)])
    assert compare(load(baseline), load(report)) == []


def test_the_baseline_never_records_a_remote_case(tmp_path: Path, monkeypatch: Any) -> None:
    """Stable Diffusion returns a different picture every call: the same `i3` scored
    196.1, 83.9 and 87.4 across three runs of unchanged code. Baselining that records
    noise and then reports it as change forever."""
    baseline = tmp_path / "baseline.json"
    report = write(
        tmp_path / "report.json",
        [row("i1"), row("i3", op="add", cost=196.1), row("i4r", op="replace")],
    )
    monkeypatch.setattr(
        "sys.argv",
        ["diff", "--report", str(report), "--baseline", str(baseline), "--update"],
    )
    assert main() == 0
    assert set(load(baseline)) == {"i1"}


def test_the_two_lists_of_remote_operations_agree() -> None:
    """`evals.run` skips them to protect the free-tier allowance and this skips them
    because they are not reproducible. Different reasons, one list — and drift would be
    silent, showing up as a permanently 'changed' case."""
    from evals.run import REMOTE_OPS as RUNNER_OPS

    assert {op.value for op in RUNNER_OPS} == set(REMOTE_OPS)


# ------------------------------------------------------------------ the image signature


def strip(
    tmp_path: Path, name: str, result_shade: int, mark: tuple[int, int] | None = None
) -> Path:
    """An `original | mask | result` strip like the one `evals.run` writes."""
    from PIL import Image

    pane = 60
    sheet = Image.new("L", (pane * 3, pane), 128)
    sheet.paste(Image.new("L", (pane, pane), result_shade), (pane * 2, 0))
    if mark is not None:
        sheet.paste(Image.new("L", (12, 12), 255), (pane * 2 + mark[0], mark[1]))
    path = tmp_path / name
    sheet.convert("RGB").save(path)
    return path


def test_an_unchanged_result_has_a_zero_distance(tmp_path: Path) -> None:
    """Measured, and the reason this check is worth having: the local half of the golden
    set is bit-exact, so any movement at all is a real change rather than noise."""
    from evals.diff import distance, signature

    a = signature(strip(tmp_path, "a.png", 90))
    b = signature(strip(tmp_path, "b.png", 90))
    assert distance(a, b) == 0.0


def test_a_localised_artefact_moves_the_signature(tmp_path: Path) -> None:
    """The `i8` case: a pale ghost of the tower left standing moved cost by 3.2% — under
    any usable tolerance — and moved this by 0.400."""
    from evals.diff import SIGNATURE_TOLERANCE, distance, signature

    clean = signature(strip(tmp_path, "clean.png", 90))
    ghosted = signature(strip(tmp_path, "ghost.png", 90, mark=(20, 20)))
    assert distance(clean, ghosted) > SIGNATURE_TOLERANCE


def test_the_signature_ignores_the_input_panes(tmp_path: Path) -> None:
    """Only the third pane is the answer; the first two are inputs and would dilute a
    localised change."""
    from PIL import Image

    from evals.diff import distance, signature

    pane = 60
    a = Image.new("L", (pane * 3, pane), 128)
    a.paste(Image.new("L", (pane, pane), 90), (pane * 2, 0))
    b = a.copy()
    b.paste(Image.new("L", (pane, pane), 10), (0, 0))  # a different *original* pane

    pa, pb = tmp_path / "a.png", tmp_path / "b.png"
    a.convert("RGB").save(pa)
    b.convert("RGB").save(pb)
    assert distance(signature(pa), signature(pb)) == 0.0


def test_a_changed_image_is_reported_but_does_not_block(tmp_path: Path) -> None:
    """The same-machine floor is zero, but the CI floor — a different CPU and ONNX build —
    is not yet measured. A tolerance that turns out too tight should cost a line in a
    comment, not a red build."""
    was = index(row("i1", signature=signature(strip(tmp_path, "x.png", 90))))
    now = index(row("i1", signature=signature(strip(tmp_path, "y.png", 90, mark=(20, 20)))))
    findings = compare(was, now)
    image = [f for f in findings if f.kind == "image"]
    assert image, "a visibly changed result went unreported"
    assert not image[0].blocking


def test_a_report_without_pictures_still_diffs(tmp_path: Path) -> None:
    """`--images` pointing nowhere is a degraded check, not a crash."""
    baseline = write(tmp_path / "baseline.json", [row("i1")])
    assert compare(load(baseline), load(baseline, tmp_path / "absent")) == []
