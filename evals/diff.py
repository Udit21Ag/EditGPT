"""Compare a golden-set report against a recorded baseline, and fail on a regression.

TD-007. `make eval` produced a table nobody diffed, so quality changes were invisible
until somebody looked at the pictures — which is how a black safety-filter frame, a
swallowed shoe and a smeared hand all reached a human rather than a check.

**What this can and cannot see.** `cost` is a photometric proxy: how far the filled
region sits from the real pixels ringing it. It is not visual quality, and this project
has the measurement to prove it — the two paired benchmarks in TD-017 disagree on the
sign of its correlation. So the thresholds here are set to catch *movement*, not to
adjudicate taste. A flagged case means "a human should look at this image", not "this is
worse", and a clean run does not mean the pictures are fine. The occluder shield in
TD-004 moved `i8` by only +1.8 cost while visibly ruining it.

What it catches reliably is the class of change nobody argues about: a case that stopped
working, a case that started failing, grounding that stopped finding the object, and a
run that quietly dropped a case.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE = REPO_ROOT / "evals" / "baseline.json"

REMOTE_OPS = frozenset({"add", "replace"})
"""Operations served by a remote provider, which are never baselined.

Not a CI-cost decision like the one in `evals.run` — a correctness one. Stable Diffusion
returns a different picture every call: the same `i3` scored 196.1, 83.9 and 87.4 across
three runs of identical code. A baseline entry for that records noise and then reports it
as change forever. Spelled as strings rather than imported from `evals.run` so this stays
a lightweight tool; `tests/test_diff.py` asserts the two cannot drift apart.
"""

COST_TOLERANCE = 0.05
"""Relative cost movement worth reporting.

Measured, not guessed: the local half of the golden set is **bit-exact** run to run. Two
runs of identical code produced the same cost to the decimal for all eleven removals and
pixel-identical images, so there is no reproducibility noise to leave headroom for. The
tolerance exists for the machine this runs on rather than the run — CPU and ONNX Runtime
build differences between a laptop and a CI worker move floating point slightly, and that
floor is not yet measured."""

SIGNATURE_SIZE = 32
"""Side of the thumbnail used to notice that a result *changed at all*.

This exists because `cost` did not do the job. The occluder shield in TD-004 left a pale
ghost of the Eiffel Tower standing in `i8` — obvious at a glance — and moved the cost by
3.2%, under any tolerance wide enough to be usable. The image noticed what the metric
could not: max cell distance 0.078 against 0.000 for two runs of unchanged code."""

SIGNATURE_TOLERANCE = 0.02
"""Max per-cell distance, on 0..1, that counts as the picture having changed.

Roughly a quarter of the `i8` regression above and well clear of the zero measured
between identical runs. Deliberately **not** blocking: the same-machine floor is zero but
the CI floor is unmeasured, so a threshold that turns out to be too tight costs a line in
a comment rather than a red build."""

IOU_TOLERANCE = 0.05
"""Absolute drop in grounding IoU worth attention. Unlike cost, this one measures a
thing with a right answer — whether the phrase found the object."""


@dataclass(frozen=True, slots=True)
class Finding:
    case: str
    kind: str
    detail: str
    blocking: bool


def signature(path: Path) -> str:
    """A thumbnail of the *result* pane, as hex, small enough to live in the baseline.

    The eval writes `original | mask | result`; only the last pane is the answer. The
    other two are inputs and would dilute exactly the localised change this is for.
    """
    from PIL import Image

    with Image.open(path) as image:
        width, height = image.size
        pane = image.crop((2 * width // 3, 0, width, height)).convert("L")
        # BOX averages the pixels it merges, so a small bright artefact survives the
        # downsample instead of being stepped over the way nearest-neighbour would.
        small = pane.resize((SIGNATURE_SIZE, SIGNATURE_SIZE), Image.Resampling.BOX)
    return bytes(small.tobytes()).hex()


def distance(left: str, right: str) -> float:
    """Largest per-cell difference between two signatures, on 0..1."""
    a, b = bytes.fromhex(left), bytes.fromhex(right)
    if len(a) != len(b):
        return 1.0
    return max(abs(x - y) for x, y in zip(a, b, strict=True)) / 255.0


def load(path: Path, images: Path | None = None) -> dict[str, dict[str, Any]]:
    """Cases from a report or baseline, with the remote ones dropped.

    Dropped on *both* sides, not just when recording. A full `make eval` includes the
    generative cases; without this they read as four new cases on every single run, and a
    report that always says something is the same as one that says nothing.

    `images` attaches a signature to each row from the pictures the run wrote. A baseline
    carries its own, recorded when it was taken; a fresh report has none until asked.
    """
    rows = [r for r in json.loads(path.read_text()) if r.get("op") not in REMOTE_OPS]
    if images is not None:
        for row in rows:
            picture = images / f"{row['id']}_{row['op']}.png"
            if picture.exists():
                row["signature"] = signature(picture)
    return {row["id"]: row for row in rows}


def compare(
    baseline: dict[str, dict[str, Any]], current: dict[str, dict[str, Any]]
) -> list[Finding]:
    """Every difference worth reporting, most serious first."""
    findings: list[Finding] = []

    findings.extend(
        Finding(case, "missing", "in the baseline but not in this run", True)
        for case in sorted(set(baseline) - set(current))
    )
    findings.extend(
        Finding(case, "new", "not in the baseline; record it if intended", False)
        for case in sorted(set(current) - set(baseline))
    )

    for case in sorted(set(baseline) & set(current)):
        was, now = baseline[case], current[case]

        if was["status"] != now["status"]:
            # Only a move *away* from a working case blocks; a fix is welcome news.
            findings.append(
                Finding(
                    case,
                    "status",
                    f"{was['status']} -> {now['status']}",
                    was["status"] == "ok",
                )
            )

        old_cost, new_cost = was.get("cost"), now.get("cost")
        if old_cost and new_cost and old_cost > 0:
            change = (new_cost - old_cost) / old_cost
            if abs(change) > COST_TOLERANCE:
                findings.append(
                    Finding(
                        case,
                        "cost",
                        f"{old_cost:.1f} -> {new_cost:.1f} ({change:+.0%})",
                        # Cost is a proxy this project has measured as unreliable, so a
                        # rise asks for eyes rather than failing the build.
                        False,
                    )
                )

        old_iou, new_iou = was.get("bbox_iou"), now.get("bbox_iou")
        if old_iou is not None and new_iou is not None:
            drop = old_iou - new_iou
            if drop > IOU_TOLERANCE:
                findings.append(
                    Finding(case, "grounding", f"IoU {old_iou:.3f} -> {new_iou:.3f}", True)
                )

        old_sig, new_sig = was.get("signature"), now.get("signature")
        if old_sig and new_sig:
            moved = distance(old_sig, new_sig)
            if moved > SIGNATURE_TOLERANCE:
                findings.append(
                    Finding(case, "image", f"the result changed (distance {moved:.3f})", False)
                )

        if was.get("passes") != now.get("passes"):
            findings.append(
                Finding(case, "route", f"{was.get('passes')} -> {now.get('passes')}", False)
            )

    return sorted(findings, key=lambda f: (not f.blocking, f.case))


def render(findings: list[Finding]) -> str:
    if not findings:
        return "No change against the baseline."
    lines = ["| case | change | detail |", "| --- | --- | --- |"]
    for f in findings:
        flag = "**blocking**" if f.blocking else f.kind
        lines.append(f"| `{f.case}` | {flag} | {f.detail} |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=REPO_ROOT / "evals" / "out" / "report.json")
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument(
        "--images",
        type=Path,
        default=REPO_ROOT / "evals" / "out",
        help="where the run wrote its pictures",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="record the current report as the new baseline instead of comparing",
    )
    args = parser.parse_args()

    if not args.report.exists():
        print(f"no report at {args.report}; run `make eval` first", file=sys.stderr)
        return 2

    if args.update:
        rows = list(load(args.report, args.images).values())
        args.baseline.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"baseline updated from {args.report}: {len(rows)} local case(s)")
        return 0

    if not args.baseline.exists():
        print(
            f"no baseline at {args.baseline}. Record one with `make eval-baseline` once the "
            "current results have been looked at.",
            file=sys.stderr,
        )
        return 2

    findings = compare(load(args.baseline), load(args.report, args.images))
    print(render(findings))

    blocking = [f for f in findings if f.blocking]
    if blocking:
        print(f"\n{len(blocking)} blocking change(s) against the baseline.", file=sys.stderr)
        return 1
    if findings:
        print("\nNothing blocking. Look at the images for anything flagged above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
