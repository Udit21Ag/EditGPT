"""How good is the planner, and what does each lane cost?

The claim the planner makes is that rules answer the decidable instructions and the model
sees only the rest. That claim is worth exactly as much as its measurement, so this reports
four things over `benchmarks/instructions.jsonl`:

* **coverage** — what fraction each lane answered, which is the cost argument;
* **accuracy** — did the operation match, and did the subject survive the trip;
* **refusal** — instructions that are not edits, or name an operation with no
  implementation, must become a question rather than a confident wrong plan;
* **cost** — seconds and tokens per lane, and how often the ten-second production deadline
  would have fired.

Two splits. `authored` is the phrasing the rules were written against and is therefore not
evidence of anything except that they still work. `holdout` is phrasing written afterwards
to be different; it is weaker than a third party's set — the same person wrote both — and
it is what is available. Report them apart, and believe the holdout.

    uv run python -m benchmarks.planner              # rules only, no key needed
    uv run python -m benchmarks.planner --model      # rules, then the model on the rest
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from editgpt_core import EditOp
from editgpt_planner import Gemini, Plan, Route, plan
from editgpt_planner.planner import OUT_OF_QUOTA

DATA = Path(__file__).with_name("instructions.jsonl")

IMPLEMENTED = frozenset(
    {EditOp.REMOVE, EditOp.ADD, EditOp.REPLACE, EditOp.BACKGROUND, EditOp.UPSCALE}
)

PRODUCTION_DEADLINE_S = 10.0
"""What `planner.PLANNING_TIMEOUT_S` cuts a real request off at. Measured here rather than
enforced, so a slow answer is counted instead of hidden."""

ARTICLES = ("the ", "a ", "an ", "my ", "this ", "that ")


def normalise(text: str | None) -> str:
    """Compare subjects the way a person would: case, articles and padding do not count."""
    if not text:
        return ""
    value = " ".join(text.lower().strip().strip(".!?").split())
    for article in ARTICLES:
        if value.startswith(article):
            value = value[len(article) :]
    return value


@dataclass(frozen=True, slots=True)
class Row:
    instruction: str
    op: str | None
    target: str | None
    content: str | None
    colour: str | None
    split: str

    @property
    def is_edit(self) -> bool:
        return self.op is not None


def load(path: Path = DATA) -> list[Row]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        raw: dict[str, Any] = json.loads(line)
        rows.append(
            Row(
                instruction=raw["instruction"],
                op=raw.get("op"),
                target=raw.get("target"),
                content=raw.get("content"),
                colour=raw.get("colour"),
                split=raw.get("split", "holdout"),
            )
        )
    return rows


def judge(row: Row, made: Plan) -> tuple[bool, str]:
    """Was the plan right? Returns the verdict and, when it is wrong, what went wrong."""
    if not row.is_edit:
        if made.actionable:
            op = made.intent.op.value if made.intent else "?"
            return False, f"planned {op} for something that is not an edit"
        return True, ""

    if made.intent is None:
        return False, "asked a question about a plain instruction"
    if made.intent.op.value != row.op:
        return False, f"{made.intent.op.value}, expected {row.op}"
    if row.target and normalise(made.intent.target) != normalise(row.target):
        return False, f"target {made.intent.target!r}, expected {row.target!r}"
    if row.colour and made.intent.colour != row.colour:
        return False, f"colour {made.intent.colour}, expected {row.colour}"
    if row.content and not made.intent.content:
        return False, "no content for an operation that needs one"
    return True, ""


def run(rows: list[Row], *, completer: Any = None, rpm: float = 0.0) -> list[dict[str, Any]]:
    """Plan every row, pacing the model calls so the run does not measure a rate limit.

    Learned the hard way: thirty calls in ninety seconds exhausted a free-tier quota and
    eight instructions came back as questions. They were not wrong answers — they were not
    answers at all, and scoring them as failures measures the free tier rather than the
    planner.
    """
    gap = 60.0 / rpm if rpm else 0.0
    results = []
    last_call = 0.0
    for row in rows:
        if gap and last_call:
            time.sleep(max(0.0, gap - (time.monotonic() - last_call)))
        started = time.monotonic()
        # Deliberately generous: this is measuring how long the model takes, and a
        # benchmark that truncates at the production deadline reports its own deadline.
        made = plan(row.instruction, available=IMPLEMENTED, completer=completer, timeout_s=60.0)
        if made.route is Route.MODEL or made.reason == OUT_OF_QUOTA:
            last_call = time.monotonic()
        correct, why = judge(row, made)
        results.append(
            {
                "instruction": row.instruction,
                "split": row.split,
                "route": made.route.value,
                "expected_op": row.op,
                "got_op": made.intent.op.value if made.intent else None,
                "correct": correct,
                "why": why,
                "reason": made.reason,
                "unmeasured": made.reason == OUT_OF_QUOTA,
                "seconds": round(time.monotonic() - started, 3),
                "tokens": made.prompt_tokens + made.output_tokens,
            }
        )
    return results


def summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Accuracy over the rows that were actually answered.

    Rows the model never saw — quota gone — are counted and excluded rather than scored.
    Including them lets a free tier's rate limit look like a planner's mistake, and a row
    scored *correct* because the model was unreachable is worse still: a refusal that
    happens to be right for no reason at all.
    """
    unmeasured = [r for r in results if r["unmeasured"]]
    results = [r for r in results if not r["unmeasured"]]
    by_route = Counter(r["route"] for r in results)
    model_calls = [r for r in results if r["route"] == "model"]
    rule_calls = [r for r in results if r["route"] == "rule"]

    def accuracy(rows: list[dict[str, Any]]) -> float:
        return round(sum(r["correct"] for r in rows) / len(rows), 3) if rows else 0.0

    edits = [r for r in results if r["expected_op"] is not None]
    refusals = [r for r in results if r["expected_op"] is None]

    return {
        "n": len(results),
        "unmeasured": len(unmeasured),
        "coverage": {
            route: round(count / len(results), 3) for route, count in sorted(by_route.items())
        },
        "accuracy": {
            "overall": accuracy(results),
            "rule": accuracy(rule_calls),
            "model": accuracy(model_calls),
            "on_edits": accuracy(edits),
            "on_refusals": accuracy(refusals),
        },
        "by_split": {
            split: accuracy([r for r in results if r["split"] == split])
            for split in sorted({r["split"] for r in results})
        },
        "seconds": {
            "rule_median": round(statistics.median([r["seconds"] for r in rule_calls]), 5)
            if rule_calls
            else 0.0,
            "model_median": round(statistics.median([r["seconds"] for r in model_calls]), 2)
            if model_calls
            else 0.0,
            "model_max": round(max([r["seconds"] for r in model_calls]), 2) if model_calls else 0.0,
            "over_production_deadline": sum(
                1 for r in model_calls if r["seconds"] > PRODUCTION_DEADLINE_S
            ),
        },
        "tokens": {
            "total": sum(r["tokens"] for r in results),
            "per_model_call": round(sum(r["tokens"] for r in model_calls) / len(model_calls), 1)
            if model_calls
            else 0.0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="store_true", help="ask the model what rules decline")
    parser.add_argument(
        "--rpm", type=float, default=8.0, help="model calls per minute; 0 to run flat out"
    )
    parser.add_argument("--json", type=Path, help="write the full per-row result here")
    args = parser.parse_args()

    rows = load()
    completer = None
    if args.model:
        completer = Gemini()
        if not completer.is_configured():
            print("no GEMINI_API_KEY; run without --model for the rules-only half", file=sys.stderr)
            return 2

    started = time.monotonic()
    results = run(rows, completer=completer, rpm=args.rpm if args.model else 0.0)
    stats = summarise(results)

    lane = "rules + model" if args.model else "rules only"
    print(f"\nplanner, {lane}, n={stats['n']}  ({time.monotonic() - started:.0f}s)")
    if stats["unmeasured"]:
        print(
            f"  {stats['unmeasured']} row(s) unmeasured: the model was out of quota. "
            "Excluded rather than scored."
        )
    print("\n  who answered:")
    for route, share in stats["coverage"].items():
        print(f"    {route:>6} {share:>6.1%}")
    print("\n  accuracy:")
    for name, value in stats["accuracy"].items():
        print(f"    {name:>12} {value:>6.3f}")
    print("\n  by split (holdout is the one that counts):")
    for split, value in stats["by_split"].items():
        print(f"    {split:>12} {value:>6.3f}")
    print("\n  cost:")
    print(f"    rule median   {stats['seconds']['rule_median'] * 1000:.3f} ms")
    if stats["seconds"]["model_median"]:
        print(f"    model median  {stats['seconds']['model_median']:.2f} s")
        print(f"    model max     {stats['seconds']['model_max']:.2f} s")
        print(
            f"    over the {PRODUCTION_DEADLINE_S:.0f}s deadline: "
            f"{stats['seconds']['over_production_deadline']} call(s) — "
            "these degrade to a question in production"
        )
        print(f"    tokens/call   {stats['tokens']['per_model_call']:.0f}")

    wrong = [r for r in results if not r["correct"] and not r["unmeasured"]]
    if wrong:
        print(f"\n  {len(wrong)} wrong:")
        for row in wrong:
            print(f"    [{row['route']}] {row['instruction']!r}\n        {row['why']}")

    if args.json:
        args.json.write_text(json.dumps({"summary": stats, "rows": results}, indent=2))
        print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
