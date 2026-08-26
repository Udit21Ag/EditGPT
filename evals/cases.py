"""Loading and validating the golden case set."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from editgpt_core import EditOp

EVALS_DIR = Path(__file__).resolve().parent
PHOTOS_DIR = EVALS_DIR / "photos"
CASES_FILE = EVALS_DIR / "cases.json"

Box = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class Case:
    """One (image, intent) pair with everything needed to score the result."""

    id: str
    path: Path
    op: EditOp
    target: str | None
    content: str | None
    box: Box | None
    """For remove: the reference bounding box of the object. For add: where to put it.
    For background: the bounding box of the SUBJECT to keep."""
    fill: str
    """What should be there, phrased for a mask+prompt model that cannot follow an
    instruction. Empty for cases that never reach a generative provider."""
    note: str
    deferred: bool = False

    @property
    def prompt(self) -> str:
        match self.op:
            case EditOp.REMOVE:
                return f"remove {self.target}"
            case EditOp.ADD:
                return f"add {self.content}"
            case EditOp.REPLACE:
                return f"replace {self.target} with {self.content}"
            case EditOp.BACKGROUND:
                return f"change the background to {self.content}"
            case EditOp.UPSCALE:
                return "upscale the image"
            case _:
                return f"{self.op.value} {self.target or self.content or ''}".strip()


def load(
    ops: set[EditOp] | None = None,
    *,
    need_box: bool = False,
    include_deferred: bool = False,
) -> list[Case]:
    raw = json.loads(CASES_FILE.read_text())
    cases: list[Case] = []
    for entry in raw:
        path = PHOTOS_DIR / entry["file"]
        if not path.exists():
            raise FileNotFoundError(f"case {entry['id']} references a missing photo: {path}")
        case = Case(
            id=entry["id"],
            path=path,
            op=EditOp(entry["op"]),
            target=entry.get("target"),
            content=entry.get("content"),
            box=tuple(entry["box"]) if entry.get("box") else None,
            fill=entry.get("fill", ""),
            note=entry.get("note", ""),
            deferred=entry.get("deferred", False),
        )
        if case.deferred and not include_deferred:
            continue
        if ops is not None and case.op not in ops:
            continue
        if need_box and case.box is None:
            continue
        cases.append(case)
    return cases
