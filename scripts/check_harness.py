"""Validate harness integrity.

The harness is production infrastructure for every future agent session, so it gets the
same treatment as code: broken references, commands that do not exist and leaked secrets
are failures, not documentation smells.

Run: uv run python scripts/check_harness.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _nested_agent_docs() -> list[Path]:
    """Directory-scoped AGENTS.md files, wherever they sit in the tree."""
    skip = {".git", ".venv", "node_modules", ".next", "spike"}
    return sorted(
        path
        for path in ROOT.rglob("AGENTS.md")
        if path != ROOT / "AGENTS.md" and not skip & set(path.relative_to(ROOT).parts)
    )


DOCS = [
    ROOT / "AGENTS.md",
    *sorted((ROOT / "harness").rglob("*.md")),
    *sorted((ROOT / "docs").glob("*.md")),
    *_nested_agent_docs(),
]

LINK = re.compile(r"\[[^\]]*\]\(([^)#][^)]*)\)")
BACKTICK_PATH = re.compile(r"`([A-Za-z0-9_./-]+/[A-Za-z0-9_./-]+\.(?:md|py|ts|tsx|json|yml|toml))`")
MAKE_TARGET = re.compile(r"`make ([a-z][a-z-]*)")
SECRET = re.compile(
    r"(AIza[0-9A-Za-z_-]{20,}|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----)"
)

failures: list[str] = []
checked = {"links": 0, "paths": 0, "commands": 0, "docs": 0}


def fail(doc: Path, message: str) -> None:
    failures.append(f"{doc.relative_to(ROOT)}: {message}")


def is_generated(path: str) -> bool:
    """Whether a referenced path is a build artifact rather than a source file.

    Git is the authority: anything gitignored is produced by a command, so its absence
    means it has not been generated yet, not that the reference is broken. Without this
    the checker fails whenever someone clears an output directory.
    """
    return (
        subprocess.run(
            ["git", "check-ignore", "-q", path], cwd=ROOT, capture_output=True
        ).returncode
        == 0
    )


def make_targets() -> set[str]:
    text = (ROOT / "Makefile").read_text()
    return set(re.findall(r"^([a-z][a-z-]*):", text, re.M))


def main() -> int:
    targets = make_targets()

    # Commands the docs may reference that are not make targets.
    scripts = set()
    for pkg in (ROOT / "package.json", ROOT / "apps/web/package.json"):
        if pkg.exists():
            scripts |= set(json.loads(pkg.read_text()).get("scripts", {}))

    for doc in DOCS:
        if not doc.exists():
            continue
        checked["docs"] += 1
        text = doc.read_text()

        if SECRET.search(text):
            fail(doc, "contains something shaped like a credential")

        for target in LINK.findall(text):
            checked["links"] += 1
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            if not (doc.parent / target).resolve().exists():
                fail(doc, f"broken link -> {target}")

        for path in BACKTICK_PATH.findall(text):
            if path.startswith("/"):
                continue  # a URL path such as /.well-known/..., not a file
            checked["paths"] += 1
            if (ROOT / path).exists() or (doc.parent / path).exists():
                continue
            if is_generated(path):
                continue
            fail(doc, f"references a path that does not exist -> {path}")

        for target in MAKE_TARGET.findall(text):
            checked["commands"] += 1
            if target not in targets:
                fail(doc, f"references `make {target}`, which is not a Makefile target")

    # AGENTS.md must stay a map. Detail belongs in harness/, loaded on demand.
    agents = (ROOT / "AGENTS.md").read_text().splitlines()
    if len(agents) > 200:
        failures.append(f"AGENTS.md: {len(agents)} lines — it has stopped being a map")

    # A directory-scoped AGENTS.md must stay scoped, not become a second root file.
    for doc in _nested_agent_docs():
        lines = len(doc.read_text().splitlines())
        if lines > 80:
            fail(doc, f"{lines} lines — a scoped file should hold only what applies here")

    # Every harness doc declares when to read it and what authority it has.
    for doc in sorted((ROOT / "harness").glob("*.md")):
        head = doc.read_text()[:600]
        for field in ("**Read when:**", "**Solves:**", "**Authority:**"):
            if field not in head:
                fail(doc, f"missing {field} header")

    # The workspace members named in pyproject must exist.
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    failures.extend(
        f"pyproject.toml: workspace member matches nothing -> {member}"
        for member in config["tool"]["uv"]["workspace"]["members"]
        if not list(ROOT.glob(member))
    )

    # architecture.md drifts first, because it is the most detailed and the code moves
    # under it. Every model it names must exist in the registry, and vice versa.
    architecture = (ROOT / "harness" / "architecture.md").read_text()
    registry_src = (ROOT / "packages/models/src/editgpt_models/registry.py").read_text()
    # Compare on alphanumerics only: the document uses display names ("MI-GAN",
    # "Big-LaMa", "Real-ESRGAN x2") where the registry uses keys.
    flat_doc = re.sub(r"[^a-z0-9]", "", architecture.lower())
    failures.extend(
        f"harness/architecture.md: model {key!r} is in the registry but undocumented"
        for key in re.findall(r'^\s*"([a-z0-9-]+)": ModelSpec\(', registry_src, re.M)
        if re.sub(r"[^a-z0-9]", "", key) not in flat_doc
    )

    # Operations the gateway advertises must be covered by the golden set, or a
    # regression in one of them is invisible.
    gateway = (ROOT / "apps/gateway/src/editgpt_gateway/app.py").read_text()
    advertised = set(re.findall(r"EditOp\.([A-Z]+),", gateway.split("unsupported=")[0]))
    covered = {c["op"].upper() for c in json.loads((ROOT / "evals/cases.json").read_text())}
    failures.extend(
        f"evals/cases.json: {op} is advertised by the gateway but has no case"
        for op in sorted(advertised - covered)
    )

    # Every open debt item needs a trigger, or it never gets paid down.
    debt = (ROOT / "harness" / "tech_debt_tracker.md").read_text()
    for block in debt.split("### TD-")[1:]:
        ident = block.split("\n", 1)[0].strip()
        if ident.startswith("00N"):
            continue  # the template in the recording instructions, not an item
        if "Status: open" in block and "**Trigger:**" not in block:
            failures.append(f"harness/tech_debt_tracker.md: TD-{ident} is open with no trigger")

    # An execution plan left in active/ for a long time is usually abandoned, and an
    # abandoned plan misleads whoever reads it next.
    for plan in (ROOT / "harness/exec-plans/active").glob("*.md"):
        age_days = (time.time() - plan.stat().st_mtime) / 86400
        if age_days > 30:
            failures.append(
                f"harness/exec-plans/active/{plan.name}: untouched for {age_days:.0f} days — "
                "complete it, or move it to completed/ with the outcome"
            )

    # .env must not be tracked by git.
    tracked = subprocess.run(
        ["git", "ls-files", ".env"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    if tracked:
        failures.append("git: .env is tracked — it must never be committed")

    print(
        f"checked {checked['docs']} docs, {checked['links']} links, "
        f"{checked['paths']} paths, {checked['commands']} commands"
    )
    if failures:
        print(f"\n{len(failures)} problem(s):")
        for problem in failures:
            print(f"  - {problem}")
        return 1
    print("harness integrity: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
