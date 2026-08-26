"""Collate out/*.json into the Phase 0 result tables.

Writes out/report.md and prints to the terminal. Deliberately does NOT touch
docs/adr/0001-model-routing.md — that file carries hand-written quality notes and
a decision, and a generator has no business overwriting either.
"""

from __future__ import annotations

import json

from rich.console import Console
from rich.table import Table

from bench.common import OUT

SLOT_MB = 1200

# Single models, measured in isolation.
MODELS = {
    "mobilesam": "MobileSAM (encoder + decoder)",
    "lama": "Big-LaMa erase",
    "migan": "MobileSAM + MI-GAN erase",
    "clipseg": "CLIPSeg text-to-mask",
}
# Whole pipelines, where more than one model is resident at once.
PIPELINES = {
    "pipeline": "box -> MobileSAM -> LaMa @1024",
    "text2edit": "text -> CLIPSeg -> MobileSAM -> LaMa",
    "fullres": "native 15.9 MP, crop vs whole",
    "erasers": "LaMa vs MI-GAN head-to-head",
    "remote": "Cloudflare Workers AI (network-bound)",
}


def load_all() -> dict:
    return {p.stem: json.loads(p.read_text()) for p in sorted(OUT.glob("*.json"))}


def main() -> None:
    console = Console()
    data = load_all()
    if not data:
        raise SystemExit("No results in out/. Run `make bench-lama` first.")

    lines: list[str] = ["# Phase 0 — measured results", ""]

    for title, group, judge in (
        ("Models in isolation", MODELS, True),
        ("Pipelines (multiple models resident)", PIPELINES, False),
    ):
        table = Table(title=title, header_style="bold")
        for col, just in (
            ("run", "left"),
            ("what", "left"),
            ("peak RSS MB", "right"),
            ("cold s", "right"),
            ("warm p50 s", "right"),
            ("vs slot", "left"),
        ):
            table.add_column(col, justify=just)
        lines += [
            f"## {title}",
            "",
            "| run | what | peak RSS (MB) | cold (s) | warm p50 (s) | vs 1200 MB slot |",
            "|---|---|---:|---:|---:|---|",
        ]
        for key, label in group.items():
            r = data.get(key)
            if not r:
                continue
            rss = r.get("peak_rss_mb", 0)
            cold = r.get("cold_load_s")
            cold = (
                "—" if cold is None else (json.dumps(cold) if isinstance(cold, dict) else f"{cold}")
            )
            warm = r.get("warm_p50_s", "—")
            if key == "remote":
                verdict = "n/a (remote)"
            elif not judge:
                verdict = "—"
            else:
                verdict = "fits" if rss <= SLOT_MB else "OVER"
            colour = {"fits": "[green]fits[/green]", "OVER": "[red]OVER[/red]"}.get(
                verdict, verdict
            )
            table.add_row(key, label, f"{rss:.0f}", cold, str(warm), colour)
            lines.append(f"| {key} | {label} | {rss:.0f} | {cold} | {warm} | {verdict} |")
        lines.append("")
        console.print(table)
        console.print()

    if "erasers" in data:
        e = data["erasers"]
        lines += [
            "## Eraser head-to-head",
            "",
            f"Wins by composite photometric cost: **{e['wins']}**",
            "",
            "| case | target | winner | LaMa cost | MI-GAN cost |",
            "|---|---|---|---:|---:|",
        ]
        t = Table(title="Eraser head-to-head (lower cost is better)", header_style="bold")
        for c in ("case", "target", "winner", "LaMa", "MI-GAN"):
            t.add_column(c, justify="right" if c in ("LaMa", "MI-GAN") else "left")
        for cid, c in e["cases"].items():
            t.add_row(cid, c["target"], c["winner"], f"{c['lama_cost']}", f"{c['migan_cost']}")
            lines.append(
                f"| {cid} | {c['target']} | {c['winner']} | {c['lama_cost']} | {c['migan_cost']} |"
            )
        console.print(t)
        lines.append("")

    if "clipseg" in data and "text2edit" in data:
        lines += [
            "## Text-to-mask",
            "",
            f"- CLIPSeg seed alone: **{data['clipseg']['hit_rate']}**",
            f"- After MobileSAM refinement: **{data['text2edit']['hit_rate']}**",
            "",
        ]
        console.print(
            f"CLIPSeg seed [bold]{data['clipseg']['hit_rate']}[/bold]  ->  "
            f"after MobileSAM [bold]{data['text2edit']['hit_rate']}[/bold]"
        )

    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    console.print(f"\nWrote [bold]{OUT / 'report.md'}[/bold]")
    console.print("[dim]docs/adr/0001-model-routing.md is hand-maintained and not touched.[/dim]")


if __name__ == "__main__":
    main()
