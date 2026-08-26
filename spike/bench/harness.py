"""Out-of-process RSS sampler.

Each model runs in its own subprocess so we measure a *clean* peak RSS —
the number that actually decides whether EditGPT fits in 8 GB. The parent
polls the child (and any grandchildren) every 50 ms; the child reports its
own timings on stdout as a single ``##BENCH##{json}`` line.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import psutil
from rich.console import Console

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
SAMPLE_HZ = 20.0

RUNNERS = {
    "lama": "bench.run_lama",
    "mobilesam": "bench.run_mobilesam",
    "clipseg": "bench.run_clipseg",
    "remote": "bench.run_remote",
    "pipeline": "bench.run_pipeline",
    "fullres": "bench.run_fullres",
    "text2edit": "bench.run_text2edit",
    "erasers": "bench.run_erasers",
    "migan": "bench.run_migan",
    "final": "bench.run_final",
}


def _tree_rss(proc: psutil.Process) -> int:
    total = proc.memory_info().rss
    for child in proc.children(recursive=True):
        try:
            total += child.memory_info().rss
        except psutil.Error:
            pass
    return total


def run(name: str, extra: list[str] | None = None) -> dict:
    module = RUNNERS[name]
    cmd = [sys.executable, "-m", module, *(extra or [])]
    started = time.perf_counter()
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, text=True)
    watcher = psutil.Process(proc.pid)

    peak = 0
    while proc.poll() is None:
        try:
            peak = max(peak, _tree_rss(watcher))
        except psutil.Error:
            break
        time.sleep(1 / SAMPLE_HZ)

    stdout, _ = proc.communicate()
    wall = time.perf_counter() - started

    payload: dict = {}
    for line in stdout.splitlines():
        if line.startswith("##BENCH##"):
            payload = json.loads(line.removeprefix("##BENCH##"))
        else:
            print(line)

    result = {
        "model": name,
        "ok": proc.returncode == 0,
        "peak_rss_mb": round(peak / 1e6, 1),
        "wall_s": round(wall, 2),
        **payload,
    }
    OUT.mkdir(exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> int:
    console = Console()
    names = sys.argv[1:] or list(RUNNERS)
    for name in names:
        if name not in RUNNERS:
            console.print(f"[red]unknown runner: {name}[/red]  choices: {', '.join(RUNNERS)}")
            return 2
        console.rule(f"[bold]{name}")
        result = run(name)
        status = "[green]ok[/green]" if result["ok"] else "[red]FAILED[/red]"
        console.print(
            f"{status}  peak RSS [bold]{result['peak_rss_mb']} MB[/bold]  "
            f"cold {result.get('cold_load_s', '—')}s  warm p50 {result.get('warm_p50_s', '—')}s"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
