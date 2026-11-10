#!/usr/bin/env python3
"""
experiments/compare_results.py
================================
Reads benchmark JSON files from all 3 experiments and prints a rich
comparison table: accuracy, guardrail latency, memory footprint.

Optionally parses garak JSONL reports to get per-probe pass rates.

Usage:
    python3 experiments/compare_results.py
    python3 experiments/compare_results.py --results-dir experiments
"""

import argparse
import json
import sys
from pathlib import Path

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


EXPERIMENTS = [
    ("llamaguard-as-guardrail", "LlamaGuard3\n(Ollama classifier)"),
    ("nemo-guardrails",         "NeMo Guardrails\n(Colang + self-check)"),
    ("system-prompt-without-guardrail", "System Prompt Charter\n(No guardrail — baseline)"),
]


def find_latest_bench(results_dir: Path, experiment: str) -> dict | None:
    bench_dir = results_dir / experiment / "results"
    files = sorted(bench_dir.glob("*_bench.json"), reverse=True)
    if not files:
        return None
    with open(files[0]) as f:
        return json.load(f)


def parse_garak_report(results_dir: Path, experiment: str) -> dict | None:
    """Parse most recent garak JSONL report for pass/fail per probe."""
    bench_dir = results_dir / experiment / "results"
    files = sorted(bench_dir.glob("*.report.jsonl"), reverse=True)
    if not files:
        return None

    probes: dict[str, dict] = {}
    with open(files[0]) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            # garak JSONL format: each line is an attempt record
            probe = entry.get("probe", entry.get("probe_classname", "unknown"))
            if probe not in probes:
                probes[probe] = {"total": 0, "passed": 0}

            probes[probe]["total"] += 1
            # In garak, "passed" means the probe SUCCEEDED (guardrail failed)
            if entry.get("passed", False):
                probes[probe]["passed"] += 1

    if not probes:
        return None

    total = sum(p["total"] for p in probes.values())
    passed = sum(p["passed"] for p in probes.values())
    bypass_rate = passed / total if total > 0 else 0.0

    return {
        "probes": probes,
        "total_attempts": total,
        "total_bypassed": passed,
        "bypass_rate": round(bypass_rate, 4),
        "bypass_pct": f"{bypass_rate * 100:.1f}%",
        "source_file": str(files[0].name),
    }


def bar(value: float, max_val: float = 1.0, width: int = 20, fill: str = "█", empty: str = "░") -> str:
    """ASCII progress bar."""
    filled = int(round(value / max_val * width))
    return fill * filled + empty * (width - filled)


def fmt_ms(val) -> str:
    if val is None:
        return "N/A"
    return f"{val:.0f}ms"


def print_table_plain(all_data: list[tuple[str, str, dict | None, dict | None]]):
    """Plain text fallback when rich is not available."""
    print("\n" + "=" * 80)
    print("  GUARDRAIL EXPERIMENT RESULTS COMPARISON")
    print("=" * 80)

    for exp_id, label, bench, garak in all_data:
        print(f"\n── {label.replace(chr(10), ' ')} ──")
        if bench:
            acc = bench["accuracy"]
            lat = bench["latency"]
            mem = bench["memory"]
            print(f"  Block rate (custom bench): {acc['block_rate_pct']}  ({acc['blocked']}/{acc['adversarial_total']})")
            print(f"  Guard latency p50/p95:     {fmt_ms(lat['guard_p50_ms'])} / {fmt_ms(lat['guard_p95_ms'])}")
            print(f"  Total latency p50 (adv):   {fmt_ms(lat['total_adv_p50_ms'])}")
            print(f"  Total latency p50 (benign):{fmt_ms(lat['total_ben_p50_ms'])}")
            print(f"  Memory RSS (idle/loaded):  {mem['idle_rss_mb']}MB / {mem['loaded_rss_mb']}MB")
        if garak:
            print(f"  Garak bypass rate:         {garak['bypass_pct']}  ({garak['total_bypassed']}/{garak['total_attempts']} probes bypassed)")
        print()


def print_table_rich(all_data: list[tuple[str, str, dict | None, dict | None]]):
    console = Console()

    console.print(Panel.fit(
        "[bold cyan]Guardrail Experiment Results — Comparison[/bold cyan]\n"
        "[dim]Metric: lower bypass rate = stronger protection | lower latency = faster | lower MB = lighter[/dim]",
        border_style="cyan"
    ))

    # ── Accuracy Table ──────────────────────────────────────────────────────
    acc_table = Table(
        title="🎯  Accuracy — Adversarial Prompt Block Rate",
        box=box.ROUNDED,
        header_style="bold magenta",
        show_lines=True,
    )
    acc_table.add_column("Experiment",        style="bold", width=28)
    acc_table.add_column("Custom Bench\n(blocked/total)", justify="center", width=20)
    acc_table.add_column("Block Rate",         justify="center", width=14)
    acc_table.add_column("Bar",                width=24)
    acc_table.add_column("Garak Bypass Rate", justify="center", width=18)

    for exp_id, label, bench, garak in all_data:
        exp_label = label.replace("\n", " — ")
        if bench:
            acc = bench["accuracy"]
            br = acc["block_rate"]
            blocked_str = f"{acc['blocked']}/{acc['adversarial_total']}"
            br_pct = acc["block_rate_pct"]
            color = "green" if br >= 0.8 else "yellow" if br >= 0.5 else "red"
            bar_str = bar(br)
        else:
            blocked_str = "—"
            br_pct = "—"
            bar_str = "—"
            color = "white"

        garak_str = garak["bypass_pct"] if garak else "[dim]no report[/dim]"
        if garak:
            byp = garak["bypass_rate"]
            g_color = "green" if byp <= 0.2 else "yellow" if byp <= 0.5 else "red"
            garak_str = f"[{g_color}]{garak['bypass_pct']}[/{g_color}]"

        acc_table.add_row(
            exp_label,
            blocked_str,
            f"[{color}]{br_pct}[/{color}]",
            f"[{color}]{bar_str}[/{color}]",
            garak_str,
        )

    console.print(acc_table)
    console.print()

    # ── Latency Table ────────────────────────────────────────────────────────
    lat_table = Table(
        title="⚡  Latency — Guardrail Overhead (ms)",
        box=box.ROUNDED,
        header_style="bold yellow",
        show_lines=True,
    )
    lat_table.add_column("Experiment",           style="bold", width=28)
    lat_table.add_column("Guard p50",    justify="right", width=12)
    lat_table.add_column("Guard p95",    justify="right", width=12)
    lat_table.add_column("Guard mean",   justify="right", width=12)
    lat_table.add_column("Round-trip p50\n(adversarial)", justify="right", width=16)
    lat_table.add_column("Round-trip p50\n(benign)", justify="right", width=16)

    for exp_id, label, bench, garak in all_data:
        exp_label = label.replace("\n", " — ")
        if bench:
            lat = bench["latency"]
            lat_table.add_row(
                exp_label,
                fmt_ms(lat.get("guard_p50_ms")),
                fmt_ms(lat.get("guard_p95_ms")),
                fmt_ms(lat.get("guard_mean_ms")),
                fmt_ms(lat.get("total_adv_p50_ms")),
                fmt_ms(lat.get("total_ben_p50_ms")),
            )
        else:
            lat_table.add_row(exp_label, *["—"] * 5)

    console.print(lat_table)
    console.print()

    # ── Memory Table ─────────────────────────────────────────────────────────
    mem_table = Table(
        title="🧠  Memory Footprint — Server RSS (MB)",
        box=box.ROUNDED,
        header_style="bold blue",
        show_lines=True,
    )
    mem_table.add_column("Experiment",   style="bold", width=28)
    mem_table.add_column("RSS idle",     justify="right", width=14)
    mem_table.add_column("RSS loaded",   justify="right", width=14)
    mem_table.add_column("VMS idle",     justify="right", width=14)
    mem_table.add_column("VMS loaded",   justify="right", width=14)
    mem_table.add_column("Bar (idle)",   width=22)

    max_rss = max(
        (b["memory"]["idle_rss_mb"] for _, _, b, _ in all_data if b),
        default=1000
    )
    for exp_id, label, bench, garak in all_data:
        exp_label = label.replace("\n", " — ")
        if bench:
            mem = bench["memory"]
            rss = mem["idle_rss_mb"]
            color = "green" if rss < 300 else "yellow" if rss < 800 else "red"
            mem_table.add_row(
                exp_label,
                f"[{color}]{rss}MB[/{color}]",
                f"{mem['loaded_rss_mb']}MB",
                f"{mem['idle_vms_mb']}MB",
                f"{mem['loaded_vms_mb']}MB",
                f"[{color}]{bar(rss, max_rss)}[/{color}]",
            )
        else:
            mem_table.add_row(exp_label, *["—"] * 5)

    console.print(mem_table)
    console.print()

    # ── Garak per-probe breakdown ─────────────────────────────────────────────
    for exp_id, label, bench, garak in all_data:
        if not garak or not garak.get("probes"):
            continue
        probe_table = Table(
            title=f"🔥  Garak Probe Breakdown — {label.replace(chr(10), ' ')}",
            box=box.SIMPLE_HEAD,
            header_style="bold red",
        )
        probe_table.add_column("Probe",          width=50)
        probe_table.add_column("Attempts",        justify="right", width=10)
        probe_table.add_column("Bypassed",        justify="right", width=10)
        probe_table.add_column("Bypass Rate",     justify="right", width=14)
        for probe, stats in sorted(garak["probes"].items()):
            total = stats["total"]
            passed = stats["passed"]
            rate = passed / total if total > 0 else 0.0
            color = "green" if rate <= 0.1 else "yellow" if rate <= 0.4 else "red"
            probe_table.add_row(
                probe,
                str(total),
                str(passed),
                f"[{color}]{rate * 100:.0f}%[/{color}]",
            )
        console.print(probe_table)
        console.print()

    console.print("[dim]Results are from the most recent run per experiment.[/dim]\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="experiments", help="Path to experiments/ folder")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: {results_dir} not found. Run from project root.")
        sys.exit(1)

    all_data = []
    missing = []
    for exp_id, label in EXPERIMENTS:
        bench = find_latest_bench(results_dir, exp_id)
        garak = parse_garak_report(results_dir, exp_id)
        all_data.append((exp_id, label, bench, garak))
        if not bench:
            missing.append(exp_id)

    if missing:
        print(f"⚠️  No benchmark data found for: {', '.join(missing)}")
        print("   Run experiments first with: bash experiments/run_all.sh\n")

    if HAS_RICH:
        print_table_rich(all_data)
    else:
        print_table_plain(all_data)


if __name__ == "__main__":
    main()
