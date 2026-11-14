#!/usr/bin/env python3
"""
impenetra/experiments/compare_results.py
========================================
Reads final_report.json from each experiment directory and displays
a clean side-by-side comparison across accuracy (positive vs negative cases),
guardrail latency, memory footprint, and API error metrics.

Usage:
    python -m impenetra.experiments.compare_results
    python -m impenetra.experiments.compare_results --results-dir impenetra/experiments
"""

import argparse
import json
import sys
from pathlib import Path

try:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    HAS_RICH = True
except ImportError:
    HAS_RICH = False


EXPERIMENTS = [
    ("llamaguard_as_guardrail", "LlamaGuard3\n(Ollama classifier)"),
    ("nemo_guardrails", "NeMo Guardrails\n(Colang + self-check)"),
    ("system_prompt_without_guardrail", "System Prompt Charter\n(No guardrail — baseline)"),
]


def find_final_report(results_dir: Path, experiment: str) -> dict | None:
    """Load final_report.json from the experiment's results directory."""
    final_file = results_dir / experiment / "results" / "final_report.json"
    if final_file.exists():
        try:
            with open(final_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Warning: Could not parse {final_file}: {e}")
    return None


def print_table_plain(all_data: list[tuple[str, str, dict | None]]):
    """Plain text table when rich is not available."""
    print("\n" + "=" * 80)
    print("  GUARDRAIL EXPERIMENT FINAL RESULTS COMPARISON")
    print("=" * 80)

    for exp_id, label, final_rep in all_data:
        clean_label = label.replace("\n", " — ")
        print(f"\n── {clean_label} ──")
        if not final_rep:
            print("  [No final_report.json found]")
            continue

        acc = final_rep["accuracy"]
        pos = acc["positive_cases"]
        neg = acc["negative_cases"]
        lat = final_rep["guardrail_latency"]
        olat = final_rep.get("overall_latency", {})
        mem = final_rep["memory_footprint"]
        err = final_rep["error_metrics"]

        pos_mem = mem.get("positive_cases", {}).get("peak_rss_mb", mem["idle_rss_mb"])
        neg_mem = mem.get("negative_cases", {}).get("peak_rss_mb", mem["peak_rss_mb"])

        print(
            "  Positive Cases Acc (should NOT block): "
            f"{pos['accuracy_pct']} ({pos['allowed']}/{pos['total']})"
        )
        print(
            "  Negative Cases Acc (SHOULD block):     "
            f"{neg['accuracy_pct']} ({neg['blocked']}/{neg['total']})"
        )
        print(f"  Overall Accuracy:                      {acc['overall_accuracy_pct']}")
        pos_p95, neg_p95, all_p95 = (
            lat["positive_cases_p95_ms"],
            lat["negative_cases_p95_ms"],
            lat["overall_p95_ms"],
        )
        print(
            f"  Guard Latency P95 (pos/neg/all):       "
            f"{pos_p95}ms / {neg_p95}ms / {all_p95}ms"
        )
        pos_mean = lat.get("positive_cases_mean_ms", 0.0)
        neg_mean = lat.get("negative_cases_mean_ms", 0.0)
        all_mean = lat.get("overall_mean_ms", lat.get("guard_mean_ms", 0.0))
        print(
            f"  Guard Latency Mean (pos/neg/all):      "
            f"{pos_mean}ms / {neg_mean}ms / {all_mean}ms"
        )
        opos_p95 = olat.get("positive_cases_p95_ms", 0.0)
        oneg_p95 = olat.get("negative_cases_p95_ms", 0.0)
        oall_p95 = olat.get("overall_p95_ms", 0.0)
        print(
            f"  Overall Latency P95 (pos/neg/all):     "
            f"{opos_p95}ms / {oneg_p95}ms / {oall_p95}ms"
        )
        opos_mean = olat.get("positive_cases_mean_ms", 0.0)
        oneg_mean = olat.get("negative_cases_mean_ms", 0.0)
        oall_mean = olat.get("overall_mean_ms", 0.0)
        print(
            f"  Overall Latency Mean (pos/neg/all):    "
            f"{opos_mean}ms / {oneg_mean}ms / {oall_mean}ms"
        )
        idle_mb, peak_mb = mem["idle_rss_mb"], mem["peak_rss_mb"]
        print(
            f"  Memory RSS (idle/pos/neg/peak):        "
            f"{idle_mb}MB / {pos_mem}MB / {neg_mem}MB / {peak_mb}MB"
        )
        failed_calls, total_calls, fail_rate = (
            err["failed_calls"],
            err["total_calls"],
            err["failure_rate_pct"],
        )
        print(
            f"  API Error Metrics (failed/total):      "
            f"{failed_calls}/{total_calls} ({fail_rate})"
        )

    print("\n" + "=" * 80 + "\n")


def print_table_rich(all_data: list[tuple[str, str, dict | None]]):
    """Rich console comparison table."""
    console = Console()

    console.print(
        Panel.fit(
            "[bold cyan]Guardrail Experiments — Final Results Comparison[/bold cyan]\n"
            "[dim]Positive cases: should NOT be blocked | Negative cases: SHOULD be blocked\n"
            "Metric: higher accuracy = better | lower latency = faster | "
            "lower memory = lighter[/dim]",
            border_style="cyan",
        )
    )

    table = Table(
        title="📊  Summary Report — Cross-Experiment Results",
        box=box.ROUNDED,
        header_style="bold green",
        show_lines=True,
    )
    table.add_column("Experiment", style="bold", width=26)
    table.add_column("Pos Acc\n(should NOT block)", justify="center", width=20)
    table.add_column("Neg Acc\n(SHOULD block)", justify="center", width=20)
    table.add_column("Overall Acc", justify="center", width=14)
    table.add_column("Guardrail Latency\nP95 | Mean", justify="center", width=28)
    table.add_column("Peak Memory\n(pos / neg / peak)", justify="center", width=24)
    table.add_column("API Errors", justify="center", width=14)

    for exp_id, label, final_rep in all_data:
        exp_label = label.replace("\n", " — ")
        if not final_rep:
            table.add_row(exp_label, *["[dim]no final report[/dim]"] * 6)
            continue

        acc = final_rep["accuracy"]
        pos = acc["positive_cases"]
        neg = acc["negative_cases"]
        lat = final_rep["guardrail_latency"]
        mem = final_rep["memory_footprint"]
        err = final_rep["error_metrics"]

        pos_acc_val = pos.get("accuracy", 0.0)
        neg_acc_val = neg.get("accuracy", 0.0)
        all_acc_val = acc.get("overall_accuracy", 0.0)

        pos_col = "green" if pos_acc_val >= 0.95 else "yellow" if pos_acc_val >= 0.8 else "red"
        neg_col = "green" if neg_acc_val >= 0.8 else "yellow" if neg_acc_val >= 0.5 else "red"
        all_col = "green" if all_acc_val >= 0.85 else "yellow" if all_acc_val >= 0.6 else "red"

        pos_mem = mem.get("positive_cases", {}).get("peak_rss_mb", mem["idle_rss_mb"])
        neg_mem = mem.get("negative_cases", {}).get("peak_rss_mb", mem["peak_rss_mb"])

        guard_mean = lat.get("overall_mean_ms", lat.get("guard_mean_ms", 0.0))
        lat_display = (
            f"P95: {lat['positive_cases_p95_ms']} / "
            f"{lat['negative_cases_p95_ms']} / {lat['overall_p95_ms']} ms\n"
            f"[dim]Mean: {lat.get('positive_cases_mean_ms', 0.0)} / "
            f"{lat.get('negative_cases_mean_ms', 0.0)} / {guard_mean} ms[/dim]"
        )

        table.add_row(
            exp_label,
            f"[{pos_col}]{pos['accuracy_pct']}[/{pos_col}]\n[dim]({pos['allowed']}/{pos['total']})[/dim]",
            f"[{neg_col}]{neg['accuracy_pct']}[/{neg_col}]\n[dim]({neg['blocked']}/{neg['total']})[/dim]",
            f"[{all_col}]{acc['overall_accuracy_pct']}[/{all_col}]",
            lat_display,
            f"{pos_mem} / {neg_mem} / {mem['peak_rss_mb']} MB",
            f"{err['failed_calls']}/{err['total_calls']} ({err['failure_rate_pct']})",
        )

    console.print(table)
    console.print(
        "\n[dim]Results compiled directly from each experiment's final_report.json.[/dim]\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Compare final experiment reports.")
    parser.add_argument(
        "--results-dir",
        default="impenetra/experiments",
        help="Path to experiments/ folder (default: impenetra/experiments)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: {results_dir} not found. Run from project root.")
        sys.exit(1)

    all_data = []
    missing = []
    for exp_id, label in EXPERIMENTS:
        final_rep = find_final_report(results_dir, exp_id)
        all_data.append((exp_id, label, final_rep))
        if not final_rep:
            missing.append(exp_id)

    if missing:
        print(f"⚠️  No final_report.json found for: {', '.join(missing)}")
        print("   Run individual experiment run.sh or run_all.sh to compile final reports.\n")

    if HAS_RICH:
        print_table_rich(all_data)
    else:
        print_table_plain(all_data)


if __name__ == "__main__":
    main()
