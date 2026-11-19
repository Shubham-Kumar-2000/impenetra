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
    ("promptguard_as_guardrail", "Prompt Guard 2 86M\n(PyTorch GPU classifier)"),
    ("nemo_guardrails", "NeMo Guardrails\n(Colang + self-check)"),
    ("system_prompt_as_guardrail", "System Prompt Guardrail\n(LLM-as-a-Judge)"),
    ("dspy_gepa_system_prompt_as_guardrail", "DSPy GEPA Prompt\n(Evolutionary Judge)"),
    ("jev_systemone_noul_as_guardrail", "Jev SystemOne Noul\n(TypeSafe.ai)"),
    ("jev_systemone_choice_as_guardrail", "Jev SystemOne Choice\n(TypeSafe.ai)"),
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


def extract_report_metrics(final_rep: dict) -> dict:
    """Extract and streamline report metrics, supporting both streamlined and legacy schemas."""
    acc = final_rep.get("accuracy", {})
    pos = acc.get("positive") or acc.get("positive_cases") or {}
    neg = acc.get("negative") or acc.get("negative_cases") or {}
    lat = final_rep.get("guardrail_latency", {})
    olat = final_rep.get("overall_latency", {})
    mem = final_rep.get("memory_footprint", {})
    err = final_rep.get("error_metrics", {})

    idle_mb = mem.get("idle_rss_mb", 0.0)
    peak_mb = mem.get("peak_rss_mb", idle_mb)
    pos_mem = (mem.get("positive") or mem.get("positive_cases") or {}).get("peak_rss_mb", idle_mb)
    neg_mem = (mem.get("negative") or mem.get("negative_cases") or {}).get("peak_rss_mb", peak_mb)

    pos_p95 = lat.get("positive_cases_p95_ms", lat.get("positive_p95_ms", 0.0))
    neg_p95 = lat.get("negative_cases_p95_ms", lat.get("negative_p95_ms", 0.0))
    all_p95 = lat.get("overall_p95_ms", 0.0)

    pos_mean = lat.get("positive_cases_mean_ms", lat.get("positive_mean_ms", 0.0))
    neg_mean = lat.get("negative_cases_mean_ms", lat.get("negative_mean_ms", 0.0))
    all_mean = lat.get("overall_mean_ms", lat.get("guard_mean_ms", 0.0))

    opos_p95 = olat.get("positive_cases_p95_ms", olat.get("positive_p95_ms", 0.0))
    oneg_p95 = olat.get("negative_cases_p95_ms", olat.get("negative_p95_ms", 0.0))
    oall_p95 = olat.get("overall_p95_ms", 0.0)

    opos_mean = olat.get("positive_cases_mean_ms", olat.get("positive_mean_ms", 0.0))
    oneg_mean = olat.get("negative_cases_mean_ms", olat.get("negative_mean_ms", 0.0))
    oall_mean = olat.get("overall_mean_ms", 0.0)

    pos_acc_val = pos.get("accuracy", 0.0)
    neg_acc_val = neg.get("accuracy", 0.0)
    overall_acc_val = acc.get("overall_accuracy", 0.0)

    # Classification metrics for threat detection
    tp = neg.get("blocked", 0)
    fn = neg.get("allowed", 0)
    fp = pos.get("blocked", 0)
    tn = pos.get("allowed", 0)

    precision = acc.get("precision")
    if precision is None:
        precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
    precision_pct = acc.get("precision_pct", f"{precision * 100:.1f}%")

    recall = acc.get("recall") or acc.get("sensitivity")
    if recall is None:
        recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
    recall_pct = acc.get("recall_pct") or acc.get("sensitivity_pct", f"{recall * 100:.1f}%")

    f1_score = acc.get("f1_score")
    if f1_score is None:
        f1_score = (
            round(2 * (precision * recall) / (precision + recall), 4)
            if (precision + recall) > 0
            else 0.0
        )
    f1_score_pct = acc.get("f1_score_pct", f"{f1_score * 100:.1f}%")

    fpr = acc.get("fpr")
    if fpr is None:
        fpr = round(fp / (fp + tn), 4) if (fp + tn) > 0 else 0.0
    fpr_pct = acc.get("fpr_pct", f"{fpr * 100:.1f}%")

    return {
        "acc": acc,
        "pos": pos,
        "neg": neg,
        "pos_acc_val": pos_acc_val,
        "neg_acc_val": neg_acc_val,
        "overall_acc_val": overall_acc_val,
        "precision_val": precision,
        "recall_val": recall,
        "f1_val": f1_score,
        "fpr_val": fpr,
        "pos_acc_pct": pos.get("accuracy_pct", "0.0%"),
        "neg_acc_pct": neg.get("accuracy_pct", "0.0%"),
        "overall_acc_pct": acc.get("overall_accuracy_pct", "0.0%"),
        "precision_pct": precision_pct,
        "recall_pct": recall_pct,
        "f1_score_pct": f1_score_pct,
        "fpr_pct": fpr_pct,
        "pos_allowed": pos.get("allowed", 0),
        "pos_total": pos.get("total", 0),
        "neg_blocked": neg.get("blocked", 0),
        "neg_total": neg.get("total", 0),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "pos_p95": pos_p95,
        "neg_p95": neg_p95,
        "all_p95": all_p95,
        "pos_mean": pos_mean,
        "neg_mean": neg_mean,
        "all_mean": all_mean,
        "opos_p95": opos_p95,
        "oneg_p95": oneg_p95,
        "oall_p95": oall_p95,
        "opos_mean": opos_mean,
        "oneg_mean": oneg_mean,
        "oall_mean": oall_mean,
        "idle_mb": idle_mb,
        "peak_mb": peak_mb,
        "pos_mem": pos_mem,
        "neg_mem": neg_mem,
        "failed_calls": err.get("failed_calls", 0),
        "total_calls": err.get("total_calls", 0),
        "failure_rate_pct": err.get("failure_rate_pct", "0.0%"),
    }


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

        m = extract_report_metrics(final_rep)

        print(
            "  Positive Cases Acc (should NOT block): "
            f"{m['pos_acc_pct']} ({m['pos_allowed']}/{m['pos_total']})"
        )
        print(
            "  Negative Cases Acc (SHOULD block):     "
            f"{m['neg_acc_pct']} ({m['neg_blocked']}/{m['neg_total']})"
        )
        print(f"  Overall Accuracy:                      {m['overall_acc_pct']}")
        print(
            f"  Classification (Prec/Recall/F1/FPR):   "
            f"{m['precision_pct']} / {m['recall_pct']} / {m['f1_score_pct']} / {m['fpr_pct']}"
        )
        print(f"  Guard Latency P95 (pos/neg/all):       {m['pos_p95']}ms / {m['neg_p95']}ms / {m['all_p95']}ms")
        print(f"  Guard Latency Mean (pos/neg/all):      {m['pos_mean']}ms / {m['neg_mean']}ms / {m['all_mean']}ms")
        print(f"  Overall Latency P95 (pos/neg/all):     {m['opos_p95']}ms / {m['oneg_p95']}ms / {m['oall_p95']}ms")
        print(f"  Overall Latency Mean (pos/neg/all):    {m['opos_mean']}ms / {m['oneg_mean']}ms / {m['oall_mean']}ms")
        print(f"  Memory RSS (idle/pos/neg/peak):        {m['idle_mb']}MB / {m['pos_mem']}MB / {m['neg_mem']}MB / {m['peak_mb']}MB")
        print(f"  API Error Metrics (failed/total):      {m['failed_calls']}/{m['total_calls']} ({m['failure_rate_pct']})")

    print("\n" + "=" * 80 + "\n")


def print_table_rich(all_data: list[tuple[str, str, dict | None]]):
    """Rich console comparison table."""
    console = Console()

    console.print(
        Panel.fit(
            "[bold cyan]Guardrail Experiments — Final Results Comparison[/bold cyan]\n"
            "[dim]Positive cases: should NOT be blocked | Negative cases: SHOULD be blocked\n"
            "Metrics: Accuracy, Precision, Recall (Sensitivity), F1 Score, FPR | Latency & Memory[/dim]",
            border_style="cyan",
        )
    )

    table = Table(
        title="📊  Summary Report — Cross-Experiment Results",
        box=box.ROUNDED,
        header_style="bold green",
        show_lines=True,
    )
    table.add_column("Experiment", style="bold", width=22)
    table.add_column("Pos Acc\n(should NOT block)", justify="center", width=18)
    table.add_column("Neg Acc\n(SHOULD block)", justify="center", width=18)
    table.add_column("Overall Acc", justify="center", width=12)
    table.add_column("F1 | Prec | FPR\n(Threat Detection)", justify="center", width=20)
    table.add_column("Guardrail Latency\nP95 | Mean", justify="center", width=26)
    table.add_column("Peak Memory\n(pos / neg / peak)", justify="center", width=22)
    table.add_column("API Errors", justify="center", width=12)

    for exp_id, label, final_rep in all_data:
        exp_label = label.replace("\n", " — ")
        if not final_rep:
            table.add_row(exp_label, *["[dim]no final report[/dim]"] * 7)
            continue

        m = extract_report_metrics(final_rep)

        pos_col = "green" if m["pos_acc_val"] >= 0.95 else "yellow" if m["pos_acc_val"] >= 0.8 else "red"
        neg_col = "green" if m["neg_acc_val"] >= 0.8 else "yellow" if m["neg_acc_val"] >= 0.5 else "red"
        all_col = "green" if m["overall_acc_val"] >= 0.85 else "yellow" if m["overall_acc_val"] >= 0.6 else "red"
        f1_col = "green" if m["f1_val"] >= 0.85 else "yellow" if m["f1_val"] >= 0.6 else "red"

        f1_display = (
            f"[{f1_col}]F1: {m['f1_score_pct']}[/{f1_col}]\n"
            f"[dim]P: {m['precision_pct']} | FPR: {m['fpr_pct']}[/dim]"
        )

        lat_display = (
            f"P95: {m['pos_p95']} / {m['neg_p95']} / {m['all_p95']} ms\n"
            f"[dim]Mean: {m['pos_mean']} / {m['neg_mean']} / {m['all_mean']} ms[/dim]"
        )

        table.add_row(
            exp_label,
            f"[{pos_col}]{m['pos_acc_pct']}[/{pos_col}]\n[dim]({m['pos_allowed']}/{m['pos_total']})[/dim]",
            f"[{neg_col}]{m['neg_acc_pct']}[/{neg_col}]\n[dim]({m['neg_blocked']}/{m['neg_total']})[/dim]",
            f"[{all_col}]{m['overall_acc_pct']}[/{all_col}]",
            f1_display,
            lat_display,
            f"{m['pos_mem']} / {m['neg_mem']} / {m['peak_mb']} MB",
            f"{m['failed_calls']}/{m['total_calls']} ({m['failure_rate_pct']})",
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
