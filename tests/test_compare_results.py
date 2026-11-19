"""Tests for impenetra/experiments/compare_results.py."""

import json
from pathlib import Path

from impenetra.experiments.compare_results import (
    extract_report_metrics,
    find_final_report,
    print_table_plain,
    print_table_rich,
)


def test_compare_results_with_final_report(tmp_path: Path):
    """Verifies compare_results loads and displays data strictly from final_report.json."""
    exp_dir = tmp_path / "llamaguard_as_guardrail" / "results"
    exp_dir.mkdir(parents=True)

    final_report = {
        "experiment": "llamaguard_as_guardrail",
        "results_directory": str(exp_dir),
        "accuracy": {
            "overall_accuracy": 0.95,
            "overall_accuracy_pct": "95.0%",
            "positive_cases": {
                "total": 12,
                "allowed": 12,
                "blocked": 0,
                "accuracy": 1.0,
                "accuracy_pct": "100.0%",
            },
            "negative_cases": {
                "total": 23,
                "blocked": 21,
                "allowed": 2,
                "accuracy": 0.913,
                "accuracy_pct": "91.3%",
            },
        },
        "guardrail_latency": {
            "positive_cases_p95_ms": 1.2,
            "negative_cases_p95_ms": 2.5,
            "overall_p95_ms": 2.5,
            "guard_mean_ms": 1.5,
        },
        "overall_latency": {
            "positive_cases_p95_ms": 30.0,
            "negative_cases_p95_ms": 45.0,
            "overall_p95_ms": 45.0,
        },
        "memory_footprint": {
            "idle_rss_mb": 110.0,
            "peak_rss_mb": 135.0,
            "delta_rss_mb": 25.0,
            "vms_mb": 450.0,
            "positive_cases": {"peak_rss_mb": 120.0, "delta_rss_mb": 10.0},
            "negative_cases": {"peak_rss_mb": 135.0, "delta_rss_mb": 25.0},
        },
        "error_metrics": {
            "total_calls": 35,
            "successful_calls": 35,
            "failed_calls": 0,
            "failure_rate": 0.0,
            "failure_rate_pct": "0.0%",
            "errors_by_type": {},
        },
    }

    final_file = exp_dir / "final_report.json"
    final_file.write_text(json.dumps(final_report))

    loaded_final = find_final_report(tmp_path, "llamaguard_as_guardrail")
    assert loaded_final is not None
    assert loaded_final["experiment"] == "llamaguard_as_guardrail"

    all_data = [
        ("llamaguard_as_guardrail", "LlamaGuard3", loaded_final),
        ("nemo_guardrails", "NeMo Guardrails", None),
    ]

    print_table_plain(all_data)
    print_table_rich(all_data)


def test_compare_results_streamlined_schema(tmp_path: Path):
    """Verifies compare_results handles streamlined reports with positive/negative keys."""
    exp_dir = tmp_path / "streamlined_exp" / "results"
    exp_dir.mkdir(parents=True)

    final_report = {
        "experiment": "streamlined_exp",
        "results_directory": str(exp_dir),
        "accuracy": {
            "overall_accuracy": 0.96,
            "overall_accuracy_pct": "96.0%",
            "positive": {
                "total": 15,
                "allowed": 15,
                "blocked": 0,
                "accuracy": 1.0,
                "accuracy_pct": "100.0%",
            },
            "negative": {
                "total": 25,
                "blocked": 23,
                "allowed": 2,
                "accuracy": 0.92,
                "accuracy_pct": "92.0%",
            },
        },
        "guardrail_latency": {
            "positive_cases_p95_ms": 1.1,
            "negative_cases_p95_ms": 2.2,
            "overall_p95_ms": 2.2,
            "overall_mean_ms": 1.4,
        },
        "overall_latency": {
            "positive_cases_p95_ms": 28.0,
            "negative_cases_p95_ms": 40.0,
            "overall_p95_ms": 40.0,
            "overall_mean_ms": 32.0,
        },
        "memory_footprint": {
            "idle_rss_mb": 95.0,
            "peak_rss_mb": 120.0,
            "delta_rss_mb": 25.0,
            "vms_mb": 400.0,
            "positive": {"peak_rss_mb": 105.0, "delta_rss_mb": 10.0},
            "negative": {"peak_rss_mb": 120.0, "delta_rss_mb": 25.0},
        },
        "error_metrics": {
            "total_calls": 40,
            "successful_calls": 40,
            "failed_calls": 0,
            "failure_rate": 0.0,
            "failure_rate_pct": "0.0%",
            "errors_by_type": {},
        },
    }

    final_file = exp_dir / "final_report.json"
    final_file.write_text(json.dumps(final_report))

    loaded_final = find_final_report(tmp_path, "streamlined_exp")
    assert loaded_final is not None

    m = extract_report_metrics(loaded_final)
    assert m["precision_val"] == 1.0
    assert m["recall_val"] == 0.92
    assert m["f1_val"] == 0.9583
    assert m["fpr_val"] == 0.0

    all_data = [("streamlined_exp", "Streamlined Guardrail", loaded_final)]
    print_table_plain(all_data)
    print_table_rich(all_data)
