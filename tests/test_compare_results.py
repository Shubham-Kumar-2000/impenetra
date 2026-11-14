"""Tests for impenetra/experiments/compare_results.py."""

import json
from pathlib import Path

from impenetra.experiments.compare_results import (
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
