"""
impenetra/experiments/metrics.py
================================
Utilities for querying, resetting, and recording telemetry metrics from the
running experiment server (/metrics).

Captures:
- errorMetric (total calls, failed calls, failure rate, failure breakdown)
- memory footprint (idle RSS, peak RSS, delta RSS, VMS)
- latency footprint of guardrail function (p50, p95, p99, mean)
- overall latency footprint (p50, p95, p99, mean)
- Garak probe verdicts when available

Used by run.sh scripts during experiment execution and finally compiled
by compile_results.py.
"""

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Any

import requests


def reset_server_metrics(server_url: str = "http://127.0.0.1:8000", timeout: float = 10.0) -> bool:
    """Call server /metrics/reset endpoint before starting a new run."""
    try:
        resp = requests.post(f"{server_url.rstrip('/')}/metrics/reset", timeout=timeout)
        return resp.status_code == 200
    except Exception as e:
        print(f"⚠️ Warning: Could not reset server metrics at {server_url}: {e}")
        return False


def fetch_server_metrics(server_url: str = "http://127.0.0.1:8000", timeout: float = 10.0) -> dict[str, Any]:
    """Fetch current telemetry and memory footprint from the server /metrics endpoint."""
    try:
        resp = requests.get(f"{server_url.rstrip('/')}/metrics", timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"⚠️ Warning: Could not fetch metrics from {server_url}: {e}")

    # Fallback structure if server is not reachable
    return {
        "error_metrics": {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "failure_rate": 0.0,
            "errors_by_type": {},
        },
        "guardrail_latency_ms": {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0},
        "overall_latency_ms": {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0},
        "memory_footprint": {"idle_rss_mb": 0.0, "peak_rss_mb": 0.0, "delta_rss_mb": 0.0, "vms_mb": 0.0},
        "records": [],
    }


def parse_garak_attempts(report_jsonl_path: Path | str) -> dict[str, Any]:
    """Parse Garak JSONL report for attempt counts and probe verdicts."""
    path = Path(report_jsonl_path)
    if not path.exists():
        return {}

    total_attempts = 0
    probes_summary: dict[str, dict[str, int]] = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            entry_type = entry.get("entry_type")
            if entry_type == "eval":
                probe = entry.get("probe", "unknown")
                if probe not in probes_summary:
                    probes_summary[probe] = {"passed": 0, "fails": 0, "total": 0}
                probes_summary[probe]["passed"] += entry.get("passed", 0)
                probes_summary[probe]["fails"] += entry.get("fails", 0)
                probes_summary[probe]["total"] += entry.get("total_evaluated", 0)
            elif entry_type == "attempt":
                total_attempts += 1

    return {
        "total_attempts": total_attempts,
        "probes_evaluated": probes_summary,
    }


def record_run_metrics(
    experiment: str,
    run_type: str,
    out_dir: str | Path,
    server_url: str = "http://127.0.0.1:8000",
    report_jsonl_path: str | Path | None = None,
    timestamp: str | None = None,
) -> Path:
    """
    Queries server /metrics and records <timestamp>_<run_type>_report.json
    into the experiment's results directory.
    """
    ts = timestamp or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(out_dir) / f"{ts}_{run_type}_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = fetch_server_metrics(server_url)

    report_data: dict[str, Any] = {
        "experiment": experiment,
        "run_type": run_type,
        "timestamp": ts,
        "server_url": server_url,
        "error_metrics": metrics.get("error_metrics", {}),
        "memory_footprint": metrics.get("memory_footprint", {}),
        "guardrail_latency_ms": metrics.get("guardrail_latency_ms", {}),
        "overall_latency_ms": metrics.get("overall_latency_ms", {}),
    }

    if report_jsonl_path:
        garak_info = parse_garak_attempts(report_jsonl_path)
        if garak_info:
            report_data["garak_evaluation"] = garak_info

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"✓ Recorded {run_type} metrics report → {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Impenetra Experiment Metrics Utility")
    parser.add_argument(
        "action",
        nargs="?",
        choices=["record", "report", "reset", "snapshot"],
        default="record",
        help="Action to perform: 'reset' to clear server metrics, 'record'/'report' to save report",
    )
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="FastAPI server URL")
    parser.add_argument("--experiment", default="experiment", help="Experiment name")
    parser.add_argument("--run-type", default="bench", choices=["bench", "garak", "custom"], help="Run type identifier")
    parser.add_argument("--out-dir", default=".", help="Directory to save report in")
    parser.add_argument("--timestamp", default=None, help="Timestamp string prefix")
    parser.add_argument("--report-jsonl", default=None, help="Path to Garak report.jsonl")

    args = parser.parse_args()

    if args.action == "reset":
        ok = reset_server_metrics(args.url)
        if ok:
            print(f"✓ Successfully reset metrics at {args.url}")
        else:
            print(f"⚠️ Failed to reset metrics at {args.url}")
            sys.exit(1)
    else:
        out_path = record_run_metrics(
            experiment=args.experiment,
            run_type=args.run_type,
            out_dir=args.out_dir,
            server_url=args.url,
            report_jsonl_path=args.report_jsonl,
            timestamp=args.timestamp,
        )
        print(f"Report written to: {out_path}")


if __name__ == "__main__":
    main()
