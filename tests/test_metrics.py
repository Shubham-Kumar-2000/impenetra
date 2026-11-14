"""Tests for impenetra/experiments/metrics.py."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from impenetra.experiments.metrics import (
    fetch_server_metrics,
    parse_garak_attempts,
    record_run_metrics,
    reset_server_metrics,
)


def test_reset_server_metrics_success():
    with patch("requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200)
        assert reset_server_metrics("http://127.0.0.1:8000") is True
        mock_post.assert_called_once_with("http://127.0.0.1:8000/metrics/reset", timeout=10.0)


def test_reset_server_metrics_failure():
    with patch("requests.post") as mock_post:
        mock_post.side_effect = Exception("Connection refused")
        assert reset_server_metrics("http://127.0.0.1:8000") is False


def test_fetch_server_metrics():
    sample_metrics = {
        "error_metrics": {
            "total_calls": 10,
            "successful_calls": 10,
            "failed_calls": 0,
            "failure_rate": 0.0,
        },
        "guardrail_latency_ms": {"p50": 1.2, "p95": 2.4, "p99": 3.0, "mean": 1.5},
        "overall_latency_ms": {"p50": 30.0, "p95": 50.0, "p99": 60.0, "mean": 35.0},
        "memory_footprint": {
            "idle_rss_mb": 100.0,
            "peak_rss_mb": 120.0,
            "delta_rss_mb": 20.0,
            "vms_mb": 300.0,
        },
    }
    with patch("requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200, json=lambda: sample_metrics)
        res = fetch_server_metrics("http://127.0.0.1:8000")
        assert res["error_metrics"]["total_calls"] == 10
        assert res["guardrail_latency_ms"]["p95"] == 2.4
        assert res["memory_footprint"]["peak_rss_mb"] == 120.0


def test_parse_garak_attempts(tmp_path: Path):
    jsonl = tmp_path / "test.report.jsonl"
    with open(jsonl, "w") as f:
        f.write(
            json.dumps(
                {
                    "entry_type": "eval",
                    "probe": "dan.DanInTheWild",
                    "passed": 8,
                    "fails": 2,
                    "total_evaluated": 10,
                }
            )
            + "\n"
        )
        f.write(json.dumps({"entry_type": "attempt", "probe": "dan.DanInTheWild"}) + "\n")
        f.write(json.dumps({"entry_type": "attempt", "probe": "dan.DanInTheWild"}) + "\n")

    res = parse_garak_attempts(jsonl)
    assert res["total_attempts"] == 2
    assert "dan.DanInTheWild" in res["probes_evaluated"]
    assert res["probes_evaluated"]["dan.DanInTheWild"]["passed"] == 8


def test_record_run_metrics(tmp_path: Path):
    sample_metrics = {
        "error_metrics": {"total_calls": 5, "failed_calls": 0, "failure_rate": 0.0},
        "guardrail_latency_ms": {"p50": 1.0, "p95": 2.0},
        "overall_latency_ms": {"p50": 20.0, "p95": 40.0},
        "memory_footprint": {"idle_rss_mb": 100.0, "peak_rss_mb": 115.0},
    }
    with patch("requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200, json=lambda: sample_metrics)
        out_file = record_run_metrics(
            experiment="mock_exp",
            run_type="bench",
            out_dir=tmp_path,
            timestamp="20261114_120000",
        )
        assert out_file.exists()
        assert out_file.name == "20261114_120000_bench_report.json"
        with open(out_file) as f:
            data = json.load(f)
            assert data["experiment"] == "mock_exp"
            assert data["run_type"] == "bench"
            assert data["memory_footprint"]["peak_rss_mb"] == 115.0
