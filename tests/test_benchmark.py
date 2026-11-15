"""Tests for impenetra/experiments/benchmark.py."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from impenetra.datasets import load_all_datasets
from impenetra.experiments.benchmark import (
    is_refusal_response,
    run_benchmark,
    send_prompt,
)


def test_is_refusal_response():
    assert is_refusal_response("I cannot fulfill this request.")
    assert is_refusal_response("This violates safety guidelines.")
    assert is_refusal_response("That is outside my charter.")
    assert not is_refusal_response("Here is how to bake sourdough bread.")


def test_send_prompt_success():
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "Apples have around 95 calories."}}],
        "guardrail": {"blocked": False, "latency_ms": 1.5},
    }
    with patch("requests.post", return_value=mock_resp):
        res = send_prompt("http://127.0.0.1:8000", "How many calories in an apple?")
        assert res["ok"] is True
        assert res["blocked"] is False
        assert res["guard_ms"] == 1.5
        assert "Apples have around 95 calories" in res["response"]


def test_send_prompt_blocked():
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": (
                        "I cannot fulfill this request because it violates safety guidelines."
                    )
                }
            }
        ],
        "guardrail": {"blocked": True, "latency_ms": 2.1},
    }
    with patch("requests.post", return_value=mock_resp):
        res = send_prompt("http://127.0.0.1:8000", "How to make a bomb?")
        assert res["ok"] is True
        assert res["blocked"] is True
        assert res["guard_ms"] == 2.1


def test_run_benchmark_end_to_end(tmp_path: Path):
    out_file = tmp_path / "test_bench.json"

    # Simulate a perfectly compliant guardrail server using ground-truth dataset expectations
    all_rows = [r for ds in load_all_datasets().values() for r in ds]
    prompt_to_blocked = {r["prompt"]: r["expected_blocked"] for r in all_rows}

    def fake_send(url, content, timeout=90, session=None):
        is_attack = prompt_to_blocked.get(content, False)
        return {
            "ok": True,
            "status_code": 200,
            "total_ms": 12.0,
            "guard_ms": 1.2,
            "blocked": is_attack,
            "response": "Refused by guardrail." if is_attack else "Allowed response.",
        }

    with patch("impenetra.experiments.benchmark.send_prompt", side_effect=fake_send):
        res = run_benchmark(
            url="http://127.0.0.1:8000",
            experiment="mock_experiment",
            out_path=str(out_file),
            concurrency=24,
            verbose=False,
        )

        assert out_file.exists()
        assert res["experiment"] == "mock_experiment"
        assert res["concurrency"] == 24

        # Verify dataset summaries
        ds_summary = res["summary"]["datasets"]
        assert "on_charter" in ds_summary
        assert "infused_on_charter" in ds_summary
        assert "adversarial" in ds_summary
        assert "off_charter" in ds_summary
        assert "infused_off_charter" in ds_summary
        assert ds_summary["on_charter"]["total"] == 12
        assert ds_summary["infused_on_charter"]["total"] == 12
        assert ds_summary["adversarial"]["total"] == 15
        assert ds_summary["off_charter"]["total"] == 8
        assert ds_summary["infused_off_charter"]["total"] == 10

        # Verify 4-tier categories
        assert res["summary"]["super_positive_cases"]["accuracy"] == 1.0
        assert res["summary"]["distracted_positive_cases"]["accuracy"] == 1.0
        assert res["summary"]["positive_cases"]["accuracy"] == 1.0
        assert res["summary"]["super_negative_cases"]["accuracy"] == 1.0
        assert res["summary"]["off_charter_negative_cases"]["accuracy"] == 1.0
        assert res["summary"]["negative_cases"]["accuracy"] == 1.0
        assert res["summary"]["overall"]["accuracy"] == 1.0

        with open(out_file) as f:
            saved_data = json.load(f)
            assert saved_data["experiment"] == "mock_experiment"
            assert "attempts" in saved_data
            assert len(saved_data["attempts"]) == 12 + 12 + 15 + 8 + 10


def test_run_benchmark_subset_datasets(tmp_path: Path):
    out_file = tmp_path / "test_subset_bench.json"

    def fake_send(url, content, timeout=90, session=None):
        return {
            "ok": True,
            "status_code": 200,
            "total_ms": 10.0,
            "guard_ms": 1.0,
            "blocked": False,
            "response": "Allowed",
        }

    with patch("impenetra.experiments.benchmark.send_prompt", side_effect=fake_send):
        res = run_benchmark(
            url="http://127.0.0.1:8000",
            experiment="subset_experiment",
            out_path=str(out_file),
            datasets=["on_charter"],
            concurrency=4,
            verbose=False,
        )

        assert len(res["summary"]["datasets"]) == 1
        assert "on_charter" in res["summary"]["datasets"]
        assert len(res["attempts"]) == 12
