"""Tests for impenetra/experiments/benchmark.py."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from impenetra.experiments.benchmark import (
    BENIGN_PROMPTS,
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

    def fake_send(url, content, timeout=90):
        # Allow benign/food prompts, block attacks
        is_attack = "DAN" in content or "bomb" in content or "Java" in content
        return {
            "ok": True,
            "status_code": 200,
            "total_ms": 15.0,
            "guard_ms": 1.2,
            "blocked": is_attack,
            "response": "Refused" if is_attack else "Allowed response",
        }

    with patch("impenetra.experiments.benchmark.send_prompt", side_effect=fake_send):
        res = run_benchmark(
            url="http://127.0.0.1:8000",
            experiment="mock_experiment",
            out_path=str(out_file),
            verbose=False,
        )

        assert out_file.exists()
        assert res["experiment"] == "mock_experiment"
        assert res["summary"]["positive_cases"]["allowed"] == len(BENIGN_PROMPTS)
        assert res["summary"]["positive_cases"]["blocked"] == 0
        assert res["summary"]["positive_cases"]["accuracy"] == 1.0
        assert res["summary"]["negative_cases"]["accuracy"] > 0.0

        with open(out_file) as f:
            saved_data = json.load(f)
            assert saved_data["experiment"] == "mock_experiment"
            assert "attempts" in saved_data
            assert len(saved_data["attempts"]) > 0
