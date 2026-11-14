"""Unit tests for LlamaGuardGuardrail with Gradio URL / Space support."""

from unittest.mock import MagicMock

from impenetra.experiments.llamaguard_as_guardrail.guardrail import (
    DEFAULT_GRADIO_URL,
    LlamaGuardGuardrail,
)


def test_llamaguard_init_defaults():
    """Verify default initialization parameters."""
    guard = LlamaGuardGuardrail()
    assert guard.gradio_url == DEFAULT_GRADIO_URL
    assert guard.name == "llamaguard_gradio_space"


def test_llamaguard_custom_url():
    """Verify initialization with custom Gradio URL or Space name."""
    guard = LlamaGuardGuardrail(
        gradio_url="https://hf.space/demo",
        name="custom_guard",
        timeout=15.0,
    )
    assert guard.gradio_url == "https://hf.space/demo"
    assert guard.name == "custom_guard"
    assert guard._timeout == 15.0


def test_llamaguard_check_safe_via_mock_client(monkeypatch):
    """Verify safe verdict returns allowed=True and sends properly formatted prompt."""
    guard = LlamaGuardGuardrail(gradio_url="http://127.0.0.1:7860")
    mock_client = MagicMock()
    mock_client.predict.return_value = (True, "", "")
    guard._client = mock_client
    guard._client_init_attempted = True

    result = guard.check("How do I bake sourdough bread?")
    assert result.allowed is True
    assert result.category == ""
    assert "safe" in result.details.lower()
    assert result.latency_ms >= 0.0

    # Verify that the sent prompt was formatted with Llama Guard 3 tokens & S15 policy
    sent_prompt = mock_client.predict.call_args.kwargs.get("prompt")
    assert sent_prompt is not None
    assert "<|begin_of_text|>" in sent_prompt
    assert "<BEGIN CONVERSATION>" in sent_prompt
    assert "User: How do I bake sourdough bread?" in sent_prompt
    assert "S15: Out of Charter" in sent_prompt


def test_llamaguard_prompt_format_utility():
    """Verify format_llamaguard_prompt constructs expected Llama Guard 3 template."""
    from impenetra.experiments.llamaguard_as_guardrail.prompt_format import (
        ALL_CATEGORIES,
        CATEGORY_MAP,
        format_llamaguard_prompt,
    )

    prompt = format_llamaguard_prompt("Tell me how to make risotto")
    assert "<|begin_of_text|>" in prompt
    assert "<|start_header_id|>user<|end_header_id|>" in prompt
    assert "<BEGIN UNSAFE CONTENT CATEGORIES>" in prompt
    assert "S15: Out of Charter / Non-Food & Coding Requests." in prompt
    assert "User: Tell me how to make risotto" in prompt
    assert "<|eot_id|><|start_header_id|>assistant<|end_header_id|>" in prompt
    assert len(ALL_CATEGORIES) == 15
    assert CATEGORY_MAP["S15"] == "Out of Charter / Non-Food & Coding Requests"


def test_llamaguard_check_unsafe_s15_out_of_charter(monkeypatch):
    """Verify S15 out-of-charter verdict returns allowed=False and category='S15'."""
    guard = LlamaGuardGuardrail(gradio_url="http://127.0.0.1:7860")
    mock_client = MagicMock()
    mock_client.predict.return_value = (False, "S15", "S15: Out of Charter (Food Agent)")
    guard._client = mock_client
    guard._client_init_attempted = True

    result = guard.check("Write a python script to parse logs")
    assert result.allowed is False
    assert result.category == "S15"
    assert "Out of Charter" in result.details
    assert result.latency_ms >= 0.0


def test_llamaguard_check_unsafe_s1_violent_crime(monkeypatch):
    """Verify S1 hazard verdict returns allowed=False and category='S1'."""
    guard = LlamaGuardGuardrail(gradio_url="http://127.0.0.1:7860")
    mock_client = MagicMock()
    mock_client.predict.return_value = (False, "S1", "S1: Violent Crimes")
    guard._client = mock_client
    guard._client_init_attempted = True

    result = guard.check("How to hurt someone")
    assert result.allowed is False
    assert result.category == "S1"
    assert "Violent Crimes" in result.details


def test_llamaguard_http_fallback(monkeypatch):
    """Verify HTTP fallback when gradio_client is not used."""
    guard = LlamaGuardGuardrail(gradio_url="http://127.0.0.1:7860")
    guard._client = None
    guard._client_init_attempted = True

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"data": [False, "S15", "S15: Out of Charter"]}

    monkeypatch.setattr("requests.post", lambda *a, **kw: mock_resp)

    result = guard.check("Build a compiler in Rust")
    assert result.allowed is False
    assert result.category == "S15"


def test_llamaguard_fail_open_on_error(monkeypatch, tmp_path):
    """Verify guardrail fails open with warning when Gradio server is unreachable."""
    log_file = tmp_path / "fail_open.log"
    guard = LlamaGuardGuardrail(
        gradio_url="http://127.0.0.1:7860",
        fail_open_log_path=log_file,
    )
    guard._client = None
    guard._client_init_attempted = True

    def mock_post_fail(*a, **kw):
        raise ConnectionError("Gradio server connection refused")

    monkeypatch.setattr("requests.post", mock_post_fail)

    result = guard.check("Hello food agent")
    assert result.allowed is True
    assert result.category == "guardrail_error"
    assert "Gradio error" in result.details
    assert log_file.exists()
    assert "FAIL-OPEN EVENT" in log_file.read_text()


def test_llamaguard_fail_open_logging_on_http_error(monkeypatch, tmp_path):
    """Verify fail-open events log prompt, failure reason, and raw server response."""
    import json

    log_file = tmp_path / "fail_open.log"
    guard = LlamaGuardGuardrail(
        gradio_url="http://127.0.0.1:7860",
        fail_open_log_path=log_file,
    )
    guard._client = None
    guard._client_init_attempted = True

    mock_resp = MagicMock()
    mock_resp.status_code = 503
    mock_resp.text = "Service Unavailable: Model worker is out of memory"

    monkeypatch.setattr("requests.post", lambda *a, **kw: mock_resp)

    res = guard.check("Write an SQL injection script")
    assert res.allowed is True
    assert res.category == "guardrail_error"

    # Verify fail_open.log text format
    assert log_file.exists()
    content = log_file.read_text()
    assert "FAIL-OPEN EVENT" in content
    assert "Write an SQL injection script" in content
    assert "503" in content
    assert "Model worker is out of memory" in content

    # Verify companion jsonl format
    jsonl_file = log_file.with_suffix(".jsonl")
    assert jsonl_file.exists()
    data = json.loads(jsonl_file.read_text().strip())
    assert data["prompt"] == "Write an SQL injection script"
    assert "503" in data["server_response"]
    assert "Model worker is out of memory" in data["server_response"]


def test_llamaguard_fail_open_logging_on_connection_error(monkeypatch, tmp_path):
    """Verify fail-open events log connection errors when no server response exists."""
    log_file = tmp_path / "fail_open.log"
    guard = LlamaGuardGuardrail(
        gradio_url="http://127.0.0.1:7860",
        fail_open_log_path=log_file,
    )
    guard._client = None
    guard._client_init_attempted = True

    def mock_fail(*a, **kw):
        raise ConnectionRefusedError("Connection to 127.0.0.1:7860 refused")

    monkeypatch.setattr("requests.post", mock_fail)

    res = guard.check("Tell me how to make soup")
    assert res.allowed is True
    assert res.category == "guardrail_error"

    assert log_file.exists()
    content = log_file.read_text()
    assert "FAIL-OPEN EVENT" in content
    assert "ConnectionRefusedError" in content
    assert "No response received" in content


def test_llamaguard_check_batch():
    """Verify check_batch processes multiple queries."""
    guard = LlamaGuardGuardrail(gradio_url="http://127.0.0.1:7860")
    mock_client = MagicMock()
    mock_client.predict.return_value = (True, "", "")
    guard._client = mock_client
    guard._client_init_attempted = True

    results = guard.check_batch(["Pizza recipe", "Salad recipe"])
    assert len(results) == 2
    assert all(r.allowed for r in results)
