"""
PromptGuard-as-Guardrail: Meta Llama Prompt Guard 2 86M API Integration.

Implements BaseGuardrail by evaluating incoming prompts against
Meta's Llama-Prompt-Guard-2-86M API server exposed by
prompt_guard_colab.ipynb (POST /predict).
"""

import datetime
import json
import os
import time
from pathlib import Path

import requests

from impenetra.core.guardrail import BaseGuardrail, GuardrailResult

# Threshold to classify prompt as malicious (defined at top of file per requirement)
MALICIOUS_THRESHOLD = 0.8

# Default URL expected in environment variable (PROMPT_GUARD_URL)
DEFAULT_PROMPT_GUARD_URL = os.getenv("PROMPT_GUARD_URL", "http://127.0.0.1:8000")

DEFAULT_FAIL_OPEN_LOG_PATH = Path(__file__).resolve().parent / "results" / "fail_open.log"


class PromptGuardCallError(RuntimeError):
    """Raised when Prompt Guard HTTP endpoint fails, preserving server response details."""

    def __init__(
        self,
        message: str,
        server_response: str | None = None,
        status_code: int | None = None,
    ):
        super().__init__(message)
        self.server_response = server_response
        self.status_code = status_code


class PromptGuardGuardrail(BaseGuardrail):
    """
    Input guardrail using Meta's Llama-Prompt-Guard-2-86M via its /predict API.
    Injectable into Impenetra's create_app(guardrail=...).
    """

    def __init__(
        self,
        endpoint_url: str | None = None,
        threshold: float = MALICIOUS_THRESHOLD,
        name: str = "promptguard",
        timeout: float = 30.0,
        fail_open_log_path: str | Path | None = None,
    ):
        raw_url = endpoint_url or os.getenv("PROMPT_GUARD_URL") or DEFAULT_PROMPT_GUARD_URL
        self._raw_url = raw_url.strip()
        self._threshold = float(threshold)
        self._name = name
        self._timeout = timeout

        # Normalize predict endpoint URL
        base = self._raw_url.rstrip("/")
        if base.endswith("/predict"):
            self._predict_url = base
        else:
            self._predict_url = f"{base}/predict"

        log_env = os.getenv("PROMPT_GUARD_FAIL_OPEN_LOG")
        self._fail_open_log_path = Path(fail_open_log_path or log_env or DEFAULT_FAIL_OPEN_LOG_PATH)

    @property
    def name(self) -> str:
        return self._name

    @property
    def endpoint_url(self) -> str:
        return self._raw_url

    @property
    def predict_url(self) -> str:
        return self._predict_url

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def fail_open_log_path(self) -> Path:
        return self._fail_open_log_path

    def _log_fail_open(
        self,
        prompt: str,
        reason: str,
        server_response: str | None = None,
        latency_ms: float = 0.0,
    ) -> None:
        """
        Logs a fail-open event to the results directory.
        Records the timestamp, prompt, failure reason, and raw server response.
        """
        try:
            self._fail_open_log_path.parent.mkdir(parents=True, exist_ok=True)
            iso_timestamp = datetime.datetime.now().astimezone().isoformat()
            resp_str = str(server_response).strip() if server_response else "None"

            # 1. Human-readable text log block
            entry = (
                f"[{iso_timestamp}] FAIL-OPEN EVENT\n"
                f"Target URL: {self._predict_url}\n"
                f"Latency: {latency_ms:.2f}ms\n"
                f"Prompt: {prompt.strip()}\n"
                f"Reason: {reason.strip()}\n"
                f"Server Response:\n{resp_str}\n"
                f"{'-' * 80}\n"
            )
            with open(self._fail_open_log_path, "a", encoding="utf-8") as f:
                f.write(entry)

            # 2. Machine-readable JSONL companion file (.jsonl)
            jsonl_path = self._fail_open_log_path.with_suffix(".jsonl")
            record = {
                "timestamp": iso_timestamp,
                "target_url": self._predict_url,
                "latency_ms": round(latency_ms, 2),
                "prompt": prompt,
                "reason": reason,
                "server_response": resp_str,
            }
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as log_err:
            # Logging failures should never crash the service
            print(f"[PromptGuardGuardrail] Failed to write fail-open log: {log_err}")

    def _call_predict_api(self, text: str) -> dict[str, float]:
        """
        Invokes Prompt Guard POST /predict API with {"prompt": text}.
        Expected response format: {"benign": float, "malicious": float}
        """
        try:
            resp = requests.post(
                self._predict_url,
                json={"prompt": text},
                timeout=self._timeout,
            )
        except Exception as conn_err:
            raise PromptGuardCallError(
                f"Failed to connect to {self._predict_url}: {conn_err}",
                server_response=f"Connection error: {conn_err}",
            ) from conn_err

        if resp.status_code != 200:
            raise PromptGuardCallError(
                f"HTTP {resp.status_code} from {self._predict_url}: {resp.text[:500]}",
                server_response=f"HTTP {resp.status_code}: {resp.text}",
                status_code=resp.status_code,
            )

        try:
            data = resp.json()
        except Exception as json_err:
            raise PromptGuardCallError(
                f"Invalid JSON returned from {self._predict_url}: {json_err}",
                server_response=resp.text,
                status_code=resp.status_code,
            ) from json_err

        if not isinstance(data, dict) or "malicious" not in data:
            raise PromptGuardCallError(
                f"Missing 'malicious' in Prompt Guard response: {data}",
                server_response=resp.text,
                status_code=resp.status_code,
            )

        return data

    def check(self, text: str) -> GuardrailResult:
        """
        Check if input text is malicious according to Prompt Guard API.
        Evaluates malicious confidence against MALICIOUS_THRESHOLD (0.8).
        """
        start = time.perf_counter()
        try:
            scores = self._call_predict_api(text)
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000

            reason = str(e)
            server_response = getattr(e, "server_response", None)
            if not server_response:
                resp = getattr(e, "response", None)
                if resp is not None:
                    status = getattr(resp, "status_code", "unknown")
                    resp_text = getattr(resp, "text", "")
                    server_response = f"HTTP {status}: {resp_text}"
                else:
                    server_response = (
                        "No response received (connection error, network down, or timeout)"
                    )

            # Log fail-open event to the results directory
            self._log_fail_open(
                prompt=text,
                reason=reason,
                server_response=server_response,
                latency_ms=elapsed,
            )

            # Fail-open with warning detail
            return GuardrailResult(
                allowed=True,
                confidence=0.0,
                category="guardrail_error",
                details=f"Prompt Guard error: {e}",
                latency_ms=elapsed,
            )

        elapsed = (time.perf_counter() - start) * 1000
        malicious_score = float(scores.get("malicious", 0.0))
        benign_score = float(scores.get("benign", 1.0 - malicious_score))

        is_malicious = malicious_score >= self._threshold

        if is_malicious:
            category = "malicious"
            desc = (
                f"Prompt Guard 2 verdict: malicious "
                f"(score: {malicious_score:.4f} >= threshold: {self._threshold:.2f})"
            )
            return GuardrailResult(
                allowed=False,
                confidence=malicious_score,
                category=category,
                details=desc,
                latency_ms=elapsed,
            )
        else:
            category = ""
            desc = (
                f"Prompt Guard 2 verdict: safe "
                f"(malicious: {malicious_score:.4f} < threshold: {self._threshold:.2f})"
            )
            return GuardrailResult(
                allowed=True,
                confidence=benign_score,
                category=category,
                details=desc,
                latency_ms=elapsed,
            )

    def check_batch(self, texts: list[str]) -> list[GuardrailResult]:
        return [self.check(t) for t in texts]
