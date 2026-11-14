"""
LlamaGuard-as-Guardrail: Gradio Space Integration.

Implements BaseGuardrail by evaluating incoming prompts against
Meta's LlamaGuard-3 8B hosted on a Gradio Space ("llama-guard-space" or custom Gradio URL).
"""

import datetime
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Tuple

import requests

from impenetra.core.guardrail import BaseGuardrail, GuardrailResult
from impenetra.experiments.llamaguard_as_guardrail.prompt_format import (
    CATEGORY_MAP,
    format_llamaguard_prompt,
)
from gradio_client import Client 

DEFAULT_GRADIO_URL = os.getenv("LLAMA_GUARD_GRADIO_URL", "http://127.0.0.1:7860")
DEFAULT_FAIL_OPEN_LOG_PATH = Path(__file__).resolve().parent / "results" / "fail_open.log"


class GradioCallError(RuntimeError):
    """Raised when Gradio Space / HTTP endpoint fails, preserving server response details."""

    def __init__(
        self,
        message: str,
        server_response: str | None = None,
        status_code: int | None = None,
    ):
        super().__init__(message)
        self.server_response = server_response
        self.status_code = status_code


class LlamaGuardGuardrail(BaseGuardrail):
    """
    Input guardrail using Meta's LlamaGuard-3 via a Gradio Space (llama-guard-space).
    Injectable into Impenetra's create_app(guardrail=...).
    """

    def __init__(
        self,
        gradio_url: str | None = None,
        name: str = "llamaguard_gradio_space",
        timeout: float = 30.0,
        fail_open_log_path: str | Path | None = None,
    ):
        raw_url = gradio_url or DEFAULT_GRADIO_URL
        self._gradio_url = raw_url.strip()
        self._name = name
        self._timeout = timeout
        self._client: Any = None
        self._client_init_attempted = False
        log_env = os.getenv("LLAMA_GUARD_FAIL_OPEN_LOG")
        self._fail_open_log_path = Path(fail_open_log_path or log_env or DEFAULT_FAIL_OPEN_LOG_PATH)

    @property
    def name(self) -> str:
        return self._name

    @property
    def gradio_url(self) -> str:
        return self._gradio_url

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
                f"Target URL: {self._gradio_url}\n"
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
                "target_url": self._gradio_url,
                "latency_ms": round(latency_ms, 2),
                "prompt": prompt,
                "reason": reason,
                "server_response": resp_str,
            }
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as log_err:
            # Logging failures should never crash the service
            print(f"[LlamaGuardGuardrail] Failed to write fail-open log: {log_err}")

    def _get_client(self) -> Any:
        """Lazily initialize the Gradio Client."""
        if not self._client_init_attempted:
            self._client_init_attempted = True
            try:
                self._client = Client(self._gradio_url)
            except Exception:
                self._client = None
        return self._client

    def _parse_gradio_output(self, out: Any) -> Tuple[bool, str, str]:
        """
        Parses output returned by LlamaGuard Gradio app.
        Expected format: (safe: bool, security_code: str, category_details: str)
        """
        if isinstance(out, (list, tuple)):
            if len(out) >= 3:
                return bool(out[0]), str(out[1] or ""), str(out[2] or "")
            elif len(out) == 2:
                return bool(out[0]), str(out[1] or ""), ""
            elif len(out) == 1:
                val = out[0]
                if isinstance(val, bool):
                    return val, "", ""
                return self._parse_string_verdict(str(val))

        if isinstance(out, dict):
            is_safe = bool(out.get("safe", True))
            code = str(out.get("security_code", ""))
            details = str(out.get("category_details", ""))
            return is_safe, code, details

        return self._parse_string_verdict(str(out))

    def _parse_string_verdict(self, raw: str) -> Tuple[bool, str, str]:
        """Fallback parser for raw string output."""
        cleaned = raw.strip()
        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        if not lines:
            return True, "", ""

        first_line = lines[0].lower()
        is_safe = ("safe" in first_line) and ("unsafe" not in first_line)
        if is_safe:
            return True, "", ""

        codes = []
        for line in lines:
            found = re.findall(r"\bS\d+\b", line, re.IGNORECASE)
            for c in found:
                c_upper = c.upper()
                if c_upper not in codes:
                    codes.append(c_upper)

        code_str = ", ".join(codes) if codes else "UNSAFE"
        details = "; ".join(lines[1:]) if len(lines) > 1 else f"Violations: {code_str}"
        return False, code_str, details

    def _call_http_api(self, text: str) -> Tuple[bool, str, str]:
        """Direct HTTP API invocation fallback for Gradio endpoints."""
        if not (self._gradio_url.startswith("http://") or self._gradio_url.startswith("https://")):
            raise ValueError(
                f"Cannot perform direct HTTP request to space name: {self._gradio_url}"
            )

        base = self._gradio_url.rstrip("/")
        candidates = [
            f"{base}/api/predict",
            f"{base}/run/predict",
            f"{base}/gradio_api/call/moderate_prompt",
        ]

        last_error = None
        last_status_code = None
        last_resp_text = None

        for endpoint in candidates:
            try:
                resp = requests.post(
                    endpoint,
                    json={"data": [text]},
                    timeout=self._timeout,
                )
                if resp.status_code == 200:
                    data = resp.json().get("data", [])
                    return self._parse_gradio_output(data)
                else:
                    last_status_code = resp.status_code
                    last_resp_text = resp.text
                    last_error = f"HTTP {resp.status_code} from {endpoint}: {resp.text[:500]}"
            except Exception as e:
                last_error = f"{type(e).__name__} connecting to {endpoint}: {e}"
                continue

        server_resp_info = (
            f"HTTP {last_status_code}: {last_resp_text}"
            if last_status_code is not None
            else "No response received (connection error, network down, or timeout)"
        )
        raise GradioCallError(
            f"All Gradio HTTP API endpoints failed: {last_error}",
            server_response=server_resp_info,
            status_code=last_status_code,
        )

    def _call_gradio(self, text: str) -> Tuple[bool, str, str]:
        """Invoke Gradio endpoint via Client or HTTP fallback."""
        client_error = None
        client = self._get_client()
        if client is not None:
            try:
                out = client.predict(prompt=text, api_name="/moderate_prompt")
                return self._parse_gradio_output(out)
            except Exception:
                try:
                    out = client.predict(text, api_name="/moderate_prompt")
                    return self._parse_gradio_output(out)
                except Exception:
                    try:
                        out = client.predict(text)
                        return self._parse_gradio_output(out)
                    except Exception as e3:
                        client_error = f"Gradio client error: {e3}"

        try:
            return self._call_http_api(text)
        except GradioCallError:
            raise
        except Exception as e:
            raise GradioCallError(
                f"HTTP API fallback failed ({client_error or 'no client'}): {e}",
                server_response=str(e),
            )

    def check(self, text: str) -> GuardrailResult:
        start = time.perf_counter()
        formatted_prompt = format_llamaguard_prompt(text)
        try:
            is_safe, sec_code, details = self._call_gradio(formatted_prompt)
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
                details=f"LlamaGuard Gradio error: {e}",
                latency_ms=elapsed,
            )

        elapsed = (time.perf_counter() - start) * 1000
        category = "" if is_safe else (sec_code or "unsafe")
        if not details and not is_safe and sec_code in CATEGORY_MAP:
            desc = f"{sec_code}: {CATEGORY_MAP[sec_code]}"
        elif details:
            desc = details
        else:
            desc = "LlamaGuard verdict: safe" if is_safe else f"LlamaGuard verdict: {category}"

        return GuardrailResult(
            allowed=is_safe,
            confidence=0.95 if is_safe else 0.99,
            category=category,
            details=desc,
            latency_ms=elapsed,
        )

    def check_batch(self, texts: list[str]) -> list[GuardrailResult]:
        return [self.check(t) for t in texts]
