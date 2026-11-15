"""
NeMo Guardrails: Injectable Guardrail Implementation.

Implements BaseGuardrail by evaluating incoming prompts through
NVIDIA NeMo Guardrails (Colang policies + input rails).
"""

import os
import time
from pathlib import Path

import nest_asyncio

from impenetra.core.guardrail import BaseGuardrail, GuardrailResult

nest_asyncio.apply()

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "guardrails_config"


def resolve_main_model_api_base(explicit_url: str | None = None) -> str:
    """
    Resolve the main model OpenAI API base URL.
    Expects https://<domain> via NEMO_MAIN_MODEL_API_BASE or explicit parameter.
    Automatically appends '/v1' path by default.
    """
    raw = (explicit_url or os.getenv("NEMO_MAIN_MODEL_API_BASE") or "").strip().rstrip("/")
    if not raw:
        raise ValueError(
            "NEMO_MAIN_MODEL_API_BASE environment variable is required (e.g. 'https://xxx.trycloudflare.com')"
        )
    if raw.endswith("/v1"):
        return raw
    return f"{raw}/v1"


class NeMoGuardrail(BaseGuardrail):
    """
    Input guardrail wrapping NVIDIA NeMo Guardrails.
    Injectable into Impenetra's create_app(guardrail=...).
    """

    def __init__(
        self,
        config_path: Path | str | None = None,
        main_model_api_base: str | None = None,
    ):
        self._config_path = Path(config_path or DEFAULT_CONFIG_PATH)
        self._name = "nemo_guardrails"
        self._main_model_api_base = resolve_main_model_api_base(main_model_api_base)
        self._rails = None
        self._init_rails()

    def _init_rails(self):
        from nemoguardrails import LLMRails, RailsConfig

        config = RailsConfig.from_path(str(self._config_path))
        # Override the main model base_url with the cloudflared OpenAI endpoint
        for model in config.models:
            if getattr(model, "type", None) == "main":
                model.parameters["base_url"] = self._main_model_api_base

        self._rails = LLMRails(config)

    @property
    def name(self) -> str:
        return self._name

    @property
    def main_model_api_base(self) -> str:
        return self._main_model_api_base

    def check(self, text: str) -> GuardrailResult:
        if self._rails is None:
            self._init_rails()
        assert self._rails is not None

        start = time.perf_counter()
        try:
            from nemoguardrails.rails.llm.options import GenerationOptions

            # Evaluate input and dialog rails
            # (with single_call: false to keep intent classification fast)
            options = GenerationOptions(rails=["input", "dialog"], log={"activated_rails": True})
            response = self._rails.generate(
                messages=[{"role": "user", "content": text}],
                options=options,
            )

            if hasattr(response, "response"):
                if response.response and isinstance(response.response[0], dict):
                    content = response.response[0].get("content", str(response.response[0]))
                else:
                    content = str(response.response)
            elif isinstance(response, dict):
                content = response.get("content", str(response))
            elif isinstance(response, list) and response:
                first = response[0]
                content = (
                    first.get("content", str(first))
                    if isinstance(first, dict)
                    else str(first)
                )
            else:
                content = str(response)

            elapsed = (time.perf_counter() - start) * 1000

            # 1. Check if an input rail triggered a stop decision
            is_blocked = False
            has_activated = (
                hasattr(response, "log")
                and response.log
                and getattr(response.log, "activated_rails", None)
            )
            if has_activated:
                for rail in response.log.activated_rails:
                    if getattr(rail, "stop", False) or "stop" in getattr(rail, "decisions", []):
                        is_blocked = True
                        break

            # 2. Check for standard refusal markers if content differed from original input
            if not is_blocked and content != text:
                refusal_markers = [
                    "i'm sorry",
                    "i cannot",
                    "i can't",
                    "i am not able",
                    "i'm not able",
                    "cannot help with that",
                    "goes against my safety guidelines",
                    "cannot fulfill this request",
                    "cannot fulfill your request",
                    "adhere to safety guidelines",
                ]
                is_blocked = any(m in content.lower() for m in refusal_markers)

            if is_blocked:
                return GuardrailResult(
                    allowed=False,
                    confidence=0.98,
                    category="nemo_rail_violation",
                    details=content,
                    latency_ms=elapsed,
                )
            else:
                return GuardrailResult(
                    allowed=True,
                    confidence=1.0,
                    category="",
                    details="Passed NeMo input rails",
                    latency_ms=elapsed,
                )

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return GuardrailResult(
                allowed=True,
                confidence=0.0,
                category="nemo_error",
                details=f"NeMo evaluation error: {e}",
                latency_ms=elapsed,
            )

    def check_batch(self, texts: list[str]) -> list[GuardrailResult]:
        return [self.check(t) for t in texts]
