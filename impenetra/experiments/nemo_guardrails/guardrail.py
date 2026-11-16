"""
NeMo Guardrails: Injectable Guardrail Implementation.

Implements BaseGuardrail by evaluating incoming prompts through
NVIDIA NeMo Guardrails (Colang policies + input rails).
"""

import contextvars
import os
import time
from pathlib import Path
from typing import Any

import nest_asyncio

from impenetra.core.guardrail import BaseGuardrail, GuardrailResult

nest_asyncio.apply()

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "guardrails_config"


class _SelfCheckTracker:
    """Mutable tracker object shared across asyncio context copies and worker threads."""

    __slots__ = ("triggered",)

    def __init__(self):
        self.triggered: bool = False


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


def is_self_check_input_triggered(response: Any, action_triggered: bool = False) -> bool:
    """
    Check if the 'self_check_input' task or action was triggered during NeMo evaluation.
    Inspects action dispatcher execution flag and response.log (llm_calls,
    activated_rails, executed_actions, and internal_events).
    """
    if action_triggered:
        return True

    if response is None:
        return False

    log = getattr(response, "log", None)
    if log is None and isinstance(response, dict):
        log = response.get("log")

    if not log:
        return False

    # 1. Inspect llm_calls for task == 'self_check_input'
    llm_calls = getattr(log, "llm_calls", None)
    if llm_calls is None and isinstance(log, dict):
        llm_calls = log.get("llm_calls")

    if llm_calls:
        for call in llm_calls:
            task = getattr(call, "task", None) or (call.get("task") if isinstance(call, dict) else None)
            if task == "self_check_input":
                return True
            prompt = getattr(call, "prompt", None) or (call.get("prompt") if isinstance(call, dict) else None)
            if isinstance(prompt, str) and ("task: self_check_input" in prompt or "self_check_input" in prompt):
                return True

    # 2. Inspect activated_rails and their executed_actions
    activated = getattr(log, "activated_rails", None)
    if activated is None and isinstance(log, dict):
        activated = log.get("activated_rails")

    if activated:
        for rail in activated:
            name = getattr(rail, "name", "") or (rail.get("name", "") if isinstance(rail, dict) else "")
            rail_type = getattr(rail, "type", "") or (rail.get("type", "") if isinstance(rail, dict) else "")
            if name in ["self check input", "self_check_input"] or rail_type == "self_check_input":
                return True

            actions = getattr(rail, "executed_actions", None)
            if actions is None and isinstance(rail, dict):
                actions = rail.get("executed_actions")

            if actions:
                for act in actions:
                    act_name = getattr(act, "action_name", "") or (
                        act.get("action_name", "") if isinstance(act, dict) else ""
                    )
                    if act_name in ["self_check_input", "self check input"]:
                        return True
                    act_calls = getattr(act, "llm_calls", None) or (
                        act.get("llm_calls") if isinstance(act, dict) else None
                    )
                    if act_calls:
                        for c in act_calls:
                            c_task = getattr(c, "task", None) or (
                                c.get("task") if isinstance(c, dict) else None
                            )
                            if c_task == "self_check_input":
                                return True

    # 3. Inspect internal_events
    events = getattr(log, "internal_events", None)
    if events is None and isinstance(log, dict):
        events = log.get("internal_events")

    if events:
        for ev in events:
            if isinstance(ev, dict):
                if ev.get("task") == "self_check_input":
                    return True
                if ev.get("action_name") in ["self_check_input", "self check input"]:
                    return True
                if ev.get("flow_id") in ["self check input", "self_check_input"]:
                    return True

    return False


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
        self._self_check_tracker: contextvars.ContextVar[_SelfCheckTracker | None] = (
            contextvars.ContextVar(f"self_check_tracker_{id(self)}", default=None)
        )
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

        # Hook action dispatcher to track action execution directly
        dispatcher = getattr(getattr(self._rails, "runtime", None), "action_dispatcher", None)
        if dispatcher and hasattr(dispatcher, "execute_action"):
            orig_execute = dispatcher.execute_action

            async def tracked_execute_action(action_name: str, params: dict):
                norm = getattr(dispatcher, "_normalize_action_name", lambda x: x)(action_name)
                if norm in ["self_check_input", "self check input"]:
                    # Mutate tracker in place so outer context sees it across asyncio boundaries
                    tracker = self._self_check_tracker.get()
                    if tracker is not None:
                        tracker.triggered = True
                return await orig_execute(action_name, params)

            dispatcher.execute_action = tracked_execute_action

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

        tracker = _SelfCheckTracker()
        token = self._self_check_tracker.set(tracker)
        start = time.perf_counter()
        try:
            from nemoguardrails.rails.llm.options import GenerationOptions

            # Evaluate input and dialog rails with activated rails, llm calls, and internal events logging
            options = GenerationOptions(
                rails=["input", "dialog"],
                log={"activated_rails": True, "llm_calls": True, "internal_events": True},
            )
            response = self._rails.generate(
                messages=[{"role": "user", "content": text}],
                options=options,
            )

            action_triggered = tracker.triggered
            judge_reached = 1 if is_self_check_input_triggered(response, action_triggered) else 0

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
                    llm_judge_layer_reached=judge_reached,
                )
            else:
                return GuardrailResult(
                    allowed=True,
                    confidence=1.0,
                    category="",
                    details="Passed NeMo input rails",
                    latency_ms=elapsed,
                    llm_judge_layer_reached=judge_reached,
                )

        except Exception as e:
            action_triggered = tracker.triggered
            judge_reached = 1 if is_self_check_input_triggered(None, action_triggered) else 0

            elapsed = (time.perf_counter() - start) * 1000
            return GuardrailResult(
                allowed=True,
                confidence=0.0,
                category="nemo_error",
                details=f"NeMo evaluation error: {e}",
                latency_ms=elapsed,
                llm_judge_layer_reached=judge_reached,
            )
        finally:
            self._self_check_tracker.reset(token)

    def check_batch(self, texts: list[str]) -> list[GuardrailResult]:
        return [self.check(t) for t in texts]
