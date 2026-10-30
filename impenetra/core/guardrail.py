"""Base guardrail interface — all guardrail implementations inherit from this."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GuardrailResult:
    """Result of a guardrail check."""

    allowed: bool  # Whether the input is allowed
    confidence: float  # Confidence score [0, 1]
    category: str = ""  # Detected category (e.g., "toxic", "jailbreak")
    details: str = ""  # Human-readable explanation
    latency_ms: float = 0.0  # Time taken for this check


class BaseGuardrail(ABC):
    """Abstract base class for all guardrail implementations."""

    @abstractmethod
    def check(self, text: str) -> GuardrailResult:
        """Check if input text passes the guardrail."""
        ...

    @abstractmethod
    def check_batch(self, texts: list[str]) -> list[GuardrailResult]:
        """Check a batch of texts."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Guardrail identifier."""
        ...


class RuleBasedGuardrail(BaseGuardrail):
    """
    Fast rule-based guardrail targeting common jailbreak, prompt injection,
    and adversarial patterns (sub-millisecond latency baseline).
    """

    DEFAULT_BLOCKED_PATTERNS = [
        r"(?i)\bignore\s+(all\s+)?(previous|prior)\s+instructions\b",
        r"(?i)\byou\s+are\s+now\s+in\s+dan\s+mode\b",
        r"(?i)\bdo\s+anything\s+now\b",
        r"(?i)\bjailbreak\b",
        r"(?i)\bdeveloper\s+mode\s+output\b",
        r"(?i)\bsystem\s+override\b",
        r"(?i)\bdisregard\s+(your\s+)?guidelines\b",
        r"(?i)\breveal\s+(your\s+)?system\s+prompt\b",
        r"(?i)\bwhat\s+(is|are)\s+your\s+hidden\s+instructions\b",
    ]

    def __init__(self, patterns: list[str] | None = None, name: str = "rule_based"):
        import re
        self._name = name
        patterns_to_compile = patterns or self.DEFAULT_BLOCKED_PATTERNS
        self._compiled_patterns = [re.compile(p) for p in patterns_to_compile]

    @property
    def name(self) -> str:
        return self._name

    def check(self, text: str) -> GuardrailResult:
        import time
        start = time.perf_counter()
        for pattern in self._compiled_patterns:
            match = pattern.search(text)
            if match:
                elapsed = (time.perf_counter() - start) * 1000
                return GuardrailResult(
                    allowed=False,
                    confidence=0.98,
                    category="prompt_injection_or_jailbreak",
                    details=f"Pattern match: {match.group(0)}",
                    latency_ms=elapsed,
                )
        elapsed = (time.perf_counter() - start) * 1000
        return GuardrailResult(
            allowed=True,
            confidence=1.0,
            category="",
            details="Passed rule check",
            latency_ms=elapsed,
        )

    def check_batch(self, texts: list[str]) -> list[GuardrailResult]:
        return [self.check(t) for t in texts]

