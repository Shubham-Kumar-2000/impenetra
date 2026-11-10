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


class OffCharterGuardrail(BaseGuardrail):
    """
    Sub-millisecond guardrail that detects off-charter utility theft and
    compute exhaustion requests (e.g. unsolicited code generation, essays,
    calculus proofs, medical prescriptions, or interactive roleplay).
    """

    DEFAULT_OFF_CHARTER_PATTERNS = [
        # Code generation
        r"(?i)\bwrite\s+(a\s+)?(complete\s+)?(java|python|sql|rust|c\+\+|go|react|typescript|javascript)\b",
        r"(?i)\b(implement|create)\s+(a\s+)?(lock-free|custom\s+memory|react\s+component|sql\s+schema|lexer|microservice)\b",
        r"(?i)\bgenerate\s+(a\s+)?(complete\s+)?sql\s+schema\b",
        r"(?i)\badvanced\s+postgresql\s+query\b",
        # Creative writing & essays
        r"(?i)\bwrite\s+(a\s+)?(comprehensive\s+)?(\d+[\s-]word\s+)?(academic\s+)?essay\b",
        r"(?i)\bwrite\s+(a\s+)?(full\s+)?(screenplay|opening\s+chapter|noir\s+novel)\b",
        r"(?i)\bcompose\s+(a\s+)?(cycle\s+of\s+)?(\w+\s+)?sonnets?\b",
        r"(?i)\bdraft\s+(a\s+)?(\d+[\s-]minute\s+)?keynote\s+address\b",
        # Academic Math & Physics
        r"(?i)\bderive\s+the\s+(maximum\s+likelihood|euler-lagrange|navier-stokes)\b",
        r"(?i)\bsolve\s+the\s+(time-independent\s+)?schrödinger\s+equation\b",
        r"(?i)\bevaluate\s+the\s+triple\s+integral\b",
        r"(?i)\bfind\s+the\s+eigenvalues\s+and\s+(corresponding\s+)?eigenvectors\b",
        # Advisory
        r"(?i)\b(prescribe|tell\s+me\s+exactly\s+which)\s+antibiotic\b",
        r"(?i)\bdraft\s+(a\s+)?legally\s+binding\s+non-compete\b",
        r"(?i)\bcreate\s+(an\s+)?aggressive\s+stock\s+portfolio\b",
        r"(?i)\bstructure\s+an\s+offshore\s+(shell\s+)?corporation\b",
        # Therapy & Roleplay
        r"(?i)\bact\s+as\s+my\s+(licensed\s+)?(clinical\s+)?(psychologist|therapist|life\s+coach)\b",
        r"(?i)\blet['’]s\s+play\s+(a\s+)?full\s+interactive\s+dungeons\s+&\s+dragons\b",
        # Bulk Synthetic Data
        r"(?i)\bgenerate\s+\d+\s+(realistic\s+)?(customer\s+profile\s+records|rows\s+of|clickbait\s+blog)\b",
        r"(?i)\bgenerate\s+a\s+(large\s+)?markdown\s+table\s+containing\s+\d+\b",
    ]

    def __init__(self, patterns: list[str] | None = None, name: str = "off_charter_guardrail"):
        import re

        self._name = name
        patterns_to_compile = patterns or self.DEFAULT_OFF_CHARTER_PATTERNS
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
                    confidence=0.95,
                    category="off_charter_utility_theft",
                    details=f"Off-charter pattern intercepted: {match.group(0)}",
                    latency_ms=elapsed,
                )
        elapsed = (time.perf_counter() - start) * 1000
        return GuardrailResult(
            allowed=True,
            confidence=1.0,
            category="",
            details="Passed off-charter check",
            latency_ms=elapsed,
        )

    def check_batch(self, texts: list[str]) -> list[GuardrailResult]:
        return [self.check(t) for t in texts]
