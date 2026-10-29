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
