"""
NeMo Guardrails baseline benchmarking.

Measures latency and accuracy of NeMo Guardrails on our test datasets,
providing the comparison baseline for Impenetra's custom guardrails.

NOTE: LLM provider is currently stubbed out. Configure a provider
(OpenAI, Azure, Ollama, etc.) in the config before running benchmarks.
"""

from pathlib import Path


class NemoBaseline:
    """Wraps NeMo Guardrails for standardized benchmarking."""

    def __init__(self, config_path: str | Path | None = None):
        self.config_path = config_path
        self._rails = None

    def setup(self):
        """
        Initialize NeMo Guardrails with config.

        Requires a valid LLM provider configuration. See NeMo Guardrails
        documentation for supported providers (OpenAI, Azure, Ollama, etc.).
        """
        # Deferred import — NeMo has heavy deps
        from nemoguardrails import RailsConfig, LLMRails

        if self.config_path:
            config = RailsConfig.from_path(str(self.config_path))
        else:
            raise ValueError(
                "No config_path provided. Create a NeMo Guardrails config "
                "directory with config.yml and Colang files. "
                "See: https://docs.nvidia.com/nemo/guardrails/getting_started/"
            )
        self._rails = LLMRails(config)

    async def generate(self, prompt: str) -> dict:
        """Generate a response through NeMo Guardrails."""
        if self._rails is None:
            raise RuntimeError("Call setup() first")
        response = await self._rails.generate_async(
            messages=[{"role": "user", "content": prompt}]
        )
        return response
