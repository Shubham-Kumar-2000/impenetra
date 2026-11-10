"""Semantic intent router for guardrail dispatch."""


class SemanticRouter:
    """Routes inputs to appropriate guardrails based on semantic similarity."""

    def __init__(self):
        self.routes: dict[str, list[str]] = {}
        self._embedder = None
        self._route_embeddings = None

    def add_route(self, name: str, exemplars: list[str]):
        """Register a route with example utterances."""
        self.routes[name] = exemplars
        self._route_embeddings = None  # Invalidate cache

    def classify(self, text: str) -> tuple[str, float]:
        """Classify text to the best matching route."""
        # Placeholder — will be implemented with GuardrailEmbedder
        raise NotImplementedError("Implement with GuardrailEmbedder")
