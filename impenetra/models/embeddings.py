"""Embedding model wrapper for semantic similarity guardrails."""

import torch
from sentence_transformers import SentenceTransformer


class GuardrailEmbedder:
    """Wraps all-MiniLM-L6-v2 for fast semantic similarity checks."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
    ):
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Encode texts into embedding vectors."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            convert_to_tensor=True,
            device=self.device,
        )

    def similarity(
        self, texts_a: list[str], texts_b: list[str]
    ) -> torch.Tensor:
        """Compute pairwise cosine similarity between two lists of texts."""
        emb_a = self.encode(texts_a)
        emb_b = self.encode(texts_b)
        return torch.nn.functional.cosine_similarity(emb_a, emb_b, dim=-1)
