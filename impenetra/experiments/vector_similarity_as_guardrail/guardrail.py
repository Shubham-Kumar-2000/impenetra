"""
Vector Similarity Guardrail: Ultra-low latency cosine similarity gate
====================================================================
Implements BaseGuardrail by evaluating incoming utterances against:
1. Short-prompt fast-path bypass (min_token_threshold)
2. Negative vector database in Qdrant (similarity >= threshold and valid excess tokens -> BLOCK)
3. In-memory intent vector matrix (similarity >= threshold and valid excess tokens -> ALLOW)
4. Default deny (unmatched intent -> BLOCK)

Strict execution: All configuration parameters and Qdrant credentials are read
directly from on_charter_intents.json. No silent fallbacks.
"""

import json
import time
from pathlib import Path
from typing import Any

import torch
from qdrant_client import QdrantClient

from impenetra.core.guardrail import BaseGuardrail, GuardrailResult
from impenetra.models.embeddings import GuardrailEmbedder

# Global constant variables per requirements
SIMILARITY_THRESHOLD = 0.95
TOKEN_DIFF_PERCENT = 0.10
MAX_TOKEN_DIFF_RATIO = 0.10  # Alias for TOKEN_DIFF_PERCENT
SECURED_MAX_EXCESS_TOKENS = 5
MIN_TOKEN_THRESHOLD = 3

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "on_charter_intents.json"


def is_token_diff_valid(
    utterance_len: int,
    matched_len: int,
    token_diff_percent: float = TOKEN_DIFF_PERCENT,
    secured_max_excess_tokens: int = SECURED_MAX_EXCESS_TOKENS,
) -> bool:
    """
    Check if the utterance has no more excess tokens than min(10%, SECURED_MAX_EXCESS_TOKENS).
    One-sided check: If utterance has fewer or equal tokens than matched, it is strictly valid.
    """
    excess = utterance_len - matched_len
    if excess <= 0:
        return True

    allowed_excess = min(
        max(1, int(round(matched_len * token_diff_percent))),
        secured_max_excess_tokens,
    )
    return excess <= allowed_excess


def load_guardrail_config(config_path: Path | str) -> dict[str, Any]:
    """
    Strictly load and validate on_charter_intents.json.
    Raises ValueError immediately if file or any required key is missing.
    """
    path = Path(config_path)
    if not path.is_file():
        raise ValueError(f"Configuration file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Validate qdrant_config
    if "qdrant_config" not in data:
        raise ValueError(f"Configuration '{path}' missing required 'qdrant_config' section.")
    q_conf = data["qdrant_config"]
    if "url" not in q_conf or not q_conf["url"]:
        raise ValueError(f"'qdrant_config' in '{path}' must specify a valid 'url'.")
    if "collection_name" not in q_conf or not q_conf["collection_name"]:
        raise ValueError(f"'qdrant_config' in '{path}' must specify a valid 'collection_name'.")

    # 2. Validate guardrail_config
    if "guardrail_config" not in data:
        raise ValueError(f"Configuration '{path}' missing required 'guardrail_config' section.")
    g_conf = data["guardrail_config"]
    required_keys = (
        "similarity_threshold",
        "token_diff_percent",
        "secured_max_excess_tokens",
        "min_token_threshold",
    )
    for key in required_keys:
        if key not in g_conf:
            raise ValueError(f"'guardrail_config' in '{path}' missing required parameter '{key}'.")

    # 3. Validate intents
    if "intents" not in data or not isinstance(data["intents"], list) or len(data["intents"]) == 0:
        raise ValueError(f"Configuration '{path}' must contain a non-empty list of 'intents'.")

    for idx, item in enumerate(data["intents"]):
        if not isinstance(item, dict) or "intent" not in item or "examples" not in item:
            raise ValueError(
                f"Intent at index {idx} in '{path}' must contain 'intent' and 'examples'."
            )
        if not isinstance(item["examples"], list) or len(item["examples"]) == 0:
            raise ValueError(f"Intent '{item.get('intent')}' in '{path}' has empty 'examples'.")

    return data


class VectorSimilarityGuardrail(BaseGuardrail):
    """
    Ultra-low latency cosine similarity guardrail for food agent interactions.
    Combines Qdrant negative probe filtering with in-memory on-charter intent tensors.
    """

    def __init__(
        self,
        config_path: Path | str = DEFAULT_CONFIG_PATH,
        qdrant_client: QdrantClient | None = None,
        name: str = "vector_similarity",
    ):
        self._name = name
        self._config_path = Path(config_path)
        self._config = load_guardrail_config(self._config_path)

        # Parse guardrail thresholds from config
        g_conf = self._config["guardrail_config"]
        self._similarity_threshold = float(g_conf["similarity_threshold"])
        self._token_diff_percent = float(g_conf["token_diff_percent"])
        self._secured_max_excess_tokens = int(g_conf["secured_max_excess_tokens"])
        self._min_token_threshold = int(g_conf["min_token_threshold"])

        # Parse Qdrant connection from config
        q_conf = self._config["qdrant_config"]
        self._qdrant_url = q_conf["url"]
        self._qdrant_api_key = q_conf.get("api_key")
        self._qdrant_timeout = int(float(q_conf.get("timeout", 10)))
        self._collection_name = q_conf["collection_name"]

        # Initialize miniLLM embedder
        self._embedder = GuardrailEmbedder()
        self._tokenizer = self._embedder.model.tokenizer

        # Boot-time in-memory vector embedding of all on-charter intent examples
        self._on_charter_examples: list[str] = []
        self._on_charter_metadata: list[dict[str, str]] = []
        self._on_charter_token_counts: list[int] = []

        for intent_obj in self._config["intents"]:
            intent_name = intent_obj["intent"]
            for ex in intent_obj["examples"]:
                clean_ex = ex.strip()
                if clean_ex:
                    self._on_charter_examples.append(clean_ex)
                    self._on_charter_metadata.append({"intent": intent_name, "text": clean_ex})
                    self._on_charter_token_counts.append(len(self._tokenizer.tokenize(clean_ex)))

        if not self._on_charter_examples:
            raise ValueError(f"No valid on-charter examples found in '{self._config_path}'.")

        # Precompute in-memory normalized dense embedding tensor
        with torch.no_grad():
            self._on_charter_embeddings = self._embedder.encode(
                self._on_charter_examples,
                normalize=True,
            )

        # Initialize Qdrant Client strictly
        if qdrant_client is not None:
            self._client = qdrant_client
        else:
            try:
                self._client = QdrantClient(
                    url=self._qdrant_url,
                    api_key=self._qdrant_api_key,
                    timeout=self._qdrant_timeout,
                )
                # Verify collection exists strictly
                self._client.get_collection(self._collection_name)
            except Exception as e:
                raise RuntimeError(
                    f"VectorSimilarityGuardrail failed to connect to Qdrant at "
                    f"'{self._qdrant_url}' or find collection '{self._collection_name}': {e}. "
                    "Ensure Qdrant is running and seeded via fill_negative_samples.py."
                ) from e

    @property
    def name(self) -> str:
        return self._name

    @property
    def similarity_threshold(self) -> float:
        return self._similarity_threshold

    @property
    def token_diff_percent(self) -> float:
        return self._token_diff_percent

    @property
    def secured_max_excess_tokens(self) -> int:
        return self._secured_max_excess_tokens

    @property
    def min_token_threshold(self) -> int:
        return self._min_token_threshold

    def check(self, text: str) -> GuardrailResult:
        """
        Evaluate utterance:
        1. Short prompt bypass check
        2. Qdrant off-charter match check
        3. In-memory on-charter intent match check
        4. Default deny
        """
        start = time.perf_counter()
        clean_text = text.strip()

        # Step 1: Tokenize
        tokens = self._tokenizer.tokenize(clean_text)
        token_len = len(tokens)

        # Step 2: Minimum token fast-path check
        if token_len <= self._min_token_threshold:
            elapsed = (time.perf_counter() - start) * 1000
            return GuardrailResult(
                allowed=True,
                confidence=1.0,
                category="short_prompt_allow",
                details=f"Passed minimum token check ({token_len} <= {self._min_token_threshold})",
                latency_ms=elapsed,
            )

        # Step 3: Embed utterance
        with torch.no_grad():
            emb = self._embedder.encode([clean_text], normalize=True)
        emb_list = emb[0].cpu().tolist()

        # Step 4: Check off-charter vectors in Qdrant
        try:
            hits = self._client.query_points(
                collection_name=self._collection_name,
                query=emb_list,
                limit=5,
                score_threshold=self._similarity_threshold,
            )
            for hit in hits.points:
                score = float(hit.score)
                payload = hit.payload or {}
                matched_tokens = int(payload.get("token_count", 0))

                if score >= self._similarity_threshold and is_token_diff_valid(
                    token_len,
                    matched_tokens,
                    self._token_diff_percent,
                    self._secured_max_excess_tokens,
                ):
                    elapsed = (time.perf_counter() - start) * 1000
                    matched_text = str(payload.get("text", ""))[:80]
                    return GuardrailResult(
                        allowed=False,
                        confidence=round(score, 4),
                        category="off_charter_vector_match",
                        details=(
                            f"Matched off-charter probe: '{matched_text}' "
                            f"(sim={score:.3f}, tokens={token_len} vs {matched_tokens})"
                        ),
                        latency_ms=elapsed,
                    )
        except Exception as e:
            msg = f"Error querying Qdrant collection '{self._collection_name}': {e}"
            raise RuntimeError(msg) from e

        # Step 5: Check in-memory on-charter intent vectors
        with torch.no_grad():
            sims = torch.matmul(self._on_charter_embeddings, emb.T).squeeze(-1)
            best_idx = int(torch.argmax(sims).item())
            best_score = float(sims[best_idx].item())

        if best_score >= self._similarity_threshold:
            matched_tokens = self._on_charter_token_counts[best_idx]
            if is_token_diff_valid(
                token_len,
                matched_tokens,
                self._token_diff_percent,
                self._secured_max_excess_tokens,
            ):
                elapsed = (time.perf_counter() - start) * 1000
                matched_intent = self._on_charter_metadata[best_idx]["intent"]
                matched_text = self._on_charter_metadata[best_idx]["text"][:80]
                return GuardrailResult(
                    allowed=True,
                    confidence=round(best_score, 4),
                    category="on_charter_intent_match",
                    details=(
                        f"Matched on-charter intent '{matched_intent}': '{matched_text}' "
                        f"(sim={best_score:.3f}, tokens={token_len} vs {matched_tokens})"
                    ),
                    latency_ms=elapsed,
                )

        # Step 6: Default Deny (unmatched intent)
        elapsed = (time.perf_counter() - start) * 1000
        return GuardrailResult(
            allowed=False,
            confidence=round(1.0 - best_score, 4),
            category="unmatched_intent_blocked",
            details=(
                f"Utterance does not match any allowed on-charter intent "
                f"(best_sim={best_score:.3f} < {self._similarity_threshold})"
            ),
            latency_ms=elapsed,
        )

    def check_batch(self, texts: list[str]) -> list[GuardrailResult]:
        return [self.check(t) for t in texts]
