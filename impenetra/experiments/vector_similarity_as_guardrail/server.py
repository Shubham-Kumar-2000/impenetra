"""
Experiment: Vector Similarity Guardrail Server
==============================================
Runs the Impenetra API framework (create_app) with VectorSimilarityGuardrail
and the Crave Craft food agent (get_default_agent).

Strict execution: Loads connection parameters and thresholds from
on_charter_intents.json. Fails fast if Qdrant is unreachable.
"""

import os

import uvicorn

from impenetra.api.app import create_app
from impenetra.experiments.vector_similarity_as_guardrail.guardrail import (
    DEFAULT_CONFIG_PATH,
    VectorSimilarityGuardrail,
)
from impenetra.food_agent import get_default_agent

config_path = os.getenv("VECTOR_SIMILARITY_CONFIG", str(DEFAULT_CONFIG_PATH))
guardrail = VectorSimilarityGuardrail(config_path=config_path)

# Inject into Impenetra FastAPI app with food agent
app = create_app(
    guardrail=guardrail,
    agent=get_default_agent(),
    model_name="vector-similarity-food-agent",
)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
