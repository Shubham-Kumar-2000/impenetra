"""
Experiment 2: NeMo Guardrails Server
=====================================
Uses the Impenetra API framework (create_app) with the injectable
NeMoGuardrail component. Safe prompts are routed to Ollama's
llama3.1:8b-instruct-q8_0.
"""

import os
import sys
from pathlib import Path

# Ensure impenetra and experiment packages are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import uvicorn
from impenetra.api.app import create_app
from guardrail import NeMoGuardrail

guardrail = NeMoGuardrail()

# Inject into Impenetra FastAPI app
app = create_app(
    guardrail=guardrail,
    backend_url="http://localhost:11434",
    model_name="llama3.1:8b-instruct-q8_0",
)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
