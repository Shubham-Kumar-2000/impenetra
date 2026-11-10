"""
Experiment 2: NeMo Guardrails Server
=====================================
Wraps NVIDIA NeMo Guardrails (LLMRails) around llama3.1:8b-instruct-q8_0
served by Ollama's OpenAI-compatible endpoint.

NeMo Guardrails handles:
  - Input rails  (self check input + custom Colang flows)
  - LLM call     (routed to Ollama via openai engine)
  - Output rails (self check output)

Architecture:
    Garak → FastAPI (8000) → NeMo LLMRails → Ollama /v1 → llama3.1:8b-instruct-q8_0
"""

import time
import uuid
import uvicorn
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

CONFIG_PATH = Path(__file__).parent / "guardrails_config"
DEFAULT_REFUSAL = "I'm sorry, I can't help with that request."

app = FastAPI(
    title="NeMo Guardrails Experiment",
    description="Experiment 2: NVIDIA NeMo Guardrails + llama3.1:8b-instruct-q8_0 via Ollama",
    version="1.0.0",
)

# Global rails instance — initialized at startup
rails = None


@app.on_event("startup")
async def startup():
    """Load NeMo Guardrails config on server startup."""
    global rails
    try:
        from nemoguardrails import RailsConfig, LLMRails

        config = RailsConfig.from_path(str(CONFIG_PATH))
        rails = LLMRails(config)
        print(f"✅ NeMo Guardrails loaded from {CONFIG_PATH}")
    except Exception as e:
        print(f"❌ Failed to load NeMo Guardrails: {e}")
        raise


class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: user, system, or assistant")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    model: str = Field(default="nemo-guardrails-experiment")
    messages: list[ChatMessage]
    temperature: float | None = Field(default=0.7)
    max_tokens: int | None = Field(default=512)


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "experiment": "nemo-guardrails",
        "rails_loaded": rails is not None,
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    """OpenAI-compatible endpoint. Garak targets this directly."""
    if rails is None:
        raise HTTPException(status_code=503, detail="NeMo Guardrails not loaded")

    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    t0 = time.perf_counter()

    try:
        response = await rails.generate_async(messages=messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NeMo error: {e}")

    latency = (time.perf_counter() - t0) * 1000

    # NeMo may return a string or dict
    if isinstance(response, str):
        content = response
    elif isinstance(response, dict):
        content = response.get("content", str(response))
    else:
        content = str(response)

    # Detect NeMo's built-in refusals (self-check responses)
    blocked = any(
        phrase in content.lower()
        for phrase in ["i'm sorry", "i cannot", "i can't", "i am not able", "i'm not able"]
    )

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "guardrail": {
            "active": True,
            "blocked": blocked,
            "latency_ms": round(latency, 2),
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
