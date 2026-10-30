"""
FastAPI application serving guardrailed endpoints for AI agents and Garak assessment.

Provides:
- POST /chat : Simple REST endpoint compatible with Garak RestGenerator
- POST /v1/chat/completions : OpenAI-compatible chat completions endpoint
- POST /guardrail/check : Direct inspection of guardrail evaluation
- GET /health : Service health check
"""

import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from impenetra.core.guardrail import BaseGuardrail, GuardrailResult, RuleBasedGuardrail


class ChatPromptRequest(BaseModel):
    """Simple chat request model, optimized for Garak RestGenerator."""

    prompt: str = Field(..., description="User prompt to evaluate and process")
    system_prompt: str | None = Field(default=None, description="Optional system prompt")


class ChatPromptResponse(BaseModel):
    """Simple chat response model."""

    response: str = Field(..., description="Model response or guardrail refusal")
    blocked: bool = Field(..., description="Whether the request was intercepted")
    category: str = Field(default="", description="Intercepted violation category")
    confidence: float = Field(default=1.0, description="Confidence of guardrail decision")
    latency_ms: float = Field(..., description="Guardrail inspection latency in ms")


class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of message sender (user, system, assistant)")
    content: str = Field(..., description="Text content of message")


class OpenAIChatRequest(BaseModel):
    """OpenAI-compatible request model for standard tools and Garak."""

    model: str = Field(default="impenetra-guardrail", description="Target model name")
    messages: list[ChatMessage] = Field(..., description="List of messages in conversation")
    temperature: float | None = Field(default=0.7)
    max_tokens: int | None = Field(default=128)


class OpenAIChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class OpenAIChatResponse(BaseModel):
    """OpenAI-compatible response structure."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[OpenAIChoice]
    usage: dict[str, int]
    guardrail: dict[str, Any]


class GuardrailCheckRequest(BaseModel):
    text: str = Field(..., description="Text to inspect with the guardrail")


class GuardrailCheckResponse(BaseModel):
    allowed: bool
    confidence: float
    category: str
    details: str
    latency_ms: float


def create_app(guardrail: BaseGuardrail | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""

    application = FastAPI(
        title="Impenetra Guardrail API",
        description="Low-latency guardrail API for AI agents and Garak red-team assessment",
        version="0.1.0",
    )

    # Use default rule-based guardrail if none provided
    active_guardrail = guardrail or RuleBasedGuardrail()
    application.state.guardrail = active_guardrail

    DEFAULT_REFUSAL = (
        "I cannot fulfill this request because it violates safety guidelines."
    )

    @application.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "service": "impenetra",
            "version": "0.1.0",
            "active_guardrail": application.state.guardrail.name,
        }

    @application.post("/guardrail/check", response_model=GuardrailCheckResponse)
    async def check_guardrail(req: GuardrailCheckRequest):
        res: GuardrailResult = application.state.guardrail.check(req.text)
        return GuardrailCheckResponse(
            allowed=res.allowed,
            confidence=res.confidence,
            category=res.category,
            details=res.details,
            latency_ms=round(res.latency_ms, 3),
        )

    @application.post("/chat", response_model=ChatPromptResponse)
    async def chat(req: ChatPromptRequest):
        """
        Primary endpoint for Garak RestGenerator.
        Inspects prompt and either blocks it or returns an approved response.
        """
        res: GuardrailResult = application.state.guardrail.check(req.prompt)

        if not res.allowed:
            return ChatPromptResponse(
                response=DEFAULT_REFUSAL,
                blocked=True,
                category=res.category,
                confidence=res.confidence,
                latency_ms=round(res.latency_ms, 3),
            )

        # In production/research this forwards to target LLM; here simulated safe response
        safe_response = f"Processed request successfully: {req.prompt[:50]}..."
        return ChatPromptResponse(
            response=safe_response,
            blocked=False,
            category="",
            confidence=res.confidence,
            latency_ms=round(res.latency_ms, 3),
        )

    @application.post("/v1/chat/completions", response_model=OpenAIChatResponse)
    async def openai_chat_completions(req: OpenAIChatRequest):
        """
        OpenAI-compatible chat completion endpoint.
        Used by Garak with --model_type openai or REST client generators.
        """
        if not req.messages:
            raise HTTPException(status_code=400, detail="Messages list cannot be empty")

        # Extract last user message
        user_messages = [m.content for m in req.messages if m.role == "user"]
        prompt_text = user_messages[-1] if user_messages else req.messages[-1].content

        res: GuardrailResult = application.state.guardrail.check(prompt_text)

        response_content = (
            DEFAULT_REFUSAL
            if not res.allowed
            else f"Response from {req.model}: Received query securely."
        )

        return OpenAIChatResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=req.model,
            choices=[
                OpenAIChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_content),
                    finish_reason="stop",
                )
            ],
            usage={
                "prompt_tokens": len(prompt_text.split()),
                "completion_tokens": len(response_content.split()),
                "total_tokens": len(prompt_text.split()) + len(response_content.split()),
            },
            guardrail={
                "allowed": res.allowed,
                "category": res.category,
                "confidence": res.confidence,
                "latency_ms": round(res.latency_ms, 3),
            },
        )

    return application


# Global application instance for uvicorn (e.g., uvicorn impenetra.api.app:app)
app = create_app()
