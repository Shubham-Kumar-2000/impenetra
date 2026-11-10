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
from typing import Any, Callable

import requests
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


def create_app(
    guardrail: BaseGuardrail | None = None,
    backend_url: str | None = "http://localhost:11434",
    model_name: str | None = "llama3.1:8b-instruct-q8_0",
    system_prompt: str | None = None,
    llm_caller: Callable[[list[ChatMessage]], str] | None = None,
    agent: Any | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application."""

    application = FastAPI(
        title="Impenetra Guardrail API",
        description="Low-latency guardrail API for AI agents and Garak red-team assessment",
        version="0.1.0",
    )

    # Use default rule-based guardrail if none provided
    active_guardrail = guardrail or RuleBasedGuardrail()
    application.state.guardrail = active_guardrail

    target_model = model_name or "llama3.1:8b-instruct-q8_0"

    active_llm_caller: Callable[[list[ChatMessage]], str] | None = None

    if agent:

        def agent_caller(messages: list[ChatMessage]) -> str:
            user_messages = [{"role": m.role, "content": m.content} for m in messages]
            if hasattr(agent, "invoke"):
                try:
                    out = agent.invoke({"messages": user_messages})
                    if isinstance(out, dict) and "messages" in out and out["messages"]:
                        last_msg = out["messages"][-1]
                        return getattr(last_msg, "content", str(last_msg))
                    elif hasattr(out, "content"):
                        return str(out.content)
                    elif isinstance(out, dict) and "output" in out:
                        return str(out["output"])
                    return str(out)
                except Exception:
                    from langchain_core.messages import (
                        AIMessage,
                        BaseMessage,
                        HumanMessage,
                        SystemMessage,
                    )

                    lc_msgs: list[BaseMessage] = []
                    for m in messages:
                        if m.role == "system":
                            lc_msgs.append(SystemMessage(content=m.content))
                        elif m.role == "assistant":
                            lc_msgs.append(AIMessage(content=m.content))
                        else:
                            lc_msgs.append(HumanMessage(content=m.content))
                    out = agent.invoke(lc_msgs)
                    if hasattr(out, "content"):
                        return str(out.content)
                    return str(out)
            elif callable(agent):
                return str(agent(messages))
            return str(agent)

        active_llm_caller = agent_caller
    elif llm_caller:
        active_llm_caller = llm_caller
    elif backend_url:

        def default_ollama_caller(messages: list[ChatMessage]) -> str:
            msg_payload = []
            if system_prompt:
                msg_payload.append({"role": "system", "content": system_prompt})
            msg_payload.extend(
                [
                    {"role": m.role, "content": m.content}
                    for m in messages
                    if not (system_prompt and m.role == "system")
                ]
            )
            resp = requests.post(
                f"{backend_url.rstrip('/')}/api/chat",
                json={
                    "model": target_model,
                    "messages": msg_payload,
                    "stream": False,
                },
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]

        active_llm_caller = default_ollama_caller
    else:
        active_llm_caller = None

    application.state.llm_caller = active_llm_caller

    DEFAULT_REFUSAL = "I cannot fulfill this request because it violates safety guidelines."

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

        if not application.state.llm_caller:
            raise HTTPException(
                status_code=503,
                detail=(
                    "LLM backend caller not configured. "
                    "Provide backend_url or llm_caller in create_app()."
                ),
            )

        safe_response = application.state.llm_caller([ChatMessage(role="user", content=req.prompt)])
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

        if not res.allowed:
            has_refusal_phrase = res.details and (
                "sorry" in res.details.lower() or "cannot" in res.details.lower()
            )
            response_content = res.details if has_refusal_phrase else DEFAULT_REFUSAL
        else:
            if not application.state.llm_caller:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "LLM backend caller not configured. "
                        "Provide backend_url or llm_caller in create_app()."
                    ),
                )
            response_content = application.state.llm_caller(req.messages)

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
                "blocked": not res.allowed,
                "category": res.category,
                "confidence": res.confidence,
                "latency_ms": round(res.latency_ms, 3),
                "details": res.details,
            },
        )

    return application


# Global application instance for uvicorn (e.g., uvicorn impenetra.api.app:app)
app = create_app()
