# Experiment 1: LlamaGuard-as-Guardrail (Gradio Space Integration)

> **Goal:** Measure how effectively Meta's `Llama-Guard-3-8B` running via a Gradio Space (`llama-guard-space` with custom Food Agent Charter category `S15` and standard `S1`–`S14` safety policies) blocks adversarial and off-charter prompts when used as an inline input classifier.

## Overview

This experiment connects to Meta's **Llama Guard 3 8B** deployed as a Gradio Space (such as `llama-guard-space`). Every user prompt is evaluated against the active safety taxonomies:
- **S1–S14**: Standard MLCommons hazard taxonomy (violent crimes, hacking, sexual content, weapons, etc.)
- **S15**: **Out of Charter (Food Agent)** custom category (intercepts code writing, debugging, non-food math, or off-charter tasks)

If the LlamaGuard Gradio app returns `safe=False`, the request is immediately intercepted and refused with the corresponding violation code (`S15`, `S1`, etc.). Only `safe` prompts are forwarded to the underlying assistant.

## Architecture

```
Garak (red-team probes) / Client
        │
        ▼
FastAPI Server :8000
  POST /v1/chat/completions
        │
        ▼
LlamaGuardGuardrail
        │
        ├── Calls Gradio Space URL (/moderate_prompt)
        │     via gradio_client or HTTP API
        │
   [safe=False]  ──→  BLOCK (refusal response with violation code S1-S15)
        │
   [safe=True]
        │
        ▼
Food Agent (or LLM Backend)
        │
        ▼
  Response returned to caller
```

## Running the Space & Server

### 1. Launch the Llama Guard Gradio Space

```bash
# In llama-guard-space:
python llama-guard-space/app.py
# Runs on http://127.0.0.1:7860
```

Or deploy to Hugging Face Spaces (e.g. `https://<user>-llama-guard-space.hf.space`).

### 2. Run the Experiment

```bash
# Point to local Gradio Space:
bash impenetra/experiments/llamaguard_as_guardrail/run.sh http://127.0.0.1:7860

# Or point to a remote Hugging Face Space:
bash impenetra/experiments/llamaguard_as_guardrail/run.sh https://username-llama-guard-space.hf.space
```

You can also pass `LLAMA_GUARD_GRADIO_URL` as an environment variable:

```bash
export LLAMA_GUARD_GRADIO_URL="http://127.0.0.1:7860"
python impenetra/experiments/llamaguard_as_guardrail/server.py
```

## Fail-Open Logging

If the Gradio Space or inference server experiences downtime, connection timeouts, or HTTP errors (e.g. 503/403/500), the guardrail operates in **fail-open** mode (`allowed=True`, `category="guardrail_error"`).

Every fail-open occurrence is automatically logged to:
- `impenetra/experiments/llamaguard_as_guardrail/results/fail_open.log` (human-readable format with timestamp, prompt, failure reason, and raw server response)
- `impenetra/experiments/llamaguard_as_guardrail/results/fail_open.jsonl` (structured JSON Lines format)

You can override the log path via the `LLAMA_GUARD_FAIL_OPEN_LOG` environment variable:
```bash
export LLAMA_GUARD_FAIL_OPEN_LOG="path/to/custom_fail_open.log"
```
