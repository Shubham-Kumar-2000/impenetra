# Experiment 1: LlamaGuard-as-Guardrail (Google Colab & Gradio Integration)

> **Goal:** Measure how effectively Meta's `Llama-Guard-3-8B` (hosted on Google Colab behind a Cloudflare Tunnel or via Gradio Space with custom Food Agent Charter category `S15` and standard `S1`–`S14` safety policies) blocks adversarial and off-charter prompts when used as an inline input classifier.

## Overview

This experiment evaluates Meta's **Llama Guard 3 8B** as an inline input guardrail for AI agents. The classifier evaluates incoming user prompts against active safety taxonomies:
- **S1–S14**: Standard MLCommons hazard taxonomy (violent crimes, hacking, sexual content, weapons, etc.)
- **S15**: **Out of Charter (Food Agent)** custom category (intercepts code writing, debugging, non-food math, or off-charter tasks)

If LlamaGuard returns `safe=False`, the request is immediately intercepted and refused with the corresponding violation code (`S15`, `S1`, etc.). Only `safe` prompts are forwarded to the underlying assistant backend.

For high-throughput benchmarking and red-teaming, Llama Guard 3 8B is hosted on **Google Colab** (using an L4 GPU and vLLM AsyncEngine) and exposed via a public **Cloudflare Tunnel (`cloudflared`)**, providing an HTTPS endpoint with no rate limits or authentication tokens required. It can also run locally or via Hugging Face Spaces.

## Architecture

```
Garak (red-team probes) / Client
        │
        ▼
FastAPI Guardrail Server :8000
  POST /v1/chat/completions
        │
        ▼
LlamaGuardGuardrail
        │
        ├── Calls Llama Guard 3 Backend:
        │     • Google Colab via cloudflared: https://<subdomain>.trycloudflare.com/api/predict
        │     • Or Local / HF Gradio Space: /gradio_api/call/moderate_prompt
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

---

## Hosting Llama Guard 3

### Option A: Google Colab via Cloudflare Tunnel (`cloudflared`) [Recommended]

Host Llama Guard 3 8B on Google Colab with GPU acceleration and vLLM continuous batching using [notebooks/llama_guard_vllm_colab.ipynb](../../../notebooks/llama_guard_vllm_colab.ipynb):

1. **Hardware Runtime**: Select **Runtime → Change runtime type → L4 GPU** (24 GB VRAM).
2. **Hugging Face Auth**: Provide a Hugging Face read token to access `meta-llama/Llama-Guard-3-8B` (gated model).
3. **Cloudflare Tunnel (`cloudflared`)**: The notebook automatically downloads the `cloudflared` binary and creates a public HTTPS tunnel (`https://<subdomain>.trycloudflare.com`) with **no account, auth token, or rate limits**.
4. **vLLM Continuous Batching**: Runs `vLLM AsyncEngine` inside FastAPI's lifespan, batching concurrent requests (e.g., `--parallel 32` from Garak) into unified GPU forward passes with CUDA graph warmup.
5. **Compatible Endpoints Exposed**:
   - `POST /api/predict` *(primary endpoint tried first by `LlamaGuardGuardrail`)*
   - `POST /run/predict`
   - `POST /gradio_api/call/moderate_prompt`
   - `GET /health` and `GET /docs` (Swagger UI)

Once started, copy the public URL printed by the notebook:
```
🌐 Public API URL : https://random-subdomain.trycloudflare.com
```

### Option B: Local Gradio Space or Hugging Face Space

Alternatively, run a Gradio Space locally or deploy to Hugging Face Spaces:

```bash
# In llama-guard-space:
python llama-guard-space/app.py
# Runs on http://127.0.0.1:7860
```
Or use a Hugging Face Space: `https://<username>-llama-guard-space.hf.space`.

---

## Running the Experiment

### 1. Execute Garak Assessment with Cloudflare Tunnel URL

Pass the Google Colab `cloudflared` URL directly to the run script:

```bash
# Point to Google Colab cloudflared URL:
bash impenetra/experiments/llamaguard_as_guardrail/run.sh https://random-subdomain.trycloudflare.com

# Or point to a local/remote Gradio Space:
bash impenetra/experiments/llamaguard_as_guardrail/run.sh http://127.0.0.1:7860
```

You can also specify the URL via the `LLAMA_GUARD_GRADIO_URL` environment variable:

```bash
export LLAMA_GUARD_GRADIO_URL="https://random-subdomain.trycloudflare.com"
bash impenetra/experiments/llamaguard_as_guardrail/run.sh
```

### 2. High-Throughput Parallel Probes

Because the Google Colab vLLM server uses continuous batching, you can scale concurrency up to 32 parallel probes:

```bash
export LLAMA_GUARD_GRADIO_URL="https://random-subdomain.trycloudflare.com"
export GARAK_PARALLEL_ATTEMPTS=32
bash impenetra/experiments/llamaguard_as_guardrail/run.sh
```

### 3. Standalone Server Execution

If you wish to run only the Impenetra FastAPI server with LlamaGuard protection (without triggering Garak):

```bash
export LLAMA_GUARD_GRADIO_URL="https://random-subdomain.trycloudflare.com"
python impenetra/experiments/llamaguard_as_guardrail/server.py
```

---

## Fail-Open Logging

If the remote Google Colab server, Cloudflare tunnel, or Gradio Space experiences downtime, network disconnections, or HTTP error statuses (e.g. 503/502/500), the guardrail operates in **fail-open** mode (`allowed=True`, `category="guardrail_error"`).

Every fail-open occurrence is automatically logged to:
- `impenetra/experiments/llamaguard_as_guardrail/results/fail_open.log` (human-readable format with timestamp, target URL, prompt, latency, failure reason, and raw server response)
- `impenetra/experiments/llamaguard_as_guardrail/results/fail_open.jsonl` (structured JSON Lines format)

You can override the log path via the `LLAMA_GUARD_FAIL_OPEN_LOG` environment variable:
```bash
export LLAMA_GUARD_FAIL_OPEN_LOG="path/to/custom_fail_open.log"
```
