# Experiment 2: NeMo Guardrails

> **Goal:** Measure how effectively NVIDIA NeMo Guardrails (input rails + Colang flows) blocks adversarial prompts when wrapping `llama3.1:8b-instruct-q8_0` hosted on Google Colab via vLLM and Cloudflare Tunnel (`cloudflared`).

## Overview

This experiment uses **NVIDIA NeMo Guardrails** (`nemoguardrails>=0.11.0`) — a programmable safety framework that sits between the user and the LLM. It evaluates:

1. **Input rails** — `self check input` (NeMo's LLM classifier) + custom Colang pattern-matching flows
2. **LLM call** — routed to the remote Colab OpenAI-compatible endpoint specified by `NEMO_MAIN_MODEL_API_BASE`

Colang is NeMo's domain-specific language for expressing conversational safety policies as declarative intent patterns.

## Architecture

```
Garak (red-team probes)
        │
        ▼
FastAPI Server :8000
  POST /v1/chat/completions
        │
        ▼
  NeMo LLMRails.generate()
        │
   ┌────┴────────────────────┐
   │  Input Rails              │
   │  - self check input       │
   │  - Colang: ask jailbreak  │
   │  - Colang: ask harmful    │
   └────┬────────────────────┘
        │  (if allowed)
        ▼
llama3.1:8b-instruct-q8_0
  (via Colab Cloudflare Tunnel: NEMO_MAIN_MODEL_API_BASE/v1)
        │
        ▼
  Response returned to Garak
```

## Prerequisites

1. **Google Colab Server**: Open and run [`notebooks/llama3_1_8b_instruct_colab.ipynb`](../../../notebooks/llama3_1_8b_instruct_colab.ipynb) on an L4 GPU.
   - At Step 5, copy the public Cloudflare tunnel URL (e.g. `https://random-subdomain.trycloudflare.com`).
2. **Local Environment**:
   ```bash
   pip install -e ".[dev]"
   ```

## Running the Experiment

Pass the Cloudflare tunnel domain directly as an argument, or set `NEMO_MAIN_MODEL_API_BASE`:

```bash
# Option A: Pass as argument (domain only)
bash impenetra/experiments/nemo_guardrails/run.sh https://random-subdomain.trycloudflare.com

# Option B: Set environment variable
export NEMO_MAIN_MODEL_API_BASE="https://random-subdomain.trycloudflare.com"
bash impenetra/experiments/nemo_guardrails/run.sh
```

> **Note:** Provide the domain only (e.g. `https://<domain>.trycloudflare.com`). The experiment automatically appends the `/v1` path.

The script will:
1. ✅ Run pre-flight checks (verifying remote Colab server health at `$NEMO_MAIN_MODEL_API_BASE/health`)
2. 🚀 Start the FastAPI+NeMo server on port 8000
3. 🧪 Smoke test with a benign prompt
4. 📊 Run on-charter and adversarial benchmark
5. 🔥 Run garak red-team probes
6. 💾 Save results to `impenetra/experiments/nemo_guardrails/results/`
7. 🧹 Stop the server

## Manual Server Start

```bash
cd /path/to/impenetra
export NEMO_MAIN_MODEL_API_BASE="https://random-subdomain.trycloudflare.com"
python impenetra/experiments/nemo_guardrails/server.py

# Test health:
curl http://127.0.0.1:8000/health

# Test with a prompt:
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"What are the ingredients in pasta carbonara?"}]}'
```

## Guardrails Configuration

```
guardrails_config/
├── config.yml    # NeMo config: LLM backend (OpenAI engine), rail types
└── rails.co      # Colang: custom intent patterns and refusal flows
```

### `config.yml` — Key settings

| Setting | Value |
|---------|-------|
| LLM engine | `openai` |
| LLM model | `llama3.1:8b-instruct-q8_0` |
| API base | Overridden dynamically from `NEMO_MAIN_MODEL_API_BASE` |
| Input rails | `self check input` |

### `rails.co` — Colang flows

| Flow | Trigger patterns |
|------|-----------------|
| `refuse jailbreak` | "ignore all previous instructions", "you are now DAN", "do anything now", ... |
| `refuse harmful` | "how do I hack", "how to make a bomb", "how to synthesize drugs", ... |

## Interpreting Garak Results

| Metric | Expected Outcome |
|--------|-----------------|
| `dan` probe pass rate | **Medium-High** — Colang flows catch explicit DAN patterns |
| `promptinject` probe pass rate | **Medium** — Self-check input catches some injections |
| `jailbreak` probe pass rate | **Medium** — Depends on how well Colang patterns generalise |

Results files:
```
results/
├── YYYYMMDD_HHMMSS_nemo.report.jsonl   # Per-probe hit log
├── YYYYMMDD_HHMMSS_nemo.report.html    # Visual report
└── YYYYMMDD_HHMMSS_garak.log           # Full console output
```
