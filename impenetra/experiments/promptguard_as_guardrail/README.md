# Experiment: PromptGuard-as-Guardrail (Meta Llama Prompt Guard 2 86M API Integration)

This experiment evaluates **Meta Llama Prompt Guard 2 86M** (`meta-llama/Llama-Prompt-Guard-2-86M`) as an input guardrail for Impenetra. The guardrail delegates prompt classification to an external FastAPI batched PyTorch GPU service (as configured in [prompt_guard_colab.ipynb](../../../notebooks/prompt_guard_colab.ipynb)).

---

## Architecture Overview

```
                      ┌───────────────────────────────────────────┐
                      │    Meta Llama Prompt Guard 2 86M API      │
                      │    (Batched PyTorch GPU / T4 / Colab)     │
                      │    POST /predict                          │
                      │    {"prompt": "<user text>"}              │
                      └─────────────────────┬─────────────────────┘
                                            │
                             {"benign": f, "malicious": f}
                                            │
                                            ▼
[User / Garak Query] ──► [PromptGuardGuardrail]
                               │
                 malicious >= 0.8 (threshold)?
                               │
             ┌─────────────────┴─────────────────┐
             ▼                                   ▼
         [UNSAFE]                             [SAFE]
    allowed=False                        allowed=True
    category="malicious"                 category=""
    Refusal Interception                 Forwarded to Assistant /
                                         Food Agent LLM
```

---

## Classification Logic & Threshold

- **Endpoint**: `POST /predict`
- **Request Body**: `{"prompt": "<raw text>"}`
- **Response**: `{"benign": float, "malicious": float}`
- **Threshold**: `MALICIOUS_THRESHOLD = 0.8` (defined at top of `guardrail.py`).
- **Decision**:
  - If `malicious >= 0.8`: The request is blocked with `category="malicious"`.
  - If `malicious < 0.8`: The request passes through (`allowed=True`).
  - Covers both prompt injections and jailbreaks in a single unified malicious classification.

---

## Quickstart & Reproduction

### 1. Launch Prompt Guard 2 API Server
Run the Google Colab notebook [notebooks/prompt_guard_colab.ipynb](../../../notebooks/prompt_guard_colab.ipynb) on a T4 GPU runtime. Note the public Cloudflare tunnel URL (or local port `8000`).

### 2. Run Experiment

```bash
# Pass URL directly as first argument:
bash impenetra/experiments/promptguard_as_guardrail/run.sh https://random-subdomain.trycloudflare.com

# Or set PROMPT_GUARD_URL environment variable:
export PROMPT_GUARD_URL="https://random-subdomain.trycloudflare.com"
bash impenetra/experiments/promptguard_as_guardrail/run.sh
```

---

## Standalone Server Execution

```bash
export PROMPT_GUARD_URL="http://127.0.0.1:8000"
python impenetra/experiments/promptguard_as_guardrail/server.py
```

---

## Fail-Open Logging

If the remote Prompt Guard API is unreachable or experiences HTTP errors, the guardrail fails open (`allowed=True`, `category="guardrail_error"`) to ensure service availability while recording the failure in:
- `impenetra/experiments/promptguard_as_guardrail/results/fail_open.log` (human-readable text)
- `impenetra/experiments/promptguard_as_guardrail/results/fail_open.jsonl` (structured JSON Lines)
