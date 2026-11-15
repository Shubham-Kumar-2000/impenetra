# Experiment: System-Prompt-as-Guardrail (LLM-as-a-Judge)

> **Goal:** Evaluate defense capability using an LLM-as-a-Judge guardrail architecture. On each incoming request, the guardrail queries the main model with a specialized safety and charter judge prompt to classify the prompt as `BENIGN` or `VIOLATION`. Allowed prompts receive `DEFAULT_ACCEPTANCE` and blocked prompts receive `DEFAULT_REFUSAL`.

## Architecture

```
Client / Garak Probe
        │
        ▼
Impenetra FastAPI (:8000)
        │
        ├── SystemPromptGuardrail.check(text)
        │       │
        │       ▼
        │   Remote Main LLM (POST {SYSTEM_PROMPT_MAIN_MODEL_API_BASE}/chat/completions)
        │   [Evaluates prompt with JUDGE_SYSTEM_PROMPT]
        │       │
        │       ▼
        │   Verdict: "BENIGN" -> allowed=True
        │            "VIOLATION" / refusal -> allowed=False
        │
        ├── If allowed: returns DEFAULT_ACCEPTANCE ("Yes, I am working on / processing your request: '{prompt}'.")
        └── If blocked: returns DEFAULT_REFUSAL ("I'm sorry, but I cannot fulfill this request...")
```

## Environment Variables

| Variable | Description | Example |
|---|---|---|
| `SYSTEM_PROMPT_MAIN_MODEL_API_BASE` | Base URL of remote OpenAI-compatible server | `https://xxxx.trycloudflare.com` |
| `MAIN_MODEL_NAME` | Model name served on OpenAI endpoint (optional) | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| `PORT` | Local FastAPI server port (optional, default: 8000) | `8000` |

## Running the Server

```bash
export SYSTEM_PROMPT_MAIN_MODEL_API_BASE="https://xxxx.trycloudflare.com"
python impenetra/experiments/system_prompt_as_guardrail/server.py
```

## Running Benchmark & Garak Probes

```bash
bash impenetra/experiments/system_prompt_as_guardrail/run.sh https://xxxx.trycloudflare.com
```

### Probe Options
`run.sh` configures the `off_charter` probe to evaluate against `mitigation.MitigationBypass`:
```bash
PROBE_OPTIONS='{"off_charter": {"primary_detector": "mitigation.MitigationBypass", "extended_detectors": []}}'
```

