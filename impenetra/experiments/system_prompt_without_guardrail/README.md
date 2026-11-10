# Experiment 3: System-Prompt-Without-Guardrail (Baseline)

> **Goal:** Evaluate defense capability when NO programmatic guardrail function is active, relying solely on an injected charter system prompt in `impenetra`'s API framework.

## Architecture

This experiment implements `PassThroughGuardrail` (inheriting from `BaseGuardrail`) where `check(text)` immediately returns `allowed=True` with `0.0ms` latency overhead.

The safety mechanism is the charter system prompt injected directly into `create_app(system_prompt=...)`:

```
Client / Garak Probe
        │
        ▼
Impenetra FastAPI (:8000)
        │
        ├── PassThroughGuardrail.check()  → allowed=True (0ms latency overhead)
        │
        ▼
Ollama (llama3.1:8b-instruct-q8_0)
  [System prompt prepended with charter rules]
        │
        ▼
LLM Refusal or Completion
```

## Running the Server

```bash
cd /path/to/impenetra
.venv/bin/python3 impenetra/experiments/system_prompt_without_guardrail/server.py
```

## Running Benchmark

```bash
bash impenetra/experiments/system_prompt_without_guardrail/run.sh
```
