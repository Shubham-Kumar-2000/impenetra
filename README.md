# Impenetra

> Low-latency, high-accuracy guardrails research for AI agents.

## Overview

Impenetra is a research platform for developing and benchmarking guardrail systems
that protect AI agents from harmful inputs (jailbreaks, prompt injection, toxic content)
with minimal latency overhead.

### Key Research Directions

- **Early-exit transformers** — Skip unnecessary computation layers when confidence is high
- **Semantic routing** — Fast intent classification via lightweight embeddings (MiniLM)
- **Benchmark suite** — Compare against NVIDIA NeMo Guardrails and evaluate with Garak LLM vulnerability probes
- **Profiling** — Measure wall-clock time, CPU time, and memory footprint per guardrail call

## Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install core dependencies (includes garak, fastapi, uvicorn)
pip install -e ".[dev]"

# (Optional) Install notebook support
pip install -e ".[notebooks]"
```

## Quick Start

### 1. Profiling Guardrail Functions

```python
from impenetra.profiling.profiler import profile_all
from impenetra.models.embeddings import GuardrailEmbedder

@profile_all
def check_similarity():
    embedder = GuardrailEmbedder()
    return embedder.similarity(
        ["How do I hack a system?"],
        ["What is cybersecurity?"]
    )

result, profile = check_similarity()
print(profile)  # ⏱ check_similarity: 45.23ms wall | 📦 12.50MB peak
```

### 2. Serving Guardrail API & Running Garak Assessment

Start the FastAPI server:

```bash
uvicorn impenetra.api.app:app --host 127.0.0.1 --port 8000
```

Run Garak against the `/chat` endpoint using the REST generator:

```bash
python -m garak --model_type rest --config impenetra/configs/garak_chat_config.json --probes dan
```

Or run Garak against the OpenAI-compatible `/v1/chat/completions` endpoint:

```bash
python -m garak --model_type rest --config impenetra/configs/garak_openai_config.json --probes dan
```

## Project Structure

```
impenetra/
├── api/          # FastAPI serving endpoints for Garak and agents
├── core/         # Base guardrail interfaces and rule baselines
├── models/       # Early-exit transformer, embeddings
├── datasets/     # Dataset registry & loaders (CLINC150, ToxicChat, XSTest)
├── benchmarks/   # NeMo baseline, latency/accuracy benchmarks
├── profiling/    # Time & memory profiling utilities
└── configs/      # YAML configuration and Garak REST JSON templates
```

## Running Tests

```bash
# Fast tests only
pytest tests/ -m "not slow"

# All tests (downloads models)
pytest tests/
```

## License

MIT
