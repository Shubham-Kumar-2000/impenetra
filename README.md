# Impenetra

> Low-latency, high-accuracy guardrails research for AI agents.

## Overview

Impenetra is a research platform for developing and benchmarking guardrail systems
that protect AI agents from harmful inputs (jailbreaks, prompt injection, toxic content)
with minimal latency overhead.

### Key Research Directions

- **Early-exit transformers** — Skip unnecessary computation layers when confidence is high
- **Semantic routing** — Fast intent classification via lightweight embeddings (MiniLM)
- **Benchmark suite** — Compare against NVIDIA NeMo Guardrails on CLINC150, ToxicChat, XSTest, AdvBench
- **Profiling** — Measure wall-clock time, CPU time, and memory footprint per guardrail call

## Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install core dependencies
pip install -e ".[dev]"

# (Optional) Install notebook support
pip install -e ".[notebooks]"
```

## Quick Start

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

## Project Structure

```
impenetra/
├── core/         # Base guardrail interfaces
├── models/       # Early-exit transformer, embeddings
├── datasets/     # Dataset registry & loaders
├── benchmarks/   # NeMo baseline, latency/accuracy benchmarks
├── profiling/    # Time & memory profiling utilities
└── configs/      # YAML configuration files
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
