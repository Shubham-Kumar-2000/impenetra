# Vector Similarity as a Low-Latency Guardrail

Ultra-low latency guardrail combining dense vector cosine similarity, token length validation, and a short-prompt bypass for the Crave Craft food agent.

## Architecture

1. **Short Prompt Fast-Path**: Prompts with $\le \text{min\_token\_threshold}$ (e.g. `hi`, `yes`, `cancel`, `thanks`) pass immediately without vector computation.
2. **Boot-Time In-Memory Embeddings**: Pre-computes normalized 384-dimensional dense vectors of on-charter food agent intents using `all-MiniLM-L6-v2`.
3. **Off-Charter Vector DB (Qdrant)**: Stores adversarial and off-charter negative samples pre-seeded by `fill_negative_samples.py` using $\min(10\%, 10)$ sampling.
4. **Strict Verification**:
   - Matches off-charter Qdrant vectors with $\ge 95\%$ cosine similarity and $\le \min(10\%, 5)$ excess tokens $\rightarrow$ **BLOCK**.
   - Matches on-charter in-memory intent vectors with $\ge 95\%$ cosine similarity and $\le \min(10\%, 5)$ excess tokens $\rightarrow$ **ALLOW**.
   - Unmatched queries $\rightarrow$ **BLOCK** (default deny).

## Configuration

All connection credentials and thresholds are configured in `on_charter_intents.json`:
- `qdrant_config.url`: Qdrant endpoint (e.g. `http://localhost:6333`).
- `guardrail_config.similarity_threshold`: `0.95`.
- `guardrail_config.token_diff_percent`: `0.10`.
- `guardrail_config.secured_max_excess_tokens`: `5`.
- `guardrail_config.min_token_threshold`: `3`.

## Usage

1. Start Qdrant:
   ```bash
   docker compose up -d
   ```
2. Populate Qdrant with negative samples:
   ```bash
   python -m impenetra.experiments.vector_similarity_as_guardrail.fill_negative_samples
   ```
3. Run benchmark and Garak evaluation:
   ```bash
   bash impenetra/experiments/vector_similarity_as_guardrail/run.sh
   ```
