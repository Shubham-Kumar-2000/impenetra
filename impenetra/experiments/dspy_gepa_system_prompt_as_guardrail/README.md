# Experiment: DSPy-GEPA Optimized System-Prompt-as-Guardrail

> **Goal:** Optimize an LLM-as-a-Judge system prompt using **DSPy GEPA (Reflective Evolutionary Prompt Optimization)**. Unlike greedy search optimizers, GEPA uses reflection over rich textual feedback to evolve instructions across an expanded training set ($\min(80\%, 100)$ prompts per probe subcategory). The winning instruction is dumped to `trained_system_prompt.txt` and evaluated via FastAPI server against benchmark prompts and Garak red-teaming probes.

---

## GEPA Optimization Highlights

- **Optimizer:** Reflective Evolutionary Prompt Optimization (`dspy.GEPA`)
- **Search Mechanics:** Genetic / evolutionary prompt mutation guided by LLM reflection
- **Feedback Signal:** Rich textual diagnostic feedback on false positives / false negatives
- **Search Space:** `auto="light"` (or `"medium"`, `"heavy"`) with Pareto frontier tracking
- **Data Sampling:** $\min(80\% \text{ per subcategory}, 100)$ (~1,440 prompts across 70 probe subcategories covering all Garak red-teaming vectors & benchmark datasets)
- **Checkpointing / Resume:** Supports `--log-dir <dir>` to checkpoint every iteration and resume interrupted runs automatically

---

## Step 1: Train & Optimize System Prompt with GEPA

```bash
# Set your remote student model (OpenAI-compatible endpoint, e.g. Llama-3.1-8B)
export SYSTEM_PROMPT_MAIN_MODEL_API_BASE="https://xxxx.trycloudflare.com"

# Set your Gemini API key for the trainer/reflection model (Gemini Flash, no fallback)
export GEMINI_API_KEY="your_gemini_api_key"

# Run DSPy GEPA optimization
python impenetra/experiments/dspy_gepa_system_prompt_as_guardrail/train_prompt.py
```

### Options:
- `--trainer-model`: Gemini Flash reflection model (default: `gemini/gemini-3.8-flash`).
- `--sample-ratio`: Ratio per subcategory (default: `0.80`).
- `--max-samples`: Max sample cap per subcategory (default: `100`).
- `--min-samples`: Min samples per subcategory (default: `1`).
- `--auto`: GEPA reflection budget: `light` (default), `medium`, `heavy`.
- `--log-dir`: Directory for saving/loading GEPA checkpoints (allows resuming).
- `--out`: Destination file path (default: `trained_system_prompt.txt`).
- `--dry-run`: Test subcategory sampling and verify pipeline without remote LLM API calls.

```bash
# Test sampling & generate baseline seed prompt:
python impenetra/experiments/dspy_gepa_system_prompt_as_guardrail/train_prompt.py --dry-run
```

---

## Step 2: Run Evaluation (Benchmark & Garak Probes)

Once `trained_system_prompt.txt` has been dumped, run the evaluation:

```bash
bash impenetra/experiments/dspy_gepa_system_prompt_as_guardrail/run.sh https://xxxx.trycloudflare.com
```
