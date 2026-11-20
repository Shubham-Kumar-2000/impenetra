#!/usr/bin/env bash
# =============================================================================
# impenetra/experiments/run_all.sh
# =============================================================================
# Master orchestration script. Runs each experiment's run.sh sequentially:
#   1. llamaguard_as_guardrail/run.sh
#   2. nemo_guardrails/run.sh
#   3. system_prompt_as_guardrail/run.sh
#   4. dspy_gepa_system_prompt_as_guardrail/run.sh
#   5. promptguard_as_guardrail/run.sh
#   6. jev_systemone_noul_as_guardrail/run.sh
#   7. jev_systemone_choice_as_guardrail/run.sh
#   8. vector_similarity_as_guardrail/run.sh
#
# Finally, compiles and compares the cross-experiment results via compare_results.py.
#
# Usage (from project root):
#   bash impenetra/experiments/run_all.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colours
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

header() {
    echo -e "\n${CYAN}${BOLD}╔══════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}${BOLD}║  $1${NC}"
    echo -e "${CYAN}${BOLD}╚══════════════════════════════════════════════════╝${NC}\n"
}

cd "$PROJECT_ROOT"

export SAMPLE_ONLY="${SAMPLE_ONLY:-false}"

# ---------------------------------------------------------------------------
# 1. Run Experiment 1: LlamaGuard
# ---------------------------------------------------------------------------
header "Experiment 1: LlamaGuard-as-Guardrail           "
bash "$SCRIPT_DIR/llamaguard_as_guardrail/run.sh" "$@"

# ---------------------------------------------------------------------------
# 2. Run Experiment 2: NeMo Guardrails
# ---------------------------------------------------------------------------
header "Experiment 2: NeMo Guardrails                  "
bash "$SCRIPT_DIR/nemo_guardrails/run.sh" "$@"

# ---------------------------------------------------------------------------
# 3. Run Experiment 3: System Prompt Guardrail (LLM-as-a-Judge)
# ---------------------------------------------------------------------------
header "Experiment 3: System Prompt Guardrail          "
if [[ -n "${SYSTEM_PROMPT_MAIN_MODEL_API_BASE:-}" ]] || [[ $# -ge 1 && "$1" == http* ]]; then
    bash "$SCRIPT_DIR/system_prompt_as_guardrail/run.sh" "$@"
else
    echo -e "${YELLOW}⚠️ Skipping System Prompt Guardrail: SYSTEM_PROMPT_MAIN_MODEL_API_BASE is not set.${NC}"
fi

# ---------------------------------------------------------------------------
# 4. Run Experiment 4: DSPy-GEPA Optimized System Prompt Guardrail
# ---------------------------------------------------------------------------
header "Experiment 4: DSPy-GEPA System Prompt Guardrail "
if [[ -n "${SYSTEM_PROMPT_MAIN_MODEL_API_BASE:-}" ]] || [[ $# -ge 1 && "$1" == http* ]]; then
    bash "$SCRIPT_DIR/dspy_gepa_system_prompt_as_guardrail/run.sh" "$@"
else
    echo -e "${YELLOW}⚠️ Skipping DSPy-GEPA Guardrail: SYSTEM_PROMPT_MAIN_MODEL_API_BASE is not set.${NC}"
fi

# ---------------------------------------------------------------------------
# 5. Run Experiment 5: Prompt Guard 2 86M
# ---------------------------------------------------------------------------
header "Experiment 5: Prompt Guard 2 86M               "
bash "$SCRIPT_DIR/promptguard_as_guardrail/run.sh" "$@"

# ---------------------------------------------------------------------------
# 6. Run Experiment 6: Jev SystemOne Noul Guardrail (TypeSafe.ai)
# ---------------------------------------------------------------------------
header "Experiment 6: Jev SystemOne Noul Guardrail      "
if [[ -n "${TYPESAFE_API_KEY:-}" ]] || [[ $# -ge 1 && "$1" != http* ]]; then
    bash "$SCRIPT_DIR/jev_systemone_noul_as_guardrail/run.sh" "$@"
else
    echo -e "${YELLOW}⚠️ Skipping Jev SystemOne Noul: TYPESAFE_API_KEY is not set.${NC}"
fi

# ---------------------------------------------------------------------------
# 7. Run Experiment 7: Jev SystemOne Choice Guardrail (TypeSafe.ai)
# ---------------------------------------------------------------------------
header "Experiment 7: Jev SystemOne Choice Guardrail    "
if [[ -n "${TYPESAFE_API_KEY:-}" ]] || [[ $# -ge 1 && "$1" != http* ]]; then
    bash "$SCRIPT_DIR/jev_systemone_choice_as_guardrail/run.sh" "$@"
else
    echo -e "${YELLOW}⚠️ Skipping Jev SystemOne Choice: TYPESAFE_API_KEY is not set.${NC}"
fi

# ---------------------------------------------------------------------------
# 8. Run Experiment 8: Vector Similarity Guardrail (Cosine + Qdrant)
# ---------------------------------------------------------------------------
header "Experiment 8: Vector Similarity Guardrail        "
bash "$SCRIPT_DIR/vector_similarity_as_guardrail/run.sh" "$@"

# ---------------------------------------------------------------------------
# 9. Compare Results Across All Experiments
# ---------------------------------------------------------------------------
header "Comparing Results Across All Experiments       "
python -m impenetra.experiments.compare_results --results-dir "$SCRIPT_DIR"

echo -e "\n${GREEN}${BOLD}🏁 All experiments and comparison complete!${NC}\n"
