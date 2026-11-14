#!/usr/bin/env bash
# =============================================================================
# impenetra/experiments/run_all.sh
# =============================================================================
# Master orchestration script. Runs each experiment's run.sh sequentially:
#   1. llamaguard_as_guardrail/run.sh
#   2. nemo_guardrails/run.sh
#   3. system_prompt_without_guardrail/run.sh
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
# 3. Run Experiment 3: System Prompt Baseline
# ---------------------------------------------------------------------------
header "Experiment 3: System Prompt Baseline           "
bash "$SCRIPT_DIR/system_prompt_without_guardrail/run.sh" "$@"

# ---------------------------------------------------------------------------
# 4. Compare Results Across Experiments
# ---------------------------------------------------------------------------
header "Comparing Results Across All Experiments       "
python -m impenetra.experiments.compare_results --results-dir "$SCRIPT_DIR"

echo -e "\n${GREEN}${BOLD}🏁 All experiments and comparison complete!${NC}\n"
