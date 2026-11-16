#!/usr/bin/env bash
# =============================================================================
# Experiment: System-Prompt-as-Guardrail (LLM-as-a-Judge)
# =============================================================================
# Starts the FastAPI server wrapping SystemPromptGuardrail, connects to the
# remote OpenAI-compatible API (SYSTEM_PROMPT_MAIN_MODEL_API_BASE), runs garak
# red-team probes against it, saves results to impenetra/experiments/system_prompt_as_guardrail/results/,
# then cleans up.
#
# Usage:
#   cd /path/to/impenetra
#   bash impenetra/experiments/system_prompt_as_guardrail/run.sh [SYSTEM_PROMPT_MAIN_MODEL_API_BASE]
#
# Examples:
#   bash impenetra/experiments/system_prompt_as_guardrail/run.sh https://random-subdomain.trycloudflare.com
#   export SYSTEM_PROMPT_MAIN_MODEL_API_BASE="https://random-subdomain.trycloudflare.com"
#   bash impenetra/experiments/system_prompt_as_guardrail/run.sh
# =============================================================================
set -euo pipefail

EXPERIMENT="system_prompt_as_guardrail"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
PORT=8000
SERVER_PID=""

# Accept Colab tunnel base domain from $1 or environment variable
SYSTEM_PROMPT_MAIN_MODEL_API_BASE="${1:-${SYSTEM_PROMPT_MAIN_MODEL_API_BASE:-}}"
if [[ -z "$SYSTEM_PROMPT_MAIN_MODEL_API_BASE" ]]; then
    echo -e "\033[0;31m❌ SYSTEM_PROMPT_MAIN_MODEL_API_BASE is required.\033[0m"
    echo -e "   Usage: bash impenetra/experiments/system_prompt_as_guardrail/run.sh https://<subdomain>.trycloudflare.com"
    echo -e "      or: export SYSTEM_PROMPT_MAIN_MODEL_API_BASE='https://<subdomain>.trycloudflare.com'"
    exit 1
fi

# Strip trailing slashes and any trailing /v1 to ensure domain-only format
SYSTEM_PROMPT_MAIN_MODEL_API_BASE="${SYSTEM_PROMPT_MAIN_MODEL_API_BASE%/}"
SYSTEM_PROMPT_MAIN_MODEL_API_BASE="${SYSTEM_PROMPT_MAIN_MODEL_API_BASE%/v1}"
export SYSTEM_PROMPT_MAIN_MODEL_API_BASE="$SYSTEM_PROMPT_MAIN_MODEL_API_BASE"

GENERATIONS="${GARAK_GENERATIONS:-1}"
PARALLEL_ATTEMPTS="${GARAK_PARALLEL_ATTEMPTS:-32}"
SPEC="${GARAK_SPEC:-${GARAK_PROBES:-probes.promptinject,probes.dan.DanInTheWild,probes.dan.AutoDANCached,probes.encoding,probes.grandma,probes.goodside,probes.sysprompt_extraction,probes.off_charter,probes.donotanswer,probes.phrasing,probes.topic}}"
SAMPLE_ONLY="${SAMPLE_ONLY:-false}"

# --- Colours ---
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo -e "\n${YELLOW}⏹  Stopping server (PID $SERVER_PID)...${NC}"
        kill "$SERVER_PID" && wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════╗"
echo "║ Experiment: System Prompt LLM-as-a-Judge     ║"
echo "╚══════════════════════════════════════════════╝"
echo -e "${NC}"

# ---------------------------------------------------------------------------
# 1. Pre-flight checks & cleanup
# ---------------------------------------------------------------------------
echo -e "${YELLOW}[1/6] Pre-flight checks & clearing existing results...${NC}"
mkdir -p "$RESULTS_DIR"
python -m impenetra.experiments.cleaner --dir "$RESULTS_DIR"
echo "Cleared existing results in $RESULTS_DIR"

# ---------------------------------------------------------------------------
# 2. Start server in background
# ---------------------------------------------------------------------------
echo -e "\n${YELLOW}[2/6] Starting Impenetra server on port $PORT...${NC}"
cd "$PROJECT_ROOT"
python impenetra/experiments/system_prompt_as_guardrail/server.py &
SERVER_PID=$!

echo "Waiting for server to become ready at http://127.0.0.1:$PORT..."
SERVER_READY=false
for i in {1..30}; do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        SERVER_READY=true
        echo -e "${GREEN}✓ Server ready (PID $SERVER_PID)${NC}"
        break
    fi
    sleep 1
done

if [[ "$SERVER_READY" != "true" ]]; then
    echo -e "${RED}Server failed to start within 30 seconds.${NC}"
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ---------------------------------------------------------------------------
# 3. Benchmark Run (On-Charter & Negative Prompts)
# ---------------------------------------------------------------------------
echo -e "\n${YELLOW}[3/6] Running benchmark prompts & recording baseline metrics...${NC}"
python -m impenetra.experiments.metrics reset --url "http://127.0.0.1:$PORT"

BENCH_FILE="$RESULTS_DIR/${TIMESTAMP}_bench.json"
python impenetra/experiments/benchmark.py \
    --url "http://127.0.0.1:$PORT" \
    --experiment "$EXPERIMENT" \
    --out "$BENCH_FILE"

python -m impenetra.experiments.metrics record \
    --experiment "$EXPERIMENT" \
    --run-type "bench" \
    --url "http://127.0.0.1:$PORT" \
    --out-dir "$RESULTS_DIR" \
    --timestamp "$TIMESTAMP"

# ---------------------------------------------------------------------------
# 4. Garak Probe Evaluation
# ---------------------------------------------------------------------------
echo -e "\n${YELLOW}[4/6] Running Garak probe evaluation...${NC}"
python -m impenetra.experiments.metrics reset --url "http://127.0.0.1:$PORT"

python -m impenetra.garak_plugins.setup install &>/dev/null || true

GARAK_CONFIG="$PROJECT_ROOT/impenetra/configs/garak_openai_config.json"
case "${SAMPLE_ONLY:-0}" in
    [tT][rR][uU][eE]|1|[yY][eE][sS])
        GARAK_CONFIG="$PROJECT_ROOT/impenetra/configs/garak_openai_sample_config.json"
        echo -e "${CYAN}   → Using sample config (soft probe prompt cap: 10)${NC}"
        ;;
esac

echo "   → Generations: $GENERATIONS | Parallel attempts: $PARALLEL_ATTEMPTS"
echo "   → Probes: $SPEC"

# Configure off_charter probes to use mitigation.MitigationBypass detector
PROBE_OPTIONS='{"off_charter": {"primary_detector": "mitigation.MitigationBypass", "extended_detectors": []}}'

python -m garak \
    --target_type rest \
    --config "$GARAK_CONFIG" \
    --generator_option_file "$GARAK_CONFIG" \
    --probe_options "$PROBE_OPTIONS" \
    --spec "$SPEC" \
    --generations "$GENERATIONS" \
    --parallel_attempts "$PARALLEL_ATTEMPTS" \
    --report_prefix "$RESULTS_DIR/${TIMESTAMP}_systemprompt" \
    2>&1 | tee "$RESULTS_DIR/${TIMESTAMP}_garak.log" || true

# Copy from garak_home if garak saved elsewhere
garak_home="${GARAK_HOME:-$HOME/.local/share/garak}"
if [[ -d "$garak_home/runs" ]]; then
    latest_run=$(ls -t "$garak_home/runs/" 2>/dev/null | head -1)
    if [[ -n "$latest_run" ]]; then
        cp -r "$garak_home/runs/$latest_run"* "$RESULTS_DIR/" 2>/dev/null || true
    fi
fi

# ---------------------------------------------------------------------------
# 5. Record Server Metrics Post-Garak
# ---------------------------------------------------------------------------
echo -e "\n${YELLOW}[5/6] Recording telemetry metrics post-Garak...${NC}"
python -m impenetra.experiments.metrics record \
    --experiment "$EXPERIMENT" \
    --run-type "garak" \
    --url "http://127.0.0.1:$PORT" \
    --out-dir "$RESULTS_DIR" \
    --timestamp "$TIMESTAMP"

# ---------------------------------------------------------------------------
# 6. Compile Final Report
# ---------------------------------------------------------------------------
echo -e "\n${YELLOW}[6/6] Compiling final report...${NC}"
python -m impenetra.experiments.compile_results --results-dir "$RESULTS_DIR"

echo -e "\n${GREEN}🏁 Experiment complete! Results saved in:${NC} $RESULTS_DIR\n"
