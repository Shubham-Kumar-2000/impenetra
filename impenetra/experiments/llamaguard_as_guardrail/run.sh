#!/usr/bin/env bash
# =============================================================================
# Experiment 1: LlamaGuard-as-Guardrail (Gradio Space Integration)
# =============================================================================
# Starts the FastAPI guardrail server connecting to a LlamaGuard Gradio Space
# (llama-guard-space or custom Gradio URL), runs garak red-team probes against it,
# saves results to impenetra/experiments/llamaguard_as_guardrail/results/, then cleans up.
#
# Usage:
#   cd /path/to/impenetra
#   bash impenetra/experiments/llamaguard_as_guardrail/run.sh [GRADIO_URL]
#
# Examples:
#   bash impenetra/experiments/llamaguard_as_guardrail/run.sh http://127.0.0.1:7860
#   bash impenetra/experiments/llamaguard_as_guardrail/run.sh https://username-llama-guard-space.hf.space
#
# Prerequisites:
#   - LlamaGuard Gradio Space running (local or Hugging Face Space)
#   - pip install -e ".[dev]"
# =============================================================================
set -euo pipefail

EXPERIMENT="llamaguard_as_guardrail"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
PORT=8000
SERVER_PID=""

# Accept Gradio URL from $1 or environment variable
GRADIO_URL="${1:-${LLAMA_GUARD_GRADIO_URL:-http://127.0.0.1:7860}}"
export LLAMA_GUARD_GRADIO_URL="$GRADIO_URL"
GENERATIONS="${GARAK_GENERATIONS:-1}"
PARALLEL_ATTEMPTS="${GARAK_PARALLEL_ATTEMPTS:-32}"
SPEC="${GARAK_SPEC:-${GARAK_PROBES:-probes.promptinject,probes.dan.DanInTheWild,probes.dan.AutoDANCached,probes.encoding,probes.grandma,probes.goodside,probes.sysprompt_extraction,probes.off_charter}}"

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
echo "║   Experiment 1: LlamaGuard (Gradio Space)    ║"
echo "╚══════════════════════════════════════════════╝"
echo -e "${NC}"

# ---------------------------------------------------------------------------
# 1. Pre-flight checks
# ---------------------------------------------------------------------------
echo -e "${YELLOW}[1/6] Pre-flight checks & clearing existing results...${NC}"

command -v python  &>/dev/null || { echo -e "${RED}❌ python not found${NC}"; exit 1; }
python -m garak --help &>/dev/null 2>&1 || { echo -e "${RED}❌ garak not installed. Run: pip install -e '.[dev]'${NC}"; exit 1; }

# Clear existing results in this experiment directory
python -m impenetra.experiments.cleaner --dir "$RESULTS_DIR"
echo -e "${GREEN}✓ Cleared existing results in $RESULTS_DIR${NC}"

echo -e "Target Gradio URL: ${CYAN}$GRADIO_URL${NC}"
echo -e "Garak generations: ${CYAN}$GENERATIONS${NC}"
echo -e "Garak parallelism (attempts): ${CYAN}$PARALLEL_ATTEMPTS${NC}"

if [[ "$GRADIO_URL" =~ ^https?:// ]]; then
    if curl -sf --max-time 3 "$GRADIO_URL" >/dev/null 2>&1; then
        echo -e "${GREEN}✓ Gradio endpoint reachable at $GRADIO_URL${NC}"
    else
        echo -e "${YELLOW}⚠️  Gradio endpoint $GRADIO_URL is not immediately responding; guardrail will attempt connection during runtime.${NC}"
    fi
fi

echo -e "${GREEN}✅ All pre-flight checks passed${NC}"

# ---------------------------------------------------------------------------
# 2. Start the guardrail server
# ---------------------------------------------------------------------------
echo -e "\n${YELLOW}[2/6] Starting LlamaGuard server on port $PORT...${NC}"
cd "$PROJECT_ROOT"
python impenetra/experiments/llamaguard_as_guardrail/server.py &
SERVER_PID=$!

# Wait for server to be ready (up to 15s)
for i in {1..15}; do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Server ready (PID $SERVER_PID)${NC}"
        break
    fi
    if [[ $i -eq 15 ]]; then
        echo -e "${RED}❌ Server did not start in time${NC}"; exit 1
    fi
    sleep 1
done

# ---------------------------------------------------------------------------
# 3. Smoke test
# ---------------------------------------------------------------------------
echo -e "\n${YELLOW}[3/6] Running smoke test...${NC}"
SMOKE=$(curl -sf -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"test","messages":[{"role":"user","content":"How many calories in an apple?"}]}')
echo "Smoke response: $(echo "$SMOKE" | python -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'][:80])")"
echo -e "${GREEN}✅ Smoke test passed${NC}"

# ---------------------------------------------------------------------------
# 4. Run Benchmark (On-charter food & benign queries + baseline attacks)
# ---------------------------------------------------------------------------
echo -e "\n${YELLOW}[4/6] Running on-charter & adversarial benchmark...${NC}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "$RESULTS_DIR"

# Reset server metrics before benchmark
python -m impenetra.experiments.metrics reset --url "http://127.0.0.1:$PORT"

# Benchmark simply tries prompts and stores prompt outcomes in bench.json
python impenetra/experiments/benchmark.py \
    --url "http://127.0.0.1:$PORT" \
    --experiment "$EXPERIMENT" \
    --out "$RESULTS_DIR/${TIMESTAMP}_bench.json"

# Record benchmark metrics from server telemetry (/metrics)
python -m impenetra.experiments.metrics record \
    --experiment "$EXPERIMENT" \
    --run-type "bench" \
    --url "http://127.0.0.1:$PORT" \
    --out-dir "$RESULTS_DIR" \
    --timestamp "$TIMESTAMP"

# ---------------------------------------------------------------------------
# 5. Run Garak (Red-team adversarial probes)
# ---------------------------------------------------------------------------
echo -e "\n${YELLOW}[5/6] Running garak red-team probes...${NC}"
echo -e "   → Generations: ${CYAN}$GENERATIONS${NC} | Parallel attempts: ${CYAN}$PARALLEL_ATTEMPTS${NC}"
REPORT_PREFIX="$RESULTS_DIR/${TIMESTAMP}_llamaguard"

# Reset server metrics before Garak
python -m impenetra.experiments.metrics reset --url "http://127.0.0.1:$PORT"

python -m impenetra.garak_plugins.setup install &>/dev/null || true

python -m garak \
    --target_type rest \
    --config "$PROJECT_ROOT/impenetra/configs/garak_openai_config.json" \
    --generator_option_file "$PROJECT_ROOT/impenetra/configs/garak_openai_config.json" \
    --spec "$SPEC" \
    --generations "$GENERATIONS" \
    --parallel_attempts "$PARALLEL_ATTEMPTS" \
    --report_prefix "$REPORT_PREFIX" \
    2>&1 | tee "$RESULTS_DIR/${TIMESTAMP}_garak.log"

echo -e "\n${GREEN}✅ Garak scan complete${NC}"

# Copy from garak_home if garak saved elsewhere
garak_home="${GARAK_HOME:-$HOME/.local/share/garak}"
if [[ -d "$garak_home/runs" ]]; then
    latest_run=$(ls -t "$garak_home/runs/" 2>/dev/null | head -1)
    if [[ -n "$latest_run" ]]; then
        cp -r "$garak_home/runs/$latest_run"* "$RESULTS_DIR/" 2>/dev/null || true
    fi
fi

# Find report.jsonl in RESULTS_DIR
REPORT_JSONL=$(ls -t "$RESULTS_DIR"/*.report.jsonl 2>/dev/null | head -1 || true)

# Record Garak metrics from server telemetry (/metrics)
python -m impenetra.experiments.metrics record \
    --experiment "$EXPERIMENT" \
    --run-type "garak" \
    --url "http://127.0.0.1:$PORT" \
    --out-dir "$RESULTS_DIR" \
    ${REPORT_JSONL:+--report-jsonl "$REPORT_JSONL"} \
    --timestamp "$TIMESTAMP"

# ---------------------------------------------------------------------------
# 6. Compile Final Report & Summary
# ---------------------------------------------------------------------------
echo -e "\n${YELLOW}[6/6] Compiling final report...${NC}"
python -m impenetra.experiments.compile_results \
    --results-dir "$RESULTS_DIR" \
    --experiment "$EXPERIMENT"

echo -e "\n${YELLOW}Results saved in $RESULTS_DIR:${NC}"
ls -lh "$RESULTS_DIR/${TIMESTAMP}"* 2>/dev/null || echo "(check $RESULTS_DIR/)"
if [[ -f "$RESULTS_DIR/fail_open.log" ]]; then
    FAIL_COUNT=$(grep -c "FAIL-OPEN EVENT" "$RESULTS_DIR/fail_open.log" || true)
    if [[ "$FAIL_COUNT" -gt 0 ]]; then
        echo -e "\n${RED}⚠️  Fail-open warning: $FAIL_COUNT fail-open event(s) logged in:${NC}"
        echo -e "   ${YELLOW}$RESULTS_DIR/fail_open.log${NC} (and .jsonl)"
    fi
fi
echo -e "\n${GREEN}🏁 Experiment '$EXPERIMENT' complete!${NC}"
echo -e "   Results → ${CYAN}$RESULTS_DIR/${NC}"
