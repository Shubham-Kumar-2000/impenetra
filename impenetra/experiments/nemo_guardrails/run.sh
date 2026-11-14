#!/usr/bin/env bash
# =============================================================================
# Experiment 2: NeMo Guardrails
# =============================================================================
# Starts the FastAPI server wrapping NVIDIA NeMo Guardrails, runs garak
# red-team probes against it, saves results to impenetra/experiments/nemo_guardrails/results/, then cleans up.
#
# Usage:
#   cd /path/to/impenetra
#   bash impenetra/experiments/nemo_guardrails/run.sh
#
# Prerequisites:
#   - ollama serve  (running in background)
#   - ollama pull llama3.1:8b-instruct-q8_0
#   - pip install -e ".[dev]"   (includes nemoguardrails)
# =============================================================================
set -euo pipefail

EXPERIMENT="nemo_guardrails"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
PORT=8000
SERVER_PID=""

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
echo "║   Experiment 2: NeMo Guardrails              ║"
echo "╚══════════════════════════════════════════════╝"
echo -e "${NC}"

# ---------------------------------------------------------------------------
# 1. Pre-flight checks
# ---------------------------------------------------------------------------
echo -e "${YELLOW}[1/6] Pre-flight checks & clearing existing results...${NC}"

command -v ollama &>/dev/null  || { echo -e "${RED}❌ ollama not found. Install from https://ollama.com${NC}"; exit 1; }
command -v python  &>/dev/null || { echo -e "${RED}❌ python not found${NC}"; exit 1; }

# Clear existing results in this experiment directory
python -m impenetra.experiments.cleaner --dir "$RESULTS_DIR"
echo -e "${GREEN}✓ Cleared existing results in $RESULTS_DIR${NC}"

# Check nemoguardrails is installed
python -c "import nemoguardrails" 2>/dev/null \
    || { echo -e "${RED}❌ nemoguardrails not installed. Run: pip install -e '.[dev]'${NC}"; exit 1; }

# Check Ollama is reachable
curl -sf "http://localhost:11434/api/tags" >/dev/null \
    || { echo -e "${RED}❌ Ollama not running. Start with: ollama serve${NC}"; exit 1; }

# Check main model is pulled
ollama list | grep -q "llama3.1:8b-instruct-q8_0" \
    || { echo -e "${RED}❌ Model not found. Run: ollama pull llama3.1:8b-instruct-q8_0${NC}"; exit 1; }

echo -e "Garak generations: ${CYAN}$GENERATIONS${NC}"
echo -e "Garak parallelism (attempts): ${CYAN}$PARALLEL_ATTEMPTS${NC}"
echo -e "${GREEN}✅ All pre-flight checks passed${NC}"

# ---------------------------------------------------------------------------
# 2. Start the NeMo server
# ---------------------------------------------------------------------------
echo -e "\n${YELLOW}[2/6] Starting NeMo Guardrails server on port $PORT...${NC}"
echo -e "       (NeMo startup may take 15–30s to load model config)"
cd "$PROJECT_ROOT"
python impenetra/experiments/nemo_guardrails/server.py &
SERVER_PID=$!

# Wait for server to be ready (up to 60s — NeMo is slower to init)
for i in {1..60}; do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Server ready (PID $SERVER_PID)${NC}"
        break
    fi
    if [[ $i -eq 60 ]]; then
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
    -d '{"model":"test","messages":[{"role":"user","content":"Hello, what is 2+2?"}]}')
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
REPORT_PREFIX="$RESULTS_DIR/${TIMESTAMP}_nemo"

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
echo -e "\n${GREEN}🏁 Experiment '$EXPERIMENT' complete!${NC}"
echo -e "   Results → ${CYAN}$RESULTS_DIR/${NC}"
