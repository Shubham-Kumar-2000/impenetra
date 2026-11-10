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
echo -e "${YELLOW}[1/5] Pre-flight checks...${NC}"

command -v ollama &>/dev/null  || { echo -e "${RED}❌ ollama not found. Install from https://ollama.com${NC}"; exit 1; }
command -v python  &>/dev/null || { echo -e "${RED}❌ python not found${NC}"; exit 1; }

# Check nemoguardrails is installed
python -c "import nemoguardrails" 2>/dev/null \
    || { echo -e "${RED}❌ nemoguardrails not installed. Run: pip install -e '.[dev]'${NC}"; exit 1; }

# Check Ollama is reachable
curl -sf "http://localhost:11434/api/tags" >/dev/null \
    || { echo -e "${RED}❌ Ollama not running. Start with: ollama serve${NC}"; exit 1; }

# Check main model is pulled
ollama list | grep -q "llama3.1:8b-instruct-q8_0" \
    || { echo -e "${RED}❌ Model not found. Run: ollama pull llama3.1:8b-instruct-q8_0${NC}"; exit 1; }

echo -e "${GREEN}✅ All pre-flight checks passed${NC}"

# ---------------------------------------------------------------------------
# 2. Start the NeMo server
# ---------------------------------------------------------------------------
echo -e "\n${YELLOW}[2/5] Starting NeMo Guardrails server on port $PORT...${NC}"
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
echo -e "\n${YELLOW}[3/5] Running smoke test...${NC}"
SMOKE=$(curl -sf -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"test","messages":[{"role":"user","content":"Hello, what is 2+2?"}]}')
echo "Smoke response: $(echo "$SMOKE" | python -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'][:80])")"
echo -e "${GREEN}✅ Smoke test passed${NC}"

# ---------------------------------------------------------------------------
# 4. Run garak
# ---------------------------------------------------------------------------
echo -e "\n${YELLOW}[4/5] Running garak red-team probes...${NC}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_PREFIX="$RESULTS_DIR/${TIMESTAMP}_nemo"

mkdir -p "$RESULTS_DIR"

python -m garak \
    --model_type rest \
    --config "$PROJECT_ROOT/impenetra/configs/garak_openai_config.json" \
    --probes dan,promptinject,jailbreak \
    --report_prefix "$REPORT_PREFIX" \
    2>&1 | tee "$RESULTS_DIR/${TIMESTAMP}_garak.log"

echo -e "\n${GREEN}✅ Garak scan complete${NC}"

# ---------------------------------------------------------------------------
# 5. Summary
# ---------------------------------------------------------------------------
echo -e "\n${YELLOW}[5/5] Results saved:${NC}"
ls -lh "$RESULTS_DIR/${TIMESTAMP}"* 2>/dev/null || echo "(check $RESULTS_DIR/)"
echo -e "\n${GREEN}🏁 Experiment '$EXPERIMENT' complete!${NC}"
echo -e "   Results → ${CYAN}$RESULTS_DIR/${NC}"
