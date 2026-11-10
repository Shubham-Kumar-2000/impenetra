#!/usr/bin/env bash
# =============================================================================
# Experiment 1: LlamaGuard-as-Guardrail
# =============================================================================
# Starts the FastAPI guardrail server, runs garak red-team probes against it,
# saves results to experiments/llamaguard-as-guardrail/results/, then cleans up.
#
# Usage:
#   cd /path/to/impenetra
#   bash experiments/llamaguard-as-guardrail/run.sh
#
# Prerequisites:
#   - ollama serve  (running in background)
#   - ollama pull llama3.1:8b-instruct-q8_0
#   - ollama pull llama-guard3:8b
#   - pip install -e ".[dev]"
# =============================================================================
set -euo pipefail

EXPERIMENT="llamaguard-as-guardrail"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
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
echo "║   Experiment 1: LlamaGuard-as-Guardrail      ║"
echo "╚══════════════════════════════════════════════╝"
echo -e "${NC}"

# ---------------------------------------------------------------------------
# 1. Pre-flight checks
# ---------------------------------------------------------------------------
echo -e "${YELLOW}[1/5] Pre-flight checks...${NC}"

command -v ollama &>/dev/null  || { echo -e "${RED}❌ ollama not found. Install from https://ollama.com${NC}"; exit 1; }
command -v python  &>/dev/null || { echo -e "${RED}❌ python not found${NC}"; exit 1; }
command -v python  -m garak --help &>/dev/null 2>&1 || { echo -e "${RED}❌ garak not installed. Run: pip install -e '.[dev]'${NC}"; exit 1; }

# Check Ollama is reachable
curl -sf "http://localhost:11434/api/tags" >/dev/null \
    || { echo -e "${RED}❌ Ollama not running. Start with: ollama serve${NC}"; exit 1; }

# Check required models are pulled
ollama list | grep -q "llama3.1:8b-instruct-q8_0" \
    || { echo -e "${RED}❌ Model not found. Run: ollama pull llama3.1:8b-instruct-q8_0${NC}"; exit 1; }
ollama list | grep -q "llama-guard3:8b" \
    || { echo -e "${RED}❌ Model not found. Run: ollama pull llama-guard3:8b${NC}"; exit 1; }

echo -e "${GREEN}✅ All pre-flight checks passed${NC}"

# ---------------------------------------------------------------------------
# 2. Start the guardrail server
# ---------------------------------------------------------------------------
echo -e "\n${YELLOW}[2/5] Starting LlamaGuard server on port $PORT...${NC}"
cd "$PROJECT_ROOT"
python experiments/llamaguard-as-guardrail/server.py &
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
REPORT_PREFIX="$RESULTS_DIR/${TIMESTAMP}_llamaguard"

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
