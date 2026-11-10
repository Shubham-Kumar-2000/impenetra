#!/usr/bin/env bash
# =============================================================================
# Experiment 3: System-Prompt-Without-Guardrail (Baseline)
# =============================================================================
# Starts the Impenetra FastAPI server with PassThroughGuardrail and charter
# system prompt, runs garak / benchmark tests, saves results, and cleans up.
# =============================================================================
set -euo pipefail

EXPERIMENT="system-prompt-without-guardrail"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
PORT=8000
SERVER_PID=""

cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Stopping server (PID $SERVER_PID)..."
        kill "$SERVER_PID" && wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "Starting server: $EXPERIMENT"
mkdir -p "$RESULTS_DIR"
cd "$PROJECT_ROOT"
.venv/bin/python3 experiments/system-prompt-without-guardrail/server.py &
SERVER_PID=$!

for i in {1..30}; do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo "Server ready (PID $SERVER_PID)"
        break
    fi
    sleep 1
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
.venv/bin/python3 experiments/benchmark.py \
    --url "http://127.0.0.1:$PORT" \
    --experiment "$EXPERIMENT" \
    --pid "$SERVER_PID" \
    --out "$RESULTS_DIR/${TIMESTAMP}_bench.json"

echo "Completed $EXPERIMENT. Results in $RESULTS_DIR/"
