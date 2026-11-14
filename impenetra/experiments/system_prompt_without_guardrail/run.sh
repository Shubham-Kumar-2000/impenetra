#!/usr/bin/env bash
# =============================================================================
# Experiment 3: System-Prompt-Without-Guardrail (Baseline)
# =============================================================================
# Starts the Impenetra FastAPI server with PassThroughGuardrail and charter
# system prompt, runs garak / benchmark tests, saves results, and cleans up.
# =============================================================================
set -euo pipefail

EXPERIMENT="system_prompt_without_guardrail"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
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
.venv/bin/python3 impenetra/experiments/system_prompt_without_guardrail/server.py &
SERVER_PID=$!

for i in {1..30}; do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo "Server ready (PID $SERVER_PID)"
        break
    fi
    sleep 1
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
.venv/bin/python3 impenetra/experiments/benchmark.py \
    --url "http://127.0.0.1:$PORT" \
    --experiment "$EXPERIMENT" \
    --pid "$SERVER_PID" \
    --out "$RESULTS_DIR/${TIMESTAMP}_bench.json"

echo "Running garak probes on $EXPERIMENT..."
GENERATIONS="${GARAK_GENERATIONS:-1}"
PARALLEL_ATTEMPTS="${GARAK_PARALLEL_ATTEMPTS:-32}"
SPEC="${GARAK_SPEC:-${GARAK_PROBES:-probes.promptinject,probes.dan.DanInTheWild,probes.dan.AutoDANCached,probes.encoding,probes.grandma,probes.goodside,probes.sysprompt_extraction,probes.off_charter}}"
echo "   → Generations: $GENERATIONS | Parallel attempts: $PARALLEL_ATTEMPTS"
.venv/bin/python3 -m impenetra.garak_plugins.setup install &>/dev/null || true

.venv/bin/python3 -m garak \
    --target_type rest \
    --config "$PROJECT_ROOT/impenetra/configs/garak_openai_config.json" \
    --generator_option_file "$PROJECT_ROOT/impenetra/configs/garak_openai_config.json" \
    --spec "$SPEC" \
    --generations "$GENERATIONS" \
    --parallel_attempts "$PARALLEL_ATTEMPTS" \
    --report_prefix "$RESULTS_DIR/${TIMESTAMP}_baseline" \
    2>&1 | tee "$RESULTS_DIR/${TIMESTAMP}_garak.log" || true

echo "Completed $EXPERIMENT. Results in $RESULTS_DIR/"
