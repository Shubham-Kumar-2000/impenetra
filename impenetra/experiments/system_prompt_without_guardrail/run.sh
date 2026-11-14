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

# Clear existing results in this experiment directory
python -m impenetra.experiments.cleaner --dir "$RESULTS_DIR"
echo "Cleared existing results in $RESULTS_DIR"

cd "$PROJECT_ROOT"
python impenetra/experiments/system_prompt_without_guardrail/server.py &
SERVER_PID=$!

for i in {1..30}; do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo "Server ready (PID $SERVER_PID)"
        break
    fi
    sleep 1
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

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

echo "Running garak probes on $EXPERIMENT..."
GENERATIONS="${GARAK_GENERATIONS:-1}"
PARALLEL_ATTEMPTS="${GARAK_PARALLEL_ATTEMPTS:-32}"
SPEC="${GARAK_SPEC:-${GARAK_PROBES:-probes.promptinject,probes.dan.DanInTheWild,probes.dan.AutoDANCached,probes.encoding,probes.grandma,probes.goodside,probes.sysprompt_extraction,probes.off_charter}}"
SAMPLE_ONLY="${SAMPLE_ONLY:-false}"
echo "   → Generations: $GENERATIONS | Parallel attempts: $PARALLEL_ATTEMPTS"

# Reset server metrics before Garak
python -m impenetra.experiments.metrics reset --url "http://127.0.0.1:$PORT"

python -m impenetra.garak_plugins.setup install &>/dev/null || true

GARAK_CONFIG="$PROJECT_ROOT/impenetra/configs/garak_openai_config.json"
case "${SAMPLE_ONLY:-0}" in
    [tT][rR][uU][eE]|1|[yY][eE][sS])
        GARAK_CONFIG="$PROJECT_ROOT/impenetra/configs/garak_openai_sample_config.json"
        ;;
esac

python -m garak \
    --target_type rest \
    --config "$GARAK_CONFIG" \
    --generator_option_file "$PROJECT_ROOT/impenetra/configs/garak_openai_config.json" \
    --spec "$SPEC" \
    --generations "$GENERATIONS" \
    --parallel_attempts "$PARALLEL_ATTEMPTS" \
    --report_prefix "$RESULTS_DIR/${TIMESTAMP}_baseline" \
    2>&1 | tee "$RESULTS_DIR/${TIMESTAMP}_garak.log" || true

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

# Compile final report
echo "Compiling final report for $EXPERIMENT..."
python -m impenetra.experiments.compile_results \
    --results-dir "$RESULTS_DIR" \
    --experiment "$EXPERIMENT"

echo "Completed $EXPERIMENT. Results in $RESULTS_DIR/"
