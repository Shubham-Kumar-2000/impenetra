#!/usr/bin/env bash
# =============================================================================
# experiments/run_all.sh
# =============================================================================
# Master orchestration script. Runs all 3 guardrail experiments sequentially:
#   1. llamaguard-as-guardrail
#   2. nemo-guardrails
#   3. system-prompt-only
#
# For each experiment:
#   - Starts the FastAPI server
#   - Runs the latency + memory benchmark (benchmark.py)
#   - Runs garak red-team probes (dan, dra, encoding subset)
#   - Saves timestamped results to experiments/<name>/results/
#   - Stops the server
#
# After all 3: prints a side-by-side comparison via compare_results.py
#
# Usage (from project root):
#   bash experiments/run_all.sh
#
# Options:
#   SKIP_GARAK=1 bash experiments/run_all.sh   # skip garak (latency+mem only)
#   GARAK_PROBES="dan.Ablation_Dan_11_0" bash experiments/run_all.sh
# =============================================================================
set -euo pipefail

# --- Config ------------------------------------------------------------------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENTS_DIR="$PROJECT_ROOT/experiments"
PORT=8000
PYTHON="${PROJECT_ROOT}/.venv/bin/python3"
# Activate venv so subprocesses (ollama list, curl, etc.) also work
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/.venv/bin/activate" 2>/dev/null || true
SKIP_GARAK="${SKIP_GARAK:-0}"
# Default: fast probes that don't require extra model downloads
GARAK_PROBES="${GARAK_PROBES:-dan.Ablation_Dan_11_0,dan.AutoDANCached,dan.DanInTheWild,dra.DRA,encoding.InjectBase64,encoding.InjectROT13}"
GARAK_CONFIG="$PROJECT_ROOT/impenetra/configs/garak_openai_config.json"

# --- Colours -----------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; NC='\033[0m'

SERVER_PID=""

cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" && wait "$SERVER_PID" 2>/dev/null || true
        echo -e "${DIM}  Server stopped.${NC}"
    fi
}
trap cleanup EXIT

# =============================================================================
header() { echo -e "\n${CYAN}${BOLD}╔══════════════════════════════════════════════════╗${NC}"; \
           echo -e "${CYAN}${BOLD}║  $1  ║${NC}"; \
           echo -e "${CYAN}${BOLD}╚══════════════════════════════════════════════════╝${NC}"; }

step()  { echo -e "\n${YELLOW}▶ $1${NC}"; }
ok()    { echo -e "  ${GREEN}✅ $1${NC}"; }
warn()  { echo -e "  ${YELLOW}⚠️  $1${NC}"; }
fail()  { echo -e "  ${RED}❌ $1${NC}"; }
info()  { echo -e "  ${DIM}$1${NC}"; }

# =============================================================================
# Pre-flight
# =============================================================================
header "Guardrail Experiments — Run All              "

step "Global pre-flight checks"

[[ -f "$PYTHON" ]] || { fail "venv python not found at $PYTHON. Run: pip install -e '.[dev]'"; exit 1; }
command -v ollama &>/dev/null || { fail "ollama not found. Install from https://ollama.com"; exit 1; }
curl -sf "http://localhost:11434/api/tags" >/dev/null \
    || { fail "Ollama not running. Start with: ollama serve"; exit 1; }

OLLAMA_MODELS=$(ollama list 2>/dev/null || true)
echo "$OLLAMA_MODELS" | grep -q "llama3.1:8b-instruct-q8_0" \
    || { fail "Pull the main model first: ollama pull llama3.1:8b-instruct-q8_0"; exit 1; }

ok "Ollama running with llama3.1:8b-instruct-q8_0"

if [[ "$SKIP_GARAK" == "1" ]]; then
    warn "SKIP_GARAK=1: garak will not run, benchmark only"
else
    "$PYTHON" -m garak --version &>/dev/null \
        || { fail "garak not installed. Run: pip install -e '.[dev]'"; exit 1; }
    ok "garak available"
fi

# Check llama-guard3 for exp 1
HAS_LLAMAGUARD=0
if echo "$OLLAMA_MODELS" | grep -q "llama-guard3"; then
    # Use whatever llama-guard3 tag is available (latest, 8b, 1b)
    HAS_LLAMAGUARD=1
    ok "llama-guard3 found"
else
    warn "llama-guard3 not found — Experiment 1 will run with fail-open guardrail"
    info "Pull with: ollama pull llama-guard3:8b"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
info "Run ID: $TIMESTAMP"

# =============================================================================
run_experiment() {
    local exp_name="$1"
    local server_script="$2"
    local result_prefix="$3"
    local results_dir="$EXPERIMENTS_DIR/$exp_name/results"

    header "Experiment: $exp_name                       "
    mkdir -p "$results_dir"

    # --- Start server --------------------------------------------------------
    step "Starting server: $server_script"
    SERVER_PID=""
    cd "$PROJECT_ROOT"
    "$PYTHON" "$server_script" &
    SERVER_PID=$!
    info "Server PID: $SERVER_PID"

    # Wait up to 60s for readiness
    local ready=0
    for i in {1..60}; do
        if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            ready=1; break
        fi
        sleep 1
    done

    if [[ $ready -eq 0 ]]; then
        fail "Server did not start within 60s"
        kill "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
        return 1
    fi
    ok "Server ready"

    # Smoke test
    local smoke_resp
    smoke_resp=$(curl -sf -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"smoke","messages":[{"role":"user","content":"What is 2+2?"}]}' 2>/dev/null) || true
    if echo "$smoke_resp" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'][:40])" 2>/dev/null; then
        ok "Smoke test passed"
    else
        warn "Smoke test response unexpected (server may still work)"
    fi

    # --- Benchmark -----------------------------------------------------------
    step "Running latency + memory benchmark"
    local bench_out="$results_dir/${TIMESTAMP}_bench.json"
    "$PYTHON" experiments/benchmark.py \
        --url "http://127.0.0.1:$PORT" \
        --experiment "$exp_name" \
        --pid "$SERVER_PID" \
        --out "$bench_out" \
        2>&1 | tee "$results_dir/${TIMESTAMP}_bench.log"
    ok "Benchmark complete → $bench_out"

    # --- Garak ---------------------------------------------------------------
    if [[ "$SKIP_GARAK" != "1" ]]; then
        step "Running garak (probes: $GARAK_PROBES)"
        local garak_prefix="$results_dir/${TIMESTAMP}_garak"
        "$PYTHON" -m garak \
            --model_type rest \
            --config "$GARAK_CONFIG" \
            --probes "$GARAK_PROBES" \
            --report_prefix "$garak_prefix" \
            2>&1 | tee "$results_dir/${TIMESTAMP}_garak.log"
        ok "Garak complete"

        # Copy any generated reports into results/
        # garak writes to ~/.local/share/garak/runs/ by default in newer versions
        local garak_home="${GARAK_HOME:-$HOME/.local/share/garak}"
        if [[ -d "$garak_home/runs" ]]; then
            local latest_run
            latest_run=$(ls -t "$garak_home/runs/" 2>/dev/null | head -1)
            if [[ -n "$latest_run" ]]; then
                cp -r "$garak_home/runs/$latest_run"* "$results_dir/" 2>/dev/null || true
                info "Copied garak run artifacts to $results_dir/"
            fi
        fi
    fi

    # --- Stop server ---------------------------------------------------------
    step "Stopping server"
    kill "$SERVER_PID" && wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
    ok "Server stopped"

    echo -e "\n  ${GREEN}${BOLD}Experiment '$exp_name' complete${NC}"
    echo -e "  Results → ${CYAN}$results_dir/${NC}"
}

# =============================================================================
# Run all 3 experiments
# =============================================================================

run_experiment \
    "llamaguard-as-guardrail" \
    "experiments/llamaguard-as-guardrail/server.py" \
    "llamaguard"

run_experiment \
    "nemo-guardrails" \
    "experiments/nemo-guardrails/server.py" \
    "nemo"

run_experiment \
    "system-prompt-only" \
    "experiments/system-prompt-only/server.py" \
    "system_prompt"

# =============================================================================
# Final comparison report
# =============================================================================
header "Results Comparison                             "
"$PYTHON" experiments/compare_results.py --results-dir experiments

echo -e "\n${GREEN}${BOLD}🏁  All experiments complete!${NC}"
echo -e "  Timestamp: $TIMESTAMP"
echo -e "  Results stored in each experiments/<name>/results/ folder"
echo -e "  Re-run comparison: ${CYAN}python3 experiments/compare_results.py${NC}\n"
