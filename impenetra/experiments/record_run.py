"""
impenetra/experiments/record_run.py
===================================
Backwards-compatible wrapper around impenetra.experiments.metrics.
"""

from impenetra.experiments.metrics import (
    fetch_server_metrics,
    main,
    parse_garak_attempts,
    record_run_metrics,
    reset_server_metrics,
)

# For backward compatibility
generate_run_report = record_run_metrics

if __name__ == "__main__":
    main()
