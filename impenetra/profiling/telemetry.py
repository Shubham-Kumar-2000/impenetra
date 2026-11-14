"""
impenetra/profiling/telemetry.py
================================
In-memory telemetry collector for tracking guardrail request latency, API errors,
and server process memory footprint (RSS/VMS) on demand without background threads.
"""

import collections
from dataclasses import asdict, dataclass
import datetime
import os
import statistics
import threading
from typing import Any

import psutil


@dataclass
class TelemetryRecord:
    prompt_snippet: str
    guardrail_latency_ms: float | None
    overall_latency_ms: float
    blocked: bool
    status_code: int
    error_message: str | None
    timestamp: str
    memory_rss_mb: float


def calculate_percentile(values: list[float], p: float) -> float:
    """
    Computes percentile using linear interpolation matching numpy/scipy.
    Returns 0.0 if values list is empty.
    """
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 1:
        return round(sorted_vals[0], 3)

    rank = (p / 100.0) * (n - 1)
    lower_idx = int(rank)
    upper_idx = min(lower_idx + 1, n - 1)
    weight = rank - lower_idx

    interpolated = sorted_vals[lower_idx] + weight * (sorted_vals[upper_idx] - sorted_vals[lower_idx])
    return round(interpolated, 3)


class TelemetryCollector:
    """
    Lightweight, thread-safe telemetry collector tracking request latency,
    API errors, and process memory footprint (RSS/VMS) on demand.

    No background threads: queries memory dynamically during record_request()
    and whenever /metrics or /metrics/summary is called. Records are cleared on
    reset() and bounded by max_records to prevent unbounded memory growth.
    """

    def __init__(self, max_records: int = 1000):
        self._lock = threading.Lock()
        self.max_records = max_records
        self.records: collections.deque[TelemetryRecord] = collections.deque(maxlen=max_records)
        self._process = psutil.Process(os.getpid())

        # Exact counters maintained across all calls
        self.total_calls: int = 0
        self.failed_calls: int = 0
        self.errors_by_type: dict[str, int] = {}
        self.guardrail_latencies: list[float] = []
        self.overall_latencies: list[float] = []

        # Memory baselines
        current_mem = self._get_current_mem()
        self.idle_rss_mb: float = current_mem["rss_mb"]
        self.peak_rss_mb: float = current_mem["rss_mb"]
        self.vms_mb: float = current_mem["vms_mb"]

    def _get_current_mem(self) -> dict[str, float]:
        try:
            mem = self._process.memory_info()
            return {
                "rss_mb": round(mem.rss / (1024 * 1024), 2),
                "vms_mb": round(mem.vms / (1024 * 1024), 2),
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return {"rss_mb": 0.0, "vms_mb": 0.0}

    def record_request(
        self,
        prompt: str,
        guardrail_latency_ms: float | None,
        overall_latency_ms: float,
        blocked: bool,
        status_code: int = 200,
        error_message: str | None = None,
    ) -> TelemetryRecord:
        """Record a single request's timing, error, and memory metrics."""
        mem = self._get_current_mem()
        record = TelemetryRecord(
            prompt_snippet=prompt[:80].replace("\n", " ").strip(),
            guardrail_latency_ms=round(guardrail_latency_ms, 3) if guardrail_latency_ms is not None else None,
            overall_latency_ms=round(overall_latency_ms, 3),
            blocked=blocked,
            status_code=status_code,
            error_message=error_message,
            timestamp=datetime.datetime.now().astimezone().isoformat(),
            memory_rss_mb=mem["rss_mb"],
        )
        with self._lock:
            self.total_calls += 1
            if status_code >= 400 or error_message:
                self.failed_calls += 1
                if error_message:
                    self.errors_by_type[error_message] = self.errors_by_type.get(error_message, 0) + 1
                elif status_code >= 400:
                    err_key = f"HTTP_{status_code}"
                    self.errors_by_type[err_key] = self.errors_by_type.get(err_key, 0) + 1

            if guardrail_latency_ms is not None:
                self.guardrail_latencies.append(round(guardrail_latency_ms, 3))
            if overall_latency_ms is not None:
                self.overall_latencies.append(round(overall_latency_ms, 3))

            if mem["rss_mb"] > self.peak_rss_mb:
                self.peak_rss_mb = mem["rss_mb"]
            self.vms_mb = mem["vms_mb"]

            self.records.append(record)
        return record

    def reset(self) -> None:
        """Reset all recorded metrics, clear records, and re-baseline memory."""
        mem = self._get_current_mem()
        with self._lock:
            self.records.clear()
            self.total_calls = 0
            self.failed_calls = 0
            self.errors_by_type.clear()
            self.guardrail_latencies.clear()
            self.overall_latencies.clear()
            self.idle_rss_mb = mem["rss_mb"]
            self.peak_rss_mb = mem["rss_mb"]
            self.vms_mb = mem["vms_mb"]

    def stop(self) -> None:
        """No-op kept for backwards compatibility (no background threads)."""
        pass

    def get_summary(self) -> dict[str, Any]:
        """Compute aggregated error metrics, latency percentiles, and current memory footprint on demand."""
        current_mem = self._get_current_mem()
        with self._lock:
            if current_mem["rss_mb"] > self.peak_rss_mb:
                self.peak_rss_mb = current_mem["rss_mb"]
            self.vms_mb = current_mem["vms_mb"]

            idle_rss = self.idle_rss_mb
            peak_rss = self.peak_rss_mb
            vms = self.vms_mb
            total_calls = self.total_calls
            failed_calls = self.failed_calls
            errors_by_type = dict(self.errors_by_type)
            guard_times = list(self.guardrail_latencies)
            overall_times = list(self.overall_latencies)

        successful_calls = total_calls - failed_calls
        failure_rate = round(failed_calls / total_calls, 4) if total_calls > 0 else 0.0

        guard_p50 = calculate_percentile(guard_times, 50)
        guard_p95 = calculate_percentile(guard_times, 95)
        guard_p99 = calculate_percentile(guard_times, 99)
        guard_mean = round(statistics.mean(guard_times), 3) if guard_times else 0.0
        guard_min = round(min(guard_times), 3) if guard_times else 0.0
        guard_max = round(max(guard_times), 3) if guard_times else 0.0

        overall_p50 = calculate_percentile(overall_times, 50)
        overall_p95 = calculate_percentile(overall_times, 95)
        overall_p99 = calculate_percentile(overall_times, 99)
        overall_mean = round(statistics.mean(overall_times), 3) if overall_times else 0.0
        overall_min = round(min(overall_times), 3) if overall_times else 0.0
        overall_max = round(max(overall_times), 3) if overall_times else 0.0

        return {
            "error_metrics": {
                "total_calls": total_calls,
                "successful_calls": successful_calls,
                "failed_calls": failed_calls,
                "failure_rate": failure_rate,
                "errors_by_type": errors_by_type,
            },
            "guardrail_latency_ms": {
                "p50": guard_p50,
                "p95": guard_p95,
                "p99": guard_p99,
                "mean": guard_mean,
                "min": guard_min,
                "max": guard_max,
            },
            "overall_latency_ms": {
                "p50": overall_p50,
                "p95": overall_p95,
                "p99": overall_p99,
                "mean": overall_mean,
                "min": overall_min,
                "max": overall_max,
            },
            "memory_footprint": {
                "idle_rss_mb": idle_rss,
                "peak_rss_mb": peak_rss,
                "delta_rss_mb": round(max(0.0, peak_rss - idle_rss), 2),
                "vms_mb": vms,
            },
        }

    def to_dict(self) -> dict[str, Any]:
        """Full representation including summary and bounded recent records."""
        summary = self.get_summary()
        with self._lock:
            raw = [asdict(r) for r in self.records]
        summary["records"] = raw
        return summary
