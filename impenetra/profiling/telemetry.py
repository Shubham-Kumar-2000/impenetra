"""
impenetra/profiling/telemetry.py
================================
In-memory telemetry collector for tracking guardrail request latency, API errors,
and server process memory footprint (RSS/VMS) on demand without background threads.

Latency statistics are computed using streaming mathematics (exact running count,
sum, mean, min, and max in O(1) space) and bounded reservoir sampling for percentiles,
preventing unbounded memory growth or expensive sorting on long-running servers.
"""

import collections
import datetime
import os
import random
import threading
from dataclasses import asdict, dataclass
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

    interpolated = sorted_vals[lower_idx] + weight * (
        sorted_vals[upper_idx] - sorted_vals[lower_idx]
    )
    return round(interpolated, 3)


class StreamingLatencyTracker:
    """
    Mathematical streaming tracker for latency metrics.

    - Count, Sum, Mean, Min, Max: Computed using exact O(1) streaming mathematics
      without retaining individual latency records.
    - Percentiles (P50, P95, P99): Computed over a bounded reservoir sample
      (Algorithm R) of at most max_samples (default 1000 floats = 8KB), ensuring
      O(1) strictly bounded memory and fast O(M log M) sorting regardless of
      total request volume (100 to 100,000+ requests).
    """

    def __init__(self, max_samples: int = 1000):
        self.max_samples = max_samples
        self.count: int = 0
        self.total_sum: float = 0.0
        self.min_val: float = 0.0
        self.max_val: float = 0.0
        self._samples: list[float] = []

    def add(self, val: float) -> None:
        """Process a new observation into running math and reservoir sample."""
        val = round(val, 3)
        if self.count == 0:
            self.min_val = val
            self.max_val = val
        else:
            if val < self.min_val:
                self.min_val = val
            if val > self.max_val:
                self.max_val = val

        self.count += 1
        self.total_sum += val

        # Bounded reservoir sampling (Algorithm R)
        if len(self._samples) < self.max_samples:
            self._samples.append(val)
        else:
            # Replace elements with decreasing probability: max_samples / count
            idx = random.randint(0, self.count - 1)
            if idx < self.max_samples:
                self._samples[idx] = val

    def reset(self) -> None:
        """Reset all math accumulators and sample buffer."""
        self.count = 0
        self.total_sum = 0.0
        self.min_val = 0.0
        self.max_val = 0.0
        self._samples.clear()

    def get_stats(self) -> dict[str, float]:
        """Compute summary statistics."""
        if self.count == 0:
            return {
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
            }

        mean = round(self.total_sum / self.count, 3)
        return {
            "p50": calculate_percentile(self._samples, 50),
            "p95": calculate_percentile(self._samples, 95),
            "p99": calculate_percentile(self._samples, 99),
            "mean": mean,
            "min": round(self.min_val, 3),
            "max": round(self.max_val, 3),
        }


class TelemetryCollector:
    """
    Lightweight, thread-safe telemetry collector tracking request latency,
    API errors, and process memory footprint (RSS/VMS) on demand.

    Uses streaming mathematics and bounded reservoir buffers to ensure strictly
    O(1) memory footprint and zero background polling threads.
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

        # Streaming mathematical latency trackers (O(1) memory)
        self.guardrail_tracker = StreamingLatencyTracker(max_samples=max_records)
        self.overall_tracker = StreamingLatencyTracker(max_samples=max_records)

        # Memory baselines
        current_mem = self._get_current_mem()
        self.idle_rss_mb: float = current_mem["rss_mb"]
        self.peak_rss_mb: float = current_mem["rss_mb"]
        self.vms_mb: float = current_mem["vms_mb"]

    @property
    def guardrail_latencies(self) -> list[float]:
        """Bounded sample of recent guardrail latencies."""
        with self._lock:
            return list(self.guardrail_tracker._samples)

    @property
    def overall_latencies(self) -> list[float]:
        """Bounded sample of recent overall latencies."""
        with self._lock:
            return list(self.overall_tracker._samples)

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
            guardrail_latency_ms=round(guardrail_latency_ms, 3)
            if guardrail_latency_ms is not None
            else None,
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
                    self.errors_by_type[error_message] = (
                        self.errors_by_type.get(error_message, 0) + 1
                    )
                elif status_code >= 400:
                    err_key = f"HTTP_{status_code}"
                    self.errors_by_type[err_key] = self.errors_by_type.get(err_key, 0) + 1

            if guardrail_latency_ms is not None:
                self.guardrail_tracker.add(guardrail_latency_ms)
            self.overall_tracker.add(overall_latency_ms)

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
            self.guardrail_tracker.reset()
            self.overall_tracker.reset()
            self.idle_rss_mb = mem["rss_mb"]
            self.peak_rss_mb = mem["rss_mb"]
            self.vms_mb = mem["vms_mb"]

    def stop(self) -> None:
        """No-op kept for backwards compatibility (no background threads)."""
        pass

    def get_summary(self) -> dict[str, Any]:
        """
        Compute aggregated error metrics, latency percentiles, and current
        memory footprint on demand.
        """
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
            guard_stats = self.guardrail_tracker.get_stats()
            overall_stats = self.overall_tracker.get_stats()

        successful_calls = total_calls - failed_calls
        failure_rate = round(failed_calls / total_calls, 4) if total_calls > 0 else 0.0

        return {
            "error_metrics": {
                "total_calls": total_calls,
                "successful_calls": successful_calls,
                "failed_calls": failed_calls,
                "failure_rate": failure_rate,
                "errors_by_type": errors_by_type,
            },
            "guardrail_latency_ms": guard_stats,
            "overall_latency_ms": overall_stats,
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
