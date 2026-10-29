"""Latency benchmarking utilities for guardrail systems."""

import time
import statistics
from dataclasses import dataclass, field
from typing import Callable, Any


@dataclass
class LatencyBenchmarkResult:
    """Results from a latency benchmark run."""

    name: str
    num_iterations: int
    mean_ms: float = 0.0
    median_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    std_ms: float = 0.0
    raw_times_ms: list[float] = field(default_factory=list)


def benchmark_latency(
    func: Callable,
    args: tuple = (),
    kwargs: dict | None = None,
    num_iterations: int = 100,
    warmup_iterations: int = 5,
    name: str | None = None,
) -> LatencyBenchmarkResult:
    """
    Benchmark the latency of a callable over multiple iterations.

    Args:
        func: The function to benchmark
        args: Positional arguments to pass to func
        kwargs: Keyword arguments to pass to func
        num_iterations: Number of timed iterations
        warmup_iterations: Number of warmup iterations (not timed)
        name: Label for the benchmark

    Returns:
        LatencyBenchmarkResult with statistical summary
    """
    if kwargs is None:
        kwargs = {}
    if name is None:
        name = getattr(func, "__qualname__", str(func))

    # Warmup
    for _ in range(warmup_iterations):
        func(*args, **kwargs)

    # Timed runs
    times_ms: list[float] = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        times_ms.append(elapsed_ms)

    sorted_times = sorted(times_ms)
    p95_idx = int(0.95 * len(sorted_times))
    p99_idx = int(0.99 * len(sorted_times))

    return LatencyBenchmarkResult(
        name=name,
        num_iterations=num_iterations,
        mean_ms=statistics.mean(times_ms),
        median_ms=statistics.median(times_ms),
        p95_ms=sorted_times[p95_idx] if p95_idx < len(sorted_times) else sorted_times[-1],
        p99_ms=sorted_times[p99_idx] if p99_idx < len(sorted_times) else sorted_times[-1],
        min_ms=min(times_ms),
        max_ms=max(times_ms),
        std_ms=statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        raw_times_ms=times_ms,
    )
