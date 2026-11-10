"""
Time and memory profiling utilities for guardrail function benchmarking.

Provides:
- @profile_time: Decorator to measure wall-clock and CPU time
- @profile_memory: Decorator to measure peak memory usage via tracemalloc
- @profile_all: Combined time + memory profiling
- benchmark_context: Context manager for fine-grained measurement
"""

import functools
import os
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Callable

import psutil


@dataclass
class ProfileResult:
    """Holds profiling measurements for a single function call or code block."""

    function_name: str
    wall_time_ms: float
    cpu_time_ms: float = 0.0
    peak_memory_mb: float = 0.0
    current_memory_mb: float = 0.0
    rss_before_mb: float = 0.0
    rss_after_mb: float = 0.0
    args_summary: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def __str__(self) -> str:
        parts = [f"⏱  {self.function_name}: {self.wall_time_ms:.2f}ms wall"]
        if self.peak_memory_mb > 0:
            parts.append(f"📦 {self.peak_memory_mb:.2f}MB peak")
        return " | ".join(parts)


def profile_time(func: Callable) -> Callable:
    """Decorator: measures wall-clock and CPU time of a function call."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> tuple[Any, ProfileResult]:
        start_wall = time.perf_counter()
        start_cpu = time.process_time()

        result = func(*args, **kwargs)

        wall_ms = (time.perf_counter() - start_wall) * 1000
        cpu_ms = (time.process_time() - start_cpu) * 1000

        profile = ProfileResult(
            function_name=func.__qualname__,
            wall_time_ms=wall_ms,
            cpu_time_ms=cpu_ms,
        )
        return result, profile

    return wrapper


def profile_memory(func: Callable) -> Callable:
    """Decorator: measures peak memory allocation via tracemalloc."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> tuple[Any, ProfileResult]:
        process = psutil.Process(os.getpid())
        rss_before = process.memory_info().rss / (1024 * 1024)

        tracemalloc.start()
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        rss_after = process.memory_info().rss / (1024 * 1024)

        profile = ProfileResult(
            function_name=func.__qualname__,
            wall_time_ms=0.0,
            peak_memory_mb=peak / (1024 * 1024),
            current_memory_mb=current / (1024 * 1024),
            rss_before_mb=rss_before,
            rss_after_mb=rss_after,
        )
        return result, profile

    return wrapper


def profile_all(func: Callable) -> Callable:
    """Decorator: combined time + memory profiling."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> tuple[Any, ProfileResult]:
        process = psutil.Process(os.getpid())
        rss_before = process.memory_info().rss / (1024 * 1024)

        tracemalloc.start()
        start_wall = time.perf_counter()
        start_cpu = time.process_time()

        result = func(*args, **kwargs)

        wall_ms = (time.perf_counter() - start_wall) * 1000
        cpu_ms = (time.process_time() - start_cpu) * 1000
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        rss_after = process.memory_info().rss / (1024 * 1024)

        profile = ProfileResult(
            function_name=func.__qualname__,
            wall_time_ms=wall_ms,
            cpu_time_ms=cpu_ms,
            peak_memory_mb=peak / (1024 * 1024),
            current_memory_mb=current / (1024 * 1024),
            rss_before_mb=rss_before,
            rss_after_mb=rss_after,
        )
        return result, profile

    return wrapper


@contextmanager
def benchmark_context(label: str = "block"):
    """Context manager for ad-hoc profiling of code blocks."""
    process = psutil.Process(os.getpid())
    rss_before = process.memory_info().rss / (1024 * 1024)
    tracemalloc.start()
    start = time.perf_counter()

    result = ProfileResult(function_name=label, wall_time_ms=0.0)
    yield result

    result.wall_time_ms = (time.perf_counter() - start) * 1000
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    result.peak_memory_mb = peak / (1024 * 1024)
    result.current_memory_mb = current / (1024 * 1024)
    result.rss_before_mb = rss_before
    result.rss_after_mb = process.memory_info().rss / (1024 * 1024)
