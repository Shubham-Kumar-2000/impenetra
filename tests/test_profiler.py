"""Tests for profiling utilities."""

import time

from impenetra.profiling.profiler import (
    ProfileResult,
    benchmark_context,
    profile_all,
    profile_memory,
    profile_time,
)


def test_profile_time():
    @profile_time
    def slow_fn():
        time.sleep(0.01)
        return 42

    result, profile = slow_fn()
    assert result == 42
    assert profile.wall_time_ms >= 9.0  # ~10ms sleep
    assert profile.function_name == "test_profile_time.<locals>.slow_fn"


def test_profile_memory():
    @profile_memory
    def alloc_fn():
        return [0] * 1_000_000

    result, profile = alloc_fn()
    assert len(result) == 1_000_000
    assert profile.peak_memory_mb > 0


def test_profile_all():
    @profile_all
    def combined_fn():
        data = [i**2 for i in range(100_000)]
        return sum(data)

    result, profile = combined_fn()
    assert result > 0
    assert profile.wall_time_ms > 0
    assert profile.peak_memory_mb > 0


def test_benchmark_context():
    with benchmark_context("test_block") as profile:
        _ = [i for i in range(500_000)]

    assert profile.wall_time_ms > 0
    assert profile.function_name == "test_block"


def test_profile_result_to_dict():
    result = ProfileResult(
        function_name="test_fn",
        wall_time_ms=10.5,
        cpu_time_ms=8.3,
        peak_memory_mb=1.2,
    )
    d = result.to_dict()
    assert d["function_name"] == "test_fn"
    assert d["wall_time_ms"] == 10.5


def test_profile_result_str():
    result = ProfileResult(
        function_name="my_func",
        wall_time_ms=42.5,
        peak_memory_mb=3.14,
    )
    s = str(result)
    assert "my_func" in s
    assert "42.50" in s
    assert "3.14" in s
