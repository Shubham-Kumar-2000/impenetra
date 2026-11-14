"""Tests for the experiment results cleaner utility."""

from pathlib import Path

from impenetra.experiments.cleaner import clear_results_dir


def test_clear_results_dir_preserves_gitkeep(tmp_path: Path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # Create dummy files
    gitkeep = results_dir / ".gitkeep"
    gitkeep.touch()

    bench_json = results_dir / "20261114_bench.json"
    bench_json.write_text('{"status": "ok"}')

    garak_log = results_dir / "20261114_garak.log"
    garak_log.write_text("log content")

    report_jsonl = results_dir / "20261114.report.jsonl"
    report_jsonl.write_text('{"entry": 1}')

    final_report = results_dir / "final_report.json"
    final_report.write_text("{}")

    deleted = clear_results_dir(results_dir)

    assert deleted == 4
    assert gitkeep.exists(), ".gitkeep must be preserved"
    assert not bench_json.exists()
    assert not garak_log.exists()
    assert not report_jsonl.exists()
    assert not final_report.exists()


def test_clear_results_dir_nonexistent_creates_dir(tmp_path: Path):
    results_dir = tmp_path / "new_results"
    assert not results_dir.exists()

    deleted = clear_results_dir(results_dir)
    assert deleted == 0
    assert results_dir.exists()
