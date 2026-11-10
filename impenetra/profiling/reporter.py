"""Aggregates profiling results into structured reports."""

import csv
import json
from pathlib import Path

from .profiler import ProfileResult


class ProfileReporter:
    """Collects ProfileResult objects and outputs JSON/CSV/rich-table reports."""

    def __init__(self):
        self.results: list[ProfileResult] = []

    def add(self, result: ProfileResult):
        """Add a profiling result to the collection."""
        self.results.append(result)

    def to_json(self, path: str | Path) -> None:
        """Export results to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2)

    def to_csv(self, path: str | Path) -> None:
        """Export results to a CSV file."""
        if not self.results:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=self.results[0].to_dict().keys()
            )
            writer.writeheader()
            for r in self.results:
                writer.writerow(r.to_dict())

    def summary(self) -> str:
        """Generate a rich-formatted summary table."""
        import io

        from rich.console import Console
        from rich.table import Table

        table = Table(title="Profiling Summary")
        table.add_column("Function", style="cyan")
        table.add_column("Wall Time (ms)", justify="right", style="green")
        table.add_column("CPU Time (ms)", justify="right")
        table.add_column("Peak Mem (MB)", justify="right", style="magenta")
        table.add_column("RSS Δ (MB)", justify="right")

        for r in self.results:
            rss_delta = r.rss_after_mb - r.rss_before_mb
            table.add_row(
                r.function_name,
                f"{r.wall_time_ms:.2f}",
                f"{r.cpu_time_ms:.2f}",
                f"{r.peak_memory_mb:.2f}",
                f"{rss_delta:+.2f}",
            )

        buf = io.StringIO()
        console = Console(file=buf, width=120)
        console.print(table)
        return buf.getvalue()
