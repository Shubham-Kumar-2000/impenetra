"""
impenetra/experiments/cleaner.py
================================
Utility for cleaning results directories prior to starting experiment runs.
Ensures that all previous result files (json, jsonl, log, html, final_report)
are cleared out, while preserving directory structure and .gitkeep files.
"""

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_RESULT_EXTENSIONS = {
    ".json",
    ".jsonl",
    ".log",
    ".html",
    ".csv",
    ".md",
    ".txt",
}

DEFAULT_PRESERVE_NAMES = {
    ".gitkeep",
    ".gitignore",
    "README.md",
}


def clear_results_dir(
    dir_path: str | Path,
    preserve_names: set[str] | None = None,
    allowed_extensions: set[str] | None = None,
) -> int:
    """
    Clears all result files from the specified experiment results directory.

    Args:
        dir_path: Path to the experiment results directory.
        preserve_names: File basenames to preserve (e.g. .gitkeep).
        allowed_extensions: Extensions of files eligible for deletion.
                            If None, DEFAULT_RESULT_EXTENSIONS are used.

    Returns:
        Number of files deleted.
    """
    path = Path(dir_path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return 0

    preserve = preserve_names if preserve_names is not None else DEFAULT_PRESERVE_NAMES
    extensions = (
        allowed_extensions if allowed_extensions is not None else DEFAULT_RESULT_EXTENSIONS
    )

    deleted_count = 0
    # Walk through directory and remove non-preserved result files
    for item in sorted(path.iterdir()):
        if item.is_file():
            if item.name in preserve:
                continue
            # Delete if extension matches or file is a known report name
            if item.suffix.lower() in extensions or "report" in item.name or "bench" in item.name:
                try:
                    item.unlink()
                    deleted_count += 1
                except OSError as e:
                    logger.warning("Failed to delete %s: %s", item, e)

    return deleted_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clear result files from an experiment results directory.")
    parser.add_argument(
        "--dir",
        required=True,
        help="Path to results directory (e.g. impenetra/experiments/llamaguard_as_guardrail/results)",
    )
    args = parser.parse_args()

    count = clear_results_dir(args.dir)
    print(f"Cleared {count} result file(s) from {args.dir}")
