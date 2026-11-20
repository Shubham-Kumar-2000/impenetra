"""
impenetra/datasets/loader.py
============================
Strict dataset loading for guardrail benchmarking from local CSV files.
Enforces a strict schema contract without loose fallbacks.
"""

import csv
from pathlib import Path
from typing import Any

from .registry import DATASET_REGISTRY, IN_CHARTER_INTENTS, DatasetInfo

REQUIRED_COLUMNS = ["prompt", "expected_blocked", "type", "subtype"]
ALLOWED_TYPES = {"positive", "negative"}


def parse_boolean(value: Any) -> bool:
    """
    Strictly parse boolean value.
    Accepts boolean or 'true'/'false' (case-insensitive string).
    Raises ValueError for any other input.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        str_val = value.strip().lower()
        if str_val == "true":
            return True
        if str_val == "false":
            return False
    raise ValueError(f"Invalid boolean value: {value!r}. Expected strictly 'true' or 'false'.")


def _load_hf_dataset(
    info: DatasetInfo,
    split: str | None = None,
) -> list[dict[str, Any]]:
    """Load and format a dataset from Hugging Face."""
    from datasets import load_dataset

    split_to_load = split or info.default_split or "test"
    kwargs: dict[str, Any] = {"path": info.hf_path, "split": split_to_load}
    if info.hf_config:
        kwargs["name"] = info.hf_config

    ds = load_dataset(**kwargs)
    rows: list[dict[str, Any]] = []
    for item in ds:
        prompt = (item.get(info.text_column) or "").strip()
        if not prompt:
            continue
        label_text = str(item.get(info.label_column) or "").strip()
        is_positive = label_text in IN_CHARTER_INTENTS
        rows.append(
            {
                "prompt": prompt,
                "expected_blocked": not is_positive,
                "type": "positive" if is_positive else "negative",
                "subtype": label_text,
                "dataset": info.name,
            }
        )
    return rows


def load_guardrail_dataset(
    name_or_path: str | Path,
    split: str | None = None,
) -> list[dict[str, Any]]:
    """
    Load and strictly validate a benchmark dataset CSV or Hugging Face dataset.

    Contract:
    - Headers must contain: 'prompt', 'expected_blocked', 'type', 'subtype'
    - 'prompt': non-empty string
    - 'expected_blocked': strictly 'true' or 'false'
    - 'type': strictly one of {'positive', 'negative'}
    - 'positive' must have expected_blocked=false
    - 'negative' must have expected_blocked=true

    Raises:
        ValueError: If file is missing or any contract check fails.
    """
    name_str = str(name_or_path)
    if name_str in DATASET_REGISTRY:
        info: DatasetInfo = DATASET_REGISTRY[name_str]
        if info.hf_path:
            return _load_hf_dataset(info, split=split)
        if info.path is None:
            raise ValueError(f"Dataset '{name_str}' has neither 'path' nor 'hf_path' defined.")
        csv_path = info.path
        dataset_name = info.name
    else:
        csv_path = Path(name_or_path)
        if not csv_path.exists():
            available = ", ".join(DATASET_REGISTRY.keys())
            raise ValueError(
                f"Unknown dataset or file '{name_or_path}'. "
                f"Available registered datasets: {available}"
            )
        dataset_name = csv_path.stem

    rows: list[dict[str, Any]] = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Dataset '{dataset_name}' CSV is empty or has no header.")

        missing_cols = [col for col in REQUIRED_COLUMNS if col not in reader.fieldnames]
        if missing_cols:
            raise ValueError(
                f"Dataset '{dataset_name}' missing required columns: {missing_cols}. "
                f"Expected columns: {REQUIRED_COLUMNS}"
            )

        for row_idx, row in enumerate(reader, start=2):  # Line 2 is first data row
            prompt = (row.get("prompt") or "").strip()
            if not prompt:
                raise ValueError(
                    f"Dataset '{dataset_name}' row {row_idx}: 'prompt' cannot be empty."
                )

            raw_blocked = row.get("expected_blocked")
            try:
                expected_blocked = parse_boolean(raw_blocked)
            except ValueError as e:
                raise ValueError(f"Dataset '{dataset_name}' row {row_idx}: {e}") from None

            raw_type = (row.get("type") or "").strip().lower()
            if raw_type not in ALLOWED_TYPES:
                raise ValueError(
                    f"Dataset '{dataset_name}' row {row_idx}: Invalid 'type' '{raw_type}'. "
                    f"Must be one of: {sorted(ALLOWED_TYPES)}"
                )

            # Strict cross-validation: expected_blocked vs type
            if raw_type == "positive" and expected_blocked is not False:
                raise ValueError(
                    f"Dataset '{dataset_name}' row {row_idx}: Type '{raw_type}' "
                    f"must have expected_blocked=false, got '{raw_blocked}'."
                )
            if raw_type == "negative" and expected_blocked is not True:
                raise ValueError(
                    f"Dataset '{dataset_name}' row {row_idx}: Type '{raw_type}' "
                    f"must have expected_blocked=true, got '{raw_blocked}'."
                )

            subtype = (row.get("subtype") or "").strip()

            rows.append(
                {
                    "prompt": prompt,
                    "expected_blocked": expected_blocked,
                    "type": raw_type,
                    "subtype": subtype,
                    "dataset": dataset_name,
                }
            )

    return rows


def load_all_datasets() -> dict[str, list[dict[str, Any]]]:
    """Load all registered benchmark datasets as a dict of {name: rows}."""
    return {name: load_guardrail_dataset(name) for name in DATASET_REGISTRY}
