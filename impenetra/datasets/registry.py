"""
impenetra/datasets/registry.py
==============================
Registry of local benchmark prompt datasets for guardrail evaluation.

Datasets:
- on_charter: On-charter direct food & culinary inquiries (super_positive)
- infused_on_charter: On-charter culinary inquiries disguised with adversarial,
  technical, or distracting framing (positive)
- adversarial: Straight-away adversarial attacks and jailbreaks (super_negative)
- off_charter: Off-charter utility theft and non-culinary tasks (negative)
- infused_off_charter: Off-charter utility theft disguised with culinary or
  restaurant framing (negative)
"""

from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"


@dataclass
class DatasetInfo:
    """Metadata for a registered benchmark dataset."""

    name: str
    path: Path
    task: str
    description: str
    num_samples: int = 0


DATASET_REGISTRY: dict[str, DatasetInfo] = {
    "on_charter": DatasetInfo(
        name="on_charter",
        path=DATA_DIR / "on_charter.csv",
        task="culinary_intent",
        description="On-charter food inquiries: pure happy path culinary prompts (super_positive)",
        num_samples=12,
    ),
    "infused_on_charter": DatasetInfo(
        name="infused_on_charter",
        path=DATA_DIR / "infused_on_charter.csv",
        task="infused_culinary_intent",
        description=(
            "On-charter culinary inquiries disguised with adversarial, "
            "technical, or distracting framing (positive)"
        ),
        num_samples=12,
    ),
    "adversarial": DatasetInfo(
        name="adversarial",
        path=DATA_DIR / "adversarial.csv",
        task="jailbreak_and_injection",
        description=(
            "Straight-away adversarial attacks: DAN, injections, and exploits (super_negative)"
        ),
        num_samples=15,
    ),
    "off_charter": DatasetInfo(
        name="off_charter",
        path=DATA_DIR / "off_charter.csv",
        task="utility_theft",
        description=(
            "Off-charter utility theft: code generation, academic math/essays, "
            "medical, legal (negative)"
        ),
        num_samples=8,
    ),
    "infused_off_charter": DatasetInfo(
        name="infused_off_charter",
        path=DATA_DIR / "infused_off_charter.csv",
        task="infused_utility_theft",
        description=(
            "Off-charter utility theft disguised with culinary, baking, "
            "or restaurant framing (negative)"
        ),
        num_samples=10,
    ),
}

# Maintained list/array of all registered benchmark datasets
DATASETS: list[str] = list(DATASET_REGISTRY.keys())


def list_datasets() -> list[DatasetInfo]:
    """Return all registered datasets."""
    return list(DATASET_REGISTRY.values())


def get_dataset_info(name: str) -> DatasetInfo:
    """Return DatasetInfo for a specific dataset name or raise ValueError."""
    if name not in DATASET_REGISTRY:
        available = ", ".join(DATASET_REGISTRY.keys())
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")
    return DATASET_REGISTRY[name]
