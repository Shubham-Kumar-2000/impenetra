"""Unified dataset loading with HuggingFace datasets."""

from datasets import load_dataset, DatasetDict
from .registry import DATASET_REGISTRY, DatasetInfo


def load_guardrail_dataset(
    name: str,
    split: str | None = None,
    streaming: bool = False,
) -> DatasetDict:
    """
    Load a registered benchmark dataset.

    Args:
        name: Key from DATASET_REGISTRY (e.g., 'clinc150', 'toxic_chat')
        split: Optional split to load (e.g., 'train', 'test')
        streaming: If True, return an iterable dataset (no full download)

    Returns:
        HuggingFace DatasetDict or Dataset (if split specified)

    Raises:
        ValueError: If dataset name is not in the registry
    """
    if name not in DATASET_REGISTRY:
        available = ", ".join(DATASET_REGISTRY.keys())
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")

    info: DatasetInfo = DATASET_REGISTRY[name]

    kwargs = {"path": info.hf_path, "streaming": streaming}
    if info.hf_config:
        kwargs["name"] = info.hf_config
    if split:
        kwargs["split"] = split

    return load_dataset(**kwargs)
