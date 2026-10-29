"""Tests for dataset registry and loading."""

import pytest

from impenetra.datasets.loader import load_guardrail_dataset
from impenetra.datasets.registry import DATASET_REGISTRY, list_datasets


def test_registry_has_datasets():
    assert len(DATASET_REGISTRY) >= 4
    assert "clinc150" in DATASET_REGISTRY
    assert "toxic_chat" in DATASET_REGISTRY
    assert "xstest" in DATASET_REGISTRY
    assert "advbench" in DATASET_REGISTRY


def test_list_datasets():
    datasets = list_datasets()
    assert len(datasets) >= 4
    names = [d.name for d in datasets]
    assert "CLINC150" in names
    assert "ToxicChat" in names


def test_dataset_info_fields():
    clinc = DATASET_REGISTRY["clinc150"]
    assert clinc.hf_path == "DeepPavlov/clinc_oos"
    assert clinc.hf_config == "plus"
    assert clinc.task == "intent_classification"
    assert clinc.num_classes == 151


def test_unknown_dataset_raises():
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_guardrail_dataset("nonexistent_dataset")


@pytest.mark.slow
def test_load_clinc150_streaming():
    ds = load_guardrail_dataset("clinc150", streaming=True)
    # Streaming returns IterableDatasetDict
    assert ds is not None
