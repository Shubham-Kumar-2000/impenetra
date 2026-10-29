"""
Dataset registry for guardrail benchmarking.

Supported datasets:
- CLINC150: 150-class intent classification + OOS detection
- ToxicChat: Real-world conversational toxicity
- XSTest: Over-refusal / false-positive testing
- AdvBench: Jailbreak & harmful instruction detection
"""

from dataclasses import dataclass


@dataclass
class DatasetInfo:
    """Metadata for a registered benchmark dataset."""

    name: str
    hf_path: str  # HuggingFace dataset identifier
    hf_config: str | None  # Dataset config/subset name
    task: str  # Classification task type
    description: str
    num_classes: int | None = None
    text_column: str = "text"
    label_column: str = "label"


DATASET_REGISTRY: dict[str, DatasetInfo] = {
    "clinc150": DatasetInfo(
        name="CLINC150",
        hf_path="DeepPavlov/clinc_oos",
        hf_config="plus",
        task="intent_classification",
        description="150 intent classes across 10 domains + out-of-scope detection",
        num_classes=151,  # 150 intents + OOS
        text_column="text",
        label_column="intent",
    ),
    "toxic_chat": DatasetInfo(
        name="ToxicChat",
        hf_path="lmsys/toxic-chat",
        hf_config="toxicchat0124",
        task="toxicity_detection",
        description="Real-world user-AI conversation toxicity (binary)",
        num_classes=2,
        text_column="user_input",
        label_column="toxicity",
    ),
    "xstest": DatasetInfo(
        name="XSTest",
        hf_path="walledai/XSTest",
        hf_config=None,
        task="over_refusal",
        description="Safe prompts that look dangerous — tests false positive rate",
        num_classes=2,
        text_column="prompt",
        label_column="label",
    ),
    "advbench": DatasetInfo(
        name="AdvBench",
        hf_path="walledai/AdvBench",
        hf_config=None,
        task="jailbreak_detection",
        description="Harmful instruction / jailbreak detection benchmark",
        num_classes=2,
        text_column="prompt",
        label_column="target",
    ),
}


def list_datasets() -> list[DatasetInfo]:
    """Return all registered datasets."""
    return list(DATASET_REGISTRY.values())
