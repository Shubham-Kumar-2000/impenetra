"""Accuracy benchmarking utilities for guardrail systems."""

from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
import numpy as np


@dataclass
class AccuracyBenchmarkResult:
    """Results from an accuracy benchmark run."""

    name: str
    accuracy: float
    precision: float  # Macro-averaged
    recall: float  # Macro-averaged
    f1: float  # Macro-averaged
    num_samples: int
    report: str = ""  # Full classification report
    confusion_matrix: np.ndarray | None = None


def benchmark_accuracy(
    y_true: list[int],
    y_pred: list[int],
    name: str = "guardrail",
    target_names: list[str] | None = None,
) -> AccuracyBenchmarkResult:
    """
    Compute accuracy metrics for a guardrail system.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        name: Label for the benchmark
        target_names: Optional class names for the report

    Returns:
        AccuracyBenchmarkResult with precision, recall, F1, etc.
    """
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    report = classification_report(
        y_true, y_pred, target_names=target_names, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    return AccuracyBenchmarkResult(
        name=name,
        accuracy=acc,
        precision=precision,
        recall=recall,
        f1=f1,
        num_samples=len(y_true),
        report=report,
        confusion_matrix=cm,
    )
