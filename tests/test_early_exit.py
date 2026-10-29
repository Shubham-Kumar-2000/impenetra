"""Tests for early-exit transformer model."""

import pytest
import torch


@pytest.fixture
def model():
    """Create a small early-exit model for testing."""
    from impenetra.models.early_exit import EarlyExitTransformer

    return EarlyExitTransformer(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        num_labels=3,
        exit_layers=[1, 3, 5],
        exit_criterion="entropy",
        exit_threshold=0.3,
    )


@pytest.fixture
def sample_inputs():
    """Create dummy tokenized inputs."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    texts = ["This is a test.", "Another example input."]
    return tokenizer(
        texts, padding=True, truncation=True, return_tensors="pt"
    )


@pytest.mark.slow
def test_training_forward(model, sample_inputs):
    labels = torch.tensor([0, 1])
    output = model.forward_train(
        input_ids=sample_inputs["input_ids"],
        attention_mask=sample_inputs["attention_mask"],
        labels=labels,
    )
    assert "loss" in output
    assert "final_logits" in output
    assert output["loss"].requires_grad


@pytest.mark.slow
def test_inference_with_early_exit(model, sample_inputs):
    result = model.forward_inference(
        input_ids=sample_inputs["input_ids"],
        attention_mask=sample_inputs["attention_mask"],
    )
    assert result.predictions.shape == (2,)
    assert result.exit_layers.shape == (2,)
    assert result.confidences.shape == (2,)
    assert (result.exit_layers >= 0).all()


@pytest.mark.slow
def test_exit_criterion_max_prob(sample_inputs):
    from impenetra.models.early_exit import EarlyExitTransformer

    model = EarlyExitTransformer(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        num_labels=2,
        exit_layers=[1, 3, 5],
        exit_criterion="max_prob",
        exit_threshold=0.8,
    )
    result = model.forward_inference(
        input_ids=sample_inputs["input_ids"],
        attention_mask=sample_inputs["attention_mask"],
    )
    assert result.predictions.shape == (2,)


@pytest.mark.slow
def test_all_logits_in_training(model, sample_inputs):
    labels = torch.tensor([0, 2])
    output = model.forward_train(
        input_ids=sample_inputs["input_ids"],
        attention_mask=sample_inputs["attention_mask"],
        labels=labels,
    )
    # Should have logits for each exit layer + final
    assert len(output["all_logits"]) == 4  # layers 1, 3, 5 + final(6)
    assert len(output["per_exit_losses"]) == 4
