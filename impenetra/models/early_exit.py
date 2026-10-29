"""
Early-Exit Transformer for low-latency guardrail classification.

Key idea: Attach lightweight classifier heads at intermediate transformer
layers. During inference, if a layer's prediction is confident enough
(low entropy / high max-prob), we skip remaining layers entirely.

Supports:
- Configurable exit points (e.g., layers 2, 4, 6 of a 6-layer model)
- Entropy-based and max-probability exit criteria
- Joint training with weighted per-exit losses
- Batch-aware early exit tracking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from dataclasses import dataclass


@dataclass
class EarlyExitResult:
    """Holds inference results with exit metadata."""

    predictions: torch.Tensor  # (batch,) predicted class indices
    exit_layers: torch.Tensor  # (batch,) which layer each sample exited at
    confidences: torch.Tensor  # (batch,) confidence score at exit
    all_logits: dict[int, torch.Tensor] | None = None  # layer -> logits (for training)


class ExitClassifier(nn.Module):
    """Lightweight classification head for an intermediate layer."""

    def __init__(self, hidden_size: int, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        # Use [CLS] token representation
        cls_output = hidden_state[:, 0, :]
        x = self.dropout(self.activation(self.dense(cls_output)))
        return self.classifier(x)


class EarlyExitTransformer(nn.Module):
    """
    Transformer with early-exit classification heads.

    Args:
        model_name: HuggingFace model identifier
        num_labels: Number of classification labels
        exit_layers: List of layer indices to attach exit heads
        exit_criterion: 'entropy' or 'max_prob'
        exit_threshold: Threshold for early exit decision
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        num_labels: int = 2,
        exit_layers: list[int] | None = None,
        exit_criterion: str = "entropy",
        exit_threshold: float = 0.3,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.output_hidden_states = True
        self.backbone = AutoModel.from_pretrained(
            model_name, config=self.config
        )
        self.num_labels = num_labels
        self.num_layers = self.config.num_hidden_layers
        self.exit_criterion = exit_criterion
        self.exit_threshold = exit_threshold

        # Default: exit at every other layer
        if exit_layers is None:
            exit_layers = list(range(1, self.num_layers, 2))
        self.exit_layers = sorted(exit_layers)

        # Create exit classifier heads
        self.exit_classifiers = nn.ModuleDict(
            {
                str(layer): ExitClassifier(
                    self.config.hidden_size,
                    num_labels,
                    self.config.hidden_dropout_prob,
                )
                for layer in self.exit_layers
            }
        )

        # Final classifier (always present)
        self.final_classifier = ExitClassifier(
            self.config.hidden_size, num_labels, self.config.hidden_dropout_prob
        )

    def _should_exit(
        self, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Determine if predictions are confident enough to exit."""
        probs = F.softmax(logits, dim=-1)

        if self.exit_criterion == "entropy":
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            confident = entropy < self.exit_threshold
            confidence = 1.0 - entropy  # Higher is more confident
        elif self.exit_criterion == "max_prob":
            max_prob, _ = torch.max(probs, dim=-1)
            confident = max_prob > self.exit_threshold
            confidence = max_prob
        else:
            raise ValueError(f"Unknown exit criterion: {self.exit_criterion}")

        return confident, confidence

    def forward_train(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        exit_weights: list[float] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Training forward pass: compute loss at ALL exit points.

        Returns dict with 'loss', 'final_logits', and per-exit losses.
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states  # (num_layers+1, batch, seq, hidden)

        all_logits = {}
        losses = {}

        # Compute logits at each exit point
        for layer_idx in self.exit_layers:
            logits = self.exit_classifiers[str(layer_idx)](
                hidden_states[layer_idx + 1]
            )
            all_logits[layer_idx] = logits
            if labels is not None:
                losses[layer_idx] = F.cross_entropy(logits, labels)

        # Final layer
        final_logits = self.final_classifier(hidden_states[-1])
        all_logits[self.num_layers] = final_logits
        if labels is not None:
            losses[self.num_layers] = F.cross_entropy(final_logits, labels)

        # Weighted loss combination
        if labels is not None:
            if exit_weights is None:
                exit_weights = [1.0] * (len(self.exit_layers) + 1)
            total_loss = sum(
                w * losses[k]
                for w, k in zip(exit_weights, list(losses.keys()))
            )
        else:
            total_loss = torch.tensor(0.0)

        return {
            "loss": total_loss,
            "final_logits": final_logits,
            "all_logits": all_logits,
            "per_exit_losses": losses,
        }

    @torch.no_grad()
    def forward_inference(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> EarlyExitResult:
        """
        Inference with dynamic early exit.
        Stops computation at the first exit layer where confidence is sufficient.
        """
        self.eval()
        batch_size = input_ids.size(0)
        device = input_ids.device

        predictions = torch.zeros(batch_size, dtype=torch.long, device=device)
        exit_layers = torch.full(
            (batch_size,), self.num_layers, dtype=torch.long, device=device
        )
        confidences = torch.zeros(batch_size, device=device)
        exited = torch.zeros(batch_size, dtype=torch.bool, device=device)

        outputs = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        )
        hidden_states = outputs.hidden_states

        for layer_idx in self.exit_layers:
            if exited.all():
                break

            logits = self.exit_classifiers[str(layer_idx)](
                hidden_states[layer_idx + 1]
            )
            confident, confidence = self._should_exit(logits)

            should_exit = confident & (~exited)
            if should_exit.any():
                preds = torch.argmax(logits, dim=-1)
                predictions[should_exit] = preds[should_exit]
                exit_layers[should_exit] = layer_idx
                confidences[should_exit] = confidence[should_exit]
                exited |= should_exit

        # Remaining samples go through final classifier
        remaining = ~exited
        if remaining.any():
            final_logits = self.final_classifier(hidden_states[-1])
            probs = F.softmax(final_logits, dim=-1)
            predictions[remaining] = torch.argmax(
                final_logits[remaining], dim=-1
            )
            confidences[remaining], _ = torch.max(probs[remaining], dim=-1)

        return EarlyExitResult(
            predictions=predictions,
            exit_layers=exit_layers,
            confidences=confidences,
        )
