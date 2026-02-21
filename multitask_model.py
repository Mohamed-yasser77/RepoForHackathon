"""
Multi-Task MiniLM — Milestone 2 ML Component
=============================================
A PyTorch multi-task learning model based on sentence-transformers/all-MiniLM-L6-v2.

Two output heads share the same transformer embeddings:
  Head 1: Classification → Billing | Technical | Legal  (softmax)
  Head 2: Regression     → Urgency/Sentiment score S ∈ [0, 1]  (sigmoid)

Includes an MLflow registry stub that integrates with the existing
SmartSupport-TicketRouter experiment without modifying it.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

CATEGORIES: List[str] = ["Billing", "Technical", "Legal"]
NUM_CLASSES: int = len(CATEGORIES)
ENCODER_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM: int = 384  # MiniLM-L6-v2 hidden size


# ─── Hyperparameters ─────────────────────────────────────────────────────────

@dataclass
class MultiTaskHyperParams:
    """All tunable knobs for the multi-task model."""
    encoder_model: str = ENCODER_MODEL
    embedding_dim: int = EMBEDDING_DIM
    cls_hidden: int = 128
    reg_hidden: int = 64
    dropout: float = 0.1
    freeze_encoder: bool = False
    learning_rate: float = 2e-5
    cls_loss_weight: float = 0.7
    reg_loss_weight: float = 0.3
    max_seq_length: int = 128


# ─── Multi-Task Model ────────────────────────────────────────────────────────

class MultiTaskMiniLM(nn.Module):
    """
    Multi-task learning model with shared MiniLM encoder and two output heads.

    Architecture:
        Input → MiniLM Encoder → [CLS] pooling (384-d)
                                    ├── Classification Head → 3-class softmax
                                    └── Regression Head     → scalar σ ∈ [0, 1]
    """

    def __init__(self, hp: Optional[MultiTaskHyperParams] = None):
        super().__init__()
        self.hp = hp or MultiTaskHyperParams()

        # ── Shared encoder ────────────────────────────────────────────────
        self.encoder = AutoModel.from_pretrained(self.hp.encoder_model)
        if self.hp.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info("Encoder weights frozen.")

        # ── Head 1: Classification (Billing, Technical, Legal) ────────────
        self.classification_head = nn.Sequential(
            nn.Linear(self.hp.embedding_dim, self.hp.cls_hidden),
            nn.ReLU(),
            nn.Dropout(self.hp.dropout),
            nn.Linear(self.hp.cls_hidden, NUM_CLASSES),
        )

        # ── Head 2: Regression (Urgency/Sentiment score) ──────────────────
        self.regression_head = nn.Sequential(
            nn.Linear(self.hp.embedding_dim, self.hp.reg_hidden),
            nn.ReLU(),
            nn.Dropout(self.hp.dropout),
            nn.Linear(self.hp.reg_hidden, 1),
            nn.Sigmoid(),
        )

        logger.info(
            "MultiTaskMiniLM initialised — encoder=%s, cls_hidden=%d, reg_hidden=%d",
            self.hp.encoder_model, self.hp.cls_hidden, self.hp.reg_hidden,
        )

    def _mean_pooling(
        self, model_output: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Mean pooling — average token embeddings weighted by attention mask."""
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through shared encoder and both heads.

        Args:
            input_ids:      (batch, seq_len) tokenised input
            attention_mask: (batch, seq_len) attention mask

        Returns:
            cls_logits:     (batch, 3) raw logits for classification
            reg_score:      (batch, 1) sigmoid output ∈ [0, 1]
        """
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # Mean-pooled sentence embedding
        embeddings = self._mean_pooling(encoder_output, attention_mask)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        cls_logits = self.classification_head(embeddings)
        reg_score = self.regression_head(embeddings)

        return cls_logits, reg_score

    def compute_loss(
        self,
        cls_logits: torch.Tensor,
        reg_score: torch.Tensor,
        cls_targets: torch.Tensor,
        reg_targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Joint loss = α · CrossEntropy(cls) + β · MSE(reg).

        Returns:
            total_loss, cls_loss, reg_loss
        """
        cls_loss = F.cross_entropy(cls_logits, cls_targets)
        reg_loss = F.mse_loss(reg_score.squeeze(-1), reg_targets)

        total_loss = (
            self.hp.cls_loss_weight * cls_loss
            + self.hp.reg_loss_weight * reg_loss
        )
        return total_loss, cls_loss, reg_loss


# ─── Inference Wrapper ────────────────────────────────────────────────────────

class MultiTaskPredictor:
    """
    High-level inference interface for the multi-task model.

    Usage:
        predictor = MultiTaskPredictor()           # loads default weights
        category, confidence, urgency = predictor.predict("My invoice is wrong")
    """

    def __init__(
        self,
        model: Optional[MultiTaskMiniLM] = None,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model or MultiTaskMiniLM()
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(ENCODER_MODEL)
        self.max_length = self.model.hp.max_seq_length
        logger.info("MultiTaskPredictor ready on %s", self.device)

    def predict(self, text: str) -> Tuple[str, float, float]:
        """
        Run inference on a single text.

        Returns:
            category:       "Billing" | "Technical" | "Legal"
            confidence:     softmax probability of the predicted class ∈ [0, 1]
            urgency_score:  regression head output S ∈ [0, 1]
        """
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            cls_logits, reg_score = self.model(input_ids, attention_mask)

        # Classification — softmax probabilities
        probs = F.softmax(cls_logits, dim=-1).squeeze(0).cpu()
        pred_idx = int(torch.argmax(probs))
        category = CATEGORIES[pred_idx]
        confidence = float(probs[pred_idx])

        # Regression — urgency/sentiment score
        urgency_score = float(reg_score.squeeze().cpu())

        return category, confidence, urgency_score

    def predict_batch(
        self, texts: List[str]
    ) -> List[Tuple[str, float, float]]:
        """Run inference on a batch of texts."""
        encoded = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            cls_logits, reg_score = self.model(input_ids, attention_mask)

        probs = F.softmax(cls_logits, dim=-1).cpu()
        scores = reg_score.squeeze(-1).cpu()

        results: List[Tuple[str, float, float]] = []
        for i in range(len(texts)):
            pred_idx = int(torch.argmax(probs[i]))
            results.append((
                CATEGORIES[pred_idx],
                float(probs[i][pred_idx]),
                float(scores[i]),
            ))
        return results


# ─── MLflow Registry Stub ────────────────────────────────────────────────────

def log_multitask_to_mlflow(
    model: MultiTaskMiniLM,
    metrics: Optional[Dict[str, float]] = None,
    experiment_name: str = "SmartSupport-TicketRouter",
    run_name: str = "multitask-minilm-v1",
) -> None:
    """
    Stub function demonstrating how to log the multi-task model
    to the existing MLflow registry without modifying the current setup.

    This integrates with the same experiment used by classifier.py.

    Usage:
        model = MultiTaskMiniLM()
        # ... train the model ...
        log_multitask_to_mlflow(model, metrics={"accuracy": 0.92, "mae": 0.08})
    """
    import mlflow
    import mlflow.pytorch
    from dataclasses import asdict

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        # ── Log hyperparameters ───────────────────────────────────────────
        hp_dict = asdict(model.hp)
        mlflow.log_params(hp_dict)
        mlflow.log_param("model_type", "MultiTaskMiniLM")
        mlflow.log_param("num_classes", NUM_CLASSES)
        mlflow.log_param("categories", str(CATEGORIES))

        # ── Log metrics (if provided from training/eval) ──────────────────
        if metrics:
            mlflow.log_metrics(metrics)

        # ── Log the PyTorch model artifact ────────────────────────────────
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="multitask_minilm",
            registered_model_name="SmartSupport-MultiTaskMiniLM",
        )

        logger.info(
            "Multi-task model logged to MLflow experiment '%s', run '%s'",
            experiment_name, run_name,
        )


# ─── Entrypoint (demo) ───────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    print("=" * 65)
    print("  Multi-Task MiniLM — Quick Sanity Check")
    print("=" * 65)

    predictor = MultiTaskPredictor()

    samples = [
        "I was billed twice this month, please refund me ASAP",
        "The API returns 500 errors and production is down",
        "I am requesting data deletion under GDPR Article 17",
        "My browser extension crashes every time I open it",
        "Threatening legal action if not resolved within 48 hours",
    ]

    for text in samples:
        category, confidence, urgency = predictor.predict(text)
        print(f"\n  Text:     {text[:70]}")
        print(f"  Category: {category}  (conf={confidence:.3f})")
        print(f"  Urgency:  {urgency:.4f}")

    print("\n✓ Multi-Task MiniLM sanity check complete.")
