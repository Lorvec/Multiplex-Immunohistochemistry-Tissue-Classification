"""Metrics used in the paper, including the exact 15-bin top-label ECE."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    bins: int = 15,
) -> float:
    """Return frequency-weighted top-label ECE with equal-width bins on [0, 1]."""

    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=int)
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correctness = (predictions == labels).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidences > lower) & (confidences <= upper)
        if mask.any():
            ece += mask.mean() * abs(
                correctness[mask].mean() - confidences[mask].mean()
            )
    return float(ece)


def metric_bundle(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    num_classes: int | None = None,
) -> dict[str, float]:
    """Compute the classification, discrimination, and calibration metrics."""

    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [samples, classes]")
    if len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must have equal sample counts")
    if num_classes is None:
        num_classes = probabilities.shape[1]
    if probabilities.shape[1] != num_classes:
        raise ValueError("probability columns do not match num_classes")

    predictions = probabilities.argmax(axis=1)
    precision, recall, _, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )
    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "macro_f1": f1_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            labels, predictions, average="weighted", zero_division=0
        ),
        "macro_precision": precision,
        "macro_recall": recall,
        "mcc": matthews_corrcoef(labels, predictions),
        "cohen_kappa": cohen_kappa_score(labels, predictions),
        "nll": log_loss(
            labels, probabilities, labels=list(range(num_classes))
        ),
        "ece_15bin": expected_calibration_error(
            probabilities, labels, bins=15
        ),
    }
    try:
        metrics["macro_roc_auc_ovr"] = roc_auc_score(
            labels,
            probabilities,
            multi_class="ovr",
            average="macro",
        )
    except ValueError:
        metrics["macro_roc_auc_ovr"] = np.nan
    return {name: float(value) for name, value in metrics.items()}
