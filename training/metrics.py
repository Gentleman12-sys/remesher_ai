"""Training-time metrics: edge classification accuracy / precision / recall / F1."""

from __future__ import annotations

from typing import Dict, List

import torch


@torch.no_grad()
def edge_accuracy(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred_labels = (pred >= threshold).float()
    target_labels = (target >= threshold).float()
    correct = (pred_labels == target_labels).float().mean()
    return float(correct.cpu())


@torch.no_grad()
def edge_precision_recall_f1(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    pred_labels = (pred >= threshold).float()
    target_labels = (target >= threshold).float()

    tp = float((pred_labels * target_labels).sum().cpu())
    fp = float((pred_labels * (1 - target_labels)).sum().cpu())
    fn = float(((1 - pred_labels) * target_labels).sum().cpu())

    precision = tp / max(1e-8, tp + fp)
    recall = tp / max(1e-8, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


def epoch_summary(batch_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """Average a list of per-batch metric dicts into a single epoch-level dict."""
    if not batch_metrics:
        return {}
    keys = batch_metrics[0].keys()
    return {k: sum(m[k] for m in batch_metrics) / len(batch_metrics) for k in keys}
