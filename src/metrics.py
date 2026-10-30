"""
Metrics module for Kaggle 'Name That LLM'.
Implements Balanced Accuracy, per-class recall, and hierarchical error reporting.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score

from .config import (
    CANDIDATE_MODELS,
    FAMILY_NAMES,
    MODEL_FAMILY_MAP,
    MODEL_TO_IDX,
    NUM_CLASSES,
)


def compute_balanced_accuracy(
    y_true: Union[np.ndarray, List[int]],
    y_pred: Union[np.ndarray, List[int]],
) -> float:
    """
    Computes Balanced Accuracy across the classes present or specified.
    Balanced Accuracy is defined as the average per-class recall:
        Balanced_Acc = (1 / K) * sum_{k=1}^K Recall_k
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    return float(balanced_accuracy_score(y_true_arr, y_pred_arr))


def evaluate_predictions(
    y_true: Union[np.ndarray, List[int]],
    y_pred: Union[np.ndarray, List[int]],
    target_names: Optional[List[str]] = None,
) -> Dict[str, Union[float, Dict[str, float], np.ndarray]]:
    """
    Detailed evaluation of multi-class predictions:
    - Overall Balanced Accuracy
    - Per-class Recall
    - Family-level Accuracy (Hierarchical)
    - Confusion Matrix
    """
    if target_names is None:
        target_names = CANDIDATE_MODELS

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    overall_bal_acc = compute_balanced_accuracy(y_true_arr, y_pred_arr)

    # Per-class recall
    recalls = recall_score(
        y_true_arr,
        y_pred_arr,
        labels=list(range(len(target_names))),
        average=None,
        zero_division=0.0,
    )
    per_class_recall = {
        name: float(rec) for name, rec in zip(target_names, recalls)
    }

    # Family-level accuracy
    y_true_fam = [MODEL_FAMILY_MAP.get(target_names[i], "Unknown") for i in y_true_arr]
    y_pred_fam = [MODEL_FAMILY_MAP.get(target_names[i], "Unknown") for i in y_pred_arr]
    family_acc = float(np.mean(np.array(y_true_fam) == np.array(y_pred_fam)))

    # Per-family recall
    per_family_recall = {}
    for fam in FAMILY_NAMES:
        mask = np.array(y_true_fam) == fam
        if np.any(mask):
            fam_rec = float(np.mean(np.array(y_pred_fam)[mask] == fam))
            per_family_recall[fam] = fam_rec

    cm = confusion_matrix(
        y_true_arr,
        y_pred_arr,
        labels=list(range(len(target_names))),
    )

    return {
        "balanced_accuracy": overall_bal_acc,
        "family_accuracy": family_acc,
        "per_class_recall": per_class_recall,
        "per_family_recall": per_family_recall,
        "confusion_matrix": cm,
    }


def format_evaluation_report(eval_results: Dict) -> str:
    """Formats evaluation results into a human- and agent-readable markdown table."""
    lines = []
    lines.append(f"### Overall Balanced Accuracy: **{eval_results['balanced_accuracy']:.4f}**")
    lines.append(f"### Family-level Accuracy:     **{eval_results['family_accuracy']:.4f}**\n")

    lines.append("#### Per-Family Performance:")
    lines.append("| Family | Recall |")
    lines.append("| :--- | :---: |")
    for fam, rec in eval_results["per_family_recall"].items():
        lines.append(f"| {fam:<12} | {rec:.4f} |")

    lines.append("\n#### Per-Model Recall:")
    lines.append("| Model | Family | Recall |")
    lines.append("| :--- | :--- | :---: |")
    for model, rec in eval_results["per_class_recall"].items():
        fam = MODEL_FAMILY_MAP.get(model, "Unknown")
        lines.append(f"| `{model}` | {fam} | {rec:.4f} |")

    return "\n".join(lines)
