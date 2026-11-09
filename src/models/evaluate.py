"""
Avaliação de modelos de classificação de emoções.
"""

from typing import List, Dict, Any
from collections import Counter
import numpy as np


def classification_report_simple(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """Relatório simples de classificação (sem dependência de sklearn se necessário)."""
    labels = sorted(set(y_true + y_pred))
    report = {}

    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        report[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(1 for t in y_true if t == label)
        }

    # Accuracy global
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true) if y_true else 0.0
    report["accuracy"] = round(accuracy, 4)
    report["macro_f1"] = round(np.mean([v["f1"] for k, v in report.items() if k != "accuracy"]), 4)

    return report


def print_report(report: Dict[str, Any]):
    print("\n=== Relatório de Classificação ===")
    print(f"Accuracy: {report.get('accuracy', 0):.4f}")
    print(f"Macro F1: {report.get('macro_f1', 0):.4f}\n")
    for label, metrics in report.items():
        if label in ("accuracy", "macro_f1"):
            continue
        print(f"{label:12}  P: {metrics['precision']:.3f}  R: {metrics['recall']:.3f}  F1: {metrics['f1']:.3f}  (n={metrics['support']})")
