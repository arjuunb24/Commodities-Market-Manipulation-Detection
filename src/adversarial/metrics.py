"""
src/adversarial/metrics.py
==========================
Helper functions for computing, comparing, and persisting per-round
detection metrics.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)


def compute_round_metrics(
    results: pd.DataFrame,
    y: pd.DataFrame,
    personas: list[str],
    threshold: float = 0.5,
) -> dict[str, dict[str, float]]:
    """
    Computes per-persona Precision, Recall, F1 and FPR by comparing 
    ensemble_flag / supervised probabilities against ground-truth labels.

    Args:
        results:   Output of DetectorPipeline.score(X) — contains ensemble_flag,
                   prob_<persona> columns, sup_flag, iso_flag.
        y:         Ground-truth label DataFrame from DetectorPipeline._generate_labels().
                   Columns are persona names, values are 0/1 integers.
        personas:  List of persona names matching y's columns.
        threshold: Probability threshold for supervised per-persona flags.

    Returns:
        Dict: {persona: {precision, recall, f1, fpr, support}}
    """
    metrics: dict[str, dict[str, float]] = {}

    # Align indexes (both share the same (commodity, timestamp) MultiIndex)
    common_idx = results.index.intersection(y.index)
    if common_idx.empty:
        logger.warning("No common index between results and labels. Returning empty metrics.")
        return {}

    results_aligned = results.loc[common_idx]
    y_aligned = y.loc[common_idx]

    for persona in personas:
        if persona not in y_aligned.columns:
            continue

        y_true = y_aligned[persona].values

        # Use per-persona probability column if available, else fall back to ensemble_flag
        prob_col = f"prob_{persona}"
        if prob_col in results_aligned.columns:
            y_pred = (results_aligned[prob_col] > threshold).astype(int).values
        else:
            y_pred = results_aligned["ensemble_flag"].astype(int).values

        # Handle degenerate case: all zeros in y_true → model can't be evaluated
        if y_true.sum() == 0:
            logger.warning(f"No positive labels for persona '{persona}' in this round. Skipping.")
            metrics[persona] = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "fpr": 0.0, "support": 0}
            continue

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall    = recall_score(y_true, y_pred, zero_division=0)
        f1        = f1_score(y_true, y_pred, zero_division=0)

        # FPR = FP / (FP + TN)
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        metrics[persona] = {
            "precision": round(float(precision), 4),
            "recall":    round(float(recall), 4),
            "f1":        round(float(f1), 4),
            "fpr":       round(fpr, 4),
            "support":   int(y_true.sum()),
        }

        logger.info(
            f"[Metrics] Round persona={persona}: "
            f"P={precision:.3f} R={recall:.3f} F1={f1:.3f} FPR={fpr:.3f}"
        )

    return metrics


def compute_shap_drift(
    shap_a: dict[str, float],
    shap_b: dict[str, float],
) -> dict[str, float]:
    """
    Computes the change in feature importance between two rounds.
    Returns a dict of {feature_name: delta} sorted by abs(delta) descending.
    """
    all_features = set(shap_a) | set(shap_b)
    drift = {}
    for feat in all_features:
        val_a = shap_a.get(feat, 0.0)
        val_b = shap_b.get(feat, 0.0)
        drift[feat] = round(val_b - val_a, 4)

    return dict(sorted(drift.items(), key=lambda kv: abs(kv[1]), reverse=True))


def save_round_metrics(
    metrics: dict[str, dict[str, float]],
    round_num: int,
    round_dir: Path,
) -> None:
    """Persists round metrics to JSON and updates the cumulative parquet file."""
    round_dir.mkdir(parents=True, exist_ok=True)

    # Save per-round JSON
    metrics_path = round_dir / "metrics.json"
    payload = {"round": round_num, "personas": metrics}
    with open(metrics_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Round {round_num} metrics saved to {metrics_path}")


def load_round_metrics(round_dir: Path) -> dict[str, dict[str, float]]:
    """Loads metrics.json from a round directory."""
    path = round_dir / "metrics.json"
    if not path.exists():
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("personas", {})


def build_summary_table(all_metrics: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Converts a list of {round, persona, precision, recall, f1} dicts
    into a tidy DataFrame for pretty-printing and parquet export.
    """
    rows = []
    for entry in all_metrics:
        rnd = entry["round"]
        for persona, m in entry["personas"].items():
            rows.append({
                "round": rnd,
                "persona": persona,
                "precision": m.get("precision", 0.0),
                "recall":    m.get("recall", 0.0),
                "f1":        m.get("f1", 0.0),
                "fpr":       m.get("fpr", 0.0),
                "support":   m.get("support", 0),
            })
    return pd.DataFrame(rows)
