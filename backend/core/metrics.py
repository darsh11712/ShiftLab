"""
Metrics factory for ShiftLab.
Returns task-appropriate metrics — no fabricated numbers, ever.
"""
from __future__ import annotations

import numpy as np
from sklearn import metrics as skm


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute standard regression metrics."""
    mae = float(skm.mean_absolute_error(y_true, y_pred))
    mse = float(skm.mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(skm.r2_score(y_true, y_pred))
    mape = float(
        np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-8, y_true))) * 100
    )
    medae = float(skm.median_absolute_error(y_true, y_pred))
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
        "median_ae": medae,
    }


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    average: str = "weighted",
) -> dict[str, float]:
    """Compute standard classification metrics."""
    result: dict[str, float] = {
        "accuracy": float(skm.accuracy_score(y_true, y_pred)),
        "f1": float(skm.f1_score(y_true, y_pred, average=average, zero_division=0)),
        "precision": float(
            skm.precision_score(y_true, y_pred, average=average, zero_division=0)
        ),
        "recall": float(
            skm.recall_score(y_true, y_pred, average=average, zero_division=0)
        ),
    }
    classes = np.unique(y_true)
    if len(classes) == 2 and y_proba is not None:
        proba_pos = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
        try:
            result["roc_auc"] = float(skm.roc_auc_score(y_true, proba_pos))
            result["avg_precision"] = float(
                skm.average_precision_score(y_true, proba_pos)
            )
            result["log_loss"] = float(skm.log_loss(y_true, proba_pos))
        except Exception:
            pass
    elif len(classes) > 2 and y_proba is not None:
        try:
            result["roc_auc"] = float(
                skm.roc_auc_score(y_true, y_proba, multi_class="ovr", average=average)
            )
            result["log_loss"] = float(skm.log_loss(y_true, y_proba))
        except Exception:
            pass
    return result


def clustering_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    y_true: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute clustering metrics (internal + external if labels available)."""
    result: dict[str, float] = {}
    unique = np.unique(labels)
    n_clusters = len(unique[unique >= 0])  # exclude noise label -1 from DBSCAN

    if n_clusters >= 2:
        try:
            result["silhouette"] = float(skm.silhouette_score(X, labels))
        except Exception:
            result["silhouette"] = float("nan")
        try:
            result["davies_bouldin"] = float(skm.davies_bouldin_score(X, labels))
        except Exception:
            result["davies_bouldin"] = float("nan")
        try:
            result["calinski_harabasz"] = float(skm.calinski_harabasz_score(X, labels))
        except Exception:
            result["calinski_harabasz"] = float("nan")

    result["n_clusters_found"] = int(n_clusters)

    if y_true is not None:
        try:
            result["adjusted_rand_index"] = float(
                skm.adjusted_rand_score(y_true, labels)
            )
            result["normalized_mutual_info"] = float(
                skm.normalized_mutual_info_score(y_true, labels)
            )
        except Exception:
            pass

    return result


def anomaly_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Compute anomaly detection metrics.
    y_true: 1 for anomaly, 0 for normal.
    y_pred: predicted labels (1 anomaly, 0 normal) or sklearn convention (-1/1).
    """
    # Normalise sklearn -1/1 to 0/1
    y_pred_norm = np.where(y_pred == -1, 1, np.where(y_pred == 1, 0, y_pred))
    result: dict[str, float] = {
        "accuracy": float(skm.accuracy_score(y_true, y_pred_norm)),
        "precision": float(
            skm.precision_score(y_true, y_pred_norm, zero_division=0)
        ),
        "recall": float(skm.recall_score(y_true, y_pred_norm, zero_division=0)),
        "f1": float(skm.f1_score(y_true, y_pred_norm, zero_division=0)),
    }
    if scores is not None:
        try:
            result["roc_auc"] = float(skm.roc_auc_score(y_true, scores))
        except Exception:
            pass
    return result


def get_metrics_for_task(
    task_type: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    X: np.ndarray | None = None,
    y_proba: np.ndarray | None = None,
    scores: np.ndarray | None = None,
) -> dict[str, float]:
    """Unified dispatcher."""
    if task_type == "regression":
        return regression_metrics(y_true, y_pred)
    elif task_type == "classification":
        return classification_metrics(y_true, y_pred, y_proba)
    elif task_type == "clustering":
        return clustering_metrics(X, y_pred, y_true)
    elif task_type == "anomaly_detection":
        return anomaly_metrics(y_true, y_pred, scores)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
