"""
ShiftLab Arena — head-to-head algorithm comparisons and leaderboards.

Everything here is derived from real fits on real data. Failed fits are
reported as errors rather than being silently turned into fake scores.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from backend.config import RANDOM_STATE
from backend.core.dataset_loader import load_dataset, split_dataset
from backend.core.experiment import list_experiments, save_experiment
from backend.core.preprocessor import PreprocessConfig, build_preprocessor, fit_transform
from backend.core.registry import AlgorithmRegistry

PRIMARY_METRIC = {
    "classification": "accuracy",
    "regression": "r2",
    "clustering": "silhouette",
    "anomaly_detection": "f1",
    "dimensionality_reduction": "explained_variance",
}


def primary_metric_for(task_type: str) -> str | None:
    return PRIMARY_METRIC.get(task_type)


def _rank(results: list[dict], metric: str) -> list[dict]:
    """Sort successful results by a metric (higher is better)."""
    successful: list[dict] = []
    failed: list[dict] = []

    def metric_value(r: dict) -> float:
        value = (r.get("metrics") or {}).get(metric)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return float("-inf")
        return value if np.isfinite(value) else float("-inf")

    for result in results:
        (successful if metric_value(result) != float("-inf") else failed).append(result)

    successful.sort(key=metric_value, reverse=True)
    for i, entry in enumerate(successful, start=1):
        entry["rank"] = i
    for entry in failed:
        entry["rank"] = None
    return successful + failed


def run_comparison(
    algorithm_slugs: list[str],
    dataset_name: str,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    hyperparams_by_slug: dict[str, dict[str, Any]] | None = None,
    scaler: str = "standard",
    imputer: str = "mean",
    save: bool = False,
) -> dict:
    """
    Train every requested algorithm on the same split/preprocessing and rank
    them by the task's primary metric.

    Raises ValueError for bad input; individual algorithm failures are captured
    in the returned `results` list with an `error` field.
    """
    if not algorithm_slugs:
        raise ValueError("At least one algorithm slug is required.")
    if len(algorithm_slugs) > 30:
        raise ValueError("At most 30 algorithms can be compared at once.")

    ds = load_dataset(dataset_name)
    if ds.task_type not in ("classification", "regression"):
        raise ValueError(
            "Arena comparison currently supports classification and regression "
            f"datasets (got '{ds.task_type}')."
        )

    AlgorithmRegistry.load_all()
    split = split_dataset(ds, test_size=test_size, random_state=random_state)
    cfg = PreprocessConfig(scaler=scaler, imputer=imputer)
    pp = build_preprocessor(cfg)
    X_train, X_test = fit_transform(pp, split.X_train, split.X_test)

    hyperparams_by_slug = hyperparams_by_slug or {}
    metric = primary_metric_for(ds.task_type)
    results: list[dict] = []

    for slug in algorithm_slugs:
        entry: dict[str, Any] = {
            "algorithm_slug": slug,
            "metrics": {},
            "train_time_s": None,
            "error": None,
        }
        try:
            klass = AlgorithmRegistry.get(slug)
        except KeyError as exc:
            entry["error"] = str(exc)
            results.append(entry)
            continue

        entry["name"] = klass.name
        entry["family"] = klass.family
        entry["task_type"] = klass.task_type

        if klass.task_type != ds.task_type:
            entry["error"] = (
                f"Task mismatch: algorithm is '{klass.task_type}' but dataset is "
                f"'{ds.task_type}'."
            )
            results.append(entry)
            continue

        try:
            algo = AlgorithmRegistry.instantiate(
                slug,
                random_state=random_state,
                **(hyperparams_by_slug.get(slug) or {}),
            )
            algo.fit(X_train, split.y_train)
            metrics = algo.score(X_test, split.y_test)
            entry["metrics"] = metrics
            entry["train_time_s"] = algo.training_time()
            entry["hyperparams"] = algo.get_params()

            if save and metric in metrics:
                entry["experiment_id"] = save_experiment(
                    algorithm_slug=slug,
                    dataset_name=ds.name,
                    hyperparams=algo.get_params(),
                    metrics=metrics,
                    train_time_s=algo.training_time(),
                    infer_time_s=algo.last_inference_time(),
                    random_state=random_state,
                    preprocess_config={"scaler": scaler, "imputer": imputer},
                    extra={"source": "arena"},
                )
        except Exception as exc:  # one bad algorithm must not abort the arena
            entry["error"] = f"{type(exc).__name__}: {exc}"
        results.append(entry)

    ranked = _rank(results, metric)
    return {
        "dataset_name": ds.name,
        "task_type": ds.task_type,
        "primary_metric": metric,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "test_size": test_size,
        "random_state": random_state,
        "results": ranked,
        "winner": next((r["algorithm_slug"] for r in ranked if r.get("rank") == 1), None),
    }


def build_leaderboard(
    dataset_name: str | None = None,
    task_type: str | None = None,
    limit: int = 50,
) -> dict:
    """
    Rank saved experiments by their task's primary metric.

    Experiments whose algorithm is no longer registered (e.g. synthetic test
    records) are ignored instead of appearing as bogus leaderboard entries.
    """
    AlgorithmRegistry.load_all()
    entries: list[dict] = []

    for exp in list_experiments():
        if dataset_name is not None and exp.get("dataset_name") != dataset_name:
            continue
        slug = exp.get("algorithm_slug", "")
        try:
            klass = AlgorithmRegistry.get(slug)
        except KeyError:
            continue
        if task_type is not None and klass.task_type != task_type:
            continue

        metric = primary_metric_for(klass.task_type)
        metrics = exp.get("metrics") or {}
        if metric is None or metric not in metrics:
            continue
        try:
            value = float(metrics[metric])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(value):
            continue

        entries.append({
            "experiment_id": exp["id"],
            "algorithm_slug": slug,
            "name": klass.name,
            "family": klass.family,
            "task_type": klass.task_type,
            "dataset_name": exp.get("dataset_name"),
            "metric": metric,
            "value": value,
            "metrics": metrics,
            "train_time_s": exp.get("train_time_s"),
            "created_at": exp.get("created_at"),
        })

    entries.sort(key=lambda e: e["value"], reverse=True)
    for i, entry in enumerate(entries[:max(0, limit)], start=1):
        entry["rank"] = i

    return {
        "task_type": task_type,
        "dataset_name": dataset_name,
        "n_entries": len(entries),
        "entries": entries[:max(0, limit)],
    }
