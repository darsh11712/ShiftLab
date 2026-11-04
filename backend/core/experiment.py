"""
Experiment persistence for ShiftLab.
Each experiment is saved as a JSON file. Results contain real metric values only.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np

from backend.config import EXPERIMENTS_DIR


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_experiment(
    algorithm_slug: str,
    dataset_name: str,
    hyperparams: dict[str, Any],
    metrics: dict[str, float],
    train_time_s: float,
    infer_time_s: float,
    random_state: int,
    preprocess_config: dict | None = None,
    extra: dict | None = None,
) -> str:
    """Save an experiment to disk. Returns the experiment ID."""
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    exp_id = str(uuid.uuid4())[:8]
    record = {
        "id": exp_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "algorithm_slug": algorithm_slug,
        "dataset_name": dataset_name,
        "hyperparams": hyperparams,
        "metrics": metrics,
        "train_time_s": train_time_s,
        "infer_time_s": infer_time_s,
        "random_state": random_state,
        "preprocess_config": preprocess_config or {},
        "extra": extra or {},
    }
    path = os.path.join(EXPERIMENTS_DIR, f"{exp_id}.json")
    with open(path, "w") as f:
        json.dump(record, f, cls=NumpyEncoder, indent=2)
    return exp_id


def load_experiment(exp_id: str) -> dict:
    path = os.path.join(EXPERIMENTS_DIR, f"{exp_id}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Experiment '{exp_id}' not found.")
    with open(path) as f:
        return json.load(f)


def list_experiments() -> list[dict]:
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    results = []
    for fname in sorted(os.listdir(EXPERIMENTS_DIR)):
        if fname.endswith(".json"):
            with open(os.path.join(EXPERIMENTS_DIR, fname)) as f:
                record = json.load(f)
            results.append({
                "id": record["id"],
                "algorithm_slug": record["algorithm_slug"],
                "dataset_name": record["dataset_name"],
                "created_at": record["created_at"],
                "metrics": record["metrics"],
                "train_time_s": record["train_time_s"],
            })
    return results
