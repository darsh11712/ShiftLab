"""
/api/train and /api/predict — core training and inference endpoints.
"""
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any

import numpy as np

from backend.core.registry import AlgorithmRegistry
from backend.core.dataset_loader import load_dataset, split_dataset
from backend.core.preprocessor import PreprocessConfig, build_preprocessor, fit_transform
from backend.core.experiment import save_experiment
from backend.config import RANDOM_STATE

router = APIRouter()


class TrainRequest(BaseModel):
    algorithm_slug: str
    dataset_name: str
    test_size: float = Field(default=0.2, ge=0.05, le=0.5)
    random_state: int = Field(default=RANDOM_STATE)
    hyperparams: dict[str, Any] = Field(default_factory=dict)
    scaler: str = Field(default="standard")
    imputer: str = Field(default="mean")
    save: bool = Field(default=True)


class TrainResponse(BaseModel):
    algorithm_slug: str
    dataset_name: str
    dataset_task_type: str | None = None
    metrics: dict[str, Any]
    train_time_s: float
    infer_time_s: float
    hyperparams: dict[str, Any]
    experiment_id: str | None = None


_UNSUPERVISED_TASKS = {"clustering", "dimensionality_reduction", "anomaly_detection"}


def _fit_task_aware(algo, X_train, y_train):
    """Unsupervised-family algorithms fit on X only; supervised get y."""
    if algo.task_type in _UNSUPERVISED_TASKS:
        return algo.fit(X_train)
    return algo.fit(X_train, y_train)


def _score_task_aware(algo, X_test, y_test):
    """Score without pretending multiclass labels are anomaly labels."""
    if algo.task_type == "anomaly_detection":
        if y_test is None or len(np.unique(y_test)) > 2:
            return algo.score(X_test, None)
        return algo.score(X_test, y_test)
    return algo.score(X_test, y_test)


@router.post("/train", response_model=TrainResponse)
def train(req: TrainRequest):
    AlgorithmRegistry.load_all()

    try:
        ds = load_dataset(req.dataset_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    split = split_dataset(ds, test_size=req.test_size, random_state=req.random_state)

    cfg = PreprocessConfig(scaler=req.scaler, imputer=req.imputer)
    pp = build_preprocessor(cfg)
    X_train, X_test = fit_transform(pp, split.X_train, split.X_test)

    try:
        algo = AlgorithmRegistry.instantiate(
            req.algorithm_slug,
            random_state=req.random_state,
            **req.hyperparams,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Guard the obvious supervised mismatch, but allow unsupervised-family
    # algorithms (anomaly/dim-reduction) on any dataset.
    if (
        ds.task_type in ("classification", "regression")
        and algo.task_type in ("classification", "regression")
        and algo.task_type != ds.task_type
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Algorithm task '{algo.task_type}' cannot be trained on a "
                f"'{ds.task_type}' dataset."
            ),
        )

    try:
        _fit_task_aware(algo, X_train, split.y_train)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")

    try:
        metrics = _score_task_aware(algo, X_test, split.y_test)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring failed: {e}")

    exp_id = None
    if req.save:
        exp_id = save_experiment(
            algorithm_slug=req.algorithm_slug,
            dataset_name=req.dataset_name,
            hyperparams=algo.get_params(),
            metrics=metrics,
            train_time_s=algo.training_time(),
            infer_time_s=algo.last_inference_time(),
            random_state=req.random_state,
            preprocess_config={"scaler": req.scaler, "imputer": req.imputer},
        )

    return TrainResponse(
        algorithm_slug=req.algorithm_slug,
        dataset_name=req.dataset_name,
        dataset_task_type=ds.task_type,
        metrics=metrics,
        train_time_s=algo.training_time(),
        infer_time_s=algo.last_inference_time(),
        hyperparams=algo.get_params(),
        experiment_id=exp_id,
    )


class PredictRequest(BaseModel):
    algorithm_slug: str
    dataset_name: str
    hyperparams: dict[str, Any] = Field(default_factory=dict)
    scaler: str = "standard"
    imputer: str = "mean"
    X: list[list[float]]


@router.post("/predict")
def predict(req: PredictRequest):
    """
    Train on full dataset, predict on provided X rows.
    For demo / exploration purposes.
    """
    AlgorithmRegistry.load_all()

    try:
        ds = load_dataset(req.dataset_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    cfg = PreprocessConfig(scaler=req.scaler, imputer=req.imputer)
    pp = build_preprocessor(cfg)
    X_all_scaled = pp.fit_transform(ds.X)

    try:
        algo = AlgorithmRegistry.instantiate(req.algorithm_slug, **req.hyperparams)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    _fit_task_aware(algo, X_all_scaled, ds.y)

    X_new = np.array(req.X, dtype=np.float64)
    X_new_scaled = pp.transform(X_new)
    preds = algo.predict(X_new_scaled)

    result = {"predictions": preds.tolist()}
    proba = algo.predict_proba(X_new_scaled)
    if proba is not None:
        result["probabilities"] = proba.tolist()

    return result
