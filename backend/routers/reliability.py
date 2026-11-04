"""
/api/reliability — distribution-shift analysis endpoints.

Exposes the real statistical routines in
`backend.reliability.distribution_shift`:

  * covariate shift (MMD² + domain-classifier two-sample test)
  * label shift (chi-squared + total-variation distance)
  * importance weights (density-ratio estimation + Kish ESS)
  * model performance under synthetic shift sweeps

No metric is mocked: every number is computed from the requested dataset.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.config import RANDOM_STATE
from backend.core.dataset_loader import load_dataset, split_dataset
from backend.core.preprocessor import PreprocessConfig, build_preprocessor, fit_transform
from backend.core.registry import AlgorithmRegistry
from backend.reliability.distribution_shift import (
    mmd_squared,
    classifier_two_sample_test,
    label_shift_analysis,
    importance_weights,
    performance_under_shift,
)

router = APIRouter()


class SplitRequest(BaseModel):
    dataset_name: str
    test_size: float = Field(default=0.25, ge=0.05, le=0.5)
    random_state: int = Field(default=RANDOM_STATE)


class CovariateShiftRequest(SplitRequest):
    sigma: float | None = Field(default=None, gt=0)


class ImportanceWeightsRequest(SplitRequest):
    method: str = Field(default="classifier")
    clip_percentile: float = Field(default=95.0, ge=50.0, le=100.0)


class ShiftSweepRequest(SplitRequest):
    algorithm_slug: str
    hyperparams: dict[str, Any] = Field(default_factory=dict)
    scaler: str = Field(default="standard")
    imputer: str = Field(default="mean")
    shift_type: str = Field(default="gaussian")
    shift_magnitudes: list[float] = Field(
        default_factory=lambda: [0.0, 0.25, 0.5, 1.0, 2.0]
    )


def _load_split(req: SplitRequest):
    try:
        ds = load_dataset(req.dataset_name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return ds, split_dataset(ds, test_size=req.test_size, random_state=req.random_state)


@router.get("/shift-types")
def list_shift_types():
    """Metadata for the synthetic shift generator."""
    return {
        "shift_types": [
            {"name": "gaussian", "description": "Add N(0, magnitude²) noise to every feature."},
            {"name": "mean_shift", "description": "Shift feature means by magnitude standard deviations."},
            {"name": "scale_shift", "description": "Multiply features by magnitude."},
        ],
        "note": "Magnitude 0 is the unshifted baseline. All results use real predictions.",
    }


@router.post("/covariate-shift")
def covariate_shift(req: CovariateShiftRequest):
    """Compare train vs test feature distributions with MMD² and a domain classifier."""
    ds, split = _load_split(req)
    if split.X_train.shape[1] == 0:
        raise HTTPException(status_code=400, detail="Dataset has no features.")

    mmd = mmd_squared(split.X_train, split.X_test, sigma=req.sigma)
    two_sample = classifier_two_sample_test(
        split.X_train, split.X_test, random_state=req.random_state
    )
    return {
        "dataset_name": ds.name,
        "n_train": int(split.X_train.shape[0]),
        "n_test": int(split.X_test.shape[0]),
        "mmd": mmd,
        "classifier_two_sample_test": two_sample,
    }


@router.post("/label-shift")
def label_shift(req: SplitRequest):
    """Compare train vs test P(y) with a chi-squared test and TV distance."""
    ds, split = _load_split(req)
    if split.y_train is None or split.y_test is None:
        raise HTTPException(status_code=400, detail="Dataset has no target labels.")
    if ds.task_type != "classification":
        raise HTTPException(
            status_code=400,
            detail=f"Label shift requires a classification dataset (got '{ds.task_type}').",
        )
    return {
        "dataset_name": ds.name,
        **label_shift_analysis(split.y_train, split.y_test),
    }


@router.post("/importance-weights")
def get_importance_weights(req: ImportanceWeightsRequest):
    """Estimate p_target(x)/p_source(x) weights for covariate-shift correction."""
    ds, split = _load_split(req)
    try:
        result = importance_weights(
            split.X_train,
            split.X_test,
            method=req.method,
            random_state=req.random_state,
            clip_percentile=req.clip_percentile,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "dataset_name": ds.name,
        "n_train": int(split.X_train.shape[0]),
        **result,
    }


@router.post("/performance-under-shift")
def performance_under_shift_endpoint(req: ShiftSweepRequest):
    """Fit a real algorithm and measure metric degradation across a shift sweep."""
    if not req.shift_magnitudes:
        raise HTTPException(status_code=400, detail="shift_magnitudes must not be empty.")
    if len(req.shift_magnitudes) > 20:
        raise HTTPException(status_code=400, detail="At most 20 shift magnitudes are allowed.")
    if any(m < 0 for m in req.shift_magnitudes):
        raise HTTPException(status_code=400, detail="Shift magnitudes must be >= 0.")
    if sorted(req.shift_magnitudes) != req.shift_magnitudes:
        raise HTTPException(status_code=400, detail="Shift magnitudes must be sorted ascending.")

    ds, split = _load_split(req)
    if ds.task_type not in ("classification", "regression"):
        raise HTTPException(
            status_code=400,
            detail="performance-under-shift supports classification/regression datasets.",
        )

    AlgorithmRegistry.load_all()
    try:
        algo = AlgorithmRegistry.instantiate(
            req.algorithm_slug,
            random_state=req.random_state,
            **req.hyperparams,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    cfg = PreprocessConfig(scaler=req.scaler, imputer=req.imputer)
    pp = build_preprocessor(cfg)
    X_train, X_test = fit_transform(pp, split.X_train, split.X_test)

    try:
        algo.fit(X_train, split.y_train)
        result = performance_under_shift(
            algo,
            X_train,
            split.y_train,
            X_test,
            split.y_test,
            shift_magnitudes=req.shift_magnitudes,
            shift_type=req.shift_type,
            task_type=ds.task_type,
            random_state=req.random_state,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Shift analysis failed: {exc}")

    return {
        "dataset_name": ds.name,
        "algorithm_slug": req.algorithm_slug,
        "train_time_s": algo.training_time(),
        **result,
    }
