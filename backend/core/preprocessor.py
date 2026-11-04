"""
Preprocessing pipelines for ShiftLab.
Fit ONLY on training data, transform both train and test.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OrdinalEncoder, OneHotEncoder,
)


ScalerType = Literal["standard", "minmax", "robust", "none"]
ImputerType = Literal["mean", "median", "most_frequent", "knn", "none"]


@dataclass
class PreprocessConfig:
    scaler: ScalerType = "standard"
    imputer: ImputerType = "mean"


def build_preprocessor(config: PreprocessConfig) -> Pipeline:
    """Build a sklearn Pipeline that scales and imputes (fit on train only)."""
    steps = []

    if config.imputer != "none":
        if config.imputer == "knn":
            steps.append(("imputer", KNNImputer(n_neighbors=5)))
        else:
            steps.append(("imputer", SimpleImputer(strategy=config.imputer)))

    if config.scaler != "none":
        scalers = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
        }
        steps.append(("scaler", scalers[config.scaler]))

    if not steps:
        # Identity — still a valid pipeline
        from sklearn.preprocessing import FunctionTransformer
        steps.append(("identity", FunctionTransformer()))

    return Pipeline(steps)


def fit_transform(
    preprocessor: Pipeline,
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit on X_train, transform both.
    Returns (X_train_processed, X_test_processed).
    No leakage: test statistics never touch the fit step.
    """
    X_train_out = preprocessor.fit_transform(X_train)
    X_test_out = preprocessor.transform(X_test)
    return X_train_out, X_test_out


def get_scaler_description(scaler: ScalerType) -> str:
    descriptions = {
        "standard": "Zero mean, unit variance (z-score normalization).",
        "minmax": "Scale to [0, 1] range.",
        "robust": "Scale using median and IQR — robust to outliers.",
        "none": "No scaling applied.",
    }
    return descriptions.get(scaler, "")


def get_imputer_description(imputer: ImputerType) -> str:
    descriptions = {
        "mean": "Replace missing values with column mean.",
        "median": "Replace missing values with column median.",
        "most_frequent": "Replace with most frequent value.",
        "knn": "K-nearest neighbour imputation (k=5).",
        "none": "No imputation. Fails if NaN values are present.",
    }
    return descriptions.get(imputer, "")
