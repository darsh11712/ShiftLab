"""
BaseAlgorithm — abstract interface every ShiftLab algorithm must implement.

Rules:
  - fit() must store all fitted state on self (no external side effects).
  - predict() must be callable after fit().
  - score() computes task-appropriate metrics on provided data.
  - explain() returns static metadata (no model needed).
  - No HistGradientBoosting* ever.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from backend.config import N_JOBS


class BaseAlgorithm(ABC):
    """Abstract base class for all ShiftLab algorithms."""

    # ------------------------------------------------------------------ #
    # Class-level metadata — override in each subclass                     #
    # ------------------------------------------------------------------ #
    name: str = "BaseAlgorithm"
    family: str = "base"          # supervised | unsupervised | anomaly | ensemble | neural | rl
    task_type: str = "base"       # regression | classification | clustering | dimensionality_reduction | anomaly_detection
    tags: list[str] = []
    sklearn_compatible: bool = True
    complexity_train: str = "?"   # e.g. "O(n·d)"
    complexity_predict: str = "?"

    def __init__(self, random_state: int = 42, **hyperparams: Any) -> None:
        self.random_state = random_state
        # Support both calling conventions:
        #   Cls(hyperparams={"key": val}, random_state=42)   ← UI / test style
        #   Cls(key=val, random_state=42)                    ← direct style
        # When a single "hyperparams" kwarg is passed containing a dict, flatten it.
        if list(hyperparams.keys()) == ["hyperparams"] and isinstance(
            hyperparams["hyperparams"], dict
        ):
            self._hyperparams: dict[str, Any] = hyperparams["hyperparams"]
        else:
            self._hyperparams = hyperparams
        self._model: Any = None
        self._is_fitted: bool = False
        self._train_time_s: float = 0.0

    # ------------------------------------------------------------------ #
    # Core interface                                                        #
    # ------------------------------------------------------------------ #
    @abstractmethod
    def _build_model(self) -> Any:
        """Instantiate and return the underlying model object."""

    def fit(self, X_train: np.ndarray, y_train: np.ndarray | None = None) -> "BaseAlgorithm":
        """Fit model on training data. Times the operation."""
        self._model = self._build_model()
        t0 = time.perf_counter()
        self._fit_impl(X_train, y_train)
        self._train_time_s = time.perf_counter() - t0
        self._is_fitted = True
        return self

    def _fit_impl(self, X_train: np.ndarray, y_train: np.ndarray | None) -> None:
        """Override if custom fit logic is needed; default delegates to sklearn API."""
        if y_train is not None:
            self._model.fit(X_train, y_train)
        else:
            self._model.fit(X_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._assert_fitted()
        t0 = time.perf_counter()
        result = self._predict_impl(X)
        self._last_infer_time_s = time.perf_counter() - t0
        return result

    def _predict_impl(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        """Return class probabilities where supported, else None."""
        self._assert_fitted()
        if hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(X)
        return None

    @abstractmethod
    def score(self, X: np.ndarray, y: np.ndarray | None = None) -> dict[str, float]:
        """Compute task-appropriate metrics. Must use real predictions only."""

    # ------------------------------------------------------------------ #
    # Hyperparameter management                                            #
    # ------------------------------------------------------------------ #
    def get_params(self) -> dict[str, Any]:
        return {"random_state": self.random_state, **self._hyperparams}

    def set_params(self, **params: Any) -> None:
        if "random_state" in params:
            self.random_state = params.pop("random_state")
        self._hyperparams.update(params)
        self._is_fitted = False  # invalidate fitted state

    @classmethod
    def hyperparameter_schema(cls) -> list[dict]:
        """
        Return list of hyperparameter descriptors for the UI.
        Each entry: {name, type, default, min, max, options, description}
        Subclasses should override.
        """
        return []

    # ------------------------------------------------------------------ #
    # Explainability / metadata                                            #
    # ------------------------------------------------------------------ #
    @classmethod
    def explain(cls) -> dict:
        """Return static metadata for the Algorithm Library."""
        return {
            "name": cls.name,
            "family": cls.family,
            "task_type": cls.task_type,
            "tags": cls.tags,
            "sklearn_compatible": cls.sklearn_compatible,
            "complexity_train": cls.complexity_train,
            "complexity_predict": cls.complexity_predict,
            "description": cls.__doc__ or "",
            "hyperparameters": cls.hyperparameter_schema(),
            "assumptions": cls._assumptions(),
            "strengths": cls._strengths(),
            "weaknesses": cls._weaknesses(),
            "use_cases": cls._use_cases(),
            "math_intuition": cls._math_intuition(),
        }

    @classmethod
    def _assumptions(cls) -> list[str]:
        return []

    @classmethod
    def _strengths(cls) -> list[str]:
        return []

    @classmethod
    def _weaknesses(cls) -> list[str]:
        return []

    @classmethod
    def _use_cases(cls) -> list[str]:
        return []

    @classmethod
    def _math_intuition(cls) -> str:
        return ""

    # ------------------------------------------------------------------ #
    # Timing                                                               #
    # ------------------------------------------------------------------ #
    def training_time(self) -> float:
        return self._train_time_s

    def last_inference_time(self) -> float:
        return getattr(self, "_last_infer_time_s", 0.0)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #
    def _n_jobs(self) -> int:
        """Resolve the parallel-worker count for estimators that support it."""
        return int(self._hyperparams.get("n_jobs", N_JOBS))

    def _assert_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(f"{self.name} has not been fitted. Call fit() first.")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(random_state={self.random_state}, **{self._hyperparams})"
