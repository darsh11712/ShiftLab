"""
Anomaly Detection Algorithms for ShiftLab — 7 algorithms.

All anomaly detectors:
- fit() on training data (expected: mostly normal).
- predict() returns labels: 1 = anomaly, 0 = normal (or -1/1 sklearn convention).
- score_samples() returns anomaly scores (higher = more anomalous).
- score() computes anomaly_metrics if y_true provided.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope

from backend.core.base_algorithm import BaseAlgorithm
from backend.core.metrics import anomaly_metrics
from backend.core.registry import AlgorithmRegistry

try:
    from pyod.models.copod import COPOD
    from pyod.models.ecod import ECOD
    _PYOD_AVAILABLE = True
except ImportError:
    _PYOD_AVAILABLE = False


def _sklearn_labels_to_binary(labels: np.ndarray) -> np.ndarray:
    """Convert sklearn -1/1 convention to 1=anomaly, 0=normal."""
    return np.where(labels == -1, 1, 0)


# ────────────────────────────────────────────────────────────────────
# A-01  Isolation Forest
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class IsolationForestAlgo(BaseAlgorithm):
    """
    Isolation Forest.
    Detects anomalies by isolating observations using random feature splits.
    Anomalies are isolated faster (shorter paths in trees) than normal points.
    """
    name = "Isolation Forest"
    family = "unsupervised"
    task_type = "anomaly_detection"
    tags = ["ensemble", "tree-based", "unsupervised", "efficient", "high-dimensional"]
    complexity_train = "O(n·trees·log n)"
    complexity_predict = "O(n·trees·log n)"

    def _build_model(self):
        return IsolationForest(
            n_estimators=self._hyperparams.get("n_estimators", 100),
            max_samples=self._hyperparams.get("max_samples", "auto"),
            contamination=self._hyperparams.get("contamination", 0.1),
            max_features=self._hyperparams.get("max_features", 1.0),
            random_state=self.random_state,
            n_jobs=self._n_jobs(),
        )

    def _fit_impl(self, X, y=None):
        self._model.fit(X)

    def _predict_impl(self, X):
        return _sklearn_labels_to_binary(self._model.predict(X))

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Anomaly scores: higher = more anomalous."""
        self._assert_fitted()
        return -self._model.score_samples(X)

    def score(self, X, y=None):
        preds = self.predict(X)
        scores = self.score_samples(X)
        if y is not None:
            return anomaly_metrics(y, preds, scores)
        n_anom = int(preds.sum())
        return {"n_anomalies_detected": n_anom, "anomaly_rate": float(n_anom / len(preds))}

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "n_estimators", "type": "int", "default": 100, "min": 10, "max": 500},
            {"name": "contamination", "type": "float", "default": 0.1, "min": 0.001, "max": 0.5,
             "description": "Expected proportion of anomalies in the dataset."},
            {"name": "max_features", "type": "float", "default": 1.0, "min": 0.1, "max": 1.0,
             "description": "Fraction of features to draw per tree."},
        ]

    @classmethod
    def _math_intuition(cls):
        return (
            "Anomaly score: s(x,n) = 2^(-E[h(x)]/c(n)) where h(x) is path length, "
            "c(n) = 2H(n-1) - 2(n-1)/n (average path length for n points). "
            "s→1 = anomaly, s→0.5 = normal."
        )

    @classmethod
    def _strengths(cls):
        return [
            "Linear time complexity.",
            "Works well in high dimensions.",
            "Does not require distance computation.",
            "Naturally handles multimodal distributions.",
        ]

    @classmethod
    def _weaknesses(cls):
        return [
            "Contamination parameter must be estimated.",
            "Can miss anomalies that cluster together.",
        ]

    @classmethod
    def _use_cases(cls):
        return ["Fraud detection", "Network intrusion", "Manufacturing defects", "General outlier detection"]

    @classmethod
    def _assumptions(cls):
        return ["Anomalies are rare and different from normal points.", "No strict distribution assumption."]


# ────────────────────────────────────────────────────────────────────
# A-02  Local Outlier Factor
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class LOFAlgo(BaseAlgorithm):
    """
    Local Outlier Factor (LOF).
    Compares the local density of a point to its neighbours.
    Points with substantially lower density than their neighbours are outliers.
    """
    name = "Local Outlier Factor"
    family = "unsupervised"
    task_type = "anomaly_detection"
    tags = ["density", "local", "neighbourhood", "unsupervised"]
    complexity_train = "O(n²) naive, O(n log n) with k-d tree"
    complexity_predict = "O(n·k)"

    def _build_model(self):
        return LocalOutlierFactor(
            n_neighbors=self._hyperparams.get("n_neighbors", 20),
            contamination=self._hyperparams.get("contamination", 0.1),
            metric=self._hyperparams.get("metric", "minkowski"),
            novelty=True,  # enable predict on new data
            n_jobs=self._n_jobs(),
        )

    def _fit_impl(self, X, y=None):
        self._model.fit(X)

    def _predict_impl(self, X):
        return _sklearn_labels_to_binary(self._model.predict(X))

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        self._assert_fitted()
        return -self._model.score_samples(X)

    def score(self, X, y=None):
        preds = self.predict(X)
        scores = self.score_samples(X)
        if y is not None:
            return anomaly_metrics(y, preds, scores)
        n_anom = int(preds.sum())
        return {"n_anomalies_detected": n_anom, "anomaly_rate": float(n_anom / len(preds))}

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "n_neighbors", "type": "int", "default": 20, "min": 2, "max": 100,
             "description": "Number of neighbours for local density estimate."},
            {"name": "contamination", "type": "float", "default": 0.1, "min": 0.001, "max": 0.5},
            {"name": "metric", "type": "select", "default": "minkowski",
             "options": ["minkowski", "euclidean", "manhattan", "cosine"]},
        ]

    @classmethod
    def _math_intuition(cls):
        return (
            "LOF(p) = [Σₒ∈Nₖ(p) lrd(o)/lrd(p)] / k. "
            "lrd(p) = 1 / [Σₒ∈Nₖ(p) reach-dist(p,o) / k]. "
            "LOF > 1 indicates lower density than neighbours → outlier."
        )

    @classmethod
    def _strengths(cls):
        return [
            "Detects local outliers in varying-density data.",
            "Does not assume global density model.",
            "Provides a scoring (LOF score) not just binary label.",
        ]

    @classmethod
    def _weaknesses(cls):
        return [
            "Slow on large datasets.",
            "Sensitive to n_neighbors.",
            "High-dimensional data: distances become meaningless.",
        ]

    @classmethod
    def _use_cases(cls):
        return ["Intrusion detection", "Medical outliers", "Sensor anomalies"]

    @classmethod
    def _assumptions(cls):
        return ["Clusters exist with varying densities.", "Scale features."]


# ────────────────────────────────────────────────────────────────────
# A-03  One-Class SVM
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class OneClassSVMAlgo(BaseAlgorithm):
    """
    One-Class SVM.
    Learns a decision boundary around the training data (assumed normal).
    Uses the kernel trick to find a hyperplane in feature space separating
    training data from the origin with maximum margin.
    """
    name = "One-Class SVM"
    family = "unsupervised"
    task_type = "anomaly_detection"
    tags = ["svm", "kernel", "boundary", "novelty-detection"]
    complexity_train = "O(n²·d) to O(n³)"
    complexity_predict = "O(n_sv·d)"

    def _build_model(self):
        return OneClassSVM(
            kernel=self._hyperparams.get("kernel", "rbf"),
            gamma=self._hyperparams.get("gamma", "scale"),
            nu=self._hyperparams.get("nu", 0.1),
        )

    def _fit_impl(self, X, y=None):
        self._model.fit(X)

    def _predict_impl(self, X):
        return _sklearn_labels_to_binary(self._model.predict(X))

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        self._assert_fitted()
        return -self._model.score_samples(X)

    def score(self, X, y=None):
        preds = self.predict(X)
        scores = self.score_samples(X)
        if y is not None:
            return anomaly_metrics(y, preds, scores)
        n_anom = int(preds.sum())
        return {"n_anomalies_detected": n_anom, "anomaly_rate": float(n_anom / len(preds))}

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "kernel", "type": "select", "default": "rbf",
             "options": ["rbf", "linear", "poly", "sigmoid"]},
            {"name": "gamma", "type": "select", "default": "scale",
             "options": ["scale", "auto"]},
            {"name": "nu", "type": "float", "default": 0.1, "min": 0.001, "max": 0.999,
             "description": "Upper bound on fraction of outliers AND lower bound on support vectors."},
        ]

    @classmethod
    def _math_intuition(cls):
        return (
            "Find hyperplane w·Φ(x) = ρ that separates training data from origin. "
            "Objective: min ½‖w‖² + (1/νn)Σ ξᵢ - ρ subject to w·Φ(xᵢ) ≥ ρ - ξᵢ."
        )

    @classmethod
    def _strengths(cls):
        return [
            "Powerful non-linear boundary via kernel trick.",
            "Works well when normal class is compact.",
            "nu directly controls fraction of outliers.",
        ]

    @classmethod
    def _weaknesses(cls):
        return [
            "Slow O(n²–n³) on large data.",
            "Requires feature scaling.",
            "Sensitive to kernel and nu.",
        ]

    @classmethod
    def _use_cases(cls):
        return ["Novelty detection", "Quality control", "Medical anomaly detection"]

    @classmethod
    def _assumptions(cls):
        return ["Training data is mostly normal.", "Scale features."]


# ────────────────────────────────────────────────────────────────────
# A-04  Elliptic Envelope
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class EllipticEnvelopeAlgo(BaseAlgorithm):
    """
    Elliptic Envelope (Robust Covariance).
    Fits a robust estimate of the data's covariance matrix and flags points
    outside the Mahalanobis distance threshold as outliers.
    Assumes the inliers follow a Gaussian distribution.
    """
    name = "Elliptic Envelope"
    family = "unsupervised"
    task_type = "anomaly_detection"
    tags = ["statistical", "gaussian", "mahalanobis", "robust-covariance"]
    complexity_train = "O(n·d²)"
    complexity_predict = "O(n·d²)"

    def _build_model(self):
        return EllipticEnvelope(
            contamination=self._hyperparams.get("contamination", 0.1),
            support_fraction=self._hyperparams.get("support_fraction", None),
            random_state=self.random_state,
        )

    def _fit_impl(self, X, y=None):
        self._model.fit(X)

    def _predict_impl(self, X):
        return _sklearn_labels_to_binary(self._model.predict(X))

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        self._assert_fitted()
        return -self._model.score_samples(X)

    def score(self, X, y=None):
        preds = self.predict(X)
        scores = self.score_samples(X)
        if y is not None:
            return anomaly_metrics(y, preds, scores)
        n_anom = int(preds.sum())
        return {"n_anomalies_detected": n_anom, "anomaly_rate": float(n_anom / len(preds))}

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "contamination", "type": "float", "default": 0.1, "min": 0.001, "max": 0.5},
            {"name": "support_fraction", "type": "float", "default": None, "min": 0.5, "max": 1.0,
             "description": "Fraction of data for MCD computation. None = (n_features+1)/2."},
        ]

    @classmethod
    def _math_intuition(cls):
        return (
            "Fits Minimum Covariance Determinant (MCD) estimator (μ̂, Σ̂). "
            "Mahalanobis distance: d(x) = √[(x-μ̂)ᵀΣ̂⁻¹(x-μ̂)]. "
            "Points with d(x) > χ²-threshold are flagged as outliers."
        )

    @classmethod
    def _strengths(cls):
        return [
            "Principled statistical approach.",
            "Robust covariance estimate handles up to 50% contamination.",
            "Clear geometric interpretation (ellipse).",
        ]

    @classmethod
    def _weaknesses(cls):
        return [
            "Assumes Gaussian inlier distribution.",
            "Fails for multimodal or non-Gaussian data.",
            "Expensive O(n·d²).",
        ]

    @classmethod
    def _use_cases(cls):
        return ["Quality control with Gaussian assumptions", "Financial return outliers"]

    @classmethod
    def _assumptions(cls):
        return ["Inliers follow a multivariate Gaussian distribution.", "Scale features."]


# ────────────────────────────────────────────────────────────────────
# A-05  COPOD (via PyOD)
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class COPODAlgo(BaseAlgorithm):
    """
    COPOD — Copula-Based Outlier Detection.
    Fast parameter-free outlier detection using empirical copulas to model
    the joint distribution of features. O(n log n) complexity.
    Requires PyOD.
    """
    name = "COPOD"
    family = "unsupervised"
    task_type = "anomaly_detection"
    tags = ["copula", "fast", "parameter-free", "statistical", "pyod"]
    complexity_train = "O(n log n · d)"
    complexity_predict = "O(n log n · d)"

    def _build_model(self):
        if not _PYOD_AVAILABLE:
            raise ImportError("pyod is required for COPOD. Install with: pip install pyod")
        return COPOD(contamination=self._hyperparams.get("contamination", 0.1))

    def _fit_impl(self, X, y=None):
        self._model.fit(X)

    def _predict_impl(self, X):
        return self._model.predict(X)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        self._assert_fitted()
        return self._model.decision_function(X)

    def score(self, X, y=None):
        preds = self.predict(X)
        scores = self.score_samples(X)
        if y is not None:
            return anomaly_metrics(y, preds, scores)
        n_anom = int((preds == 1).sum())
        return {"n_anomalies_detected": n_anom, "anomaly_rate": float(n_anom / len(preds))}

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "contamination", "type": "float", "default": 0.1, "min": 0.001, "max": 0.5},
        ]

    @classmethod
    def _math_intuition(cls):
        return (
            "Models the joint distribution as an empirical copula: "
            "F(x₁,...,xd) = C(F₁(x₁), ..., Fd(xd)). "
            "Outlier score = -log C(u) where u = empirical CDF values."
        )

    @classmethod
    def _strengths(cls):
        return [
            "Parameter-free (only contamination needed).",
            "O(n log n) — very fast.",
            "Handles both left- and right-tail anomalies.",
        ]

    @classmethod
    def _weaknesses(cls):
        return ["Assumes features are independent after copula transformation.", "Requires PyOD."]

    @classmethod
    def _use_cases(cls):
        return ["High-throughput anomaly detection", "Financial data", "Tabular outlier detection"]

    @classmethod
    def _assumptions(cls):
        return ["Anomalies lie in the tails of the marginal distributions."]


# ────────────────────────────────────────────────────────────────────
# A-06  ECOD (via PyOD)
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class ECODAlgo(BaseAlgorithm):
    """
    ECOD — Empirical Cumulative-distribution-based Outlier Detection.
    Uses the empirical CDF of each feature to compute tail probabilities.
    Combines left- and right-tail scores. Very fast and parameter-free.
    Requires PyOD.
    """
    name = "ECOD"
    family = "unsupervised"
    task_type = "anomaly_detection"
    tags = ["empirical-cdf", "fast", "parameter-free", "statistical", "pyod"]
    complexity_train = "O(n log n · d)"
    complexity_predict = "O(n log n · d)"

    def _build_model(self):
        if not _PYOD_AVAILABLE:
            raise ImportError("pyod is required for ECOD. Install with: pip install pyod")
        return ECOD(contamination=self._hyperparams.get("contamination", 0.1))

    def _fit_impl(self, X, y=None):
        self._model.fit(X)

    def _predict_impl(self, X):
        return self._model.predict(X)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        self._assert_fitted()
        return self._model.decision_function(X)

    def score(self, X, y=None):
        preds = self.predict(X)
        scores = self.score_samples(X)
        if y is not None:
            return anomaly_metrics(y, preds, scores)
        n_anom = int((preds == 1).sum())
        return {"n_anomalies_detected": n_anom, "anomaly_rate": float(n_anom / len(preds))}

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "contamination", "type": "float", "default": 0.1, "min": 0.001, "max": 0.5},
        ]

    @classmethod
    def _math_intuition(cls):
        return (
            "Outlier score = -log[P(X ≤ x) · P(X ≥ x)] "
            "= sum of left- and right-tail log probabilities across features."
        )

    @classmethod
    def _strengths(cls):
        return ["Extremely fast.", "Parameter-free.", "Interpretable marginal CDF scores."]

    @classmethod
    def _weaknesses(cls):
        return ["Ignores feature dependencies.", "Requires PyOD."]

    @classmethod
    def _use_cases(cls):
        return ["Large-scale tabular anomaly detection", "Fast baseline outlier detection"]

    @classmethod
    def _assumptions(cls):
        return ["Anomalies have extreme marginal values in at least one feature."]


# ────────────────────────────────────────────────────────────────────
# A-07  Z-Score / IQR Statistical Detector
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class ZScoreIQRAlgo(BaseAlgorithm):
    """
    Z-Score / IQR Statistical Anomaly Detector.
    Uses classical univariate statistics to flag outliers.
    Z-Score method: flag points with |z| > threshold.
    IQR method: flag points outside [Q1 - k·IQR, Q3 + k·IQR].
    Interpretable and assumption-free baseline.
    """
    name = "Z-Score / IQR Detector"
    family = "unsupervised"
    task_type = "anomaly_detection"
    tags = ["statistical", "univariate", "baseline", "interpretable", "fast"]
    complexity_train = "O(n·d)"
    complexity_predict = "O(n·d)"
    sklearn_compatible = False

    def _build_model(self):
        return None  # Pure numpy/scipy implementation

    def _fit_impl(self, X, y=None):
        method = self._hyperparams.get("method", "zscore")
        if method == "zscore":
            self._means = X.mean(axis=0)
            self._stds = X.std(axis=0)
            self._stds[self._stds == 0] = 1e-8
        elif method == "iqr":
            self._q1 = np.percentile(X, 25, axis=0)
            self._q3 = np.percentile(X, 75, axis=0)
            self._iqr = self._q3 - self._q1
            self._iqr[self._iqr == 0] = 1e-8

    def _compute_scores(self, X: np.ndarray) -> np.ndarray:
        method = self._hyperparams.get("method", "zscore")
        if method == "zscore":
            z = np.abs((X - self._means) / self._stds)
            return z.max(axis=1)
        elif method == "iqr":
            k = self._hyperparams.get("iqr_factor", 1.5)
            lower = self._q1 - k * self._iqr
            upper = self._q3 + k * self._iqr
            is_out = (X < lower) | (X > upper)
            return is_out.sum(axis=1).astype(float)
        raise ValueError(f"Unknown method: {method}")

    def _predict_impl(self, X: np.ndarray) -> np.ndarray:
        method = self._hyperparams.get("method", "zscore")
        scores = self._compute_scores(X)
        if method == "zscore":
            threshold = self._hyperparams.get("z_threshold", 3.0)
            return (scores > threshold).astype(int)
        elif method == "iqr":
            # Flag if outlier in ANY feature
            return (scores > 0).astype(int)

    def fit(self, X, y=None):
        import time
        t0 = time.perf_counter()
        self._fit_impl(X, y)
        self._train_time_s = time.perf_counter() - t0
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._assert_fitted()
        import time
        t0 = time.perf_counter()
        result = self._predict_impl(X)
        self._last_infer_time_s = time.perf_counter() - t0
        return result

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        self._assert_fitted()
        return self._compute_scores(X)

    def score(self, X, y=None):
        preds = self.predict(X)
        scores = self.score_samples(X)
        if y is not None:
            return anomaly_metrics(y, preds, scores)
        n_anom = int(preds.sum())
        return {"n_anomalies_detected": n_anom, "anomaly_rate": float(n_anom / len(preds))}

    def predict_proba(self, X):
        return None

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "method", "type": "select", "default": "zscore",
             "options": ["zscore", "iqr"],
             "description": "'zscore' uses standard deviations, 'iqr' uses interquartile range."},
            {"name": "z_threshold", "type": "float", "default": 3.0, "min": 1.0, "max": 10.0,
             "description": "Z-Score threshold above which a point is flagged as anomalous."},
            {"name": "iqr_factor", "type": "float", "default": 1.5, "min": 0.5, "max": 5.0,
             "description": "IQR multiplier k for [Q1-k·IQR, Q3+k·IQR] bounds."},
        ]

    @classmethod
    def _math_intuition(cls):
        return (
            "Z-Score: z = (x - μ) / σ. Flag if max |z| > threshold. "
            "IQR: flag if x < Q1 - k·IQR or x > Q3 + k·IQR."
        )

    @classmethod
    def _strengths(cls):
        return [
            "Extremely simple and interpretable.",
            "Zero hyperparameters (z_threshold=3 is standard).",
            "Very fast O(n·d).",
            "Excellent baseline.",
        ]

    @classmethod
    def _weaknesses(cls):
        return [
            "Univariate — ignores feature interactions.",
            "Z-Score assumes Gaussian distribution.",
            "Sensitive to existing outliers in training data (use robust z-score for contaminated data).",
        ]

    @classmethod
    def _use_cases(cls):
        return ["Quick outlier check", "Data quality validation", "EDA baseline"]

    @classmethod
    def _assumptions(cls):
        return [
            "Z-Score: Gaussian marginal distributions.",
            "IQR: No strict distributional assumption.",
        ]
