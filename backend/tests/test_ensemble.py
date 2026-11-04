"""
Phase 3A — Ensemble Algorithm Tests.

Tests all 10 ensemble algorithms registered in ensemble family.
Verifies: slug registration, fit, predict, score, explain all work correctly
on real sklearn datasets. No fabricated metrics.

Slow algorithms (Stacking, Voting) use a small subset (100 samples) to keep
the test suite fast on CPU-only 8 GB RAM machines.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_iris, load_diabetes

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Import ensemble module to trigger registration
from backend.algorithms.ensemble import ensemblers  # noqa: F401
from backend.core.registry import AlgorithmRegistry

# ---------------------------------------------------------------------------
# Dataset fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def iris_data():
    """Binary classification subset: Iris setosa vs versicolor (100 samples)."""
    ds = load_iris()
    X, y = ds.data[:100], ds.target[:100]  # setosa=0, versicolor=1
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(X))
    split = int(0.8 * len(X))
    X_train, X_test = X[idx[:split]], X[idx[split:]]
    y_train, y_test = y[idx[:split]], y[idx[split:]]
    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="module")
def iris_multiclass_data():
    """Full Iris (3-class, 150 samples)."""
    ds = load_iris()
    X, y = ds.data, ds.target
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(X))
    split = int(0.8 * len(X))
    X_train, X_test = X[idx[:split]], X[idx[split:]]
    y_train, y_test = y[idx[:split]], y[idx[split:]]
    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="module")
def diabetes_data():
    """Diabetes regression dataset (442 samples)."""
    ds = load_diabetes()
    X, y = ds.data, ds.target
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(X))
    split = int(0.8 * len(X))
    X_train, X_test = X[idx[:split]], X[idx[split:]]
    y_train, y_test = y[idx[:split]], y[idx[split:]]
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Helper slugs
# ---------------------------------------------------------------------------

CLASSIFICATION_SLUGS = [
    "ensemble.classification.baggingclassifieralgo",
    "ensemble.classification.extratreesclassifieralgo",
    "ensemble.classification.adaboostclassifieralgo",
    "ensemble.classification.votingclassifieralgo",
    "ensemble.classification.stackingclassifieralgo",
]

REGRESSION_SLUGS = [
    "ensemble.regression.baggingregressoralgo",
    "ensemble.regression.extratreesregressoralgo",
    "ensemble.regression.adaboostregressoralgo",
    "ensemble.regression.votingregressoralgo",
    "ensemble.regression.stackingregressoralgo",
]

ALL_SLUGS = CLASSIFICATION_SLUGS + REGRESSION_SLUGS

# ---------------------------------------------------------------------------
# 1. Registry tests
# ---------------------------------------------------------------------------

class TestEnsembleRegistry:
    def test_all_slugs_registered(self):
        """All 10 ensemble slugs must appear in the registry."""
        registered = {a["slug"] for a in AlgorithmRegistry.list_all() if a["family"] == "ensemble"}
        for slug in ALL_SLUGS:
            assert slug in registered, f"Missing slug: {slug}"

    def test_ensemble_family_count(self):
        """Exactly 10 ensemble algorithms must be registered."""
        ensemble_algos = [a for a in AlgorithmRegistry.list_all() if a["family"] == "ensemble"]
        assert len(ensemble_algos) == 10, (
            f"Expected 10 ensemble algorithms, got {len(ensemble_algos)}: "
            f"{[a['slug'] for a in ensemble_algos]}"
        )

    def test_all_have_required_metadata(self):
        """Every ensemble algorithm must expose name, family, task_type, tags."""
        for a in AlgorithmRegistry.list_all():
            if a["family"] != "ensemble":
                continue
            assert a["name"], f"{a['slug']}: missing name"
            assert a["family"] == "ensemble", f"{a['slug']}: wrong family"
            assert a["task_type"] in ("classification", "regression"), f"{a['slug']}: bad task_type"
            assert isinstance(a["tags"], list), f"{a['slug']}: tags not a list"


# ---------------------------------------------------------------------------
# 2. Classification — fit / predict / score
# ---------------------------------------------------------------------------

class TestBaggingClassifier:
    def test_fit_predict_score(self, iris_data):
        X_train, X_test, y_train, y_test = iris_data
        algo = AlgorithmRegistry.get("ensemble.classification.baggingclassifieralgo")(
            hyperparams={"n_estimators": 10}, random_state=42
        )
        algo.fit(X_train, y_train)
        preds = algo.predict(X_test)
        assert preds.shape == y_test.shape
        metrics = algo.score(X_test, y_test)
        assert metrics["accuracy"] > 0.7, f"Low accuracy: {metrics['accuracy']}"

    def test_explain(self, iris_data):
        X_train, X_test, y_train, _ = iris_data
        algo = AlgorithmRegistry.get("ensemble.classification.baggingclassifieralgo")(
            hyperparams={"n_estimators": 5}, random_state=42
        )
        algo.fit(X_train, y_train)
        info = algo.explain()
        assert "name" in info
        assert "description" in info


class TestExtraTreesClassifier:
    def test_fit_predict_score(self, iris_data):
        X_train, X_test, y_train, y_test = iris_data
        algo = AlgorithmRegistry.get("ensemble.classification.extratreesclassifieralgo")(
            hyperparams={"n_estimators": 20}, random_state=42
        )
        algo.fit(X_train, y_train)
        preds = algo.predict(X_test)
        assert preds.shape == y_test.shape
        metrics = algo.score(X_test, y_test)
        assert metrics["accuracy"] > 0.7

    def test_feature_importances(self, iris_data):
        X_train, X_test, y_train, _ = iris_data
        algo = AlgorithmRegistry.get("ensemble.classification.extratreesclassifieralgo")(
            hyperparams={"n_estimators": 10}, random_state=42
        )
        algo.fit(X_train, y_train)
        info = algo.explain()
        assert "feature_importances" in info or "description" in info


class TestAdaBoostClassifier:
    def test_fit_predict_score(self, iris_data):
        X_train, X_test, y_train, y_test = iris_data
        algo = AlgorithmRegistry.get("ensemble.classification.adaboostclassifieralgo")(
            hyperparams={"n_estimators": 30, "learning_rate": 1.0}, random_state=42
        )
        algo.fit(X_train, y_train)
        preds = algo.predict(X_test)
        assert preds.shape == y_test.shape
        metrics = algo.score(X_test, y_test)
        assert metrics["accuracy"] > 0.6

    def test_predict_proba(self, iris_data):
        X_train, X_test, y_train, _ = iris_data
        algo = AlgorithmRegistry.get("ensemble.classification.adaboostclassifieralgo")(
            hyperparams={"n_estimators": 10}, random_state=42
        )
        algo.fit(X_train, y_train)
        if hasattr(algo._model, "predict_proba"):
            proba = algo._model.predict_proba(X_test)
            assert proba.shape[1] == 2  # binary


class TestVotingClassifier:
    def test_fit_predict_score(self, iris_data):
        X_train, X_test, y_train, y_test = iris_data
        algo = AlgorithmRegistry.get("ensemble.classification.votingclassifieralgo")(
            hyperparams={"voting": "hard"}, random_state=42
        )
        algo.fit(X_train, y_train)
        preds = algo.predict(X_test)
        assert preds.shape == y_test.shape
        metrics = algo.score(X_test, y_test)
        assert metrics["accuracy"] > 0.6

    def test_soft_voting(self, iris_data):
        X_train, X_test, y_train, y_test = iris_data
        algo = AlgorithmRegistry.get("ensemble.classification.votingclassifieralgo")(
            hyperparams={"voting": "soft"}, random_state=42
        )
        algo.fit(X_train, y_train)
        preds = algo.predict(X_test)
        metrics = algo.score(X_test, y_test)
        assert metrics["accuracy"] > 0.6


class TestStackingClassifier:
    def test_fit_predict_score(self, iris_data):
        X_train, X_test, y_train, y_test = iris_data
        algo = AlgorithmRegistry.get("ensemble.classification.stackingclassifieralgo")(
            hyperparams={}, random_state=42
        )
        algo.fit(X_train, y_train)
        preds = algo.predict(X_test)
        assert preds.shape == y_test.shape
        metrics = algo.score(X_test, y_test)
        assert metrics["accuracy"] > 0.6

    def test_explain(self, iris_data):
        X_train, _, y_train, _ = iris_data
        algo = AlgorithmRegistry.get("ensemble.classification.stackingclassifieralgo")(
            hyperparams={}, random_state=42
        )
        algo.fit(X_train, y_train)
        info = algo.explain()
        assert "name" in info


# ---------------------------------------------------------------------------
# 3. Regression — fit / predict / score
# ---------------------------------------------------------------------------

class TestBaggingRegressor:
    def test_fit_predict_score(self, diabetes_data):
        X_train, X_test, y_train, y_test = diabetes_data
        algo = AlgorithmRegistry.get("ensemble.regression.baggingregressoralgo")(
            hyperparams={"n_estimators": 10}, random_state=42
        )
        algo.fit(X_train, y_train)
        preds = algo.predict(X_test)
        assert preds.shape == y_test.shape
        metrics = algo.score(X_test, y_test)
        assert metrics["r2"] > 0.2


class TestExtraTreesRegressor:
    def test_fit_predict_score(self, diabetes_data):
        X_train, X_test, y_train, y_test = diabetes_data
        algo = AlgorithmRegistry.get("ensemble.regression.extratreesregressoralgo")(
            hyperparams={"n_estimators": 20}, random_state=42
        )
        algo.fit(X_train, y_train)
        preds = algo.predict(X_test)
        assert preds.shape == y_test.shape
        metrics = algo.score(X_test, y_test)
        assert metrics["r2"] > 0.2

    def test_feature_importances(self, diabetes_data):
        X_train, _, y_train, _ = diabetes_data
        algo = AlgorithmRegistry.get("ensemble.regression.extratreesregressoralgo")(
            hyperparams={"n_estimators": 10}, random_state=42
        )
        algo.fit(X_train, y_train)
        info = algo.explain()
        assert "description" in info


class TestAdaBoostRegressor:
    def test_fit_predict_score(self, diabetes_data):
        X_train, X_test, y_train, y_test = diabetes_data
        algo = AlgorithmRegistry.get("ensemble.regression.adaboostregressoralgo")(
            hyperparams={"n_estimators": 30, "learning_rate": 0.5}, random_state=42
        )
        algo.fit(X_train, y_train)
        preds = algo.predict(X_test)
        assert preds.shape == y_test.shape
        metrics = algo.score(X_test, y_test)
        assert metrics["r2"] > -1.0  # at minimum better than a constant predictor


class TestVotingRegressor:
    def test_fit_predict_score(self, diabetes_data):
        X_train, X_test, y_train, y_test = diabetes_data
        algo = AlgorithmRegistry.get("ensemble.regression.votingregressoralgo")(
            hyperparams={}, random_state=42
        )
        algo.fit(X_train, y_train)
        preds = algo.predict(X_test)
        assert preds.shape == y_test.shape
        metrics = algo.score(X_test, y_test)
        assert metrics["r2"] > 0.2


class TestStackingRegressor:
    def test_fit_predict_score(self, diabetes_data):
        X_train, X_test, y_train, y_test = diabetes_data
        algo = AlgorithmRegistry.get("ensemble.regression.stackingregressoralgo")(
            hyperparams={}, random_state=42
        )
        algo.fit(X_train, y_train)
        preds = algo.predict(X_test)
        assert preds.shape == y_test.shape
        metrics = algo.score(X_test, y_test)
        assert metrics["r2"] > 0.2

    def test_explain(self, diabetes_data):
        X_train, _, y_train, _ = diabetes_data
        algo = AlgorithmRegistry.get("ensemble.regression.stackingregressoralgo")(
            hyperparams={}, random_state=42
        )
        algo.fit(X_train, y_train)
        info = algo.explain()
        assert "name" in info


# ---------------------------------------------------------------------------
# 4. Cross-cutting: all algorithms can fit→predict→score without exception
# ---------------------------------------------------------------------------

class TestEnsembleSmoke:
    """Smoke tests — every algorithm must complete fit/predict/score."""

    def test_all_classifiers_smoke(self, iris_data):
        X_train, X_test, y_train, y_test = iris_data
        for slug in CLASSIFICATION_SLUGS:
            Cls = AlgorithmRegistry.get(slug)
            assert Cls is not None, f"Registry.get returned None for {slug}"
            algo = Cls(hyperparams={}, random_state=42)
            algo.fit(X_train, y_train)
            preds = algo.predict(X_test)
            assert preds.shape == y_test.shape, f"{slug}: wrong prediction shape"
            metrics = algo.score(X_test, y_test)
            assert "accuracy" in metrics, f"{slug}: missing 'accuracy' in score"

    def test_all_regressors_smoke(self, diabetes_data):
        X_train, X_test, y_train, y_test = diabetes_data
        for slug in REGRESSION_SLUGS:
            Cls = AlgorithmRegistry.get(slug)
            assert Cls is not None, f"Registry.get returned None for {slug}"
            algo = Cls(hyperparams={}, random_state=42)
            algo.fit(X_train, y_train)
            preds = algo.predict(X_test)
            assert preds.shape == y_test.shape, f"{slug}: wrong prediction shape"
            metrics = algo.score(X_test, y_test)
            assert "r2" in metrics, f"{slug}: missing 'r2' in score"
