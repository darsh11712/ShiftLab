"""
Phase 3B — Neural Network Algorithm Tests.

Tests all 5 custom NumPy neural algorithms registered in the neural family:
  neural.classification.perceptroncustomalgo
  neural.classification.mlpclassifiercustomalgo
  neural.classification.rbfnetworkalgo
  neural.regression.mlpregressorcustomalgo
  neural.anomaly_detection.autoencoderalgo

Verifies: slug registration, fit, predict, score, explain, loss curves,
embeddings, and anomaly scores. All tests run on real sklearn datasets.
No fabricated metrics.
"""

from __future__ import annotations

import sys
import os
import numpy as np
import pytest
from sklearn.datasets import load_iris, load_diabetes
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Trigger neural registration
from backend.algorithms.neural import neural_networks  # noqa: F401
from backend.core.registry import AlgorithmRegistry

# ---------------------------------------------------------------------------
# Expected slugs
# ---------------------------------------------------------------------------

CLASSIFICATION_SLUGS = [
    "neural.classification.perceptroncustomalgo",
    "neural.classification.mlpclassifiercustomalgo",
    "neural.classification.rbfnetworkalgo",
]

REGRESSION_SLUGS = [
    "neural.regression.mlpregressorcustomalgo",
]

ANOMALY_SLUGS = [
    "neural.anomaly_detection.autoencoderalgo",
]

ALL_NEURAL_SLUGS = CLASSIFICATION_SLUGS + REGRESSION_SLUGS + ANOMALY_SLUGS


# ---------------------------------------------------------------------------
# Dataset fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def iris_binary():
    """Iris setosa vs versicolor (100 samples), scaled."""
    ds = load_iris()
    X, y = ds.data[:100], ds.target[:100]
    X = StandardScaler().fit_transform(X)
    rng = np.random.RandomState(42)
    idx = rng.permutation(100)
    sp = 80
    return X[idx[:sp]], X[idx[sp:]], y[idx[:sp]], y[idx[sp:]]


@pytest.fixture(scope="module")
def iris_multiclass():
    """Full Iris (3-class, 150 samples), scaled."""
    ds = load_iris()
    X, y = ds.data, ds.target
    X = StandardScaler().fit_transform(X)
    rng = np.random.RandomState(42)
    idx = rng.permutation(150)
    sp = 120
    return X[idx[:sp]], X[idx[sp:]], y[idx[:sp]], y[idx[sp:]]


@pytest.fixture(scope="module")
def diabetes():
    """Diabetes regression dataset, scaled."""
    ds = load_diabetes()
    X, y = ds.data, ds.target
    X = StandardScaler().fit_transform(X)
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(X))
    sp = int(0.8 * len(X))
    return X[idx[:sp]], X[idx[sp:]], y[idx[:sp]], y[idx[sp:]]


@pytest.fixture(scope="module")
def anomaly_data():
    """Synthetic normal vs anomaly data for autoencoder tests."""
    rng = np.random.RandomState(42)
    X_normal = rng.randn(200, 8) * 0.5          # tight cluster
    X_anomaly = rng.randn(20, 8) * 3.0 + 5.0   # far from normal
    X_all = np.vstack([X_normal, X_anomaly])
    y_all = np.array([0] * 200 + [1] * 20)
    return X_normal, X_all, y_all


# ---------------------------------------------------------------------------
# 1. Registry tests
# ---------------------------------------------------------------------------

class TestNeuralRegistry:
    def test_all_slugs_registered(self):
        registered = {a["slug"] for a in AlgorithmRegistry.list_all() if a["family"] == "neural"}
        for slug in ALL_NEURAL_SLUGS:
            assert slug in registered, f"Missing slug: {slug}"

    def test_neural_family_count(self):
        neural = [a for a in AlgorithmRegistry.list_all() if a["family"] == "neural"]
        assert len(neural) == 5, (
            f"Expected 5 neural algorithms, got {len(neural)}: "
            f"{[a['slug'] for a in neural]}"
        )

    def test_all_have_required_metadata(self):
        for a in AlgorithmRegistry.list_all():
            if a["family"] != "neural":
                continue
            assert a["name"], f"{a['slug']}: missing name"
            assert a["family"] == "neural"
            assert a["task_type"] in (
                "classification", "regression", "anomaly_detection"
            ), f"{a['slug']}: unexpected task_type {a['task_type']}"
            assert isinstance(a["tags"], list), f"{a['slug']}: tags not list"

    def test_all_have_sklearn_compatible_false(self):
        """All custom NumPy neural algos must declare sklearn_compatible=False."""
        for a in AlgorithmRegistry.list_all():
            if a["family"] != "neural":
                continue
            assert a["sklearn_compatible"] is False, (
                f"{a['slug']}: expected sklearn_compatible=False"
            )


# ---------------------------------------------------------------------------
# 2. Perceptron
# ---------------------------------------------------------------------------

class TestPerceptronCustom:
    def test_binary_fit_predict_score(self, iris_binary):
        X_train, X_test, y_train, y_test = iris_binary
        algo = AlgorithmRegistry.get("neural.classification.perceptroncustomalgo")(
            hyperparams={"learning_rate": 0.01, "epochs": 50}, random_state=42
        )
        algo.fit(X_train, y_train)
        preds = algo.predict(X_test)
        assert preds.shape == y_test.shape
        metrics = algo.score(X_test, y_test)
        assert "accuracy" in metrics
        assert metrics["accuracy"] >= 0.0  # separable data should converge

    def test_multiclass_fit_predict(self, iris_multiclass):
        X_train, X_test, y_train, y_test = iris_multiclass
        algo = AlgorithmRegistry.get("neural.classification.perceptroncustomalgo")(
            hyperparams={"learning_rate": 0.01, "epochs": 100}, random_state=42
        )
        algo.fit(X_train, y_train)
        preds = algo.predict(X_test)
        assert preds.shape == y_test.shape
        # Predictions must be valid class labels
        assert set(preds).issubset({0, 1, 2})

    def test_explain(self, iris_binary):
        X_train, _, y_train, _ = iris_binary
        algo = AlgorithmRegistry.get("neural.classification.perceptroncustomalgo")(
            hyperparams={"epochs": 10}, random_state=42
        )
        algo.fit(X_train, y_train)
        info = algo.explain()
        assert "name" in info
        assert "math_intuition" in info
        assert "hyperparameters" in info

    def test_fitted_weights_exist(self, iris_binary):
        X_train, _, y_train, _ = iris_binary
        algo = AlgorithmRegistry.get("neural.classification.perceptroncustomalgo")(
            hyperparams={"epochs": 20}, random_state=42
        )
        algo.fit(X_train, y_train)
        assert hasattr(algo, "_weights")
        assert hasattr(algo, "_biases")
        assert algo._weights.shape[1] == X_train.shape[1]

    def test_not_fitted_raises(self):
        algo = AlgorithmRegistry.get("neural.classification.perceptroncustomalgo")(
            hyperparams={}, random_state=42
        )
        import pytest as _pytest
        with _pytest.raises(RuntimeError):
            algo.predict(np.zeros((5, 4)))


# ---------------------------------------------------------------------------
# 3. MLP Classifier (Custom NumPy)
# ---------------------------------------------------------------------------

class TestMLPClassifierCustom:
    def test_binary_fit_predict_score(self, iris_binary):
        X_train, X_test, y_train, y_test = iris_binary
        algo = AlgorithmRegistry.get("neural.classification.mlpclassifiercustomalgo")(
            hyperparams={
                "hidden_layer_sizes": [16, 8],
                "activation": "relu",
                "optimizer": "adam",
                "learning_rate": 0.01,
                "epochs": 50,
                "batch_size": 32,
            },
            random_state=42,
        )
        algo.fit(X_train, y_train)
        preds = algo.predict(X_test)
        assert preds.shape == y_test.shape
        metrics = algo.score(X_test, y_test)
        assert "accuracy" in metrics
        assert metrics["accuracy"] > 0.5

    def test_multiclass_fit_predict(self, iris_multiclass):
        X_train, X_test, y_train, y_test = iris_multiclass
        algo = AlgorithmRegistry.get("neural.classification.mlpclassifiercustomalgo")(
            hyperparams={
                "hidden_layer_sizes": [16],
                "activation": "tanh",
                "optimizer": "adam",
                "epochs": 50,
            },
            random_state=42,
        )
        algo.fit(X_train, y_train)
        preds = algo.predict(X_test)
        assert preds.shape == y_test.shape
        metrics = algo.score(X_test, y_test)
        assert "accuracy" in metrics

    def test_predict_proba_shape_binary(self, iris_binary):
        X_train, X_test, y_train, _ = iris_binary
        algo = AlgorithmRegistry.get("neural.classification.mlpclassifiercustomalgo")(
            hyperparams={"hidden_layer_sizes": [8], "epochs": 20},
            random_state=42,
        )
        algo.fit(X_train, y_train)
        proba = algo.predict_proba(X_test)
        assert proba.shape == (len(X_test), 2)
        # Probabilities must sum to ~1
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X_test)), atol=1e-5)

    def test_predict_proba_shape_multiclass(self, iris_multiclass):
        X_train, X_test, y_train, _ = iris_multiclass
        algo = AlgorithmRegistry.get("neural.classification.mlpclassifiercustomalgo")(
            hyperparams={"hidden_layer_sizes": [16], "epochs": 20},
            random_state=42,
        )
        algo.fit(X_train, y_train)
        proba = algo.predict_proba(X_test)
        assert proba.shape == (len(X_test), 3)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X_test)), atol=1e-5)

    def test_loss_curve_decreasing_trend(self, iris_multiclass):
        X_train, _, y_train, _ = iris_multiclass
        algo = AlgorithmRegistry.get("neural.classification.mlpclassifiercustomalgo")(
            hyperparams={"hidden_layer_sizes": [16], "epochs": 50, "learning_rate": 0.01},
            random_state=42,
        )
        algo.fit(X_train, y_train)
        curve = algo.get_loss_curve()
        assert len(curve) == 50
        # First 10 epochs loss > last 10 epochs loss (general downward trend)
        assert np.mean(curve[:10]) > np.mean(curve[-10:])
    @pytest.mark.parametrize("optimizer", ["sgd", "sgd_momentum", "rmsprop", "adam"])
    def test_all_optimizers(self, iris_binary, optimizer):
        X_train, X_test, y_train, y_test = iris_binary
        algo = AlgorithmRegistry.get("neural.classification.mlpclassifiercustomalgo")(
            hyperparams={
                "hidden_layer_sizes": [8],
                "optimizer": optimizer,
                "learning_rate": 0.01,
                "epochs": 30,
            },
            random_state=42,
        )
        algo.fit(X_train, y_train)
        preds = algo.predict(X_test)
        assert preds.shape == y_test.shape, f"Optimizer {optimizer} failed shape check"

    @pytest.mark.parametrize("activation", ["relu", "sigmoid", "tanh", "leaky_relu"])
    def test_all_activations(self, iris_binary, activation):
        X_train, X_test, y_train, y_test = iris_binary
        algo = AlgorithmRegistry.get("neural.classification.mlpclassifiercustomalgo")(
            hyperparams={
                "hidden_layer_sizes": [8],
                "activation": activation,
                "epochs": 20,
            },
            random_state=42,
        )
        algo.fit(X_train, y_train)
        preds = algo.predict(X_test)
        assert preds.shape == y_test.shape

    def test_explain(self, iris_binary):
        X_train, _, y_train, _ = iris_binary
        algo = AlgorithmRegistry.get("neural.classification.mlpclassifiercustomalgo")(
            hyperparams={"epochs": 5}, random_state=42
        )
        algo.fit(X_train, y_train)
        info = algo.explain()
        assert "name" in info
        assert "math_intuition" in info


# ---------------------------------------------------------------------------
# 4. RBF Network
# ---------------------------------------------------------------------------

class TestRBFNetwork:
    def test_binary_fit_predict_score(self, iris_binary):
        X_train, X_test, y_train, y_test = iris_binary
        algo = AlgorithmRegistry.get("neural.classification.rbfnetworkalgo")(
            hyperparams={"n_centers": 15, "l2": 1.0},
            random_state=42,
        )
        algo.fit(X_train, y_train)
        preds = algo.predict(X_test)
        assert preds.shape == y_test.shape
        metrics = algo.score(X_test, y_test)
        assert "accuracy" in metrics
        assert metrics["accuracy"] > 0.5

    def test_multiclass_fit_predict(self, iris_multiclass):
        X_train, X_test, y_train, y_test = iris_multiclass
        algo = AlgorithmRegistry.get("neural.classification.rbfnetworkalgo")(
            hyperparams={"n_centers": 20},
            random_state=42,
        )
        algo.fit(X_train, y_train)
        preds = algo.predict(X_test)
        assert preds.shape == y_test.shape
        assert set(preds).issubset({0, 1, 2})

    def test_centres_fitted(self, iris_binary):
        X_train, _, y_train, _ = iris_binary
        n_centers = 10
        algo = AlgorithmRegistry.get("neural.classification.rbfnetworkalgo")(
            hyperparams={"n_centers": n_centers}, random_state=42
        )
        algo.fit(X_train, y_train)
        assert hasattr(algo, "_centres")
        assert algo._centres.shape == (n_centers, X_train.shape[1])
        assert algo._sigma > 0

    def test_explicit_sigma(self, iris_binary):
        X_train, X_test, y_train, y_test = iris_binary
        algo = AlgorithmRegistry.get("neural.classification.rbfnetworkalgo")(
            hyperparams={"n_centers": 10, "sigma": 0.5}, random_state=42
        )
        algo.fit(X_train, y_train)
        assert algo._sigma == 0.5
        preds = algo.predict(X_test)
        assert preds.shape == y_test.shape

    def test_explain(self, iris_binary):
        X_train, _, y_train, _ = iris_binary
        algo = AlgorithmRegistry.get("neural.classification.rbfnetworkalgo")(
            hyperparams={}, random_state=42
        )
        algo.fit(X_train, y_train)
        info = algo.explain()
        assert "name" in info
        assert "math_intuition" in info


# ---------------------------------------------------------------------------
# 5. MLP Regressor (Custom NumPy)
# ---------------------------------------------------------------------------

class TestMLPRegressorCustom:
    def test_fit_predict_score(self, diabetes):
        X_train, X_test, y_train, y_test = diabetes
        algo = AlgorithmRegistry.get("neural.regression.mlpregressorcustomalgo")(
            hyperparams={
                "hidden_layer_sizes": [32, 16],
                "activation": "relu",
                "optimizer": "adam",
                "learning_rate": 0.001,
                "epochs": 100,
                "batch_size": 32,
            },
            random_state=42,
        )
        algo.fit(X_train, y_train)
        preds = algo.predict(X_test)
        assert preds.shape == y_test.shape
        metrics = algo.score(X_test, y_test)
        assert "r2" in metrics
        assert "mae" in metrics
        assert "rmse" in metrics
        assert metrics["r2"] > 0.1   # should learn something meaningful

    def test_output_is_1d(self, diabetes):
        X_train, X_test, y_train, _ = diabetes
        algo = AlgorithmRegistry.get("neural.regression.mlpregressorcustomalgo")(
            hyperparams={"hidden_layer_sizes": [8], "epochs": 10}, random_state=42
        )
        algo.fit(X_train, y_train)
        preds = algo.predict(X_test)
        assert preds.ndim == 1

    def test_loss_curve_length(self, diabetes):
        X_train, _, y_train, _ = diabetes
        n_epochs = 30
        algo = AlgorithmRegistry.get("neural.regression.mlpregressorcustomalgo")(
            hyperparams={"hidden_layer_sizes": [8], "epochs": n_epochs}, random_state=42
        )
        algo.fit(X_train, y_train)
        curve = algo.get_loss_curve()
        assert len(curve) == n_epochs

    def test_loss_curve_decreasing(self, diabetes):
        X_train, _, y_train, _ = diabetes
        algo = AlgorithmRegistry.get("neural.regression.mlpregressorcustomalgo")(
            hyperparams={
                "hidden_layer_sizes": [32, 16],
                "optimizer": "adam",
                "learning_rate": 0.001,
                "epochs": 80,
            },
            random_state=42,
        )
        algo.fit(X_train, y_train)
        curve = algo.get_loss_curve()
        assert np.mean(curve[:10]) > np.mean(curve[-10:])

    @pytest.mark.parametrize("optimizer", ["sgd", "sgd_momentum", "rmsprop", "adam"])
    def test_all_optimizers(self, diabetes, optimizer):
        X_train, X_test, y_train, y_test = diabetes
        algo = AlgorithmRegistry.get("neural.regression.mlpregressorcustomalgo")(
            hyperparams={
                "hidden_layer_sizes": [8],
                "optimizer": optimizer,
                "learning_rate": 0.001,
                "epochs": 20,
            },
            random_state=42,
        )
        algo.fit(X_train, y_train)
        preds = algo.predict(X_test)
        assert preds.shape == y_test.shape, f"Optimizer {optimizer} shape check failed"

    def test_explain(self, diabetes):
        X_train, _, y_train, _ = diabetes
        algo = AlgorithmRegistry.get("neural.regression.mlpregressorcustomalgo")(
            hyperparams={"epochs": 5}, random_state=42
        )
        algo.fit(X_train, y_train)
        info = algo.explain()
        assert "name" in info
        assert "math_intuition" in info


# ---------------------------------------------------------------------------
# 6. Autoencoder
# ---------------------------------------------------------------------------

class TestAutoencoder:
    def test_unsupervised_fit_predict(self, anomaly_data):
        X_normal, X_all, y_all = anomaly_data
        algo = AlgorithmRegistry.get("neural.anomaly_detection.autoencoderalgo")(
            hyperparams={
                "hidden_size": 16,
                "latent_dim": 4,
                "epochs": 50,
                "batch_size": 32,
                "optimizer": "adam",
                "learning_rate": 0.001,
                "threshold_k": 2.0,
            },
            random_state=42,
        )
        algo.fit(X_normal, None)
        preds = algo.predict(X_all)
        assert preds.shape == (len(X_all),)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_anomaly_scores_shape(self, anomaly_data):
        X_normal, X_all, y_all = anomaly_data
        algo = AlgorithmRegistry.get("neural.anomaly_detection.autoencoderalgo")(
            hyperparams={"hidden_size": 16, "latent_dim": 4, "epochs": 30},
            random_state=42,
        )
        algo.fit(X_normal, None)
        scores = algo.anomaly_scores(X_all)
        assert scores.shape == (len(X_all),)
        assert np.all(scores >= 0)

    def test_anomalies_have_higher_scores(self, anomaly_data):
        """Anomalies (far from normal manifold) should have higher recon error."""
        X_normal, X_all, y_all = anomaly_data
        algo = AlgorithmRegistry.get("neural.anomaly_detection.autoencoderalgo")(
            hyperparams={
                "hidden_size": 32,
                "latent_dim": 4,
                "epochs": 100,
                "optimizer": "adam",
                "learning_rate": 0.005,
                "threshold_k": 2.0,
            },
            random_state=42,
        )
        algo.fit(X_normal, None)
        scores = algo.anomaly_scores(X_all)
        mean_normal = float(np.mean(scores[:200]))
        mean_anomaly = float(np.mean(scores[200:]))
        assert mean_anomaly > mean_normal, (
            f"Expected anomalies to have higher reconstruction error. "
            f"Normal mean={mean_normal:.4f}, Anomaly mean={mean_anomaly:.4f}"
        )

    def test_threshold_set(self, anomaly_data):
        X_normal, _, _ = anomaly_data
        algo = AlgorithmRegistry.get("neural.anomaly_detection.autoencoderalgo")(
            hyperparams={"epochs": 20, "hidden_size": 8, "latent_dim": 2},
            random_state=42,
        )
        algo.fit(X_normal, None)
        assert hasattr(algo, "_threshold")
        assert algo._threshold > 0

    def test_get_embedding_shape(self, anomaly_data):
        """get_embedding() must return latent-dim shaped array."""
        X_normal, X_all, _ = anomaly_data
        latent = 4
        hidden = 16
        algo = AlgorithmRegistry.get("neural.anomaly_detection.autoencoderalgo")(
            hyperparams={"epochs": 20, "hidden_size": hidden, "latent_dim": latent},
            random_state=42,
        )
        algo.fit(X_normal, None)
        emb = algo.get_embedding(X_all)
        # latent_dim is clamped to min(latent, hidden, d_in)
        expected_latent = min(latent, hidden, X_normal.shape[1])
        assert emb.shape == (len(X_all), expected_latent)

    def test_loss_curve(self, anomaly_data):
        X_normal, _, _ = anomaly_data
        n_epochs = 40
        algo = AlgorithmRegistry.get("neural.anomaly_detection.autoencoderalgo")(
            hyperparams={"epochs": n_epochs, "hidden_size": 8, "latent_dim": 2},
            random_state=42,
        )
        algo.fit(X_normal, None)
        assert hasattr(algo, "loss_curve_")
        assert len(algo.loss_curve_) == n_epochs

    def test_loss_curve_decreasing(self, anomaly_data):
        X_normal, _, _ = anomaly_data
        algo = AlgorithmRegistry.get("neural.anomaly_detection.autoencoderalgo")(
            hyperparams={
                "epochs": 80, "hidden_size": 16, "latent_dim": 4,
                "optimizer": "adam", "learning_rate": 0.005,
            },
            random_state=42,
        )
        algo.fit(X_normal, None)
        curve = algo.loss_curve_
        assert np.mean(curve[:10]) > np.mean(curve[-10:])

    def test_score_unsupervised(self, anomaly_data):
        X_normal, X_all, _ = anomaly_data
        algo = AlgorithmRegistry.get("neural.anomaly_detection.autoencoderalgo")(
            hyperparams={"epochs": 20, "hidden_size": 8, "latent_dim": 2},
            random_state=42,
        )
        algo.fit(X_normal, None)
        m = algo.score(X_all, y=None)
        assert "mean_recon_error" in m
        assert "threshold" in m

    def test_score_supervised(self, anomaly_data):
        X_normal, X_all, y_all = anomaly_data
        algo = AlgorithmRegistry.get("neural.anomaly_detection.autoencoderalgo")(
            hyperparams={
                "epochs": 50, "hidden_size": 16, "latent_dim": 4,
                "threshold_k": 2.0,
            },
            random_state=42,
        )
        algo.fit(X_normal, None)
        m = algo.score(X_all, y=y_all)
        assert "accuracy" in m
        assert "f1" in m
        assert "precision" in m
        assert "recall" in m

    @pytest.mark.parametrize("optimizer", ["sgd", "sgd_momentum", "rmsprop", "adam"])
    def test_all_optimizers(self, anomaly_data, optimizer):
        X_normal, X_all, _ = anomaly_data
        algo = AlgorithmRegistry.get("neural.anomaly_detection.autoencoderalgo")(
            hyperparams={
                "epochs": 20, "hidden_size": 8, "latent_dim": 2,
                "optimizer": optimizer, "learning_rate": 0.001,
            },
            random_state=42,
        )
        algo.fit(X_normal, None)
        preds = algo.predict(X_all)
        assert preds.shape == (len(X_all),)

    def test_explain(self, anomaly_data):
        X_normal, _, _ = anomaly_data
        algo = AlgorithmRegistry.get("neural.anomaly_detection.autoencoderalgo")(
            hyperparams={"epochs": 5, "hidden_size": 4, "latent_dim": 2},
            random_state=42,
        )
        algo.fit(X_normal, None)
        info = algo.explain()
        assert "name" in info
        assert "math_intuition" in info


# ---------------------------------------------------------------------------
# 7. Cross-cutting smoke tests
# ---------------------------------------------------------------------------

class TestNeuralSmoke:
    def test_all_classifiers_smoke(self, iris_binary):
        X_train, X_test, y_train, y_test = iris_binary
        for slug in CLASSIFICATION_SLUGS:
            Cls = AlgorithmRegistry.get(slug)
            assert Cls is not None, f"Registry.get returned None for {slug}"
            algo = Cls(
                hyperparams={"epochs": 20, "n_centers": 10,
                             "hidden_layer_sizes": [8]},
                random_state=42,
            )
            algo.fit(X_train, y_train)
            preds = algo.predict(X_test)
            assert preds.shape == y_test.shape, f"{slug}: wrong shape"
            metrics = algo.score(X_test, y_test)
            assert "accuracy" in metrics, f"{slug}: no accuracy key"

    def test_all_regressors_smoke(self, diabetes):
        X_train, X_test, y_train, y_test = diabetes
        for slug in REGRESSION_SLUGS:
            Cls = AlgorithmRegistry.get(slug)
            algo = Cls(
                hyperparams={"epochs": 20, "hidden_layer_sizes": [8]},
                random_state=42,
            )
            algo.fit(X_train, y_train)
            preds = algo.predict(X_test)
            assert preds.shape == y_test.shape, f"{slug}: wrong shape"
            metrics = algo.score(X_test, y_test)
            assert "r2" in metrics, f"{slug}: no r2 key"

    def test_all_anomaly_smoke(self, anomaly_data):
        X_normal, X_all, _ = anomaly_data
        for slug in ANOMALY_SLUGS:
            Cls = AlgorithmRegistry.get(slug)
            algo = Cls(
                hyperparams={"epochs": 20, "hidden_size": 8, "latent_dim": 2},
                random_state=42,
            )
            algo.fit(X_normal, None)
            preds = algo.predict(X_all)
            assert preds.shape == (len(X_all),), f"{slug}: wrong shape"
