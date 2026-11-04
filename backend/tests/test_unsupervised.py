"""
Phase 2 Test Suite — Unsupervised Clustering, Dimensionality Reduction, Anomaly Detection.

Rules:
- Every algorithm must pass fit → predict → score on real data.
- Metric values must be finite floats.
- No fabricated numbers.
"""
import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.core.registry import AlgorithmRegistry
from backend.core.dataset_loader import load_dataset, split_dataset
from backend.core.preprocessor import PreprocessConfig, build_preprocessor, fit_transform


# ──────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def cluster_data():
    """Blobs dataset — 4 clusters, 2 features, 300 samples."""
    ds = load_dataset("blobs")
    cfg = PreprocessConfig(scaler="standard")
    pp = build_preprocessor(cfg)
    X_scaled = pp.fit_transform(ds.X)
    return X_scaled, ds.y


@pytest.fixture(scope="session")
def iris_data():
    """Iris — for dim-reduction and supervised dim-reduction (LDA)."""
    ds = load_dataset("iris")
    cfg = PreprocessConfig(scaler="standard")
    pp = build_preprocessor(cfg)
    X_scaled = pp.fit_transform(ds.X)
    return X_scaled, ds.y


@pytest.fixture(scope="session")
def nonneg_data():
    """MinMax-scaled iris — required for NMF."""
    ds = load_dataset("iris")
    cfg = PreprocessConfig(scaler="minmax")
    pp = build_preprocessor(cfg)
    X_scaled = pp.fit_transform(ds.X)
    return X_scaled, ds.y


@pytest.fixture(scope="session")
def anomaly_data():
    """Breast cancer: normal=1(benign), anomaly=0(malignant) — for supervised anomaly metrics."""
    ds = load_dataset("breast_cancer")
    cfg = PreprocessConfig(scaler="standard")
    pp = build_preprocessor(cfg)
    split = split_dataset(ds, test_size=0.3, random_state=42)
    X_tr = pp.fit_transform(split.X_train)
    X_te = pp.transform(split.X_test)
    # Convert: malignant(0) = anomaly(1), benign(1) = normal(0)
    y_te = (split.y_test == 0).astype(int)
    return X_tr, X_te, y_te


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────

def _assert_finite_metrics(metrics: dict, algo_slug: str):
    assert isinstance(metrics, dict), f"{algo_slug}: score() must return dict"
    assert len(metrics) > 0, f"{algo_slug}: score() returned empty dict"
    for k, v in metrics.items():
        if isinstance(v, list):
            for item in v:
                assert np.isfinite(float(item)), f"{algo_slug}: metric '{k}' contains non-finite value"
        elif isinstance(v, (int, float)):
            assert np.isfinite(float(v)), f"{algo_slug}: metric '{k}' = {v} is not finite"


# ──────────────────────────────────────────────────────────
# CLUSTERING TESTS (U-01 → U-11)
# ──────────────────────────────────────────────────────────

# Algorithms that auto-determine k (no n_clusters param)
NO_K_SLUGS = {
    "unsupervised.clustering.dbscanalgo",
    "unsupervised.clustering.hdbscanalgo",
    "unsupervised.clustering.meanshiftalgo",
    "unsupervised.clustering.opticsalgo",
    "unsupervised.clustering.affinitypropagationalgo",
}

CLUSTERING_SLUGS = [
    "unsupervised.clustering.kmeansalgo",
    "unsupervised.clustering.minibatchkmeansalgo",
    "unsupervised.clustering.dbscanalgo",
    "unsupervised.clustering.hdbscanalgo",
    "unsupervised.clustering.agglomerativeclusteringalgo",
    "unsupervised.clustering.gaussianmixturealgo",
    "unsupervised.clustering.meanshiftalgo",
    "unsupervised.clustering.opticsalgo",
    "unsupervised.clustering.spectralclusteringalgo",
    "unsupervised.clustering.birchalgo",
    "unsupervised.clustering.affinitypropagationalgo",
]


@pytest.mark.parametrize("slug", CLUSTERING_SLUGS)
def test_clustering_algorithm(slug, cluster_data):
    AlgorithmRegistry.load_all()
    X, y_true = cluster_data

    # Limit data size for slow algorithms
    if slug in {"unsupervised.clustering.spectralclusteringalgo",
                "unsupervised.clustering.affinitypropagationalgo"}:
        X, y_true = X[:80], y_true[:80]
    elif slug in {"unsupervised.clustering.meanshiftalgo"}:
        X, y_true = X[:150], y_true[:150]

    # Build hyperparams
    hyperparams = {}
    if slug not in NO_K_SLUGS:
        if "gaussian" in slug:
            hyperparams["n_components"] = 4
        elif "birch" in slug:
            hyperparams["n_clusters"] = 4
            hyperparams["threshold"] = 0.5
        else:
            hyperparams["n_clusters"] = 4

    algo = AlgorithmRegistry.instantiate(slug, random_state=42, **hyperparams)
    algo.fit(X)

    # Get labels from training
    if hasattr(algo, "_train_labels"):
        labels = algo._train_labels
    elif hasattr(algo, "_labels"):
        labels = algo._labels
    else:
        labels = algo.predict(X)

    assert labels is not None, f"{slug}: no labels produced"
    assert len(labels) == len(X), f"{slug}: label count mismatch"

    n_unique = len(np.unique(labels[labels >= 0]))
    assert n_unique >= 1, f"{slug}: produced 0 clusters"

    metrics = algo.score(X, y_true)
    _assert_finite_metrics(metrics, slug)

    print(f"✅ {slug}: {n_unique} clusters | silhouette={metrics.get('silhouette', 'N/A')} | "
          f"train={algo.training_time():.3f}s")


# ──────────────────────────────────────────────────────────
# DIMENSIONALITY REDUCTION TESTS (D-01 → D-10)
# ──────────────────────────────────────────────────────────

# Algorithms that require non-negative input
NONNEG_DIM_SLUGS = {"unsupervised.dimensionality_reduction.nmfalgo"}

# Supervised reducers that need y
SUPERVISED_DIM_SLUGS = {"unsupervised.dimensionality_reduction.ldareduceralgo"}

# Algorithms with no out-of-sample transform (train embedding only)
TRANSDUCTIVE_SLUGS = {"unsupervised.dimensionality_reduction.tsnealgo"}

DIM_REDUCTION_SLUGS_ACTUAL = [
    "unsupervised.dimensionality_reduction.pcaalgo",
    "unsupervised.dimensionality_reduction.kernelpcaalgo",
    "unsupervised.dimensionality_reduction.truncatedsvdalgo",
    "unsupervised.dimensionality_reduction.fasticaalgo",
    "unsupervised.dimensionality_reduction.nmfalgo",
    "unsupervised.dimensionality_reduction.tsnealgo",
    "unsupervised.dimensionality_reduction.umapalgo",
    "unsupervised.dimensionality_reduction.isomapalgo",
    "unsupervised.dimensionality_reduction.llealgo",
    "unsupervised.dimensionality_reduction.ldareduceralgo",
]


@pytest.mark.parametrize("slug", DIM_REDUCTION_SLUGS_ACTUAL)
def test_dim_reduction_algorithm(slug, iris_data, nonneg_data):
    AlgorithmRegistry.load_all()

    needs_nonneg = slug in NONNEG_DIM_SLUGS
    X, y = nonneg_data if needs_nonneg else iris_data
    needs_supervised = slug in SUPERVISED_DIM_SLUGS
    is_transductive = slug in TRANSDUCTIVE_SLUGS

    # Limit size for slow algorithms
    if slug in {"unsupervised.dimensionality_reduction.tsnealgo",
                "unsupervised.dimensionality_reduction.kernelpcaalgo"}:
        X, y = X[:100], y[:100]

    hyperparams = {"n_components": 2}
    if "tsne" in slug:
        hyperparams["perplexity"] = 15.0
        hyperparams["max_iter"] = 300

    algo = AlgorithmRegistry.instantiate(slug, random_state=42, **hyperparams)

    if needs_supervised:
        algo.fit(X, y)
    else:
        algo.fit(X)

    if is_transductive:
        embedding = algo.get_embedding()
    else:
        embedding = algo.predict(X)

    assert embedding is not None, f"{slug}: predict() returned None"
    assert embedding.shape[0] == X.shape[0], f"{slug}: embedding row count mismatch"
    assert embedding.shape[1] == 2, f"{slug}: expected 2 components, got {embedding.shape[1]}"
    assert not np.any(np.isnan(embedding)), f"{slug}: embedding contains NaN"

    metrics = algo.score(X, y if needs_supervised else None)
    _assert_finite_metrics(metrics, slug)

    print(f"✅ {slug}: shape={embedding.shape} | metrics={metrics} | train={algo.training_time():.3f}s")


# ──────────────────────────────────────────────────────────
# ANOMALY DETECTION TESTS (A-01 → A-07)
# ──────────────────────────────────────────────────────────

ANOMALY_SLUGS = [
    "unsupervised.anomaly_detection.isolationforestalgo",
    "unsupervised.anomaly_detection.lofalgo",
    "unsupervised.anomaly_detection.oneclasssvmalgo",
    "unsupervised.anomaly_detection.ellipticenvelopealgo",
    "unsupervised.anomaly_detection.copodalgo",
    "unsupervised.anomaly_detection.ecodalgo",
    "unsupervised.anomaly_detection.zscoreiqralgo",
]


@pytest.mark.parametrize("slug", ANOMALY_SLUGS)
def test_anomaly_detection_algorithm(slug, anomaly_data):
    AlgorithmRegistry.load_all()
    X_tr, X_te, y_te = anomaly_data

    # OC-SVM is slow; use subset
    if "oneclasssvm" in slug:
        X_tr, X_te = X_tr[:100], X_te[:40]
        y_te = y_te[:40]

    algo = AlgorithmRegistry.instantiate(slug, random_state=42, contamination=0.15)
    algo.fit(X_tr)

    preds = algo.predict(X_te)
    assert preds is not None
    assert preds.shape[0] == X_te.shape[0]
    assert set(np.unique(preds)).issubset({0, 1}), f"{slug}: predictions not binary 0/1"

    metrics = algo.score(X_te, y_te)
    _assert_finite_metrics(metrics, slug)

    assert 0.0 <= metrics.get("accuracy", 0.5) <= 1.0

    # Test score_samples
    if hasattr(algo, "score_samples"):
        scores = algo.score_samples(X_te)
        assert scores.shape[0] == X_te.shape[0]
        assert not np.any(np.isnan(scores)), f"{slug}: score_samples contains NaN"

    print(f"✅ {slug}: acc={metrics.get('accuracy', '?'):.3f} | "
          f"f1={metrics.get('f1', '?'):.3f} | train={algo.training_time():.3f}s")


# ──────────────────────────────────────────────────────────
# Registry coverage tests
# ──────────────────────────────────────────────────────────

def test_clustering_in_registry():
    AlgorithmRegistry.load_all()
    clustering = AlgorithmRegistry.list_by_task("clustering")
    slugs = [a["slug"] for a in clustering]
    assert len(slugs) >= 11, f"Expected 11+ clustering algos, got {len(slugs)}: {slugs}"


def test_dimensionality_reduction_in_registry():
    AlgorithmRegistry.load_all()
    dr = AlgorithmRegistry.list_by_task("dimensionality_reduction")
    assert len(dr) >= 10, f"Expected 10+ dim-reduction algos, got {len(dr)}"


def test_anomaly_detection_in_registry():
    AlgorithmRegistry.load_all()
    an = AlgorithmRegistry.list_by_task("anomaly_detection")
    assert len(an) >= 7, f"Expected 7+ anomaly algos, got {len(an)}"


def test_all_phase2_have_explain():
    AlgorithmRegistry.load_all()
    phase2_tasks = {"clustering", "dimensionality_reduction", "anomaly_detection"}
    for entry in AlgorithmRegistry.list_all():
        if entry["task_type"] in phase2_tasks:
            klass = AlgorithmRegistry.get(entry["slug"])
            exp = klass.explain()
            assert "name" in exp
            assert "math_intuition" in exp
            assert isinstance(exp.get("hyperparameters", []), list)


def test_total_registry_count():
    AlgorithmRegistry.load_all()
    all_algos = AlgorithmRegistry.list_all()
    print(f"\nTotal algorithms in registry: {len(all_algos)}")
    assert len(all_algos) >= 55, f"Expected 55+ total algorithms, got {len(all_algos)}"

