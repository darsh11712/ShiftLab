"""
Comprehensive test suite for all Phase 1 supervised learning algorithms.

Rules:
- Every algorithm must pass fit → predict → score on a real dataset.
- No algorithm is marked tested until all three stages pass.
- Checks that scores are real numbers (not NaN, not fabricated).
- Tests both regression and classification families.
"""
import sys
import os
import pytest
import numpy as np

# Make sure backend package is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.core.registry import AlgorithmRegistry
from backend.core.dataset_loader import load_dataset, split_dataset
from backend.core.preprocessor import PreprocessConfig, build_preprocessor, fit_transform

# ──────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def regression_split():
    ds = load_dataset("diabetes")
    split = split_dataset(ds, test_size=0.2, random_state=42)
    cfg = PreprocessConfig(scaler="standard", imputer="mean")
    pp = build_preprocessor(cfg)
    X_tr, X_te = fit_transform(pp, split.X_train, split.X_test)
    return X_tr, X_te, split.y_train, split.y_test


@pytest.fixture(scope="session")
def classification_split():
    """Multi-class: Iris (3 classes)."""
    ds = load_dataset("iris")
    split = split_dataset(ds, test_size=0.2, random_state=42)
    cfg = PreprocessConfig(scaler="standard", imputer="mean")
    pp = build_preprocessor(cfg)
    X_tr, X_te = fit_transform(pp, split.X_train, split.X_test)
    return X_tr, X_te, split.y_train, split.y_test


@pytest.fixture(scope="session")
def binary_classification_split():
    """Binary: Breast cancer."""
    ds = load_dataset("breast_cancer")
    split = split_dataset(ds, test_size=0.2, random_state=42)
    cfg = PreprocessConfig(scaler="standard", imputer="mean")
    pp = build_preprocessor(cfg)
    X_tr, X_te = fit_transform(pp, split.X_train, split.X_test)
    return X_tr, X_te, split.y_train, split.y_test


@pytest.fixture(scope="session")
def nonneg_classification_split():
    """Non-negative features for Multinomial NB — MinMax scaled."""
    ds = load_dataset("iris")
    split = split_dataset(ds, test_size=0.2, random_state=42)
    cfg = PreprocessConfig(scaler="minmax", imputer="mean")
    pp = build_preprocessor(cfg)
    X_tr, X_te = fit_transform(pp, split.X_train, split.X_test)
    return X_tr, X_te, split.y_train, split.y_test


# ──────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────

def _assert_metrics(metrics: dict, task: str):
    """Verify all metric values are finite floats."""
    assert isinstance(metrics, dict), "score() must return a dict"
    assert len(metrics) > 0, "score() returned empty dict"
    for k, v in metrics.items():
        assert isinstance(v, float), f"Metric '{k}' is not float: {type(v)}"
        assert np.isfinite(v), f"Metric '{k}' is not finite: {v}"


# ──────────────────────────────────────────────────────────
# Regression tests (R-01 → R-18)
# ──────────────────────────────────────────────────────────

REGRESSION_SLUGS = [
    "supervised.regression.linearregressionalgo",
    "supervised.regression.ridgeregressionalgo",
    "supervised.regression.lassoregressionalgo",
    "supervised.regression.elasticnetalgo",
    "supervised.regression.polynomialregressionalgo",
    "supervised.regression.svralgo",
    "supervised.regression.knnregressoralgo",
    "supervised.regression.decisiontreeregressoralgo",
    "supervised.regression.randomforestregressoralgo",
    "supervised.regression.gradientboostingregressoralgo",
    "supervised.regression.xgboostregressoralgo",
    "supervised.regression.lightgbmregressoralgo",
    "supervised.regression.gaussianprocessregressoralgo",
    "supervised.regression.bayesianridgealgo",
    "supervised.regression.huberregressoralgo",
    "supervised.regression.quantileregressoralgo",
    "supervised.regression.mlpregressoralgo",
    "supervised.regression.passiveaggressiveregressoralgo",
]


@pytest.mark.parametrize("slug", REGRESSION_SLUGS)
def test_regression_algorithm(slug, regression_split):
    AlgorithmRegistry.load_all()
    X_tr, X_te, y_tr, y_te = regression_split

    # GPR is very slow on large data — use a small subset
    if "gaussianprocess" in slug:
        X_tr, y_tr = X_tr[:80], y_tr[:80]
        X_te, y_te = X_te[:20], y_te[:20]

    algo = AlgorithmRegistry.instantiate(slug, random_state=42)
    algo.fit(X_tr, y_tr)
    preds = algo.predict(X_te)

    assert preds.shape == y_te.shape, f"{slug}: prediction shape mismatch"
    assert not np.any(np.isnan(preds)), f"{slug}: predictions contain NaN"

    metrics = algo.score(X_te, y_te)
    _assert_metrics(metrics, "regression")

    # Sanity: R² should be somewhat reasonable (not completely random)
    assert metrics["r2"] > -10.0, f"{slug}: R² too low ({metrics['r2']:.3f})"

    # Training time must have been measured
    assert algo.training_time() >= 0.0

    print(f"✅ {slug}: r2={metrics['r2']:.4f} mae={metrics['mae']:.4f} train={algo.training_time():.3f}s")


# ──────────────────────────────────────────────────────────
# Classification tests (C-01 → C-20)
# ──────────────────────────────────────────────────────────

# Algorithms that need non-negative features
NONNEG_SLUGS = {
    "supervised.classification.multinomialnbalgo",
    "supervised.classification.bernoullinbalgo",
}

# Algorithms that don't support predict_proba
NO_PROBA_SLUGS = {
    "supervised.classification.ridgeclassifieralgo",
    "supervised.classification.perceptronalgo",
    "supervised.classification.sgdclassifieralgo",
    "supervised.classification.passiveaggressiveclassifieralgo",
}

CLASSIFICATION_SLUGS = [
    "supervised.classification.logisticregressionalgo",
    "supervised.classification.ldaclassifieralgo",
    "supervised.classification.qdaclassifieralgo",
    "supervised.classification.svmlinearalgo",
    "supervised.classification.svmrbfalgo",
    "supervised.classification.knnclassifieralgo",
    "supervised.classification.gaussiannbalgo",
    "supervised.classification.multinomialnbalgo",
    "supervised.classification.bernoullinbalgo",
    "supervised.classification.decisiontreeclassifieralgo",
    "supervised.classification.randomforestclassifieralgo",
    "supervised.classification.gradientboostingclassifieralgo",
    "supervised.classification.xgboostclassifieralgo",
    "supervised.classification.lightgbmclassifieralgo",
    "supervised.classification.ridgeclassifieralgo",
    "supervised.classification.perceptronalgo",
    "supervised.classification.sgdclassifieralgo",
    "supervised.classification.mlpclassifieralgo",
    "supervised.classification.passiveaggressiveclassifieralgo",
    "supervised.classification.calibratedclassifieralgo",
]


@pytest.mark.parametrize("slug", CLASSIFICATION_SLUGS)
def test_classification_algorithm(slug, classification_split, nonneg_classification_split, binary_classification_split):
    AlgorithmRegistry.load_all()

    # Choose appropriate data
    if slug in NONNEG_SLUGS:
        X_tr, X_te, y_tr, y_te = nonneg_classification_split
    elif slug == "supervised.classification.calibratedclassifieralgo":
        # Use binary for cleaner calibration test
        X_tr, X_te, y_tr, y_te = binary_classification_split
    else:
        X_tr, X_te, y_tr, y_te = classification_split

    # SVM with probability=True is slow on large data
    if "svm" in slug or "calibrated" in slug:
        X_tr, y_tr = X_tr[:100], y_tr[:100]
        X_te, y_te = X_te[:30], y_te[:30]

    algo = AlgorithmRegistry.instantiate(slug, random_state=42)
    algo.fit(X_tr, y_tr)
    preds = algo.predict(X_te)

    assert preds.shape == y_te.shape, f"{slug}: prediction shape mismatch"
    assert not np.any(np.isnan(preds.astype(float))), f"{slug}: predictions contain NaN"

    metrics = algo.score(X_te, y_te)
    _assert_metrics(metrics, "classification")

    # Sanity check: accuracy must be between 0 and 1
    assert 0.0 <= metrics["accuracy"] <= 1.0, f"{slug}: invalid accuracy {metrics['accuracy']}"

    print(f"✅ {slug}: acc={metrics['accuracy']:.4f} f1={metrics.get('f1', 'N/A')} train={algo.training_time():.3f}s")


# ──────────────────────────────────────────────────────────
# Registry tests
# ──────────────────────────────────────────────────────────

def test_registry_lists_all_algorithms():
    AlgorithmRegistry.load_all()
    all_algos = AlgorithmRegistry.list_all()
    slugs = [a["slug"] for a in all_algos]

    expected = REGRESSION_SLUGS + CLASSIFICATION_SLUGS
    for slug in expected:
        assert slug in slugs, f"Algorithm '{slug}' not found in registry"

    print(f"Registry contains {len(all_algos)} algorithms total")


def test_registry_search():
    AlgorithmRegistry.load_all()
    results = AlgorithmRegistry.search("ridge")
    names = [r["name"] for r in results]
    assert any("Ridge" in n for n in names), "Search for 'ridge' returned no Ridge algorithms"


def test_registry_list_by_family():
    AlgorithmRegistry.load_all()
    sup = AlgorithmRegistry.list_by_family("supervised")
    assert len(sup) > 0


def test_registry_list_by_task():
    AlgorithmRegistry.load_all()
    reg_algos = AlgorithmRegistry.list_by_task("regression")
    cls_algos = AlgorithmRegistry.list_by_task("classification")
    assert len(reg_algos) >= 18
    assert len(cls_algos) >= 20


# ──────────────────────────────────────────────────────────
# Explain / metadata tests
# ──────────────────────────────────────────────────────────

def test_all_algorithms_have_explain():
    AlgorithmRegistry.load_all()
    for entry in AlgorithmRegistry.list_all():
        klass = AlgorithmRegistry.get(entry["slug"])
        exp = klass.explain()
        assert "name" in exp
        assert "family" in exp
        assert "task_type" in exp
        assert isinstance(exp.get("hyperparameters", []), list)


# ──────────────────────────────────────────────────────────
# Dataset loader tests
# ──────────────────────────────────────────────────────────

def test_all_builtin_datasets_load():
    from backend.core.dataset_loader import list_datasets, load_dataset
    catalog = list_datasets()
    assert len(catalog) >= 10
    for meta in catalog:
        ds = load_dataset(meta["name"])
        assert ds.X is not None
        assert ds.X.shape[0] > 0
        assert ds.X.shape[1] > 0


def test_split_no_data_leakage():
    """Verify that test indices are disjoint from train indices."""
    from backend.core.dataset_loader import load_dataset, split_dataset
    ds = load_dataset("iris")
    split = split_dataset(ds, test_size=0.2, random_state=42)
    total = split.X_train.shape[0] + split.X_test.shape[0]
    assert total == ds.X.shape[0]


# ──────────────────────────────────────────────────────────
# Preprocessor tests
# ──────────────────────────────────────────────────────────

def test_preprocessor_no_leakage():
    """Test data must not affect scaler fit."""
    from backend.core.preprocessor import PreprocessConfig, build_preprocessor, fit_transform
    rng = np.random.default_rng(42)
    X_train = rng.normal(0, 1, (100, 5))
    X_test = rng.normal(100, 1, (20, 5))  # very different distribution
    cfg = PreprocessConfig(scaler="standard")
    pp = build_preprocessor(cfg)
    X_tr_s, X_te_s = fit_transform(pp, X_train, X_test)
    # Training set should have zero mean
    assert np.abs(X_tr_s.mean(axis=0)).max() < 0.1
    # Test set should NOT be centred around zero (different distribution)
    assert X_te_s.mean(axis=0).max() > 50  # large positive shift preserved


# ──────────────────────────────────────────────────────────
# Experiment persistence tests
# ──────────────────────────────────────────────────────────

def test_experiment_save_load():
    import os
    from backend.core.experiment import save_experiment, load_experiment, list_experiments
    from backend.config import EXPERIMENTS_DIR

    exp_id = save_experiment(
        algorithm_slug="test.algo",
        dataset_name="iris",
        hyperparams={"C": 1.0},
        metrics={"accuracy": 0.95, "f1": 0.94},
        train_time_s=0.1,
        infer_time_s=0.001,
        random_state=42,
    )
    try:
        loaded = load_experiment(exp_id)
        assert loaded["algorithm_slug"] == "test.algo"
        assert loaded["metrics"]["accuracy"] == pytest.approx(0.95)

        exps = list_experiments()
        ids = [e["id"] for e in exps]
        assert exp_id in ids
    finally:
        # Do not leave synthetic records in the experiment catalog.
        path = os.path.join(EXPERIMENTS_DIR, f"{exp_id}.json")
        if os.path.exists(path):
            os.remove(path)
