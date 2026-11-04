"""
API integration tests for ShiftLab.

Covers the original routes plus the recovered features:
  * arena comparison / leaderboard
  * reliability endpoints
  * uploaded-CSV registration + training
  * bundled frontend
"""
from __future__ import annotations

import os
import uuid

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.core.dataset_loader import USER_UPLOAD_DIR

client = TestClient(app)


@pytest.fixture()
def uploaded_csv():
    """Create an upload through the API and clean the files afterwards."""
    name = f"pytest_upload_{uuid.uuid4().hex[:8]}.csv"
    content = (
        "f1,f2,f3,target\n"
        "1,2,3,a\n2,1,4,a\n5,6,7,b\n6,5,8,b\n9,8,7,b\n1,1,2,a\n"
    )
    response = client.post(
        "/api/datasets/upload",
        files={"file": (name, content, "text/csv")},
        data={"target_column": "target", "task_type": "classification"},
    )
    assert response.status_code == 200, response.text
    yield name, response.json()
    for path in (os.path.join(USER_UPLOAD_DIR, name), os.path.join(USER_UPLOAD_DIR, name + ".meta.json")):
        if os.path.exists(path):
            os.remove(path)


# ---------------------------------------------------------------------------
# Health / static frontend
# ---------------------------------------------------------------------------

def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_frontend_is_served():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "ShiftLab" in response.text


# ---------------------------------------------------------------------------
# Algorithms
# ---------------------------------------------------------------------------

def test_list_algorithms():
    response = client.get("/api/algorithms")
    assert response.status_code == 200
    algorithms = response.json()
    assert len(algorithms) == 81
    slugs = {a["slug"] for a in algorithms}
    assert "supervised.classification.logisticregressionalgo" in slugs
    assert "ensemble.classification.stackingclassifieralgo" in slugs
    assert "neural.classification.mlpclassifiercustomalgo" in slugs


def test_algorithm_filters():
    family = client.get("/api/algorithms", params={"family": "neural"}).json()
    assert len(family) == 5

    task = client.get("/api/algorithms", params={"task_type": "regression"}).json()
    assert len(task) >= 18

    search = client.get("/api/algorithms", params={"q": "ridge"}).json()
    assert any("Ridge" in entry["name"] for entry in search)


def test_algorithm_explain_and_404():
    response = client.get("/api/algorithms/supervised.classification.logisticregressionalgo")
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "Logistic Regression"
    assert isinstance(body["hyperparameters"], list)

    missing = client.get("/api/algorithms/does.not.exist")
    assert missing.status_code == 404


# ---------------------------------------------------------------------------
# Datasets / upload
# ---------------------------------------------------------------------------

def test_list_datasets_includes_builtins():
    response = client.get("/api/datasets")
    assert response.status_code == 200
    names = {d["name"] for d in response.json()}
    assert {"iris", "wine", "diabetes", "blobs"} <= names


def test_uploaded_dataset_is_trainable(uploaded_csv):
    name, meta = uploaded_csv
    assert meta["task_type"] == "classification"
    assert meta["target_column"] == "target"

    listing = client.get("/api/datasets").json()
    uploaded = next(d for d in listing if d["name"] == name)
    assert uploaded["source"] == "upload"

    info = client.get(f"/api/datasets/{name}")
    assert info.status_code == 200
    assert info.json()["n_samples"] == 6

    trained = client.post(
        "/api/train",
        json={
            "algorithm_slug": "supervised.classification.logisticregressionalgo",
            "dataset_name": name,
            "test_size": 0.34,
            "save": False,
        },
    )
    assert trained.status_code == 200, trained.text
    assert 0.0 <= trained.json()["metrics"]["accuracy"] <= 1.0


def test_upload_rejects_bad_target(uploaded_csv):
    name = f"pytest_bad_{uuid.uuid4().hex[:8]}.csv"
    response = client.post(
        "/api/datasets/upload",
        files={"file": (name, "a,b\n1,2\n", "text/csv")},
        data={"target_column": "missing"},
    )
    assert response.status_code == 400
    assert not os.path.exists(os.path.join(USER_UPLOAD_DIR, name))


# ---------------------------------------------------------------------------
# Train / predict
# ---------------------------------------------------------------------------

def test_train_and_predict():
    response = client.post(
        "/api/train",
        json={
            "algorithm_slug": "supervised.classification.logisticregressionalgo",
            "dataset_name": "iris",
            "test_size": 0.2,
            "save": False,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["dataset_task_type"] == "classification"
    assert 0.0 <= body["metrics"]["accuracy"] <= 1.0
    assert body["experiment_id"] is None

    prediction = client.post(
        "/api/predict",
        json={
            "algorithm_slug": "supervised.classification.logisticregressionalgo",
            "dataset_name": "iris",
            "X": [[5.1, 3.5, 1.4, 0.2], [6.0, 3.0, 4.8, 1.8]],
        },
    )
    assert prediction.status_code == 200, prediction.text
    assert len(prediction.json()["predictions"]) == 2


def test_train_rejects_task_mismatch():
    response = client.post(
        "/api/train",
        json={
            "algorithm_slug": "supervised.classification.logisticregressionalgo",
            "dataset_name": "diabetes",
            "save": False,
        },
    )
    assert response.status_code == 400


def test_train_unknown_dataset_and_algorithm():
    dataset = client.post(
        "/api/train",
        json={"algorithm_slug": "supervised.classification.logisticregressionalgo",
              "dataset_name": "nope", "save": False},
    )
    assert dataset.status_code == 404

    algorithm = client.post(
        "/api/train",
        json={"algorithm_slug": "nope", "dataset_name": "iris", "save": False},
    )
    assert algorithm.status_code == 404


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def test_experiments_list_and_missing():
    response = client.get("/api/experiments")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert client.get("/api/experiments/doesnotexist").status_code == 404


# ---------------------------------------------------------------------------
# Arena
# ---------------------------------------------------------------------------

def test_arena_compare_and_leaderboard():
    response = client.post(
        "/api/arena/compare",
        json={
            "algorithm_slugs": [
                "supervised.classification.logisticregressionalgo",
                "supervised.classification.knnclassifieralgo",
                "supervised.classification.decisiontreeclassifieralgo",
            ],
            "dataset_name": "iris",
            "test_size": 0.2,
            "save": False,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["primary_metric"] == "accuracy"
    assert len(body["results"]) == 3
    assert body["winner"] is not None
    ranks = [entry["rank"] for entry in body["results"] if entry["error"] is None]
    assert ranks == [1, 2, 3]

    leaderboard = client.get("/api/arena/leaderboard")
    assert leaderboard.status_code == 200
    assert "entries" in leaderboard.json()


def test_arena_rejects_unsupported_dataset():
    response = client.post(
        "/api/arena/compare",
        json={
            "algorithm_slugs": ["unsupervised.clustering.kmeansalgo"],
            "dataset_name": "blobs",
        },
    )
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# Reliability
# ---------------------------------------------------------------------------

def test_reliability_covariate_shift():
    response = client.post(
        "/api/reliability/covariate-shift",
        json={"dataset_name": "iris", "test_size": 0.25},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert "mmd2" in body["mmd"]
    assert 0.0 <= body["classifier_two_sample_test"]["auc_mean"] <= 1.0


def test_reliability_label_shift():
    response = client.post(
        "/api/reliability/label-shift",
        json={"dataset_name": "iris", "test_size": 0.25},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["classes"]) == 3
    assert 0.0 <= body["tv_distance"] <= 1.0


def test_reliability_label_shift_rejects_regression():
    response = client.post(
        "/api/reliability/label-shift",
        json={"dataset_name": "diabetes", "test_size": 0.25},
    )
    assert response.status_code == 400


def test_reliability_importance_weights():
    response = client.post(
        "/api/reliability/importance-weights",
        json={"dataset_name": "iris", "test_size": 0.25, "method": "classifier"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["weights"]) == body["n_train"]
    assert 0.0 < body["ess_ratio"] <= 1.0


def test_reliability_performance_under_shift():
    response = client.post(
        "/api/reliability/performance-under-shift",
        json={
            "dataset_name": "iris",
            "algorithm_slug": "supervised.classification.logisticregressionalgo",
            "shift_magnitudes": [0.0, 0.5, 1.0],
            "shift_type": "gaussian",
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["results"]) == 3
    assert 0.0 <= body["baseline_performance"] <= 1.0
    assert body["degradation"] >= 0.0
