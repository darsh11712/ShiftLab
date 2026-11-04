"""
Dataset loader for ShiftLab.
Provides built-in sklearn datasets and user-uploaded CSVs.
No data leakage: split is performed here with a fixed random seed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn import datasets as skds
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from backend.config import RANDOM_STATE, DATA_DIR


@dataclass
class Dataset:
    name: str
    X: np.ndarray
    y: np.ndarray | None
    feature_names: list[str]
    target_name: str | None
    task_type: str               # regression | classification | clustering
    description: str = ""
    n_classes: int | None = None
    class_names: list[str] | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class SplitDataset:
    dataset: Dataset
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray | None
    y_test: np.ndarray | None
    test_size: float
    random_state: int


# ------------------------------------------------------------------ #
# Built-in dataset definitions                                         #
# ------------------------------------------------------------------ #

def _load_iris() -> Dataset:
    d = skds.load_iris()
    return Dataset(
        name="iris",
        X=d.data.astype(np.float64),
        y=d.target,
        feature_names=list(d.feature_names),
        target_name="species",
        task_type="classification",
        description="Classic 3-class flower classification (150 samples, 4 features).",
        n_classes=3,
        class_names=list(d.target_names),
    )


def _load_wine() -> Dataset:
    d = skds.load_wine()
    return Dataset(
        name="wine",
        X=d.data.astype(np.float64),
        y=d.target,
        feature_names=list(d.feature_names),
        target_name="wine_class",
        task_type="classification",
        description="3-class wine recognition (178 samples, 13 chemical features).",
        n_classes=3,
        class_names=list(d.target_names),
    )


def _load_breast_cancer() -> Dataset:
    d = skds.load_breast_cancer()
    return Dataset(
        name="breast_cancer",
        X=d.data.astype(np.float64),
        y=d.target,
        feature_names=list(d.feature_names),
        target_name="diagnosis",
        task_type="classification",
        description="Binary classification: malignant vs benign (569 samples, 30 features).",
        n_classes=2,
        class_names=list(d.target_names),
    )


def _load_digits() -> Dataset:
    d = skds.load_digits()
    return Dataset(
        name="digits",
        X=d.data.astype(np.float64),
        y=d.target,
        feature_names=[f"pixel_{i}" for i in range(d.data.shape[1])],
        target_name="digit",
        task_type="classification",
        description="Handwritten digit recognition 0-9 (1797 samples, 64 features).",
        n_classes=10,
        class_names=[str(i) for i in range(10)],
    )


def _load_diabetes() -> Dataset:
    d = skds.load_diabetes()
    return Dataset(
        name="diabetes",
        X=d.data.astype(np.float64),
        y=d.target.astype(np.float64),
        feature_names=list(d.feature_names),
        target_name="disease_progression",
        task_type="regression",
        description="Diabetes disease progression regression (442 samples, 10 features).",
    )


def _load_california_housing() -> Dataset:
    d = skds.fetch_california_housing()
    return Dataset(
        name="california_housing",
        X=d.data.astype(np.float64),
        y=d.target.astype(np.float64),
        feature_names=list(d.feature_names),
        target_name="median_house_value",
        task_type="regression",
        description="California housing price regression (20,640 samples, 8 features).",
    )


def _load_blobs() -> Dataset:
    X, y = skds.make_blobs(
        n_samples=300, centers=4, cluster_std=1.0, random_state=RANDOM_STATE
    )
    return Dataset(
        name="blobs",
        X=X.astype(np.float64),
        y=y,
        feature_names=["x1", "x2"],
        target_name="cluster",
        task_type="clustering",
        description="Synthetic isotropic Gaussian blobs (300 samples, 4 clusters).",
        n_classes=4,
    )


def _load_moons() -> Dataset:
    X, y = skds.make_moons(n_samples=300, noise=0.2, random_state=RANDOM_STATE)
    return Dataset(
        name="moons",
        X=X.astype(np.float64),
        y=y,
        feature_names=["x1", "x2"],
        target_name="class",
        task_type="classification",
        description="Synthetic interleaved half-moons (300 samples, binary).",
        n_classes=2,
        class_names=["moon_0", "moon_1"],
    )


def _load_circles() -> Dataset:
    X, y = skds.make_circles(n_samples=300, noise=0.1, factor=0.5, random_state=RANDOM_STATE)
    return Dataset(
        name="circles",
        X=X.astype(np.float64),
        y=y,
        feature_names=["x1", "x2"],
        target_name="class",
        task_type="classification",
        description="Synthetic concentric circles (300 samples, binary).",
        n_classes=2,
        class_names=["outer", "inner"],
    )


def _load_make_classification() -> Dataset:
    X, y = skds.make_classification(
        n_samples=500, n_features=10, n_informative=5, n_redundant=2,
        n_classes=3, random_state=RANDOM_STATE
    )
    return Dataset(
        name="make_classification",
        X=X.astype(np.float64),
        y=y,
        feature_names=[f"feature_{i}" for i in range(10)],
        target_name="class",
        task_type="classification",
        description="Synthetic multi-class classification (500 samples, 10 features, 3 classes).",
        n_classes=3,
    )


def _load_make_regression() -> Dataset:
    X, y = skds.make_regression(
        n_samples=500, n_features=10, n_informative=5, noise=10.0, random_state=RANDOM_STATE
    )
    return Dataset(
        name="make_regression",
        X=X.astype(np.float64),
        y=y.astype(np.float64),
        feature_names=[f"feature_{i}" for i in range(10)],
        target_name="target",
        task_type="regression",
        description="Synthetic regression (500 samples, 10 features, 5 informative).",
    )


_BUILTIN_LOADERS: dict[str, callable] = {
    "iris": _load_iris,
    "wine": _load_wine,
    "breast_cancer": _load_breast_cancer,
    "digits": _load_digits,
    "diabetes": _load_diabetes,
    "california_housing": _load_california_housing,
    "blobs": _load_blobs,
    "moons": _load_moons,
    "circles": _load_circles,
    "make_classification": _load_make_classification,
    "make_regression": _load_make_regression,
}

USER_UPLOAD_DIR = os.path.join(DATA_DIR, "user_uploads")


# ------------------------------------------------------------------ #
# User-uploaded datasets                                               #
# ------------------------------------------------------------------ #

def _upload_path(filename: str) -> str:
    """Resolve an upload filename safely (no path traversal)."""
    safe = os.path.basename(filename)
    if not safe or safe != filename:
        raise ValueError("Invalid dataset filename.")
    return os.path.join(USER_UPLOAD_DIR, safe)


def _meta_path(filename: str) -> str:
    return _upload_path(filename) + ".meta.json"


def infer_target_and_task(
    df: pd.DataFrame,
    target_column: str | None = None,
    task_type: str | None = None,
) -> tuple[str, str]:
    """Infer a sensible target column and task when the uploader omits them."""
    target = target_column or str(df.columns[-1])
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not in CSV.")

    if task_type not in ("classification", "regression"):
        series = df[target]
        n_unique = int(series.nunique(dropna=True))
        # Numeric target with many distinct values → regression.
        if pd.api.types.is_numeric_dtype(series) and n_unique > max(20, int(0.05 * len(df))):
            task_type = "regression"
        else:
            task_type = "classification"
    return target, task_type


def register_user_dataset(
    filename: str,
    target_column: str | None = None,
    task_type: str | None = None,
    description: str | None = None,
) -> dict:
    """Validate an uploaded CSV and persist its metadata sidecar."""
    path = _upload_path(filename)
    if not os.path.exists(path):
        raise ValueError(f"Uploaded file '{filename}' not found.")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Uploaded CSV is empty.")
    target, task = infer_target_and_task(df, target_column, task_type)

    # Validate that the file can actually be loaded/encoded now, not later.
    load_csv(path, target, task)

    meta = {
        "name": filename,
        "filename": filename,
        "target_column": target,
        "task_type": task,
        "n_samples": int(df.shape[0]),
        "n_features": int(df.shape[1] - 1),
        "description": description or f"User-uploaded: {filename}",
        "source": "upload",
    }
    os.makedirs(USER_UPLOAD_DIR, exist_ok=True)
    with open(_meta_path(filename), "w", encoding="utf-8") as f:
        import json
        json.dump(meta, f, indent=2)
    return meta


def list_user_datasets() -> list[dict]:
    """Return metadata for every uploaded CSV, inferring missing sidecars."""
    if not os.path.isdir(USER_UPLOAD_DIR):
        return []

    results: list[dict] = []
    for fname in sorted(os.listdir(USER_UPLOAD_DIR)):
        if not fname.lower().endswith(".csv"):
            continue
        try:
            meta_file = _meta_path(fname)
            if os.path.exists(meta_file):
                import json
                with open(meta_file, encoding="utf-8") as f:
                    meta = json.load(f)
            else:
                meta = register_user_dataset(fname)
            results.append({
                "name": meta["name"],
                "task_type": meta["task_type"],
                "n_samples": meta.get("n_samples"),
                "n_features": meta.get("n_features"),
                "description": meta.get("description", f"User-uploaded: {fname}"),
                "n_classes": None,
                "target_column": meta.get("target_column"),
                "source": "upload",
            })
        except Exception as exc:  # malformed CSV should not break the catalog
            results.append({
                "name": fname,
                "task_type": "unknown",
                "n_samples": None,
                "n_features": None,
                "description": f"User-uploaded (unreadable): {exc}",
                "n_classes": None,
                "source": "upload",
            })
    return results


def _load_user_dataset(name: str) -> Dataset | None:
    try:
        path = _upload_path(name)
    except ValueError:
        return None
    if not os.path.exists(path) or not name.lower().endswith(".csv"):
        return None

    meta_file = _meta_path(name)
    if os.path.exists(meta_file):
        import json
        with open(meta_file, encoding="utf-8") as f:
            meta = json.load(f)
        target = meta.get("target_column")
        task = meta.get("task_type")
    else:
        df = pd.read_csv(path)
        target, task = infer_target_and_task(df)
    return load_csv(path, target, task)


# ------------------------------------------------------------------ #
# Public API                                                           #
# ------------------------------------------------------------------ #

def list_datasets() -> list[dict]:
    """List built-in and user-uploaded datasets (metadata only, no data)."""
    results = []
    for name, loader in _BUILTIN_LOADERS.items():
        ds = loader()
        results.append({
            "name": ds.name,
            "task_type": ds.task_type,
            "n_samples": ds.X.shape[0],
            "n_features": len(ds.feature_names),
            "description": ds.description,
            "n_classes": ds.n_classes,
            "source": "builtin",
        })
    results.extend(list_user_datasets())
    return results


def load_dataset(name: str) -> Dataset:
    """Load a built-in or user-uploaded dataset by name."""
    if name in _BUILTIN_LOADERS:
        return _BUILTIN_LOADERS[name]()

    user_ds = _load_user_dataset(name)
    if user_ds is not None:
        return user_ds

    available = list(_BUILTIN_LOADERS.keys()) + [d["name"] for d in list_user_datasets()]
    raise ValueError(f"Unknown dataset '{name}'. Available: {available}")


def split_dataset(
    dataset: Dataset,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> SplitDataset:
    """Train/test split — no leakage, reproducible."""
    if dataset.y is not None:
        stratify = dataset.y if dataset.task_type == "classification" else None
        X_tr, X_te, y_tr, y_te = train_test_split(
            dataset.X, dataset.y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    else:
        X_tr, X_te = train_test_split(
            dataset.X, test_size=test_size, random_state=random_state
        )
        y_tr = y_te = None

    return SplitDataset(
        dataset=dataset,
        X_train=X_tr, X_test=X_te,
        y_train=y_tr, y_test=y_te,
        test_size=test_size,
        random_state=random_state,
    )


def load_csv(path: str, target_column: str, task_type: str) -> Dataset:
    """Load a user-uploaded CSV. Encodes categoricals automatically."""
    df = pd.read_csv(path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not in CSV.")

    y_raw = df[target_column].values
    X_df = df.drop(columns=[target_column])

    # Encode object columns
    for col in X_df.select_dtypes(include="object").columns:
        X_df[col] = LabelEncoder().fit_transform(X_df[col].astype(str))

    X = X_df.to_numpy(dtype=np.float64)

    class_names = None
    n_classes = None
    if task_type == "classification":
        le = LabelEncoder()
        y = le.fit_transform(y_raw.astype(str))
        class_names = list(le.classes_)
        n_classes = len(class_names)
    else:
        y = y_raw.astype(np.float64)

    return Dataset(
        name=os.path.basename(path),
        X=X,
        y=y,
        feature_names=list(X_df.columns),
        target_name=target_column,
        task_type=task_type,
        description=f"User-uploaded: {os.path.basename(path)}",
        n_classes=n_classes,
        class_names=class_names,
    )
