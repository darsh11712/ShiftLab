"""
Dimensionality Reduction Algorithms for ShiftLab — 10 algorithms.

All reducers:
- Implement BaseAlgorithm with fit() / predict() (transform) / score().
- predict() returns the low-dimensional embedding.
- score() returns reconstruction error or explained variance where applicable.
- task_type = "dimensionality_reduction"
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import (
    PCA, KernelPCA, TruncatedSVD, FastICA, NMF,
)
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from backend.core.base_algorithm import BaseAlgorithm
from backend.core.registry import AlgorithmRegistry


def _reconstruction_error(X_orig: np.ndarray, X_reduced: np.ndarray, inverse_fn) -> dict:
    """Helper: compute reconstruction MSE and explained-variance proxy."""
    try:
        X_recon = inverse_fn(X_reduced)
        mse = float(np.mean((X_orig - X_recon) ** 2))
    except Exception:
        mse = float("nan")
    return {"reconstruction_mse": mse, "n_components_used": int(X_reduced.shape[1])}


# ────────────────────────────────────────────────────────────────────
# D-01  PCA
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class PCAAlgo(BaseAlgorithm):
    """
    Principal Component Analysis (PCA).
    Linear dimensionality reduction via SVD. Projects data onto the directions
    of maximum variance (principal components).
    """
    name = "PCA"
    family = "unsupervised"
    task_type = "dimensionality_reduction"
    tags = ["linear", "variance", "projection", "fast", "interpretable"]
    complexity_train = "O(min(n,d)²·max(n,d))"
    complexity_predict = "O(n·d·k)"

    def _build_model(self):
        return PCA(
            n_components=self._hyperparams.get("n_components", 2),
            whiten=self._hyperparams.get("whiten", False),
            random_state=self.random_state,
        )

    def _fit_impl(self, X, y=None):
        self._model.fit(X)

    def _predict_impl(self, X):
        return self._model.transform(X)

    def score(self, X, y=None):
        X_red = self.predict(X)
        result = _reconstruction_error(X, X_red, self._model.inverse_transform)
        result["explained_variance_ratio"] = float(
            self._model.explained_variance_ratio_.sum()
        )
        result["per_component_variance"] = self._model.explained_variance_ratio_.tolist()
        return result

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "n_components", "type": "int", "default": 2, "min": 1, "max": 100,
             "description": "Number of principal components to retain."},
            {"name": "whiten", "type": "bool", "default": False,
             "description": "If True, components are divided by their std (unit variance)."},
        ]

    @classmethod
    def _math_intuition(cls):
        return (
            "Compute covariance C = XᵀX/(n-1). "
            "Eigendecompose: C = VΛVᵀ. "
            "Project: Z = XV_k (top-k eigenvectors)."
        )

    @classmethod
    def _assumptions(cls):
        return [
            "Linear relationships in data.",
            "Variance = information (may not hold).",
            "Scale features before applying.",
        ]

    @classmethod
    def _strengths(cls):
        return [
            "Deterministic and fast.",
            "Invertible — exact reconstruction error.",
            "Explained variance ratio gives interpretable component importance.",
            "Removes multicollinearity.",
        ]

    @classmethod
    def _weaknesses(cls):
        return ["Only linear structure.", "Components may not be interpretable.", "Variance ≠ relevance."]

    @classmethod
    def _use_cases(cls):
        return ["Preprocessing for ML", "Noise reduction", "Visualisation (n_components=2)"]


# ────────────────────────────────────────────────────────────────────
# D-02  Kernel PCA
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class KernelPCAAlgo(BaseAlgorithm):
    """
    Kernel PCA.
    Extension of PCA using the kernel trick to capture non-linear structure.
    Projects data into a (potentially infinite-dimensional) feature space
    and applies PCA there.
    """
    name = "Kernel PCA"
    family = "unsupervised"
    task_type = "dimensionality_reduction"
    tags = ["non-linear", "kernel", "manifold", "rbf"]
    complexity_train = "O(n²·d) to O(n³)"
    complexity_predict = "O(n_train·d)"

    def _build_model(self):
        return KernelPCA(
            n_components=self._hyperparams.get("n_components", 2),
            kernel=self._hyperparams.get("kernel", "rbf"),
            gamma=self._hyperparams.get("gamma", None),
            fit_inverse_transform=True,
            random_state=self.random_state,
            n_jobs=self._n_jobs(),
        )

    def _fit_impl(self, X, y=None):
        self._model.fit(X)

    def _predict_impl(self, X):
        return self._model.transform(X)

    def score(self, X, y=None):
        X_red = self.predict(X)
        try:
            X_recon = self._model.inverse_transform(X_red)
            mse = float(np.mean((X - X_recon) ** 2))
        except Exception:
            mse = float("nan")
        return {"reconstruction_mse": mse, "n_components_used": int(X_red.shape[1])}

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "n_components", "type": "int", "default": 2, "min": 1, "max": 100},
            {"name": "kernel", "type": "select", "default": "rbf",
             "options": ["linear", "poly", "rbf", "sigmoid", "cosine"]},
            {"name": "gamma", "type": "float", "default": None, "min": 0.0001, "max": 100.0,
             "description": "Kernel coefficient. None = 1/n_features."},
        ]

    @classmethod
    def _math_intuition(cls):
        return (
            "Kernelise: K(x,x') = φ(x)·φ(x'). "
            "Centred kernel matrix K̃ = K - 1ₙK - K1ₙ + 1ₙK1ₙ. "
            "Eigendecompose K̃ and project."
        )

    @classmethod
    def _strengths(cls):
        return ["Captures non-linear structure.", "Flexible kernel choice.", "Extends PCA naturally."]

    @classmethod
    def _weaknesses(cls):
        return ["O(n²) memory for kernel matrix.", "Slow for large n.", "No explicit feature loading."]

    @classmethod
    def _use_cases(cls):
        return ["Non-linear manifold reduction", "Preprocessing for kernel SVM"]

    @classmethod
    def _assumptions(cls):
        return ["Data lies on a non-linear manifold.", "Scale features."]


# ────────────────────────────────────────────────────────────────────
# D-03  Truncated SVD (LSA)
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class TruncatedSVDAlgo(BaseAlgorithm):
    """
    Truncated SVD (LSA — Latent Semantic Analysis).
    Linear reduction via SVD without centring. Works on sparse matrices directly —
    ideal for TF-IDF document-term matrices.
    """
    name = "Truncated SVD"
    family = "unsupervised"
    task_type = "dimensionality_reduction"
    tags = ["linear", "sparse", "text", "lsa", "svd"]
    complexity_train = "O(n·d·k)"
    complexity_predict = "O(n·d·k)"

    def _build_model(self):
        return TruncatedSVD(
            n_components=self._hyperparams.get("n_components", 2),
            algorithm=self._hyperparams.get("algorithm", "randomized"),
            n_iter=self._hyperparams.get("n_iter", 5),
            random_state=self.random_state,
        )

    def _fit_impl(self, X, y=None):
        self._model.fit(X)

    def _predict_impl(self, X):
        return self._model.transform(X)

    def score(self, X, y=None):
        X_red = self.predict(X)
        result = _reconstruction_error(X, X_red, self._model.inverse_transform)
        result["explained_variance_ratio"] = float(
            self._model.explained_variance_ratio_.sum()
        )
        return result

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "n_components", "type": "int", "default": 2, "min": 1, "max": 200},
            {"name": "algorithm", "type": "select", "default": "randomized",
             "options": ["randomized", "arpack"]},
            {"name": "n_iter", "type": "int", "default": 5, "min": 1, "max": 20,
             "description": "Number of iterations for randomised SVD."},
        ]

    @classmethod
    def _math_intuition(cls):
        return "X ≈ UₖΣₖVₖᵀ where Uₖ,Σₖ,Vₖ are the top-k singular triplets. No centring required."

    @classmethod
    def _strengths(cls):
        return ["Handles sparse matrices directly.", "Fast with randomised algorithm.", "Scales to large vocabularies."]

    @classmethod
    def _weaknesses(cls):
        return ["Linear structure only.", "Components not as interpretable as PCA for dense data."]

    @classmethod
    def _use_cases(cls):
        return ["Text/NLP (TF-IDF → LSA)", "Sparse matrix compression", "Collaborative filtering"]

    @classmethod
    def _assumptions(cls):
        return ["Linear relationships.", "Sparse or dense input OK."]


# ────────────────────────────────────────────────────────────────────
# D-04  FastICA
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class FastICAAlgo(BaseAlgorithm):
    """
    FastICA — Independent Component Analysis.
    Separates a multivariate signal into maximally statistically independent components.
    Useful for blind source separation (e.g. cocktail party problem).
    """
    name = "FastICA"
    family = "unsupervised"
    task_type = "dimensionality_reduction"
    tags = ["ica", "source-separation", "non-gaussian", "blind"]
    complexity_train = "O(n·d·k·iter)"
    complexity_predict = "O(n·d·k)"

    def _build_model(self):
        return FastICA(
            n_components=self._hyperparams.get("n_components", 2),
            algorithm=self._hyperparams.get("algorithm", "parallel"),
            fun=self._hyperparams.get("fun", "logcosh"),
            max_iter=self._hyperparams.get("max_iter", 200),
            random_state=self.random_state,
            whiten="unit-variance",
        )

    def _fit_impl(self, X, y=None):
        self._model.fit(X)

    def _predict_impl(self, X):
        return self._model.transform(X)

    def score(self, X, y=None):
        X_red = self.predict(X)
        try:
            X_recon = self._model.inverse_transform(X_red)
            mse = float(np.mean((X - X_recon) ** 2))
        except Exception:
            mse = float("nan")
        return {"reconstruction_mse": mse, "n_components_used": int(X_red.shape[1])}

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "n_components", "type": "int", "default": 2, "min": 1, "max": 100},
            {"name": "algorithm", "type": "select", "default": "parallel",
             "options": ["parallel", "deflation"]},
            {"name": "fun", "type": "select", "default": "logcosh",
             "options": ["logcosh", "exp", "cube"],
             "description": "Non-linearity for approximating negentropy."},
            {"name": "max_iter", "type": "int", "default": 200, "min": 50, "max": 1000},
        ]

    @classmethod
    def _math_intuition(cls):
        return (
            "Maximise non-Gaussianity of each component (by negentropy proxy). "
            "Fixed-point iteration: w ← E[x·g(wᵀx)] - E[g'(wᵀx)]·w, then orthogonalise."
        )

    @classmethod
    def _strengths(cls):
        return ["Blind source separation.", "Components are statistically independent (not just uncorrelated)."]

    @classmethod
    def _weaknesses(cls):
        return [
            "Requires non-Gaussian sources.",
            "Signs of components are arbitrary.",
            "Sensitive to outliers.",
        ]

    @classmethod
    def _use_cases(cls):
        return ["EEG/brain signal separation", "Audio source separation", "Feature extraction"]

    @classmethod
    def _assumptions(cls):
        return ["Sources are statistically independent.", "At most one source is Gaussian."]


# ────────────────────────────────────────────────────────────────────
# D-05  NMF
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class NMFAlgo(BaseAlgorithm):
    """
    Non-Negative Matrix Factorisation (NMF).
    Decomposes X ≈ WH where W, H ≥ 0. Components are parts-based and additive —
    naturally interpretable for images, text, and spectral data.
    """
    name = "NMF"
    family = "unsupervised"
    task_type = "dimensionality_reduction"
    tags = ["non-negative", "parts-based", "interpretable", "additive"]
    complexity_train = "O(n·d·k·iter)"
    complexity_predict = "O(n·d·k)"

    def _build_model(self):
        return NMF(
            n_components=self._hyperparams.get("n_components", 2),
            init=self._hyperparams.get("init", "nndsvda"),
            max_iter=self._hyperparams.get("max_iter", 200),
            random_state=self.random_state,
        )

    def _fit_impl(self, X, y=None):
        # NMF requires non-negative input
        self._model.fit(X)

    def _predict_impl(self, X):
        return self._model.transform(X)

    def score(self, X, y=None):
        X_red = self.predict(X)
        result = _reconstruction_error(X, X_red, self._model.inverse_transform)
        result["reconstruction_error"] = float(self._model.reconstruction_err_)
        return result

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "n_components", "type": "int", "default": 2, "min": 1, "max": 100},
            {"name": "init", "type": "select", "default": "nndsvda",
             "options": ["random", "nndsvda", "nndsvd", "nndsvdar"],
             "description": "'nndsvda' is good for sparse data."},
            {"name": "max_iter", "type": "int", "default": 200, "min": 50, "max": 1000},
        ]

    @classmethod
    def _math_intuition(cls):
        return (
            "Minimise ‖X - WH‖²_F subject to W, H ≥ 0. "
            "Multiplicative update rules: H ← H ⊙ (WᵀX)/(WᵀWH), W ← W ⊙ (XHᵀ)/(WHHᵀ)."
        )

    @classmethod
    def _strengths(cls):
        return [
            "Non-negativity produces parts-based representation.",
            "Interpretable components (e.g., face parts, topics).",
            "Good for sparse data.",
        ]

    @classmethod
    def _weaknesses(cls):
        return ["Requires non-negative input.", "Non-convex — local optima.", "Slow convergence."]

    @classmethod
    def _use_cases(cls):
        return ["Topic modelling", "Image decomposition", "Spectroscopy", "Audio analysis"]

    @classmethod
    def _assumptions(cls):
        return ["Input must be non-negative.", "Parts-based representation is meaningful."]


# ────────────────────────────────────────────────────────────────────
# D-06  t-SNE
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class TSNEAlgo(BaseAlgorithm):
    """
    t-SNE — t-Distributed Stochastic Neighbour Embedding.
    Non-linear dimensionality reduction that preserves local structure.
    Exceptionally useful for visualising high-dimensional data in 2D/3D.
    Cannot be used for out-of-sample projection.
    """
    name = "t-SNE"
    family = "unsupervised"
    task_type = "dimensionality_reduction"
    tags = ["non-linear", "visualization", "local-structure", "manifold"]
    complexity_train = "O(n² log n)"
    complexity_predict = "O(n) — no out-of-sample"

    def _build_model(self):
        return TSNE(
            n_components=self._hyperparams.get("n_components", 2),
            perplexity=self._hyperparams.get("perplexity", 30.0),
            learning_rate=self._hyperparams.get("learning_rate", "auto"),
            max_iter=self._hyperparams.get("max_iter", 1000),
            metric=self._hyperparams.get("metric", "euclidean"),
            random_state=self.random_state,
        )

    def _fit_impl(self, X, y=None):
        self._embedding = self._model.fit_transform(X)
        self._train_X = X

    def _predict_impl(self, X):
        # t-SNE doesn't support out-of-sample projection; return training embedding
        if np.array_equal(X, self._train_X):
            return self._embedding
        raise NotImplementedError(
            "t-SNE does not support out-of-sample projection. "
            "Use the training embedding or apply PCA first."
        )

    def fit(self, X, y=None):
        import time
        self._model = self._build_model()
        t0 = time.perf_counter()
        self._fit_impl(X, y)
        self._train_time_s = time.perf_counter() - t0
        self._is_fitted = True
        return self

    def score(self, X, y=None):
        return {
            "kl_divergence": float(self._model.kl_divergence_),
            "n_iter": int(self._model.n_iter_),
            "n_components_used": int(self._embedding.shape[1]),
        }

    def get_embedding(self) -> np.ndarray:
        """Return the low-dimensional embedding of training data."""
        self._assert_fitted()
        return self._embedding

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "n_components", "type": "int", "default": 2, "min": 2, "max": 3,
             "description": "Embedding dimension (2 or 3 for visualisation)."},
            {"name": "perplexity", "type": "float", "default": 30.0, "min": 5.0, "max": 100.0,
             "description": "Effective number of neighbours. Typical: 5–50."},
            {"name": "max_iter", "type": "int", "default": 1000, "min": 250, "max": 5000},
            {"name": "metric", "type": "select", "default": "euclidean",
             "options": ["euclidean", "cosine", "manhattan"]},
        ]

    @classmethod
    def _math_intuition(cls):
        return (
            "High-d similarities: pⱼ|ᵢ = exp(-‖xᵢ-xⱼ‖²/2σᵢ²)/Σₖ exp(-‖xᵢ-xₖ‖²/2σᵢ²). "
            "Low-d similarities: qᵢⱼ = (1+‖yᵢ-yⱼ‖²)⁻¹/Z (Student-t kernel). "
            "Minimise KL(P‖Q) via gradient descent."
        )

    @classmethod
    def _strengths(cls):
        return [
            "Reveals cluster structure invisible to linear methods.",
            "Preserves local (neighbourhood) structure.",
            "Standard tool for high-dimensional visualisation.",
        ]

    @classmethod
    def _weaknesses(cls):
        return [
            "No out-of-sample extension.",
            "Global distances not preserved.",
            "Slow O(n² log n) — limit to ~10k points.",
            "Random seed dependent.",
        ]

    @classmethod
    def _use_cases(cls):
        return ["Data exploration/visualisation", "Cluster inspection", "Embedding quality check"]

    @classmethod
    def _assumptions(cls):
        return ["Only for visualisation/exploration — not for downstream ML."]


# ────────────────────────────────────────────────────────────────────
# D-07  UMAP
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class UMAPAlgo(BaseAlgorithm):
    """
    UMAP — Uniform Manifold Approximation and Projection.
    Non-linear dimensionality reduction that preserves both local and global
    structure better than t-SNE. Supports out-of-sample projection.
    """
    name = "UMAP"
    family = "unsupervised"
    task_type = "dimensionality_reduction"
    tags = ["non-linear", "manifold", "global-structure", "fast", "out-of-sample"]
    complexity_train = "O(n^1.14) empirically"
    complexity_predict = "O(n)"

    def _build_model(self):
        from umap import UMAP as _UMAP
        return _UMAP(
            n_components=self._hyperparams.get("n_components", 2),
            n_neighbors=self._hyperparams.get("n_neighbors", 15),
            min_dist=self._hyperparams.get("min_dist", 0.1),
            metric=self._hyperparams.get("metric", "euclidean"),
            random_state=self.random_state,
        )

    def _fit_impl(self, X, y=None):
        self._model.fit(X)

    def _predict_impl(self, X):
        return self._model.transform(X)

    def score(self, X, y=None):
        X_red = self.predict(X)
        return {"n_components_used": int(X_red.shape[1]), "embedding_shape": list(X_red.shape)}

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "n_components", "type": "int", "default": 2, "min": 2, "max": 10},
            {"name": "n_neighbors", "type": "int", "default": 15, "min": 2, "max": 200,
             "description": "Local neighbourhood size. Larger = more global structure."},
            {"name": "min_dist", "type": "float", "default": 0.1, "min": 0.0, "max": 0.99,
             "description": "Minimum distance between points in embedding."},
            {"name": "metric", "type": "select", "default": "euclidean",
             "options": ["euclidean", "cosine", "manhattan", "correlation"]},
        ]

    @classmethod
    def _math_intuition(cls):
        return (
            "Constructs a weighted k-NN graph in high-d space. "
            "Optimises low-d layout to match the fuzzy topological representation "
            "by minimising cross-entropy between high-d and low-d graphs."
        )

    @classmethod
    def _strengths(cls):
        return [
            "Preserves global structure better than t-SNE.",
            "Supports out-of-sample projection (transform()).",
            "Much faster than t-SNE for large datasets.",
            "Supports supervised/semi-supervised mode.",
        ]

    @classmethod
    def _weaknesses(cls):
        return [
            "Stochastic — results vary with seed.",
            "Harder to interpret than PCA.",
            "Several hyperparameters to tune.",
        ]

    @classmethod
    def _use_cases(cls):
        return ["Large-scale visualisation", "Preprocessing for clustering", "Embedding analysis"]

    @classmethod
    def _assumptions(cls):
        return ["Data lies on a low-dimensional manifold.", "Scale features."]


# ────────────────────────────────────────────────────────────────────
# D-08  Isomap
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class IsomapAlgo(BaseAlgorithm):
    """
    Isomap — Isometric Mapping.
    Manifold learning: computes geodesic distances along the data manifold
    (via shortest paths in k-NN graph) then applies MDS to embed.
    """
    name = "Isomap"
    family = "unsupervised"
    task_type = "dimensionality_reduction"
    tags = ["manifold", "geodesic", "mds", "non-linear"]
    complexity_train = "O(n²·k + n²·log n)"
    complexity_predict = "O(n·train_n)"

    def _build_model(self):
        return Isomap(
            n_components=self._hyperparams.get("n_components", 2),
            n_neighbors=self._hyperparams.get("n_neighbors", 5),
            metric=self._hyperparams.get("metric", "minkowski"),
            n_jobs=self._n_jobs(),
        )

    def _fit_impl(self, X, y=None):
        self._model.fit(X)

    def _predict_impl(self, X):
        return self._model.transform(X)

    def score(self, X, y=None):
        X_red = self.predict(X)
        err = float(self._model.reconstruction_error())
        return {"reconstruction_error": err, "n_components_used": int(X_red.shape[1])}

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "n_components", "type": "int", "default": 2, "min": 1, "max": 50},
            {"name": "n_neighbors", "type": "int", "default": 5, "min": 2, "max": 50,
             "description": "Number of neighbours for the k-NN graph."},
            {"name": "metric", "type": "select", "default": "minkowski",
             "options": ["minkowski", "euclidean", "cosine"]},
        ]

    @classmethod
    def _math_intuition(cls):
        return (
            "Build k-NN graph. Compute all-pairs shortest paths (geodesic distances). "
            "Apply Multidimensional Scaling (MDS) to embed geodesic distances in R^k."
        )

    @classmethod
    def _strengths(cls):
        return [
            "Respects intrinsic manifold geometry.",
            "Isometric embedding — preserves geodesic distances.",
            "Supports out-of-sample extension.",
        ]

    @classmethod
    def _weaknesses(cls):
        return [
            "O(n²) — slow for large datasets.",
            "Sensitive to noise and n_neighbors.",
            "Fails on non-convex manifolds (short-circuit paths).",
        ]

    @classmethod
    def _use_cases(cls):
        return ["Curved manifolds (e.g. Swiss roll)", "Robotics pose estimation"]

    @classmethod
    def _assumptions(cls):
        return ["Data lies on a convex manifold.", "Scale features."]


# ────────────────────────────────────────────────────────────────────
# D-09  LLE — Locally Linear Embedding
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class LLEAlgo(BaseAlgorithm):
    """
    Locally Linear Embedding (LLE).
    Manifold learning: reconstructs each point as a linear combination of
    its neighbours, then finds a low-dimensional embedding that preserves
    these local linear relationships.
    """
    name = "LLE"
    family = "unsupervised"
    task_type = "dimensionality_reduction"
    tags = ["manifold", "local-linear", "non-linear", "neighbourhood"]
    complexity_train = "O(d·n·k² + n²·k)"
    complexity_predict = "O(n)"

    def _build_model(self):
        return LocallyLinearEmbedding(
            n_components=self._hyperparams.get("n_components", 2),
            n_neighbors=self._hyperparams.get("n_neighbors", 5),
            method=self._hyperparams.get("method", "standard"),
            reg=self._hyperparams.get("reg", 1e-3),
            random_state=self.random_state,
            n_jobs=self._n_jobs(),
        )

    def _fit_impl(self, X, y=None):
        self._model.fit(X)

    def _predict_impl(self, X):
        return self._model.transform(X)

    def score(self, X, y=None):
        X_red = self.predict(X)
        err = float(self._model.reconstruction_error_)
        return {"reconstruction_error": err, "n_components_used": int(X_red.shape[1])}

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "n_components", "type": "int", "default": 2, "min": 1, "max": 50},
            {"name": "n_neighbors", "type": "int", "default": 5, "min": 2, "max": 50},
            {"name": "method", "type": "select", "default": "standard",
             "options": ["standard", "modified", "hessian", "ltsa"],
             "description": "LLE variant. 'modified' and 'hessian' handle sparse manifolds better."},
            {"name": "reg", "type": "float", "default": 1e-3, "min": 1e-6, "max": 1.0,
             "description": "Regularisation for singular weight matrix."},
        ]

    @classmethod
    def _math_intuition(cls):
        return (
            "Step 1: Find weights Wᵢⱼ that best reconstruct xᵢ from its neighbours: "
            "min Σᵢ ‖xᵢ - ΣⱼWᵢⱼxⱼ‖². "
            "Step 2: Find embedding Y that preserves W: min Σᵢ ‖yᵢ - ΣⱼWᵢⱼyⱼ‖²."
        )

    @classmethod
    def _strengths(cls):
        return [
            "Captures local non-linear structure.",
            "Computationally efficient (sparse eigendecomposition).",
            "Supports out-of-sample extension.",
        ]

    @classmethod
    def _weaknesses(cls):
        return [
            "Sensitive to n_neighbors.",
            "Fails on non-uniformly sampled manifolds.",
            "Standard LLE fails on some manifold shapes.",
        ]

    @classmethod
    def _use_cases(cls):
        return ["Face pose manifold", "Handwriting recognition", "Non-linear structure discovery"]

    @classmethod
    def _assumptions(cls):
        return ["Data lies on a smooth, uniformly sampled manifold.", "Scale features."]


# ────────────────────────────────────────────────────────────────────
# D-10  LDA (as Dimensionality Reducer)
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class LDAReducerAlgo(BaseAlgorithm):
    """
    Linear Discriminant Analysis — Dimensionality Reduction mode.
    Uses class labels to find projections that maximally separate classes.
    Supervised dim-reduction — reduces to at most (n_classes - 1) dimensions.
    """
    name = "LDA (Dim Reduction)"
    family = "unsupervised"
    task_type = "dimensionality_reduction"
    tags = ["supervised", "linear", "class-separating", "lda"]
    complexity_train = "O(n·d²)"
    complexity_predict = "O(n·d·k)"

    def _build_model(self):
        return LinearDiscriminantAnalysis(
            n_components=self._hyperparams.get("n_components", 2),
            solver=self._hyperparams.get("solver", "svd"),
        )

    def _fit_impl(self, X, y=None):
        self._model.fit(X, y)

    def _predict_impl(self, X):
        return self._model.transform(X)

    def score(self, X, y=None):
        X_red = self.predict(X)
        return {
            "n_components_used": int(X_red.shape[1]),
            "explained_variance_ratio": self._model.explained_variance_ratio_.tolist()
            if hasattr(self._model, "explained_variance_ratio_") else [],
        }

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "n_components", "type": "int", "default": 2, "min": 1, "max": 50,
             "description": "Number of components (max = n_classes - 1)."},
            {"name": "solver", "type": "select", "default": "svd",
             "options": ["svd", "lsqr", "eigen"]},
        ]

    @classmethod
    def _math_intuition(cls):
        return "Maximise J(w) = wᵀSBw / wᵀSWw. Between-class scatter / within-class scatter ratio."

    @classmethod
    def _strengths(cls):
        return [
            "Supervised — uses class labels for optimal separation.",
            "Guaranteed at most (C-1) components.",
            "Fast and analytical.",
        ]

    @classmethod
    def _weaknesses(cls):
        return [
            "Requires class labels (supervised).",
            "Assumes Gaussian class-conditional distributions.",
            "Linear boundaries only.",
        ]

    @classmethod
    def _use_cases(cls):
        return ["Pre-classification projection", "Visualising class separability"]

    @classmethod
    def _assumptions(cls):
        return ["Gaussian class distributions.", "Equal covariance matrices.", "Scale features."]
