"""
Unsupervised Clustering Algorithms for ShiftLab — 11 algorithms.

All clusterers:
- Implement BaseAlgorithm with fit() / predict() / score().
- predict() returns cluster labels.
- score() returns real clustering metrics (silhouette, Davies-Bouldin, etc.).
- Register themselves via @AlgorithmRegistry.register.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import (
    KMeans, MiniBatchKMeans, DBSCAN, HDBSCAN,
    AgglomerativeClustering, MeanShift, OPTICS,
    SpectralClustering, Birch, AffinityPropagation,
)
from sklearn.mixture import GaussianMixture

from backend.core.base_algorithm import BaseAlgorithm
from backend.core.metrics import clustering_metrics
from backend.core.registry import AlgorithmRegistry


# ────────────────────────────────────────────────────────────────────
# U-01  K-Means
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class KMeansAlgo(BaseAlgorithm):
    """
    K-Means Clustering.
    Partitions n observations into k clusters by iteratively assigning
    each point to its nearest centroid and updating centroids as cluster means.
    Convergence is guaranteed; global optimum is not.
    """
    name = "K-Means"
    family = "unsupervised"
    task_type = "clustering"
    tags = ["centroid", "partitional", "flat", "euclidean", "classic"]
    complexity_train = "O(n·k·d·iter)"
    complexity_predict = "O(n·k·d)"

    def _build_model(self):
        return KMeans(
            n_clusters=self._hyperparams.get("n_clusters", 8),
            init=self._hyperparams.get("init", "k-means++"),
            n_init=self._hyperparams.get("n_init", 10),
            max_iter=self._hyperparams.get("max_iter", 300),
            random_state=self.random_state,
        )

    def _fit_impl(self, X, y=None):
        self._model.fit(X)
        self._labels = self._model.labels_

    def _predict_impl(self, X):
        return self._model.predict(X)

    def score(self, X, y=None):
        labels = self.predict(X)
        return clustering_metrics(X, labels, y)

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "n_clusters", "type": "int", "default": 8, "min": 2, "max": 50,
             "description": "Number of clusters k."},
            {"name": "init", "type": "select", "default": "k-means++",
             "options": ["k-means++", "random"],
             "description": "Centroid initialisation strategy."},
            {"name": "n_init", "type": "int", "default": 10, "min": 1, "max": 30,
             "description": "Number of random initialisations to try."},
            {"name": "max_iter", "type": "int", "default": 300, "min": 50, "max": 1000},
        ]

    @classmethod
    def _math_intuition(cls):
        return (
            "Objective: minimise within-cluster sum of squares (WCSS) = Σₖ Σᵢ∈Cₖ ‖xᵢ - μₖ‖². "
            "E-step assigns points to nearest centroid; M-step recomputes centroids as cluster means."
        )

    @classmethod
    def _assumptions(cls):
        return [
            "Clusters are roughly spherical and of similar size.",
            "Features should be scaled (Euclidean distance is scale-sensitive).",
            "k must be known in advance.",
        ]

    @classmethod
    def _strengths(cls):
        return [
            "Simple and fast — O(n·k·d·iter).",
            "Scales to large datasets.",
            "Centroids are interpretable.",
        ]

    @classmethod
    def _weaknesses(cls):
        return [
            "Sensitive to initialisation and outliers.",
            "Assumes convex, isotropic clusters.",
            "Requires k to be specified.",
            "Fails on non-convex shapes.",
        ]

    @classmethod
    def _use_cases(cls):
        return [
            "Customer segmentation",
            "Image compression via colour quantisation",
            "Document clustering",
            "Vector quantisation",
        ]


# ────────────────────────────────────────────────────────────────────
# U-02  Mini-Batch K-Means
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class MiniBatchKMeansAlgo(BaseAlgorithm):
    """
    Mini-Batch K-Means.
    Approximates K-Means using random mini-batches instead of the full dataset.
    Drastically reduces training time on large data with small quality trade-off.
    """
    name = "Mini-Batch K-Means"
    family = "unsupervised"
    task_type = "clustering"
    tags = ["centroid", "partitional", "scalable", "online", "approximate"]
    complexity_train = "O(n·k·d·iter / batch_size)"
    complexity_predict = "O(n·k·d)"

    def _build_model(self):
        return MiniBatchKMeans(
            n_clusters=self._hyperparams.get("n_clusters", 8),
            batch_size=self._hyperparams.get("batch_size", 1024),
            n_init=self._hyperparams.get("n_init", 3),
            max_iter=self._hyperparams.get("max_iter", 100),
            random_state=self.random_state,
        )

    def _fit_impl(self, X, y=None):
        self._model.fit(X)

    def _predict_impl(self, X):
        return self._model.predict(X)

    def score(self, X, y=None):
        return clustering_metrics(X, self.predict(X), y)

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "n_clusters", "type": "int", "default": 8, "min": 2, "max": 50},
            {"name": "batch_size", "type": "int", "default": 1024, "min": 32, "max": 10000,
             "description": "Size of mini-batches."},
            {"name": "n_init", "type": "int", "default": 3, "min": 1, "max": 20},
            {"name": "max_iter", "type": "int", "default": 100, "min": 10, "max": 500},
        ]

    @classmethod
    def _math_intuition(cls):
        return (
            "Each iteration samples a mini-batch B, assigns to nearest centroid, "
            "then updates centroids with a per-centroid learning rate ηₜ = 1/vₜ "
            "where vₜ is the number of points assigned to that centroid so far."
        )

    @classmethod
    def _strengths(cls):
        return ["Much faster than K-Means on large n.", "Low memory.", "Online-friendly."]

    @classmethod
    def _weaknesses(cls):
        return ["Slightly less accurate than full K-Means.", "Still requires k."]

    @classmethod
    def _use_cases(cls):
        return ["Large-scale clustering", "Streaming data", "Real-time segmentation"]

    @classmethod
    def _assumptions(cls):
        return ["Scale features.", "Spherical clusters of similar size."]


# ────────────────────────────────────────────────────────────────────
# U-03  DBSCAN
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class DBSCANAlgo(BaseAlgorithm):
    """
    DBSCAN — Density-Based Spatial Clustering of Applications with Noise.
    Groups together points that are closely packed and marks low-density outliers as noise (-1).
    Does not require specifying k.
    """
    name = "DBSCAN"
    family = "unsupervised"
    task_type = "clustering"
    tags = ["density", "noise-robust", "arbitrary-shape", "no-k"]
    complexity_train = "O(n log n) with index, O(n²) naive"
    complexity_predict = "O(n·d) — predict assigns to nearest cluster or noise"

    def _build_model(self):
        return DBSCAN(
            eps=self._hyperparams.get("eps", 0.5),
            min_samples=self._hyperparams.get("min_samples", 5),
            metric=self._hyperparams.get("metric", "euclidean"),
            algorithm=self._hyperparams.get("algorithm", "auto"),
        )

    def _fit_impl(self, X, y=None):
        self._model.fit(X)
        self._train_labels = self._model.labels_
        self._train_X = X

    def _predict_impl(self, X):
        # DBSCAN doesn't have a native predict — assign to nearest core point's cluster
        from sklearn.metrics.pairwise import euclidean_distances
        if not hasattr(self, "_train_X"):
            raise RuntimeError("fit() must be called before predict()")
        dists = euclidean_distances(X, self._train_X)
        nearest_idx = np.argmin(dists, axis=1)
        return self._train_labels[nearest_idx]

    def score(self, X, y=None):
        labels = self._train_labels if hasattr(self, "_train_labels") else self.predict(X)
        return clustering_metrics(X if not hasattr(self, "_train_X") else self._train_X, labels, y)

    def fit(self, X, y=None):
        import time
        self._model = self._build_model()
        t0 = time.perf_counter()
        self._fit_impl(X, y)
        self._train_time_s = time.perf_counter() - t0
        self._is_fitted = True
        return self

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "eps", "type": "float", "default": 0.5, "min": 0.01, "max": 10.0,
             "description": "Maximum distance between two points to be considered neighbours."},
            {"name": "min_samples", "type": "int", "default": 5, "min": 1, "max": 50,
             "description": "Minimum number of points to form a dense region (core point)."},
            {"name": "metric", "type": "select", "default": "euclidean",
             "options": ["euclidean", "manhattan", "cosine"]},
            {"name": "algorithm", "type": "select", "default": "auto",
             "options": ["auto", "ball_tree", "kd_tree", "brute"]},
        ]

    @classmethod
    def _math_intuition(cls):
        return (
            "A point p is a core point if |N_ε(p)| ≥ min_samples. "
            "Border points are reachable from a core point. Noise points are neither. "
            "Clusters form as maximal sets of density-connected points."
        )

    @classmethod
    def _strengths(cls):
        return [
            "Discovers arbitrary-shape clusters.",
            "Identifies noise/outliers explicitly.",
            "No need to specify k.",
            "Robust to outliers.",
        ]

    @classmethod
    def _weaknesses(cls):
        return [
            "Sensitive to eps and min_samples.",
            "Struggles with varying-density clusters.",
            "O(n²) without spatial index.",
        ]

    @classmethod
    def _use_cases(cls):
        return [
            "Geospatial clustering",
            "Anomaly detection",
            "Non-convex cluster shapes",
            "Image segmentation",
        ]

    @classmethod
    def _assumptions(cls):
        return ["Clusters have uniform density.", "Scale features."]


# ────────────────────────────────────────────────────────────────────
# U-04  HDBSCAN
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class HDBSCANAlgo(BaseAlgorithm):
    """
    HDBSCAN — Hierarchical Density-Based Spatial Clustering of Applications with Noise.
    Extends DBSCAN by building a hierarchy of cluster densities, making it robust to
    varying-density clusters without requiring eps.
    """
    name = "HDBSCAN"
    family = "unsupervised"
    task_type = "clustering"
    tags = ["density", "hierarchical", "noise-robust", "no-k", "variable-density"]
    complexity_train = "O(n log n)"
    complexity_predict = "O(n)"

    def _build_model(self):
        return HDBSCAN(
            min_cluster_size=self._hyperparams.get("min_cluster_size", 5),
            min_samples=self._hyperparams.get("min_samples", None),
            cluster_selection_epsilon=self._hyperparams.get("cluster_selection_epsilon", 0.0),
            cluster_selection_method=self._hyperparams.get("cluster_selection_method", "eom"),
        )

    def _fit_impl(self, X, y=None):
        self._model.fit(X)
        self._train_labels = self._model.labels_
        self._train_X = X

    def _predict_impl(self, X):
        # HDBSCAN sklearn: use labels from fit for train data; approximate for new
        from sklearn.metrics.pairwise import euclidean_distances
        dists = euclidean_distances(X, self._train_X)
        nearest_idx = np.argmin(dists, axis=1)
        return self._train_labels[nearest_idx]

    def score(self, X, y=None):
        labels = self._train_labels
        return clustering_metrics(self._train_X, labels, y)

    def fit(self, X, y=None):
        import time
        self._model = self._build_model()
        t0 = time.perf_counter()
        self._fit_impl(X, y)
        self._train_time_s = time.perf_counter() - t0
        self._is_fitted = True
        return self

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "min_cluster_size", "type": "int", "default": 5, "min": 2, "max": 100,
             "description": "Minimum size of a cluster."},
            {"name": "min_samples", "type": "int", "default": None, "min": 1, "max": 50,
             "description": "Conservativeness of clustering. Default=min_cluster_size."},
            {"name": "cluster_selection_epsilon", "type": "float", "default": 0.0, "min": 0.0, "max": 2.0,
             "description": "A distance threshold. Clusters below will not be split."},
            {"name": "cluster_selection_method", "type": "select", "default": "eom",
             "options": ["eom", "leaf"],
             "description": "'eom'=excess of mass (default), 'leaf'=smallest stable clusters."},
        ]

    @classmethod
    def _math_intuition(cls):
        return (
            "Builds a minimum spanning tree of the mutual reachability graph, "
            "then extracts a condensed cluster tree and selects clusters via excess-of-mass."
        )

    @classmethod
    def _strengths(cls):
        return [
            "Handles variable-density clusters.",
            "No eps parameter needed.",
            "Soft cluster membership probabilities available.",
            "Outperforms DBSCAN in practice.",
        ]

    @classmethod
    def _weaknesses(cls):
        return [
            "min_cluster_size still required.",
            "Slower than K-Means.",
            "Harder to interpret hierarchy.",
        ]

    @classmethod
    def _use_cases(cls):
        return ["Complex density structures", "Natural language embeddings", "Genomic data"]

    @classmethod
    def _assumptions(cls):
        return ["Scale features.", "Clusters exist at multiple density scales."]


# ────────────────────────────────────────────────────────────────────
# U-05  Agglomerative Clustering
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class AgglomerativeClusteringAlgo(BaseAlgorithm):
    """
    Agglomerative (Hierarchical) Clustering.
    Bottom-up: starts with each point as its own cluster and merges the
    closest pair at each step. Produces a dendrogram.
    """
    name = "Agglomerative Clustering"
    family = "unsupervised"
    task_type = "clustering"
    tags = ["hierarchical", "bottom-up", "dendrogram", "no-scaling"]
    complexity_train = "O(n² log n)"
    complexity_predict = "O(n)"

    def _build_model(self):
        return AgglomerativeClustering(
            n_clusters=self._hyperparams.get("n_clusters", 2),
            linkage=self._hyperparams.get("linkage", "ward"),
            metric=self._hyperparams.get("metric", "euclidean"),
        )

    def _fit_impl(self, X, y=None):
        self._labels = self._model.fit_predict(X)
        self._train_X = X

    def _predict_impl(self, X):
        # AgglomerativeClustering doesn't support predict on new data
        # Assign to nearest cluster centroid from training
        unique_labels = np.unique(self._labels)
        centroids = np.array([
            self._train_X[self._labels == lbl].mean(axis=0)
            for lbl in unique_labels if lbl >= 0
        ])
        dists = np.linalg.norm(X[:, np.newaxis] - centroids[np.newaxis, :], axis=2)
        return unique_labels[np.argmin(dists, axis=1)]

    def score(self, X, y=None):
        return clustering_metrics(self._train_X, self._labels, y)

    def fit(self, X, y=None):
        import time
        self._model = self._build_model()
        t0 = time.perf_counter()
        self._fit_impl(X, y)
        self._train_time_s = time.perf_counter() - t0
        self._is_fitted = True
        return self

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "n_clusters", "type": "int", "default": 2, "min": 2, "max": 50,
             "description": "Number of clusters to find."},
            {"name": "linkage", "type": "select", "default": "ward",
             "options": ["ward", "complete", "average", "single"],
             "description": "Linkage criterion: ward minimises variance, complete uses max distance."},
            {"name": "metric", "type": "select", "default": "euclidean",
             "options": ["euclidean", "manhattan", "cosine"],
             "description": "Distance metric (ward requires euclidean)."},
        ]

    @classmethod
    def _math_intuition(cls):
        return (
            "Ward linkage: merge clusters that minimise increase in total within-cluster variance. "
            "Δ(A,B) = (nA·nB)/(nA+nB) · ‖μA - μB‖²."
        )

    @classmethod
    def _strengths(cls):
        return [
            "No need to specify k upfront (can cut dendrogram post-hoc).",
            "Produces hierarchy of clusters.",
            "Ward linkage handles many practical cases well.",
        ]

    @classmethod
    def _weaknesses(cls):
        return [
            "O(n² log n) — slow for large datasets.",
            "Cannot reassign points once merged.",
            "Sensitive to outliers (single linkage).",
        ]

    @classmethod
    def _use_cases(cls):
        return ["Gene expression analysis", "Document taxonomy", "Market basket analysis"]

    @classmethod
    def _assumptions(cls):
        return ["Ward linkage assumes Euclidean geometry.", "Scale features."]


# ────────────────────────────────────────────────────────────────────
# U-06  Gaussian Mixture Model (GMM / EM)
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class GaussianMixtureAlgo(BaseAlgorithm):
    """
    Gaussian Mixture Model (GMM) via Expectation-Maximisation.
    Models the data as a mixture of k Gaussian distributions.
    Provides soft (probabilistic) cluster assignments.
    """
    name = "Gaussian Mixture Model"
    family = "unsupervised"
    task_type = "clustering"
    tags = ["probabilistic", "soft-assignment", "generative", "em", "bayesian"]
    complexity_train = "O(n·k·d²·iter)"
    complexity_predict = "O(n·k·d²)"

    def _build_model(self):
        return GaussianMixture(
            n_components=self._hyperparams.get("n_components", 3),
            covariance_type=self._hyperparams.get("covariance_type", "full"),
            max_iter=self._hyperparams.get("max_iter", 100),
            n_init=self._hyperparams.get("n_init", 1),
            random_state=self.random_state,
        )

    def _fit_impl(self, X, y=None):
        self._model.fit(X)

    def _predict_impl(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        self._assert_fitted()
        return self._model.predict_proba(X)

    def score(self, X, y=None):
        labels = self.predict(X)
        result = clustering_metrics(X, labels, y)
        result["bic"] = float(self._model.bic(X))
        result["aic"] = float(self._model.aic(X))
        return result

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "n_components", "type": "int", "default": 3, "min": 1, "max": 50,
             "description": "Number of Gaussian components (clusters)."},
            {"name": "covariance_type", "type": "select", "default": "full",
             "options": ["full", "tied", "diag", "spherical"],
             "description": "Shape of covariance matrix per component."},
            {"name": "max_iter", "type": "int", "default": 100, "min": 10, "max": 1000},
            {"name": "n_init", "type": "int", "default": 1, "min": 1, "max": 10,
             "description": "Number of random initialisations."},
        ]

    @classmethod
    def _math_intuition(cls):
        return (
            "p(x) = Σₖ πₖ N(x; μₖ, Σₖ). "
            "E-step: compute responsibilities rᵢₖ = πₖ N(xᵢ; μₖ, Σₖ) / p(xᵢ). "
            "M-step: update πₖ, μₖ, Σₖ from responsibilities. Maximises log-likelihood."
        )

    @classmethod
    def _strengths(cls):
        return [
            "Soft cluster assignments (probabilities).",
            "Flexible covariance shapes.",
            "Model selection via BIC/AIC.",
            "Generative model — can sample new data.",
        ]

    @classmethod
    def _weaknesses(cls):
        return [
            "Requires k (n_components).",
            "Local optima — sensitive to initialisation.",
            "Slow with full covariance on high-d data.",
        ]

    @classmethod
    def _use_cases(cls):
        return ["Density estimation", "Soft clustering", "Anomaly scoring via log-likelihood"]

    @classmethod
    def _assumptions(cls):
        return ["Each cluster is approximately Gaussian.", "Scale features."]


# ────────────────────────────────────────────────────────────────────
# U-07  Mean Shift
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class MeanShiftAlgo(BaseAlgorithm):
    """
    Mean Shift Clustering.
    Non-parametric algorithm that finds cluster centres as modes of the
    kernel density estimate. Does not require specifying k.
    """
    name = "Mean Shift"
    family = "unsupervised"
    task_type = "clustering"
    tags = ["density", "mode-seeking", "non-parametric", "no-k", "kernel"]
    complexity_train = "O(n²·iter)"
    complexity_predict = "O(n·clusters)"

    def _build_model(self):
        bw = self._hyperparams.get("bandwidth", None)
        return MeanShift(
            bandwidth=bw,
            bin_seeding=self._hyperparams.get("bin_seeding", False),
            cluster_all=self._hyperparams.get("cluster_all", True),
            n_jobs=self._n_jobs(),
        )

    def _fit_impl(self, X, y=None):
        self._model.fit(X)

    def _predict_impl(self, X):
        return self._model.predict(X)

    def score(self, X, y=None):
        labels = self.predict(X)
        return clustering_metrics(X, labels, y)

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "bandwidth", "type": "float", "default": None, "min": 0.01, "max": 10.0,
             "description": "Kernel bandwidth. None = estimated from data automatically."},
            {"name": "bin_seeding", "type": "bool", "default": False,
             "description": "Speed up by discretising space into bins as seeds."},
            {"name": "cluster_all", "type": "bool", "default": True,
             "description": "If False, points not in any kernel are labelled -1 (noise)."},
        ]

    @classmethod
    def _math_intuition(cls):
        return (
            "Each point iterates: m(x) = [Σᵢ K(xᵢ-x)·xᵢ] / [Σᵢ K(xᵢ-x)] "
            "until convergence to a mode. K is typically a Gaussian/flat kernel."
        )

    @classmethod
    def _strengths(cls):
        return [
            "No need to specify k.",
            "Robust to outliers.",
            "Finds arbitrarily shaped clusters.",
        ]

    @classmethod
    def _weaknesses(cls):
        return [
            "O(n²) — slow on large data.",
            "Bandwidth selection is critical.",
            "Not suitable for high dimensions.",
        ]

    @classmethod
    def _use_cases(cls):
        return ["Image segmentation", "Object tracking", "Exploratory clustering"]

    @classmethod
    def _assumptions(cls):
        return ["Clusters correspond to modes of the density.", "Scale features."]


# ────────────────────────────────────────────────────────────────────
# U-08  OPTICS
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class OPTICSAlgo(BaseAlgorithm):
    """
    OPTICS — Ordering Points To Identify the Clustering Structure.
    Generalisation of DBSCAN that handles varying density by producing
    a reachability plot from which clusters can be extracted at multiple eps levels.
    """
    name = "OPTICS"
    family = "unsupervised"
    task_type = "clustering"
    tags = ["density", "varying-density", "no-eps", "noise-robust"]
    complexity_train = "O(n log n) with index"
    complexity_predict = "O(n)"

    def _build_model(self):
        return OPTICS(
            min_samples=self._hyperparams.get("min_samples", 5),
            xi=self._hyperparams.get("xi", 0.05),
            min_cluster_size=self._hyperparams.get("min_cluster_size", 0.05),
            metric=self._hyperparams.get("metric", "minkowski"),
            cluster_method=self._hyperparams.get("cluster_method", "xi"),
        )

    def _fit_impl(self, X, y=None):
        self._model.fit(X)
        self._train_labels = self._model.labels_
        self._train_X = X

    def _predict_impl(self, X):
        from sklearn.metrics.pairwise import euclidean_distances
        dists = euclidean_distances(X, self._train_X)
        nearest_idx = np.argmin(dists, axis=1)
        return self._train_labels[nearest_idx]

    def score(self, X, y=None):
        return clustering_metrics(self._train_X, self._train_labels, y)

    def fit(self, X, y=None):
        import time
        self._model = self._build_model()
        t0 = time.perf_counter()
        self._fit_impl(X, y)
        self._train_time_s = time.perf_counter() - t0
        self._is_fitted = True
        return self

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "min_samples", "type": "int", "default": 5, "min": 2, "max": 50},
            {"name": "xi", "type": "float", "default": 0.05, "min": 0.001, "max": 0.5,
             "description": "Minimum steepness to define a cluster boundary (xi method)."},
            {"name": "min_cluster_size", "type": "float", "default": 0.05, "min": 0.01, "max": 0.5,
             "description": "Minimum cluster size as fraction of total points."},
            {"name": "cluster_method", "type": "select", "default": "xi",
             "options": ["xi", "dbscan"]},
        ]

    @classmethod
    def _math_intuition(cls):
        return (
            "Sorts points by reachability distance r(p,o) = max(core_dist(o), dist(p,o)). "
            "The reachability plot's valleys correspond to clusters."
        )

    @classmethod
    def _strengths(cls):
        return [
            "Handles varying-density clusters.",
            "Does not require eps.",
            "Produces a rich reachability structure.",
        ]

    @classmethod
    def _weaknesses(cls):
        return ["Xi and min_cluster_size require tuning.", "Slow on large data."]

    @classmethod
    def _use_cases(cls):
        return ["Geographic data", "Multi-scale cluster discovery"]

    @classmethod
    def _assumptions(cls):
        return ["Scale features.", "Clusters are density-connected regions."]


# ────────────────────────────────────────────────────────────────────
# U-09  Spectral Clustering
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class SpectralClusteringAlgo(BaseAlgorithm):
    """
    Spectral Clustering.
    Embeds data into the low-dimensional eigenspace of the graph Laplacian,
    then applies K-Means in that space. Effective for non-convex clusters.
    """
    name = "Spectral Clustering"
    family = "unsupervised"
    task_type = "clustering"
    tags = ["graph", "laplacian", "non-convex", "manifold"]
    complexity_train = "O(n³) eigen decomp + O(n·k)"
    complexity_predict = "O(n)"

    def _build_model(self):
        return SpectralClustering(
            n_clusters=self._hyperparams.get("n_clusters", 3),
            affinity=self._hyperparams.get("affinity", "rbf"),
            gamma=self._hyperparams.get("gamma", 1.0),
            n_init=self._hyperparams.get("n_init", 10),
            assign_labels=self._hyperparams.get("assign_labels", "kmeans"),
            random_state=self.random_state,
            n_jobs=self._n_jobs(),
        )

    def _fit_impl(self, X, y=None):
        self._labels = self._model.fit_predict(X)
        self._train_X = X

    def _predict_impl(self, X):
        # Spectral clustering has no native predict
        unique_labels = np.unique(self._labels[self._labels >= 0])
        centroids = np.array([
            self._train_X[self._labels == lbl].mean(axis=0)
            for lbl in unique_labels
        ])
        dists = np.linalg.norm(X[:, np.newaxis] - centroids[np.newaxis, :], axis=2)
        return unique_labels[np.argmin(dists, axis=1)]

    def score(self, X, y=None):
        return clustering_metrics(self._train_X, self._labels, y)

    def fit(self, X, y=None):
        import time
        self._model = self._build_model()
        t0 = time.perf_counter()
        self._fit_impl(X, y)
        self._train_time_s = time.perf_counter() - t0
        self._is_fitted = True
        return self

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "n_clusters", "type": "int", "default": 3, "min": 2, "max": 50},
            {"name": "affinity", "type": "select", "default": "rbf",
             "options": ["rbf", "nearest_neighbors", "precomputed"],
             "description": "How to construct the affinity matrix."},
            {"name": "gamma", "type": "float", "default": 1.0, "min": 0.001, "max": 100.0,
             "description": "Kernel coefficient for rbf affinity."},
            {"name": "assign_labels", "type": "select", "default": "kmeans",
             "options": ["kmeans", "discretize", "cluster_qr"]},
        ]

    @classmethod
    def _math_intuition(cls):
        return (
            "Construct affinity W. Compute normalised Laplacian L = I - D^{-1/2}WD^{-1/2}. "
            "Embed data using k smallest eigenvectors of L. Apply K-Means in embedded space."
        )

    @classmethod
    def _strengths(cls):
        return [
            "Handles non-convex clusters.",
            "Works on graph-structured data.",
            "Theoretically grounded in graph partitioning.",
        ]

    @classmethod
    def _weaknesses(cls):
        return [
            "O(n³) eigendecomposition — slow for large n.",
            "Requires k.",
            "Sensitive to affinity construction.",
        ]

    @classmethod
    def _use_cases(cls):
        return ["Image segmentation", "Community detection in graphs", "Non-linearly separable clusters"]

    @classmethod
    def _assumptions(cls):
        return ["Data has a meaningful graph structure.", "Scale features."]


# ────────────────────────────────────────────────────────────────────
# U-10  BIRCH
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class BIRCHAlgo(BaseAlgorithm):
    """
    BIRCH — Balanced Iterative Reducing and Clustering using Hierarchies.
    Memory-efficient clustering that incrementally builds a CF-Tree (Clustering Feature Tree).
    Designed for very large datasets.
    """
    name = "BIRCH"
    family = "unsupervised"
    task_type = "clustering"
    tags = ["hierarchical", "incremental", "scalable", "large-data", "memory-efficient"]
    complexity_train = "O(n)"
    complexity_predict = "O(n)"

    def _build_model(self):
        return Birch(
            n_clusters=self._hyperparams.get("n_clusters", 3),
            threshold=self._hyperparams.get("threshold", 0.5),
            branching_factor=self._hyperparams.get("branching_factor", 50),
        )

    def _fit_impl(self, X, y=None):
        self._model.fit(X)

    def _predict_impl(self, X):
        return self._model.predict(X)

    def score(self, X, y=None):
        labels = self.predict(X)
        return clustering_metrics(X, labels, y)

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "n_clusters", "type": "int", "default": 3, "min": 2, "max": 100,
             "description": "Number of final clusters. None returns subcluster labels."},
            {"name": "threshold", "type": "float", "default": 0.5, "min": 0.01, "max": 5.0,
             "description": "Max radius of subcluster. Controls CF-Tree density."},
            {"name": "branching_factor", "type": "int", "default": 50, "min": 10, "max": 200,
             "description": "Max subclusters per CF-Tree node."},
        ]

    @classmethod
    def _math_intuition(cls):
        return (
            "A CF entry stores (N, LS, SS): count, linear sum, squared sum of a cluster. "
            "Each new point either joins an existing subcluster or creates a new leaf entry."
        )

    @classmethod
    def _strengths(cls):
        return [
            "O(n) — very fast on large datasets.",
            "Memory-efficient CF-Tree.",
            "Online/incremental learning.",
        ]

    @classmethod
    def _weaknesses(cls):
        return [
            "Assumes spherical clusters.",
            "Threshold must be tuned.",
            "Not suitable for high-dimensional sparse data.",
        ]

    @classmethod
    def _use_cases(cls):
        return ["Very large tabular datasets", "Streaming clustering", "Pre-clustering step"]

    @classmethod
    def _assumptions(cls):
        return ["Euclidean distance.", "Scale features.", "Approximately spherical clusters."]


# ────────────────────────────────────────────────────────────────────
# U-11  Affinity Propagation
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class AffinityPropagationAlgo(BaseAlgorithm):
    """
    Affinity Propagation.
    Message-passing algorithm that identifies exemplars (cluster centres) by
    exchanging responsibility and availability messages between data points.
    Does not require specifying k.
    """
    name = "Affinity Propagation"
    family = "unsupervised"
    task_type = "clustering"
    tags = ["message-passing", "exemplar", "no-k", "graph"]
    complexity_train = "O(n²·iter)"
    complexity_predict = "O(n·exemplars)"

    def _build_model(self):
        return AffinityPropagation(
            damping=self._hyperparams.get("damping", 0.5),
            max_iter=self._hyperparams.get("max_iter", 200),
            convergence_iter=self._hyperparams.get("convergence_iter", 15),
            preference=self._hyperparams.get("preference", None),
            random_state=self.random_state,
        )

    def _fit_impl(self, X, y=None):
        self._model.fit(X)

    def _predict_impl(self, X):
        return self._model.predict(X)

    def score(self, X, y=None):
        labels = self.predict(X)
        return clustering_metrics(X, labels, y)

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "damping", "type": "float", "default": 0.5, "min": 0.5, "max": 0.99,
             "description": "Damping factor [0.5, 1). Higher = more stable, slower convergence."},
            {"name": "max_iter", "type": "int", "default": 200, "min": 50, "max": 1000},
            {"name": "preference", "type": "float", "default": None,
             "description": "Points with larger preference become exemplars more easily."},
        ]

    @classmethod
    def _math_intuition(cls):
        return (
            "Responsibility r(i,k): how well k serves as exemplar for i relative to others. "
            "Availability a(i,k): how appropriate for i to choose k as exemplar. "
            "Exemplars = argmax_k r(k,k)+a(k,k)."
        )

    @classmethod
    def _strengths(cls):
        return [
            "Automatically determines number of clusters.",
            "Identified exemplars are actual data points.",
            "Works with precomputed similarity matrices.",
        ]

    @classmethod
    def _weaknesses(cls):
        return [
            "O(n²) memory and time.",
            "Can produce too many clusters if preference is high.",
            "Not suitable for large datasets.",
        ]

    @classmethod
    def _use_cases(cls):
        return ["Representative sentence selection", "Exemplar-based image clustering"]

    @classmethod
    def _assumptions(cls):
        return ["All points can potentially be exemplars.", "Scale features."]
