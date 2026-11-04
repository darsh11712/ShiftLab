"""
Ensemble Learning Algorithms for ShiftLab — Phase 3A.

Covers: Bagging, ExtraTrees, AdaBoost, Voting, and Stacking.
XGBoost, LightGBM, GradientBoosting, and RandomForest are already registered
in the supervised module — this module provides the dedicated ensemble-family view.

All algorithms register with family="ensemble" for clear taxonomy.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import (
    BaggingClassifier, BaggingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    VotingClassifier, VotingRegressor,
    StackingClassifier, StackingRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from backend.core.base_algorithm import BaseAlgorithm
from backend.core.metrics import regression_metrics, classification_metrics
from backend.core.registry import AlgorithmRegistry


# ────────────────────────────────────────────────────────────────────
# E-01  Bagging Classifier
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class BaggingClassifierAlgo(BaseAlgorithm):
    """
    Bagging Classifier (Bootstrap Aggregating).
    Trains multiple base classifiers on random bootstrap samples of the
    training data and aggregates predictions by majority vote.
    Reduces variance without increasing bias — most effective with high-variance
    base learners (e.g. unpruned decision trees).
    """
    name = "Bagging Classifier"
    family = "ensemble"
    task_type = "classification"
    tags = ["bagging", "bootstrap", "variance-reduction", "parallel"]
    complexity_train = "O(n·estimators·base_complexity)"
    complexity_predict = "O(estimators·base_predict)"

    def _build_model(self):
        return BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=self.random_state),
            n_estimators=self._hyperparams.get("n_estimators", 10),
            max_samples=self._hyperparams.get("max_samples", 1.0),
            max_features=self._hyperparams.get("max_features", 1.0),
            bootstrap=self._hyperparams.get("bootstrap", True),
            bootstrap_features=self._hyperparams.get("bootstrap_features", False),
            random_state=self.random_state,
            n_jobs=self._n_jobs(),
        )

    def score(self, X, y=None):
        proba = self.predict_proba(X)
        return classification_metrics(y, self.predict(X), proba)

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "n_estimators", "type": "int", "default": 10, "min": 2, "max": 200,
             "description": "Number of base estimators."},
            {"name": "max_samples", "type": "float", "default": 1.0, "min": 0.1, "max": 1.0,
             "description": "Fraction of samples to draw for each base estimator."},
            {"name": "max_features", "type": "float", "default": 1.0, "min": 0.1, "max": 1.0,
             "description": "Fraction of features to draw for each base estimator."},
            {"name": "bootstrap", "type": "bool", "default": True,
             "description": "Sample with replacement (bootstrap) or without."},
        ]

    @classmethod
    def _math_intuition(cls):
        return (
            "Given training set D, create T bootstrap samples D₁,...,Dₜ. "
            "Train hₜ on each Dₜ. Predict via majority vote: H(x) = majority{h₁(x),...,hₜ(x)}. "
            "Bias–Variance: E[err] ≈ Bias² + Var/T + noise (variance decreases with T)."
        )

    @classmethod
    def _strengths(cls):
        return [
            "Reduces variance of high-variance base models.",
            "Embarrassingly parallel.",
            "Works with any base estimator.",
            "Reduces overfitting without complex tuning.",
        ]

    @classmethod
    def _weaknesses(cls):
        return [
            "Doesn't reduce bias.",
            "Slower than a single model.",
            "Not as accurate as boosting on most benchmarks.",
        ]

    @classmethod
    def _use_cases(cls):
        return ["Unstable learners (decision trees)", "High-variance models", "General classification"]

    @classmethod
    def _assumptions(cls):
        return ["Base learner should have high variance (unpruned trees are ideal)."]


# ────────────────────────────────────────────────────────────────────
# E-02  Bagging Regressor
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class BaggingRegressorAlgo(BaseAlgorithm):
    """
    Bagging Regressor.
    Aggregates predictions of multiple base regressors trained on bootstrap
    samples by averaging. Reduces variance of high-variance regressors.
    """
    name = "Bagging Regressor"
    family = "ensemble"
    task_type = "regression"
    tags = ["bagging", "bootstrap", "variance-reduction", "parallel"]
    complexity_train = "O(n·estimators·base_complexity)"
    complexity_predict = "O(estimators·base_predict)"

    def _build_model(self):
        return BaggingRegressor(
            estimator=DecisionTreeRegressor(random_state=self.random_state),
            n_estimators=self._hyperparams.get("n_estimators", 10),
            max_samples=self._hyperparams.get("max_samples", 1.0),
            max_features=self._hyperparams.get("max_features", 1.0),
            bootstrap=self._hyperparams.get("bootstrap", True),
            random_state=self.random_state,
            n_jobs=self._n_jobs(),
        )

    def score(self, X, y=None):
        return regression_metrics(y, self.predict(X))

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "n_estimators", "type": "int", "default": 10, "min": 2, "max": 200},
            {"name": "max_samples", "type": "float", "default": 1.0, "min": 0.1, "max": 1.0},
            {"name": "max_features", "type": "float", "default": 1.0, "min": 0.1, "max": 1.0},
            {"name": "bootstrap", "type": "bool", "default": True},
        ]

    @classmethod
    def _math_intuition(cls):
        return "ŷ = (1/T) · Σₜ hₜ(x). Averaging T independent estimators: Var[ŷ] ≈ σ²/T."

    @classmethod
    def _strengths(cls):
        return ["Reduces variance.", "Parallel training.", "Works with any base regressor."]

    @classmethod
    def _weaknesses(cls):
        return ["No bias reduction.", "Less powerful than boosting on most datasets."]

    @classmethod
    def _use_cases(cls):
        return ["High-variance regression models", "Decision tree regressors", "Variance control"]

    @classmethod
    def _assumptions(cls):
        return ["Base learner should have high variance."]


# ────────────────────────────────────────────────────────────────────
# E-03  Extra Trees Classifier
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class ExtraTreesClassifierAlgo(BaseAlgorithm):
    """
    Extra Trees Classifier (Extremely Randomised Trees).
    Like Random Forest but split thresholds are chosen randomly
    rather than optimally — further reduces variance and speeds up training.
    """
    name = "Extra Trees Classifier"
    family = "ensemble"
    task_type = "classification"
    tags = ["ensemble", "trees", "random-splits", "fast", "low-variance"]
    complexity_train = "O(n·trees·d·log n) — faster than RF"
    complexity_predict = "O(trees·depth)"

    def _build_model(self):
        return ExtraTreesClassifier(
            n_estimators=self._hyperparams.get("n_estimators", 100),
            max_depth=self._hyperparams.get("max_depth", None),
            min_samples_split=self._hyperparams.get("min_samples_split", 2),
            max_features=self._hyperparams.get("max_features", "sqrt"),
            criterion=self._hyperparams.get("criterion", "gini"),
            random_state=self.random_state,
            n_jobs=self._n_jobs(),
        )

    def score(self, X, y=None):
        proba = self.predict_proba(X)
        return classification_metrics(y, self.predict(X), proba)

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "n_estimators", "type": "int", "default": 100, "min": 10, "max": 500},
            {"name": "max_depth", "type": "int", "default": None, "min": 1, "max": 50},
            {"name": "max_features", "type": "select", "default": "sqrt",
             "options": ["sqrt", "log2", "none"]},
            {"name": "criterion", "type": "select", "default": "gini",
             "options": ["gini", "entropy"]},
        ]

    @classmethod
    def _math_intuition(cls):
        return (
            "Like RF: train T trees on bootstrap samples with random feature subsets. "
            "Difference: instead of finding optimal split threshold, draw it uniformly "
            "at random from the feature range → faster, more randomisation, lower variance."
        )

    @classmethod
    def _strengths(cls):
        return [
            "Faster than Random Forest (no optimal split search).",
            "Lower variance than RF.",
            "Good feature importance.",
            "Works well with many features.",
        ]

    @classmethod
    def _weaknesses(cls):
        return ["Slightly higher bias than RF.", "Less interpretable than single trees."]

    @classmethod
    def _use_cases(cls):
        return ["Large feature sets", "Fast training requirement", "Feature importance ranking"]

    @classmethod
    def _assumptions(cls):
        return ["None (non-parametric)."]


# ────────────────────────────────────────────────────────────────────
# E-04  Extra Trees Regressor
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class ExtraTreesRegressorAlgo(BaseAlgorithm):
    """
    Extra Trees Regressor.
    Extremely randomised trees for regression — random split thresholds
    reduce variance even further than Random Forest.
    """
    name = "Extra Trees Regressor"
    family = "ensemble"
    task_type = "regression"
    tags = ["ensemble", "trees", "random-splits", "fast"]
    complexity_train = "O(n·trees·d·log n)"
    complexity_predict = "O(trees·depth)"

    def _build_model(self):
        return ExtraTreesRegressor(
            n_estimators=self._hyperparams.get("n_estimators", 100),
            max_depth=self._hyperparams.get("max_depth", None),
            min_samples_split=self._hyperparams.get("min_samples_split", 2),
            max_features=self._hyperparams.get("max_features", 1.0),
            random_state=self.random_state,
            n_jobs=self._n_jobs(),
        )

    def score(self, X, y=None):
        return regression_metrics(y, self.predict(X))

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "n_estimators", "type": "int", "default": 100, "min": 10, "max": 500},
            {"name": "max_depth", "type": "int", "default": None, "min": 1, "max": 50},
            {"name": "max_features", "type": "float", "default": 1.0, "min": 0.1, "max": 1.0},
        ]

    @classmethod
    def _math_intuition(cls):
        return "ŷ = mean of T extremely-randomised trees. Random thresholds → lower variance than RF."

    @classmethod
    def _strengths(cls):
        return ["Faster than RF.", "Low variance.", "Feature importance."]

    @classmethod
    def _weaknesses(cls):
        return ["Slightly higher bias than RF."]

    @classmethod
    def _use_cases(cls):
        return ["Fast regression with many features", "Variance-sensitive problems"]

    @classmethod
    def _assumptions(cls):
        return ["None (non-parametric)."]


# ────────────────────────────────────────────────────────────────────
# E-05  AdaBoost Classifier
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class AdaBoostClassifierAlgo(BaseAlgorithm):
    """
    AdaBoost Classifier (Adaptive Boosting).
    Sequentially trains weak learners (decision stumps by default),
    re-weighting misclassified samples at each step. Final prediction
    is a weighted majority vote of all weak learners.
    """
    name = "AdaBoost Classifier"
    family = "ensemble"
    task_type = "classification"
    tags = ["boosting", "sequential", "weighted", "weak-learners"]
    complexity_train = "O(n·T·base_complexity)"
    complexity_predict = "O(T·base_predict)"

    def _build_model(self):
        return AdaBoostClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=self._hyperparams.get("base_max_depth", 1),
                random_state=self.random_state,
            ),
            n_estimators=self._hyperparams.get("n_estimators", 50),
            learning_rate=self._hyperparams.get("learning_rate", 1.0),
            # sklearn 1.9: algorithm param removed — SAMME is the only supported method
            random_state=self.random_state,
        )

    def score(self, X, y=None):
        proba = self.predict_proba(X)
        return classification_metrics(y, self.predict(X), proba)

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "n_estimators", "type": "int", "default": 50, "min": 10, "max": 500,
             "description": "Number of weak learners (boosting rounds)."},
            {"name": "learning_rate", "type": "float", "default": 1.0, "min": 0.01, "max": 2.0,
             "description": "Shrinks contribution of each classifier."},
            {"name": "base_max_depth", "type": "int", "default": 1, "min": 1, "max": 5,
             "description": "Depth of base decision tree (1 = stump)."},
            {"name": "algorithm", "type": "select", "default": "SAMME",
             "options": ["SAMME"],
             "description": "SAMME: discrete AdaBoost."},
        ]

    @classmethod
    def _math_intuition(cls):
        return (
            "Initialise weights wᵢ = 1/n. For t=1..T: "
            "(1) Fit hₜ on weighted data. "
            "(2) Compute error εₜ = Σᵢ wᵢ I(hₜ(xᵢ) ≠ yᵢ). "
            "(3) αₜ = ½ log[(1-εₜ)/εₜ]. "
            "(4) Update: wᵢ ← wᵢ exp(-αₜ yᵢ hₜ(xᵢ)), normalise. "
            "Final: H(x) = sign(Σₜ αₜ hₜ(x))."
        )

    @classmethod
    def _strengths(cls):
        return [
            "Simple and effective boosting.",
            "Reduces both bias and variance.",
            "Works well with weak learners.",
            "Less prone to overfitting than single trees.",
        ]

    @classmethod
    def _weaknesses(cls):
        return [
            "Sensitive to noisy data and outliers (high-weight misclassified points).",
            "Sequential — cannot parallelise.",
            "Slower convergence than GBT on complex tasks.",
        ]

    @classmethod
    def _use_cases(cls):
        return ["Binary/multi-class classification", "Face detection", "Text categorisation"]

    @classmethod
    def _assumptions(cls):
        return ["Base learner is only slightly better than random (weak learner condition)."]


# ────────────────────────────────────────────────────────────────────
# E-06  AdaBoost Regressor
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class AdaBoostRegressorAlgo(BaseAlgorithm):
    """
    AdaBoost Regressor (AdaBoost.R2).
    Sequential boosting for regression: re-weights samples with large
    residuals so subsequent trees focus on harder examples.
    """
    name = "AdaBoost Regressor"
    family = "ensemble"
    task_type = "regression"
    tags = ["boosting", "sequential", "weighted"]
    complexity_train = "O(n·T·base_complexity)"
    complexity_predict = "O(T·base_predict)"

    def _build_model(self):
        return AdaBoostRegressor(
            estimator=DecisionTreeRegressor(
                max_depth=self._hyperparams.get("base_max_depth", 3),
                random_state=self.random_state,
            ),
            n_estimators=self._hyperparams.get("n_estimators", 50),
            learning_rate=self._hyperparams.get("learning_rate", 1.0),
            loss=self._hyperparams.get("loss", "linear"),
            random_state=self.random_state,
        )

    def score(self, X, y=None):
        return regression_metrics(y, self.predict(X))

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "n_estimators", "type": "int", "default": 50, "min": 10, "max": 500},
            {"name": "learning_rate", "type": "float", "default": 1.0, "min": 0.01, "max": 2.0},
            {"name": "base_max_depth", "type": "int", "default": 3, "min": 1, "max": 10},
            {"name": "loss", "type": "select", "default": "linear",
             "options": ["linear", "square", "exponential"],
             "description": "Loss function for weight update."},
        ]

    @classmethod
    def _math_intuition(cls):
        return (
            "AdaBoost.R2: weight update based on normalised loss Lᵢ = |yᵢ - hₜ(xᵢ)| / max_loss. "
            "Larger residuals get higher weights for the next round."
        )

    @classmethod
    def _strengths(cls):
        return ["Reduces bias and variance.", "Focuses on hard samples."]

    @classmethod
    def _weaknesses(cls):
        return ["Sensitive to outliers.", "Sequential training."]

    @classmethod
    def _use_cases(cls):
        return ["Regression with complex patterns", "Baseline boosting comparison"]

    @classmethod
    def _assumptions(cls):
        return ["Base learner performs better than random on the training distribution."]


# ────────────────────────────────────────────────────────────────────
# E-07  Voting Classifier (Hard + Soft)
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class VotingClassifierAlgo(BaseAlgorithm):
    """
    Voting Classifier.
    Combines predictions from multiple diverse classifiers.
    Hard voting: majority vote of predicted labels.
    Soft voting: averages predicted probabilities (generally better).
    """
    name = "Voting Classifier"
    family = "ensemble"
    task_type = "classification"
    tags = ["voting", "heterogeneous", "ensemble", "model-combination"]
    complexity_train = "O(Σ base_complexity)"
    complexity_predict = "O(Σ base_predict)"

    def _build_model(self):
        from sklearn.naive_bayes import GaussianNB
        voting = self._hyperparams.get("voting", "soft")
        estimators = [
            ("lr", LogisticRegression(max_iter=500, random_state=self.random_state)),
            ("knn", KNeighborsClassifier(n_neighbors=5)),
            ("gnb", GaussianNB()),
        ]
        return VotingClassifier(
            estimators=estimators,
            voting=voting,
            n_jobs=self._n_jobs(),
        )

    def score(self, X, y=None):
        proba = self.predict_proba(X) if self._hyperparams.get("voting", "soft") == "soft" else None
        return classification_metrics(y, self.predict(X), proba)

    def predict_proba(self, X):
        self._assert_fitted()
        if self._hyperparams.get("voting", "soft") == "soft":
            return self._model.predict_proba(X)
        return None

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "voting", "type": "select", "default": "soft",
             "options": ["hard", "soft"],
             "description": "'soft' averages probabilities (better). 'hard' uses majority vote."},
        ]

    @classmethod
    def _math_intuition(cls):
        return (
            "Hard: ŷ = argmax_c Σᵢ I(hᵢ(x)=c). "
            "Soft: ŷ = argmax_c (1/k) Σᵢ pᵢ(y=c|x). "
            "Soft voting leverages calibrated probability estimates."
        )

    @classmethod
    def _strengths(cls):
        return [
            "Combines diverse models — reduces error if models have uncorrelated errors.",
            "Soft voting outperforms hard when probabilities are well-calibrated.",
            "Simple and interpretable ensemble strategy.",
        ]

    @classmethod
    def _weaknesses(cls):
        return [
            "Fixed base estimators (LR + KNN + GNB in this implementation).",
            "Does not learn how to weight classifiers.",
            "All estimators must support predict_proba for soft voting.",
        ]

    @classmethod
    def _use_cases(cls):
        return ["Combining diverse models", "Robust predictions", "Competition ML (model blending)"]

    @classmethod
    def _assumptions(cls):
        return ["Base classifiers make independent errors.", "All classifiers trained on same features."]


# ────────────────────────────────────────────────────────────────────
# E-08  Voting Regressor
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class VotingRegressorAlgo(BaseAlgorithm):
    """
    Voting Regressor.
    Averages predictions from multiple diverse regressors.
    Combines a Ridge, KNN, and decision tree regressor by default.
    """
    name = "Voting Regressor"
    family = "ensemble"
    task_type = "regression"
    tags = ["voting", "heterogeneous", "ensemble", "averaging"]
    complexity_train = "O(Σ base_complexity)"
    complexity_predict = "O(Σ base_predict)"

    def _build_model(self):
        estimators = [
            ("ridge", Ridge(alpha=1.0)),
            ("knn", KNeighborsRegressor(n_neighbors=5)),
            ("tree", DecisionTreeRegressor(max_depth=5, random_state=self.random_state)),
        ]
        return VotingRegressor(estimators=estimators, n_jobs=self._n_jobs())

    def score(self, X, y=None):
        return regression_metrics(y, self.predict(X))

    @classmethod
    def hyperparameter_schema(cls):
        return []  # Ensemble composition is fixed for this implementation

    @classmethod
    def _math_intuition(cls):
        return "ŷ = (1/k) Σᵢ hᵢ(x). Averaging reduces variance if errors are uncorrelated."

    @classmethod
    def _strengths(cls):
        return ["Simple.", "Robust.", "Reduces variance across diverse models."]

    @classmethod
    def _weaknesses(cls):
        return ["No learned weighting.", "Fixed base estimators.", "Less powerful than stacking."]

    @classmethod
    def _use_cases(cls):
        return ["Model averaging", "Robust baseline", "Hedging predictions"]

    @classmethod
    def _assumptions(cls):
        return ["Base regressors make uncorrelated errors on the same data."]


# ────────────────────────────────────────────────────────────────────
# E-09  Stacking Classifier
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class StackingClassifierAlgo(BaseAlgorithm):
    """
    Stacking Classifier (Stacked Generalisation).
    Level-0: trains diverse base classifiers using cross-validation to
    generate out-of-fold predictions (meta-features).
    Level-1: a meta-learner is trained on these meta-features to make
    the final prediction. Learns optimal weighting of base models.
    """
    name = "Stacking Classifier"
    family = "ensemble"
    task_type = "classification"
    tags = ["stacking", "meta-learning", "generalisation", "blending"]
    complexity_train = "O(cv_folds · Σ base_complexity + meta_complexity)"
    complexity_predict = "O(Σ base_predict + meta_predict)"

    def _build_model(self):
        from sklearn.naive_bayes import GaussianNB
        base_estimators = [
            ("lr", LogisticRegression(max_iter=300, random_state=self.random_state)),
            ("gnb", GaussianNB()),
            ("knn", KNeighborsClassifier(n_neighbors=5)),
        ]
        final_estimator = LogisticRegression(max_iter=300, random_state=self.random_state)
        return StackingClassifier(
            estimators=base_estimators,
            final_estimator=final_estimator,
            cv=self._hyperparams.get("cv", 5),
            stack_method=self._hyperparams.get("stack_method", "predict_proba"),
            n_jobs=self._n_jobs(),
        )

    def score(self, X, y=None):
        proba = self.predict_proba(X)
        return classification_metrics(y, self.predict(X), proba)

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "cv", "type": "int", "default": 5, "min": 2, "max": 10,
             "description": "Cross-validation folds for generating meta-features."},
            {"name": "stack_method", "type": "select", "default": "predict_proba",
             "options": ["predict_proba", "decision_function", "predict"],
             "description": "Method used to generate meta-features from base estimators."},
        ]

    @classmethod
    def _math_intuition(cls):
        return (
            "Level-0: For each fold f, train {hₜ} on D\\Df. Generate ẑᵢ = hₜ(xᵢ) for i∈Df. "
            "Stack meta-features Z = [ẑ₁,...,ẑₜ]. "
            "Level-1: fit meta-learner g on (Z, y). Final: H(x) = g(h₁(x),...,hₜ(x))."
        )

    @classmethod
    def _strengths(cls):
        return [
            "Learns optimal combination of base models.",
            "More powerful than Voting (learned weights).",
            "Reduces bias and variance simultaneously.",
        ]

    @classmethod
    def _weaknesses(cls):
        return [
            "Slower to train (cross-validation required).",
            "Risk of overfitting if not careful.",
            "More complex pipeline.",
        ]

    @classmethod
    def _use_cases(cls):
        return ["Competition ML (top performance)", "Robust generalisation", "Heterogeneous model fusion"]

    @classmethod
    def _assumptions(cls):
        return ["Base models have diverse error structures.", "Enough data for cross-validation."]


# ────────────────────────────────────────────────────────────────────
# E-10  Stacking Regressor
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class StackingRegressorAlgo(BaseAlgorithm):
    """
    Stacking Regressor.
    Trains diverse base regressors via cross-validation to produce
    out-of-fold predictions, then trains a meta-learner (Ridge by default)
    to combine them into the final prediction.
    """
    name = "Stacking Regressor"
    family = "ensemble"
    task_type = "regression"
    tags = ["stacking", "meta-learning", "generalisation", "blending"]
    complexity_train = "O(cv_folds · Σ base_complexity + meta_complexity)"
    complexity_predict = "O(Σ base_predict + meta_predict)"

    def _build_model(self):
        base_estimators = [
            ("ridge", Ridge(alpha=1.0)),
            ("knn", KNeighborsRegressor(n_neighbors=5)),
            ("tree", DecisionTreeRegressor(max_depth=4, random_state=self.random_state)),
        ]
        final_estimator = Ridge(alpha=1.0)
        return StackingRegressor(
            estimators=base_estimators,
            final_estimator=final_estimator,
            cv=self._hyperparams.get("cv", 5),
            n_jobs=self._n_jobs(),
        )

    def score(self, X, y=None):
        return regression_metrics(y, self.predict(X))

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "cv", "type": "int", "default": 5, "min": 2, "max": 10,
             "description": "Cross-validation folds for meta-feature generation."},
        ]

    @classmethod
    def _math_intuition(cls):
        return "Meta-learner (Ridge) trained on out-of-fold predictions [h₁(x),...,hₜ(x)] as features."

    @classmethod
    def _strengths(cls):
        return ["Learns optimal combination.", "Diverse base models reduce correlated errors."]

    @classmethod
    def _weaknesses(cls):
        return ["Slow (CV-based training).", "Risk of overfitting on small datasets."]

    @classmethod
    def _use_cases(cls):
        return ["High-performance regression", "Competition", "Multi-model fusion"]

    @classmethod
    def _assumptions(cls):
        return ["Enough data for k-fold CV.", "Base models have diverse prediction patterns."]
