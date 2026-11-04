"""
Supervised Regression Algorithms for ShiftLab.
All 18 regression algorithms in one file for Phase 1.
Each class:
  - Wraps the canonical sklearn / xgboost / lightgbm estimator.
  - Implements BaseAlgorithm interface.
  - Registers itself with @registry.register.
  - Returns real metrics from self.score().
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    BayesianRidge, HuberRegressor, QuantileRegressor,
    SGDRegressor,
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from backend.core.base_algorithm import BaseAlgorithm
from backend.core.metrics import regression_metrics
from backend.core.registry import AlgorithmRegistry

# Lazy imports for optional heavy libs
def _xgb_regressor():
    from xgboost import XGBRegressor
    return XGBRegressor

def _lgb_regressor():
    from lightgbm import LGBMRegressor
    return LGBMRegressor


# ────────────────────────────────────────────────────────────────────
# R-01  Linear Regression
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class LinearRegressionAlgo(BaseAlgorithm):
    """
    Ordinary Least Squares Linear Regression.
    Fits a hyperplane by minimising the sum of squared residuals.
    """
    name = "Linear Regression"
    family = "supervised"
    task_type = "regression"
    tags = ["linear", "parametric", "interpretable", "fast"]
    complexity_train = "O(n·d²)"
    complexity_predict = "O(d)"

    def _build_model(self):
        return LinearRegression(
            fit_intercept=self._hyperparams.get("fit_intercept", True),
        )

    def score(self, X, y=None):
        return regression_metrics(y, self.predict(X))

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "fit_intercept", "type": "bool", "default": True,
             "description": "Whether to calculate the intercept term."},
        ]

    @classmethod
    def _assumptions(cls):
        return [
            "Linear relationship between features and target.",
            "Homoscedasticity (constant error variance).",
            "Independence of residuals.",
            "No severe multicollinearity.",
        ]

    @classmethod
    def _strengths(cls):
        return ["Highly interpretable coefficients.", "Very fast to train.", "No hyperparameters by default.", "Extrapolates linearly."]

    @classmethod
    def _weaknesses(cls):
        return ["Fails on non-linear relationships.", "Sensitive to outliers.", "Assumes linearity."]

    @classmethod
    def _use_cases(cls):
        return ["Baseline model", "Economics", "Simple trend estimation"]

    @classmethod
    def _math_intuition(cls):
        return "Minimise RSS = Σ(yᵢ - ŷᵢ)². Closed-form solution: β = (XᵀX)⁻¹Xᵀy."


# ────────────────────────────────────────────────────────────────────
# R-02  Ridge Regression
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class RidgeRegressionAlgo(BaseAlgorithm):
    """
    Ridge Regression (L2 regularisation).
    Adds a penalty λ·‖β‖² to shrink coefficients and reduce overfitting.
    """
    name = "Ridge Regression"
    family = "supervised"
    task_type = "regression"
    tags = ["linear", "regularisation", "l2", "parametric"]
    complexity_train = "O(n·d²)"
    complexity_predict = "O(d)"

    def _build_model(self):
        return Ridge(
            alpha=self._hyperparams.get("alpha", 1.0),
            fit_intercept=self._hyperparams.get("fit_intercept", True),
            solver=self._hyperparams.get("solver", "auto"),
        )

    def score(self, X, y=None):
        return regression_metrics(y, self.predict(X))

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "alpha", "type": "float", "default": 1.0, "min": 0.0001, "max": 1000.0,
             "description": "Regularisation strength. Larger = more shrinkage."},
            {"name": "solver", "type": "select", "default": "auto",
             "options": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
             "description": "Solver algorithm."},
        ]

    @classmethod
    def _math_intuition(cls):
        return "Minimise RSS + λ·‖β‖². Solution: β = (XᵀX + λI)⁻¹Xᵀy. Never singular."

    @classmethod
    def _assumptions(cls):
        return ["Linear relationship.", "Features should be scaled."]

    @classmethod
    def _strengths(cls):
        return ["Handles multicollinearity well.", "Stable with many features.", "Analytical solution."]

    @classmethod
    def _weaknesses(cls):
        return ["Does not perform feature selection (keeps all non-zero).", "Hyperparameter alpha requires tuning."]

    @classmethod
    def _use_cases(cls):
        return ["Multicollinear features", "High-dimensional data", "Genomics"]


# ────────────────────────────────────────────────────────────────────
# R-03  Lasso Regression
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class LassoRegressionAlgo(BaseAlgorithm):
    """
    Lasso Regression (L1 regularisation).
    Penalty λ·‖β‖₁ drives some coefficients exactly to zero — automatic feature selection.
    """
    name = "Lasso Regression"
    family = "supervised"
    task_type = "regression"
    tags = ["linear", "regularisation", "l1", "sparse", "feature-selection"]
    complexity_train = "O(n·d)"
    complexity_predict = "O(d)"

    def _build_model(self):
        return Lasso(
            alpha=self._hyperparams.get("alpha", 1.0),
            max_iter=self._hyperparams.get("max_iter", 1000),
            fit_intercept=self._hyperparams.get("fit_intercept", True),
        )

    def score(self, X, y=None):
        return regression_metrics(y, self.predict(X))

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "alpha", "type": "float", "default": 1.0, "min": 0.0001, "max": 1000.0,
             "description": "Regularisation strength."},
            {"name": "max_iter", "type": "int", "default": 1000, "min": 100, "max": 10000,
             "description": "Maximum number of iterations for coordinate descent."},
        ]

    @classmethod
    def _math_intuition(cls):
        return "Minimise RSS + λ·Σ|βⱼ|. Solved via coordinate descent. Sparsity emerges from the L1 geometry."

    @classmethod
    def _strengths(cls):
        return ["Automatic feature selection.", "Produces sparse models.", "Interpretable."]

    @classmethod
    def _weaknesses(cls):
        return ["With correlated features, arbitrarily picks one.", "Can be unstable."]

    @classmethod
    def _use_cases(cls):
        return ["Feature selection", "Sparse signal recovery", "Genomics"]

    @classmethod
    def _assumptions(cls):
        return ["Linear relationship.", "Features should be scaled for fair penalisation."]


# ────────────────────────────────────────────────────────────────────
# R-04  ElasticNet
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class ElasticNetAlgo(BaseAlgorithm):
    """
    ElasticNet Regression.
    Combines L1 and L2 penalties: α·l1_ratio·‖β‖₁ + α·(1-l1_ratio)·‖β‖₂².
    """
    name = "ElasticNet"
    family = "supervised"
    task_type = "regression"
    tags = ["linear", "regularisation", "l1", "l2", "sparse"]
    complexity_train = "O(n·d)"
    complexity_predict = "O(d)"

    def _build_model(self):
        return ElasticNet(
            alpha=self._hyperparams.get("alpha", 1.0),
            l1_ratio=self._hyperparams.get("l1_ratio", 0.5),
            max_iter=self._hyperparams.get("max_iter", 1000),
        )

    def score(self, X, y=None):
        return regression_metrics(y, self.predict(X))

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "alpha", "type": "float", "default": 1.0, "min": 0.0001, "max": 1000.0,
             "description": "Overall regularisation strength."},
            {"name": "l1_ratio", "type": "float", "default": 0.5, "min": 0.0, "max": 1.0,
             "description": "Mix ratio: 0=Ridge, 1=Lasso, (0,1)=ElasticNet."},
            {"name": "max_iter", "type": "int", "default": 1000, "min": 100, "max": 10000,
             "description": "Max coordinate descent iterations."},
        ]

    @classmethod
    def _math_intuition(cls):
        return "Minimise RSS + α·l1·|β| + α·(1-l1)·β². Balances sparsity and stability."

    @classmethod
    def _strengths(cls):
        return ["Sparsity + stability.", "Handles correlated features better than Lasso."]

    @classmethod
    def _weaknesses(cls):
        return ["Two hyperparameters to tune.", "Slower than pure Ridge/Lasso."]

    @classmethod
    def _use_cases(cls):
        return ["Correlated features", "Gene expression", "Text regression"]

    @classmethod
    def _assumptions(cls):
        return ["Linear relationship.", "Scale your features."]


# ────────────────────────────────────────────────────────────────────
# R-05  Polynomial Regression
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class PolynomialRegressionAlgo(BaseAlgorithm):
    """
    Polynomial Regression.
    Expands features to degree d then fits OLS — captures non-linear trends.
    """
    name = "Polynomial Regression"
    family = "supervised"
    task_type = "regression"
    tags = ["linear", "non-linear", "feature-engineering"]
    complexity_train = "O(n·C(d+p,d)²)"
    complexity_predict = "O(C(d+p,d))"

    def _build_model(self):
        degree = self._hyperparams.get("degree", 2)
        return Pipeline([
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("lr", LinearRegression()),
        ])

    def score(self, X, y=None):
        return regression_metrics(y, self.predict(X))

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "degree", "type": "int", "default": 2, "min": 2, "max": 8,
             "description": "Polynomial degree. Higher = more complex, risk of overfitting."},
        ]

    @classmethod
    def _math_intuition(cls):
        return "Maps x → [x, x², …, xᵈ] then fits OLS. Non-linear in x but linear in parameters."

    @classmethod
    def _strengths(cls):
        return ["Captures polynomial relationships.", "Still interpretable at low degrees."]

    @classmethod
    def _weaknesses(cls):
        return ["Overfits at high degrees.", "Feature explosion with many input dimensions."]

    @classmethod
    def _use_cases(cls):
        return ["Curve fitting", "Physics models", "Signal modelling"]

    @classmethod
    def _assumptions(cls):
        return ["Relationship is polynomial (not arbitrary)."]


# ────────────────────────────────────────────────────────────────────
# R-06  Support Vector Regression
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class SVRAlgo(BaseAlgorithm):
    """
    Support Vector Regression (SVR).
    Finds a tube of width ε and penalises points outside it.
    Kernel trick enables non-linear regression.
    """
    name = "Support Vector Regression"
    family = "supervised"
    task_type = "regression"
    tags = ["svm", "kernel", "non-linear", "robust"]
    complexity_train = "O(n²·d) to O(n³)"
    complexity_predict = "O(n_sv·d)"

    def _build_model(self):
        return SVR(
            kernel=self._hyperparams.get("kernel", "rbf"),
            C=self._hyperparams.get("C", 1.0),
            epsilon=self._hyperparams.get("epsilon", 0.1),
            gamma=self._hyperparams.get("gamma", "scale"),
        )

    def score(self, X, y=None):
        return regression_metrics(y, self.predict(X))

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "kernel", "type": "select", "default": "rbf",
             "options": ["linear", "poly", "rbf", "sigmoid"],
             "description": "Kernel function."},
            {"name": "C", "type": "float", "default": 1.0, "min": 0.001, "max": 1000.0,
             "description": "Regularisation parameter. Larger = less margin tolerance."},
            {"name": "epsilon", "type": "float", "default": 0.1, "min": 0.0, "max": 10.0,
             "description": "Epsilon-tube width. Points inside tube incur no loss."},
            {"name": "gamma", "type": "select", "default": "scale",
             "options": ["scale", "auto"],
             "description": "Kernel coefficient (for RBF/poly/sigmoid)."},
        ]

    @classmethod
    def _math_intuition(cls):
        return "Minimise ½‖w‖² + C·Σξᵢ subject to |yᵢ - f(xᵢ)| ≤ ε + ξᵢ. Dual solved via QP."

    @classmethod
    def _strengths(cls):
        return ["Robust to outliers (ε-insensitive loss).", "Works well in high-dim.", "Flexible kernels."]

    @classmethod
    def _weaknesses(cls):
        return ["Slow on large datasets.", "Requires feature scaling.", "Hard to interpret."]

    @classmethod
    def _use_cases(cls):
        return ["Small-to-medium datasets", "Non-linear regression", "Time-series prediction"]

    @classmethod
    def _assumptions(cls):
        return ["Features should be scaled.", "Kernel choice matters."]


# ────────────────────────────────────────────────────────────────────
# R-07  K-Nearest Neighbours Regression
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class KNNRegressorAlgo(BaseAlgorithm):
    """
    K-Nearest Neighbours Regression.
    Predicts target as the (weighted) average of k nearest training samples.
    """
    name = "KNN Regressor"
    family = "supervised"
    task_type = "regression"
    tags = ["instance-based", "lazy", "non-parametric", "non-linear"]
    complexity_train = "O(1)"
    complexity_predict = "O(n·d)"

    def _build_model(self):
        return KNeighborsRegressor(
            n_neighbors=self._hyperparams.get("n_neighbors", 5),
            weights=self._hyperparams.get("weights", "uniform"),
            algorithm=self._hyperparams.get("algorithm", "auto"),
            metric=self._hyperparams.get("metric", "minkowski"),
        )

    def score(self, X, y=None):
        return regression_metrics(y, self.predict(X))

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "n_neighbors", "type": "int", "default": 5, "min": 1, "max": 50,
             "description": "Number of neighbours k."},
            {"name": "weights", "type": "select", "default": "uniform",
             "options": ["uniform", "distance"],
             "description": "Weight function: uniform or inverse-distance."},
            {"name": "metric", "type": "select", "default": "minkowski",
             "options": ["minkowski", "euclidean", "manhattan", "chebyshev"],
             "description": "Distance metric."},
        ]

    @classmethod
    def _math_intuition(cls):
        return "ŷ = (1/k)·Σ yᵢ for k nearest neighbours. With distance weights: ŷ = Σ(wᵢ·yᵢ)/Σwᵢ."

    @classmethod
    def _strengths(cls):
        return ["No training time.", "Naturally non-linear.", "No assumptions on data distribution."]

    @classmethod
    def _weaknesses(cls):
        return ["Slow prediction on large data.", "Sensitive to irrelevant features.", "High memory."]

    @classmethod
    def _use_cases(cls):
        return ["Small datasets", "Recommendation systems", "Missing value imputation"]

    @classmethod
    def _assumptions(cls):
        return ["Scale features.", "Similar samples have similar targets."]


# ────────────────────────────────────────────────────────────────────
# R-08  Decision Tree Regressor
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class DecisionTreeRegressorAlgo(BaseAlgorithm):
    """
    Decision Tree Regressor.
    Recursively partitions the feature space and predicts leaf mean.
    """
    name = "Decision Tree Regressor"
    family = "supervised"
    task_type = "regression"
    tags = ["tree", "non-linear", "interpretable", "no-scaling"]
    complexity_train = "O(n·d·log n)"
    complexity_predict = "O(depth)"

    def _build_model(self):
        return DecisionTreeRegressor(
            max_depth=self._hyperparams.get("max_depth", None),
            min_samples_split=self._hyperparams.get("min_samples_split", 2),
            min_samples_leaf=self._hyperparams.get("min_samples_leaf", 1),
            criterion=self._hyperparams.get("criterion", "squared_error"),
            random_state=self.random_state,
        )

    def score(self, X, y=None):
        return regression_metrics(y, self.predict(X))

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "max_depth", "type": "int", "default": None, "min": 1, "max": 50,
             "description": "Max tree depth. None = grow until pure leaves."},
            {"name": "min_samples_split", "type": "int", "default": 2, "min": 2, "max": 100,
             "description": "Minimum samples to split an internal node."},
            {"name": "min_samples_leaf", "type": "int", "default": 1, "min": 1, "max": 50,
             "description": "Minimum samples in a leaf node."},
            {"name": "criterion", "type": "select", "default": "squared_error",
             "options": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
             "description": "Split quality criterion."},
        ]

    @classmethod
    def _math_intuition(cls):
        return "Greedy split selection minimising MSE at each node. Prediction = mean(y) in leaf."

    @classmethod
    def _strengths(cls):
        return ["No feature scaling needed.", "Handles mixed types.", "Visualisable.", "Fast."]

    @classmethod
    def _weaknesses(cls):
        return ["High variance (overfits without depth limit).", "Axis-aligned splits only."]

    @classmethod
    def _use_cases(cls):
        return ["Baseline tree model", "Feature importance ranking", "Explainability"]

    @classmethod
    def _assumptions(cls):
        return ["None (non-parametric)."]


# ────────────────────────────────────────────────────────────────────
# R-09  Random Forest Regressor
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class RandomForestRegressorAlgo(BaseAlgorithm):
    """
    Random Forest Regressor.
    Bagging of decision trees with random feature subsets — reduces variance.
    """
    name = "Random Forest Regressor"
    family = "supervised"
    task_type = "regression"
    tags = ["ensemble", "bagging", "non-linear", "robust"]
    complexity_train = "O(n·trees·d·log n)"
    complexity_predict = "O(trees·depth)"

    def _build_model(self):
        return RandomForestRegressor(
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
            {"name": "n_estimators", "type": "int", "default": 100, "min": 10, "max": 500,
             "description": "Number of trees in the forest."},
            {"name": "max_depth", "type": "int", "default": None, "min": 1, "max": 50,
             "description": "Maximum depth of each tree."},
            {"name": "max_features", "type": "float", "default": 1.0, "min": 0.1, "max": 1.0,
             "description": "Fraction of features considered at each split."},
        ]

    @classmethod
    def _math_intuition(cls):
        return "ŷ = (1/T)·Σ hₜ(x) where each hₜ is a tree trained on a bootstrap sample."

    @classmethod
    def _strengths(cls):
        return ["Low variance.", "Feature importance.", "Robust to outliers.", "Handles missing data via imputation."]

    @classmethod
    def _weaknesses(cls):
        return ["Slower than single tree.", "Less interpretable.", "Memory-heavy with many trees."]

    @classmethod
    def _use_cases(cls):
        return ["General-purpose regression", "Feature importance", "Tabular data"]

    @classmethod
    def _assumptions(cls):
        return ["None (non-parametric)."]


# ────────────────────────────────────────────────────────────────────
# R-10  Gradient Boosting Regressor
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class GradientBoostingRegressorAlgo(BaseAlgorithm):
    """
    Gradient Boosting Regressor (sklearn GBT).
    Sequentially fits trees on residuals with gradient descent in function space.
    """
    name = "Gradient Boosting Regressor"
    family = "supervised"
    task_type = "regression"
    tags = ["ensemble", "boosting", "non-linear", "powerful"]
    complexity_train = "O(n·trees·d·log n)"
    complexity_predict = "O(trees·depth)"

    def _build_model(self):
        return GradientBoostingRegressor(
            n_estimators=self._hyperparams.get("n_estimators", 100),
            learning_rate=self._hyperparams.get("learning_rate", 0.1),
            max_depth=self._hyperparams.get("max_depth", 3),
            subsample=self._hyperparams.get("subsample", 1.0),
            loss=self._hyperparams.get("loss", "squared_error"),
            random_state=self.random_state,
        )

    def score(self, X, y=None):
        return regression_metrics(y, self.predict(X))

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "n_estimators", "type": "int", "default": 100, "min": 10, "max": 500},
            {"name": "learning_rate", "type": "float", "default": 0.1, "min": 0.001, "max": 1.0,
             "description": "Shrinkage (step size) for each tree."},
            {"name": "max_depth", "type": "int", "default": 3, "min": 1, "max": 10},
            {"name": "subsample", "type": "float", "default": 1.0, "min": 0.1, "max": 1.0,
             "description": "Fraction of samples per tree (< 1 adds stochasticity)."},
            {"name": "loss", "type": "select", "default": "squared_error",
             "options": ["squared_error", "absolute_error", "huber", "quantile"]},
        ]

    @classmethod
    def _math_intuition(cls):
        return "F_m(x) = F_{m-1}(x) + η·h_m(x) where h_m fits the negative gradient of the loss."

    @classmethod
    def _strengths(cls):
        return ["Very accurate.", "Feature importance.", "Handles mixed data."]

    @classmethod
    def _weaknesses(cls):
        return ["Slow to train sequentially.", "Many hyperparameters.", "Can overfit."]

    @classmethod
    def _use_cases(cls):
        return ["Kaggle competitions", "Tabular data", "Ranking problems"]

    @classmethod
    def _assumptions(cls):
        return ["None (non-parametric)."]


# ────────────────────────────────────────────────────────────────────
# R-11  XGBoost Regressor
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class XGBoostRegressorAlgo(BaseAlgorithm):
    """
    XGBoost Regressor.
    Extreme Gradient Boosting with regularisation, column sampling, and
    approximate tree learning — faster and more regularised than vanilla GBT.
    """
    name = "XGBoost Regressor"
    family = "supervised"
    task_type = "regression"
    tags = ["ensemble", "boosting", "xgboost", "fast", "powerful"]
    complexity_train = "O(n·trees·d)"
    complexity_predict = "O(trees·depth)"

    def _build_model(self):
        XGBRegressor = _xgb_regressor()
        return XGBRegressor(
            n_estimators=self._hyperparams.get("n_estimators", 100),
            learning_rate=self._hyperparams.get("learning_rate", 0.1),
            max_depth=self._hyperparams.get("max_depth", 6),
            subsample=self._hyperparams.get("subsample", 1.0),
            colsample_bytree=self._hyperparams.get("colsample_bytree", 1.0),
            reg_alpha=self._hyperparams.get("reg_alpha", 0.0),
            reg_lambda=self._hyperparams.get("reg_lambda", 1.0),
            random_state=self.random_state,
            verbosity=0,
            n_jobs=self._n_jobs(),
        )

    def score(self, X, y=None):
        return regression_metrics(y, self.predict(X))

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "n_estimators", "type": "int", "default": 100, "min": 10, "max": 500},
            {"name": "learning_rate", "type": "float", "default": 0.1, "min": 0.001, "max": 1.0},
            {"name": "max_depth", "type": "int", "default": 6, "min": 1, "max": 15},
            {"name": "subsample", "type": "float", "default": 1.0, "min": 0.1, "max": 1.0},
            {"name": "colsample_bytree", "type": "float", "default": 1.0, "min": 0.1, "max": 1.0,
             "description": "Column subsampling ratio per tree."},
            {"name": "reg_alpha", "type": "float", "default": 0.0, "min": 0.0, "max": 10.0,
             "description": "L1 regularisation on leaf weights."},
            {"name": "reg_lambda", "type": "float", "default": 1.0, "min": 0.0, "max": 10.0,
             "description": "L2 regularisation on leaf weights."},
        ]

    @classmethod
    def _math_intuition(cls):
        return "Adds regularisation term Ω(f)=γT + ½λΣwⱼ² to the GBT objective. Uses 2nd-order Taylor expansion."

    @classmethod
    def _strengths(cls):
        return ["Best-in-class on tabular data.", "Built-in regularisation.", "Handles NaNs natively.", "Very fast."]

    @classmethod
    def _weaknesses(cls):
        return ["Many hyperparameters.", "Less interpretable than linear models."]

    @classmethod
    def _use_cases(cls):
        return ["Structured/tabular data competitions", "Feature importance at scale"]

    @classmethod
    def _assumptions(cls):
        return ["None (non-parametric)."]


# ────────────────────────────────────────────────────────────────────
# R-12  LightGBM Regressor
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class LightGBMRegressorAlgo(BaseAlgorithm):
    """
    LightGBM Regressor.
    Leaf-wise gradient boosting with histogram-based splits — very fast on large data.
    """
    name = "LightGBM Regressor"
    family = "supervised"
    task_type = "regression"
    tags = ["ensemble", "boosting", "lightgbm", "fast", "large-data"]
    complexity_train = "O(n·trees·bins)"
    complexity_predict = "O(trees·depth)"

    def _build_model(self):
        LGBMRegressor = _lgb_regressor()
        return LGBMRegressor(
            n_estimators=self._hyperparams.get("n_estimators", 100),
            learning_rate=self._hyperparams.get("learning_rate", 0.1),
            max_depth=self._hyperparams.get("max_depth", -1),
            num_leaves=self._hyperparams.get("num_leaves", 31),
            subsample=self._hyperparams.get("subsample", 1.0),
            colsample_bytree=self._hyperparams.get("colsample_bytree", 1.0),
            reg_alpha=self._hyperparams.get("reg_alpha", 0.0),
            reg_lambda=self._hyperparams.get("reg_lambda", 0.0),
            random_state=self.random_state,
            verbosity=-1,
            n_jobs=self._n_jobs(),
        )

    def score(self, X, y=None):
        return regression_metrics(y, self.predict(X))

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "n_estimators", "type": "int", "default": 100, "min": 10, "max": 500},
            {"name": "learning_rate", "type": "float", "default": 0.1, "min": 0.001, "max": 1.0},
            {"name": "num_leaves", "type": "int", "default": 31, "min": 2, "max": 300,
             "description": "Max leaves per tree. Controls complexity (leaf-wise growth)."},
            {"name": "max_depth", "type": "int", "default": -1, "min": -1, "max": 50,
             "description": "-1 = unlimited depth."},
            {"name": "subsample", "type": "float", "default": 1.0, "min": 0.1, "max": 1.0},
            {"name": "colsample_bytree", "type": "float", "default": 1.0, "min": 0.1, "max": 1.0},
        ]

    @classmethod
    def _math_intuition(cls):
        return "Histogram approximation of exact split search + leaf-wise growth reduces O(n) per split to O(bins)."

    @classmethod
    def _strengths(cls):
        return ["Extremely fast.", "Low memory.", "High accuracy.", "Handles categorical natively."]

    @classmethod
    def _weaknesses(cls):
        return ["Leaf-wise can overfit on small data.", "Less tuned defaults than XGBoost for small N."]

    @classmethod
    def _use_cases(cls):
        return ["Large datasets", "Fast iteration", "Production ML pipelines"]

    @classmethod
    def _assumptions(cls):
        return ["None (non-parametric)."]


# ────────────────────────────────────────────────────────────────────
# R-13  Gaussian Process Regression
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class GaussianProcessRegressorAlgo(BaseAlgorithm):
    """
    Gaussian Process Regression (GPR).
    Non-parametric Bayesian model that provides uncertainty estimates alongside predictions.
    """
    name = "Gaussian Process Regressor"
    family = "supervised"
    task_type = "regression"
    tags = ["bayesian", "uncertainty", "non-parametric", "kernel"]
    complexity_train = "O(n³)"
    complexity_predict = "O(n²)"

    def _build_model(self):
        kernel = RBF(
            length_scale=self._hyperparams.get("length_scale", 1.0)
        ) + WhiteKernel(
            noise_level=self._hyperparams.get("noise_level", 0.1)
        )
        return GaussianProcessRegressor(
            kernel=kernel,
            alpha=self._hyperparams.get("alpha", 1e-10),
            n_restarts_optimizer=self._hyperparams.get("n_restarts_optimizer", 0),
            random_state=self.random_state,
            normalize_y=True,
        )

    def score(self, X, y=None):
        return regression_metrics(y, self.predict(X))

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "length_scale", "type": "float", "default": 1.0, "min": 0.01, "max": 100.0,
             "description": "RBF kernel length scale."},
            {"name": "noise_level", "type": "float", "default": 0.1, "min": 1e-5, "max": 10.0,
             "description": "Noise variance in WhiteKernel."},
            {"name": "n_restarts_optimizer", "type": "int", "default": 0, "min": 0, "max": 10,
             "description": "Restarts for kernel hyperparameter optimisation."},
        ]

    @classmethod
    def _math_intuition(cls):
        return "Models f(x) ~ GP(μ, k). Posterior: f*|X,y,x* is Gaussian with computable mean and variance."

    @classmethod
    def _strengths(cls):
        return ["Uncertainty quantification.", "Flexible via kernel choice.", "Works well on small data."]

    @classmethod
    def _weaknesses(cls):
        return ["O(n³) training — unusable for large datasets.", "Kernel choice critical."]

    @classmethod
    def _use_cases(cls):
        return ["Bayesian optimisation", "Small experimental datasets", "Scientific computing"]

    @classmethod
    def _assumptions(cls):
        return ["Observations are noisy evaluations of a GP.", "Stationarity (for RBF kernel)."]


# ────────────────────────────────────────────────────────────────────
# R-14  Bayesian Ridge Regression
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class BayesianRidgeAlgo(BaseAlgorithm):
    """
    Bayesian Ridge Regression.
    Places Gaussian priors on weights and estimates regularisation hyperparameters
    from the data via evidence approximation.
    """
    name = "Bayesian Ridge Regression"
    family = "supervised"
    task_type = "regression"
    tags = ["bayesian", "linear", "regularisation", "uncertainty"]
    complexity_train = "O(n·d²)"
    complexity_predict = "O(d)"

    def _build_model(self):
        return BayesianRidge(
            max_iter=self._hyperparams.get("max_iter", 300),
            tol=self._hyperparams.get("tol", 1e-3),
        )

    def score(self, X, y=None):
        return regression_metrics(y, self.predict(X))

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "max_iter", "type": "int", "default": 300, "min": 50, "max": 1000},
            {"name": "tol", "type": "float", "default": 1e-3, "min": 1e-6, "max": 0.1},
        ]

    @classmethod
    def _math_intuition(cls):
        return "p(w|α)=N(0,α⁻¹I). Hyperparameters α,β estimated by maximising marginal likelihood (evidence)."

    @classmethod
    def _strengths(cls):
        return ["No need to cross-validate regularisation.", "Gives prediction uncertainty.", "Robust."]

    @classmethod
    def _weaknesses(cls):
        return ["Assumes linear relationship.", "Slower than Ridge."]

    @classmethod
    def _use_cases(cls):
        return ["Automatic regularisation tuning", "Uncertainty-aware linear regression"]

    @classmethod
    def _assumptions(cls):
        return ["Gaussian noise.", "Gaussian priors on weights."]


# ────────────────────────────────────────────────────────────────────
# R-15  Huber Regressor
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class HuberRegressorAlgo(BaseAlgorithm):
    """
    Huber Regressor.
    Robust linear regression using the Huber loss — insensitive to large outliers.
    """
    name = "Huber Regressor"
    family = "supervised"
    task_type = "regression"
    tags = ["linear", "robust", "outlier-resistant"]
    complexity_train = "O(n·d)"
    complexity_predict = "O(d)"

    def _build_model(self):
        return HuberRegressor(
            epsilon=self._hyperparams.get("epsilon", 1.35),
            alpha=self._hyperparams.get("alpha", 0.0001),
            max_iter=self._hyperparams.get("max_iter", 100),
        )

    def score(self, X, y=None):
        return regression_metrics(y, self.predict(X))

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "epsilon", "type": "float", "default": 1.35, "min": 1.0, "max": 10.0,
             "description": "Transition from quadratic (|r|≤ε) to linear loss (|r|>ε). 1.35 → ~95% efficiency."},
            {"name": "alpha", "type": "float", "default": 0.0001, "min": 1e-6, "max": 1.0,
             "description": "L2 regularisation strength."},
        ]

    @classmethod
    def _math_intuition(cls):
        return "L_ε(r) = r²/2 if |r|≤ε, else ε(|r|-ε/2). Quadratic near zero, linear for large residuals."

    @classmethod
    def _strengths(cls):
        return ["Robust to outliers.", "Still linear (interpretable).", "Efficient."]

    @classmethod
    def _weaknesses(cls):
        return ["Only linear relationship.", "Epsilon requires tuning."]

    @classmethod
    def _use_cases(cls):
        return ["Data with outliers", "Noisy sensor data", "Financial regression"]

    @classmethod
    def _assumptions(cls):
        return ["Linear relationship.", "Outliers are present but not systematic."]


# ────────────────────────────────────────────────────────────────────
# R-16  Quantile Regression
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class QuantileRegressorAlgo(BaseAlgorithm):
    """
    Quantile Regression.
    Estimates a specified quantile (e.g. median, 90th percentile) of the conditional distribution.
    """
    name = "Quantile Regressor"
    family = "supervised"
    task_type = "regression"
    tags = ["linear", "quantile", "robust", "uncertainty"]
    complexity_train = "O(n·d)"
    complexity_predict = "O(d)"

    def _build_model(self):
        return QuantileRegressor(
            quantile=self._hyperparams.get("quantile", 0.5),
            alpha=self._hyperparams.get("alpha", 1.0),
            solver=self._hyperparams.get("solver", "highs"),
        )

    def score(self, X, y=None):
        return regression_metrics(y, self.predict(X))

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "quantile", "type": "float", "default": 0.5, "min": 0.01, "max": 0.99,
             "description": "Target quantile. 0.5 = median regression."},
            {"name": "alpha", "type": "float", "default": 1.0, "min": 0.0, "max": 100.0,
             "description": "L1 regularisation strength."},
        ]

    @classmethod
    def _math_intuition(cls):
        return "Minimise Σ ρ_q(yᵢ - ŷᵢ) where ρ_q(u) = u·q if u≥0, else u·(q-1)."

    @classmethod
    def _strengths(cls):
        return ["Robust to outliers.", "Prediction intervals.", "Median regression is more robust than OLS."]

    @classmethod
    def _weaknesses(cls):
        return ["LP optimisation can be slow.", "Only linear."]

    @classmethod
    def _use_cases(cls):
        return ["Confidence intervals", "Insurance risk", "Extreme event modelling"]

    @classmethod
    def _assumptions(cls):
        return ["Linear quantile function."]


# ────────────────────────────────────────────────────────────────────
# R-17  MLP Regressor
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class MLPRegressorAlgo(BaseAlgorithm):
    """
    Multi-Layer Perceptron Regressor (sklearn).
    Feedforward neural network trained with backpropagation — universal approximator.
    """
    name = "MLP Regressor"
    family = "supervised"
    task_type = "regression"
    tags = ["neural-network", "mlp", "non-linear", "deep"]
    complexity_train = "O(n·epochs·layers)"
    complexity_predict = "O(layers·units)"

    def _build_model(self):
        return MLPRegressor(
            hidden_layer_sizes=tuple(self._hyperparams.get("hidden_layer_sizes", [100])),
            activation=self._hyperparams.get("activation", "relu"),
            solver=self._hyperparams.get("solver", "adam"),
            alpha=self._hyperparams.get("alpha", 0.0001),
            learning_rate_init=self._hyperparams.get("learning_rate_init", 0.001),
            max_iter=self._hyperparams.get("max_iter", 200),
            random_state=self.random_state,
        )

    def score(self, X, y=None):
        return regression_metrics(y, self.predict(X))

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "hidden_layer_sizes", "type": "list_int", "default": [100],
             "description": "Sizes of hidden layers, e.g. [100, 50]."},
            {"name": "activation", "type": "select", "default": "relu",
             "options": ["relu", "tanh", "logistic"],
             "description": "Activation function for hidden layers."},
            {"name": "solver", "type": "select", "default": "adam",
             "options": ["adam", "sgd", "lbfgs"]},
            {"name": "alpha", "type": "float", "default": 0.0001, "min": 1e-6, "max": 1.0,
             "description": "L2 regularisation parameter."},
            {"name": "max_iter", "type": "int", "default": 200, "min": 10, "max": 2000},
            {"name": "learning_rate_init", "type": "float", "default": 0.001, "min": 1e-5, "max": 0.1},
        ]

    @classmethod
    def _math_intuition(cls):
        return "Universal approximator via composition of linear + nonlinear transforms. Trained via backpropagation + Adam."

    @classmethod
    def _strengths(cls):
        return ["Universal approximator.", "Can model complex non-linearities.", "No feature engineering needed."]

    @classmethod
    def _weaknesses(cls):
        return ["Many hyperparameters.", "Needs feature scaling.", "Sensitive to initialization."]

    @classmethod
    def _use_cases(cls):
        return ["Complex non-linear regression", "Baseline neural network"]

    @classmethod
    def _assumptions(cls):
        return ["Scale features.", "Sufficient training data."]


# ────────────────────────────────────────────────────────────────────
# R-18  Passive Aggressive Regressor
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class PassiveAggressiveRegressorAlgo(BaseAlgorithm):
    """
    Passive Aggressive Regressor.
    Online learning algorithm: stays passive on correct predictions,
    updates aggressively on errors. Suitable for streaming data.
    Implemented via SGDRegressor with epsilon-insensitive loss and PA update rule.
    """
    name = "Passive Aggressive Regressor"
    family = "supervised"
    task_type = "regression"
    tags = ["online", "linear", "streaming", "fast"]
    complexity_train = "O(n·d)"
    complexity_predict = "O(d)"

    def _build_model(self):
        # Use SGDRegressor with PA1 update rule (sklearn 1.9+ preferred approach)
        return SGDRegressor(
            loss="epsilon_insensitive",
            penalty=None,
            learning_rate="pa1",
            eta0=1.0,
            epsilon=self._hyperparams.get("epsilon", 0.1),
            max_iter=self._hyperparams.get("max_iter", 1000),
            tol=self._hyperparams.get("tol", 1e-3),
            random_state=self.random_state,
        )

    def score(self, X, y=None):
        return regression_metrics(y, self.predict(X))

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "C", "type": "float", "default": 1.0, "min": 0.0001, "max": 1000.0,
             "description": "Aggressiveness parameter. Larger = bigger updates."},
            {"name": "epsilon", "type": "float", "default": 0.1, "min": 0.0, "max": 10.0,
             "description": "Insensitivity margin."},
            {"name": "max_iter", "type": "int", "default": 1000, "min": 10, "max": 5000},
        ]

    @classmethod
    def _math_intuition(cls):
        return "Update rule: w ← w + τ·x where τ = min(C, L_ε/‖x‖²). Passive (τ=0) if |residual|≤ε."

    @classmethod
    def _strengths(cls):
        return ["Online/streaming learning.", "Very fast.", "No need to store all data."]

    @classmethod
    def _weaknesses(cls):
        return ["Only linear.", "Sensitive to outliers and noisy labels.", "Order-dependent."]

    @classmethod
    def _use_cases(cls):
        return ["Streaming regression", "Online learning systems", "Large-scale text regression"]

    @classmethod
    def _assumptions(cls):
        return ["Linear relationship.", "Data may arrive as a stream."]

