"""
Supervised Classification Algorithms for ShiftLab — all 20 classifiers.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import (
    LogisticRegression, RidgeClassifier, Perceptron,
    SGDClassifier,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV

from backend.core.base_algorithm import BaseAlgorithm
from backend.core.metrics import classification_metrics
from backend.core.registry import AlgorithmRegistry

def _xgb_classifier():
    from xgboost import XGBClassifier
    return XGBClassifier

def _lgb_classifier():
    from lightgbm import LGBMClassifier
    return LGBMClassifier


# ────────────────────────────────────────────────────────────────────
# C-01  Logistic Regression
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class LogisticRegressionAlgo(BaseAlgorithm):
    """
    Logistic Regression.
    Models log-odds as a linear function of features. Trained via L-BFGS / Newton.
    """
    name = "Logistic Regression"
    family = "supervised"
    task_type = "classification"
    tags = ["linear", "parametric", "probabilistic", "interpretable"]
    complexity_train = "O(n·d·iter)"
    complexity_predict = "O(d)"

    def _build_model(self):
        return LogisticRegression(
            C=self._hyperparams.get("C", 1.0),
            penalty=self._hyperparams.get("penalty", "l2"),
            solver=self._hyperparams.get("solver", "lbfgs"),
            max_iter=self._hyperparams.get("max_iter", 1000),
            random_state=self.random_state,
        )

    def score(self, X, y=None):
        proba = self.predict_proba(X)
        return classification_metrics(y, self.predict(X), proba)

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "C", "type": "float", "default": 1.0, "min": 0.0001, "max": 1000.0,
             "description": "Inverse regularisation strength. Smaller = stronger regularisation."},
            {"name": "penalty", "type": "select", "default": "l2",
             "options": ["l1", "l2", "elasticnet", "none"],
             "description": "Regularisation type."},
            {"name": "solver", "type": "select", "default": "lbfgs",
             "options": ["lbfgs", "liblinear", "saga", "newton-cg"],
             "description": "Optimisation algorithm."},
            {"name": "max_iter", "type": "int", "default": 1000, "min": 100, "max": 5000},
        ]

    @classmethod
    def _math_intuition(cls):
        return "P(y=1|x) = σ(wᵀx + b) = 1/(1+exp(-(wᵀx+b))). Maximise log-likelihood."

    @classmethod
    def _assumptions(cls):
        return ["Linear decision boundary.", "Features should be scaled.", "No severe multicollinearity."]

    @classmethod
    def _strengths(cls):
        return ["Probabilistic output.", "Fast.", "Interpretable coefficients.", "Regularisation built-in."]

    @classmethod
    def _weaknesses(cls):
        return ["Linear boundary only.", "Sensitive to outliers.", "Assumes feature independence (loosely)."]

    @classmethod
    def _use_cases(cls):
        return ["Binary/multi-class classification", "Credit scoring", "Medical diagnosis", "Baseline"]


# ────────────────────────────────────────────────────────────────────
# C-02  Linear Discriminant Analysis
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class LDAClassifierAlgo(BaseAlgorithm):
    """
    Linear Discriminant Analysis (LDA) Classifier.
    Finds a linear combination of features that maximises class separability.
    Assumes equal covariance matrices across classes.
    """
    name = "Linear Discriminant Analysis"
    family = "supervised"
    task_type = "classification"
    tags = ["linear", "parametric", "generative", "dimensionality-reduction"]
    complexity_train = "O(n·d²)"
    complexity_predict = "O(d)"

    def _build_model(self):
        return LinearDiscriminantAnalysis(
            solver=self._hyperparams.get("solver", "svd"),
            shrinkage=self._hyperparams.get("shrinkage", None),
        )

    def score(self, X, y=None):
        proba = self.predict_proba(X)
        return classification_metrics(y, self.predict(X), proba)

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "solver", "type": "select", "default": "svd",
             "options": ["svd", "lsqr", "eigen"],
             "description": "Solver. Use 'lsqr' or 'eigen' with shrinkage."},
            {"name": "shrinkage", "type": "select", "default": None,
             "options": [None, "auto"],
             "description": "Ledoit-Wolf shrinkage. Use when n < d."},
        ]

    @classmethod
    def _math_intuition(cls):
        return "Maximise J(w) = wᵀS_Bw / wᵀS_Ww (between-class / within-class scatter ratio)."

    @classmethod
    def _assumptions(cls):
        return ["Gaussian class-conditional distributions.", "Equal covariance matrices.", "Features scaled."]

    @classmethod
    def _strengths(cls):
        return ["Fast.", "Works as both classifier and dim-reduction.", "Probabilistic output."]

    @classmethod
    def _weaknesses(cls):
        return ["Assumes equal covariance.", "Linear boundary only.", "Sensitive to outliers."]

    @classmethod
    def _use_cases(cls):
        return ["Face recognition", "Medical classification", "Dimensionality reduction preprocessing"]


# ────────────────────────────────────────────────────────────────────
# C-03  Quadratic Discriminant Analysis
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class QDAClassifierAlgo(BaseAlgorithm):
    """
    Quadratic Discriminant Analysis (QDA).
    Like LDA but allows each class its own covariance — quadratic decision boundary.
    """
    name = "Quadratic Discriminant Analysis"
    family = "supervised"
    task_type = "classification"
    tags = ["quadratic", "parametric", "generative", "non-linear"]
    complexity_train = "O(n·d²)"
    complexity_predict = "O(d²)"

    def _build_model(self):
        return QuadraticDiscriminantAnalysis(
            reg_param=self._hyperparams.get("reg_param", 0.0),
        )

    def score(self, X, y=None):
        proba = self.predict_proba(X)
        return classification_metrics(y, self.predict(X), proba)

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "reg_param", "type": "float", "default": 0.0, "min": 0.0, "max": 1.0,
             "description": "Regularise covariance estimate (0=none, >0=shrink toward diagonal)."},
        ]

    @classmethod
    def _math_intuition(cls):
        return "Each class k has its own Σₖ. Decision: argmax_k log p(x|k) + log p(k)."

    @classmethod
    def _assumptions(cls):
        return ["Gaussian class-conditional distributions.", "Separate covariance per class."]

    @classmethod
    def _strengths(cls):
        return ["Quadratic boundary — more flexible than LDA.", "Fast.", "No hyperparameters (unless regularised)."]

    @classmethod
    def _weaknesses(cls):
        return ["More parameters than LDA.", "Unstable with small class samples."]

    @classmethod
    def _use_cases(cls):
        return ["When LDA assumptions are too strict", "Biometric classification"]


# ────────────────────────────────────────────────────────────────────
# C-04  SVM — Linear kernel
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class SVMLinearAlgo(BaseAlgorithm):
    """
    Support Vector Machine — Linear Kernel.
    Finds the maximum-margin hyperplane separating classes.
    """
    name = "SVM (Linear)"
    family = "supervised"
    task_type = "classification"
    tags = ["svm", "linear", "margin", "kernel"]
    complexity_train = "O(n²·d)"
    complexity_predict = "O(n_sv·d)"

    def _build_model(self):
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.svm import SVC as _SVC
        base = _SVC(kernel="linear", C=self._hyperparams.get("C", 1.0), random_state=self.random_state)
        return CalibratedClassifierCV(estimator=base, ensemble=False)

    def score(self, X, y=None):
        proba = self.predict_proba(X)
        return classification_metrics(y, self.predict(X), proba)

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "C", "type": "float", "default": 1.0, "min": 0.0001, "max": 1000.0,
             "description": "Regularisation. Larger = harder margin, smaller = softer."},
        ]

    @classmethod
    def _math_intuition(cls):
        return "Maximise 2/‖w‖ subject to yᵢ(wᵀxᵢ+b)≥1. Dual via QP with Lagrange multipliers."

    @classmethod
    def _assumptions(cls):
        return ["Data should be linearly separable (or soft margin via C).", "Scale features."]

    @classmethod
    def _strengths(cls):
        return ["Maximum margin = good generalisation.", "Effective in high dimensions.", "Memory efficient (uses only support vectors)."]

    @classmethod
    def _weaknesses(cls):
        return ["Slow on large datasets.", "Requires feature scaling.", "Hard to interpret."]

    @classmethod
    def _use_cases(cls):
        return ["Text classification", "High-dimensional data", "Binary classification"]


# ────────────────────────────────────────────────────────────────────
# C-05  SVM — RBF kernel
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class SVMRBFAlgo(BaseAlgorithm):
    """
    Support Vector Machine — RBF (Gaussian) Kernel.
    Uses the kernel trick to project into infinite-dimensional feature space.
    """
    name = "SVM (RBF)"
    family = "supervised"
    task_type = "classification"
    tags = ["svm", "kernel", "non-linear", "rbf"]
    complexity_train = "O(n²·d) to O(n³)"
    complexity_predict = "O(n_sv·d)"

    def _build_model(self):
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.svm import SVC as _SVC
        base = _SVC(
            kernel="rbf",
            C=self._hyperparams.get("C", 1.0),
            gamma=self._hyperparams.get("gamma", "scale"),
            random_state=self.random_state,
        )
        return CalibratedClassifierCV(estimator=base, ensemble=False)

    def score(self, X, y=None):
        proba = self.predict_proba(X)
        return classification_metrics(y, self.predict(X), proba)

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "C", "type": "float", "default": 1.0, "min": 0.0001, "max": 1000.0},
            {"name": "gamma", "type": "select", "default": "scale",
             "options": ["scale", "auto"],
             "description": "'scale' = 1/(n_features·var(X))."},
        ]

    @classmethod
    def _math_intuition(cls):
        return "K(x,x') = exp(-γ‖x-x'‖²). Infinite-dimensional feature map via Mercer's theorem."

    @classmethod
    def _assumptions(cls):
        return ["Features should be scaled.", "C and gamma need tuning."]

    @classmethod
    def _strengths(cls):
        return ["Highly flexible non-linear boundary.", "Works well with clear margin."]

    @classmethod
    def _weaknesses(cls):
        return ["Very slow O(n³) on large data.", "Two hyperparameters to tune."]

    @classmethod
    def _use_cases(cls):
        return ["Image classification (small datasets)", "Bioinformatics", "Non-linear patterns"]


# ────────────────────────────────────────────────────────────────────
# C-06  K-Nearest Neighbours Classifier
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class KNNClassifierAlgo(BaseAlgorithm):
    """
    K-Nearest Neighbours Classifier.
    Classifies by majority vote among k nearest training samples.
    """
    name = "KNN Classifier"
    family = "supervised"
    task_type = "classification"
    tags = ["instance-based", "lazy", "non-parametric", "non-linear"]
    complexity_train = "O(1)"
    complexity_predict = "O(n·d)"

    def _build_model(self):
        return KNeighborsClassifier(
            n_neighbors=self._hyperparams.get("n_neighbors", 5),
            weights=self._hyperparams.get("weights", "uniform"),
            metric=self._hyperparams.get("metric", "minkowski"),
        )

    def score(self, X, y=None):
        proba = self.predict_proba(X)
        return classification_metrics(y, self.predict(X), proba)

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "n_neighbors", "type": "int", "default": 5, "min": 1, "max": 50},
            {"name": "weights", "type": "select", "default": "uniform",
             "options": ["uniform", "distance"]},
            {"name": "metric", "type": "select", "default": "minkowski",
             "options": ["minkowski", "euclidean", "manhattan", "chebyshev"]},
        ]

    @classmethod
    def _math_intuition(cls):
        return "ŷ = argmax_c Σ_{i∈Nₖ(x)} I(yᵢ=c). Decision boundary is a Voronoi mosaic."

    @classmethod
    def _assumptions(cls):
        return ["Scale features.", "Similar inputs have similar outputs."]

    @classmethod
    def _strengths(cls):
        return ["No training.", "Naturally multi-class.", "Non-linear boundary."]

    @classmethod
    def _weaknesses(cls):
        return ["Slow prediction.", "Memory-intensive.", "Curse of dimensionality."]

    @classmethod
    def _use_cases(cls):
        return ["Recommendation systems", "Anomaly detection", "Image recognition (small)"]


# ────────────────────────────────────────────────────────────────────
# C-07  Gaussian Naive Bayes
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class GaussianNBAlgo(BaseAlgorithm):
    """
    Gaussian Naive Bayes.
    Assumes feature independence and Gaussian class-conditional distributions.
    Very fast generative classifier.
    """
    name = "Gaussian Naive Bayes"
    family = "supervised"
    task_type = "classification"
    tags = ["bayesian", "generative", "fast", "probabilistic", "naive-bayes"]
    complexity_train = "O(n·d)"
    complexity_predict = "O(d)"

    def _build_model(self):
        return GaussianNB(
            var_smoothing=self._hyperparams.get("var_smoothing", 1e-9),
        )

    def score(self, X, y=None):
        proba = self.predict_proba(X)
        return classification_metrics(y, self.predict(X), proba)

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "var_smoothing", "type": "float", "default": 1e-9, "min": 1e-12, "max": 0.01,
             "description": "Portion of the largest variance added for numerical stability."},
        ]

    @classmethod
    def _math_intuition(cls):
        return "P(y|x) ∝ P(y)·∏ P(xⱼ|y) where P(xⱼ|y) = N(μⱼᵧ, σⱼᵧ²)."

    @classmethod
    def _assumptions(cls):
        return ["Feature independence given class (naive assumption).", "Gaussian feature distributions per class."]

    @classmethod
    def _strengths(cls):
        return ["Extremely fast.", "Works well with small data.", "Handles many features.", "Probabilistic."]

    @classmethod
    def _weaknesses(cls):
        return ["Strong independence assumption often violated.", "Gaussian assumption may not hold."]

    @classmethod
    def _use_cases(cls):
        return ["Text classification", "Real-time classification", "Medical diagnosis baseline"]


# ────────────────────────────────────────────────────────────────────
# C-08  Multinomial Naive Bayes
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class MultinomialNBAlgo(BaseAlgorithm):
    """
    Multinomial Naive Bayes.
    Models feature counts (e.g. word frequencies in text). Requires non-negative features.
    """
    name = "Multinomial Naive Bayes"
    family = "supervised"
    task_type = "classification"
    tags = ["bayesian", "text", "count-data", "naive-bayes"]
    complexity_train = "O(n·d)"
    complexity_predict = "O(d)"

    def _build_model(self):
        return MultinomialNB(alpha=self._hyperparams.get("alpha", 1.0))

    def score(self, X, y=None):
        proba = self.predict_proba(X)
        return classification_metrics(y, self.predict(X), proba)

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "alpha", "type": "float", "default": 1.0, "min": 0.0, "max": 10.0,
             "description": "Laplace / Lidstone smoothing. 0 = no smoothing."},
        ]

    @classmethod
    def _math_intuition(cls):
        return "P(xⱼ|y) = (count(xⱼ,y) + α) / (count(y) + α·d). Multinomial likelihood."

    @classmethod
    def _assumptions(cls):
        return ["Features are non-negative counts.", "Feature independence."]

    @classmethod
    def _strengths(cls):
        return ["Excellent for text (TF/TF-IDF features).", "Very fast.", "Works with sparse data."]

    @classmethod
    def _weaknesses(cls):
        return ["Requires non-negative features.", "Independence assumption."]

    @classmethod
    def _use_cases(cls):
        return ["Spam detection", "Document classification", "Sentiment analysis"]


# ────────────────────────────────────────────────────────────────────
# C-09  Bernoulli Naive Bayes
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class BernoulliNBAlgo(BaseAlgorithm):
    """
    Bernoulli Naive Bayes.
    Models binary features (word presence/absence). Penalises absent features explicitly.
    """
    name = "Bernoulli Naive Bayes"
    family = "supervised"
    task_type = "classification"
    tags = ["bayesian", "binary-features", "text", "naive-bayes"]
    complexity_train = "O(n·d)"
    complexity_predict = "O(d)"

    def _build_model(self):
        return BernoulliNB(
            alpha=self._hyperparams.get("alpha", 1.0),
            binarize=self._hyperparams.get("binarize", 0.0),
        )

    def score(self, X, y=None):
        proba = self.predict_proba(X)
        return classification_metrics(y, self.predict(X), proba)

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "alpha", "type": "float", "default": 1.0, "min": 0.0, "max": 10.0},
            {"name": "binarize", "type": "float", "default": 0.0, "min": 0.0, "max": 1.0,
             "description": "Threshold for binarising features. 0 = already binary."},
        ]

    @classmethod
    def _math_intuition(cls):
        return "P(x|y) = ∏ pⱼᵧ^xⱼ · (1-pⱼᵧ)^(1-xⱼ). Explicitly models feature absence."

    @classmethod
    def _assumptions(cls):
        return ["Binary features.", "Feature independence."]

    @classmethod
    def _strengths(cls):
        return ["Good for binary document features.", "Very fast.", "Simple."]

    @classmethod
    def _weaknesses(cls):
        return ["Requires binary or binarisable features.", "Strong independence assumption."]

    @classmethod
    def _use_cases(cls):
        return ["Short text / sentiment", "Binary feature classification"]


# ────────────────────────────────────────────────────────────────────
# C-10  Decision Tree Classifier
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class DecisionTreeClassifierAlgo(BaseAlgorithm):
    """
    Decision Tree Classifier.
    Recursively partitions the feature space using Gini impurity or entropy.
    Fully visualisable.
    """
    name = "Decision Tree Classifier"
    family = "supervised"
    task_type = "classification"
    tags = ["tree", "non-linear", "interpretable", "no-scaling"]
    complexity_train = "O(n·d·log n)"
    complexity_predict = "O(depth)"

    def _build_model(self):
        return DecisionTreeClassifier(
            max_depth=self._hyperparams.get("max_depth", None),
            criterion=self._hyperparams.get("criterion", "gini"),
            min_samples_split=self._hyperparams.get("min_samples_split", 2),
            min_samples_leaf=self._hyperparams.get("min_samples_leaf", 1),
            random_state=self.random_state,
        )

    def score(self, X, y=None):
        proba = self.predict_proba(X)
        return classification_metrics(y, self.predict(X), proba)

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "max_depth", "type": "int", "default": None, "min": 1, "max": 50},
            {"name": "criterion", "type": "select", "default": "gini",
             "options": ["gini", "entropy", "log_loss"]},
            {"name": "min_samples_split", "type": "int", "default": 2, "min": 2, "max": 100},
            {"name": "min_samples_leaf", "type": "int", "default": 1, "min": 1, "max": 50},
        ]

    @classmethod
    def _math_intuition(cls):
        return "Gini: 1 - Σpₖ². Entropy: -Σpₖ log pₖ. Greedy split maximises purity gain."

    @classmethod
    def _assumptions(cls):
        return ["None (non-parametric)."]

    @classmethod
    def _strengths(cls):
        return ["Fully visualisable.", "No scaling needed.", "Handles mixed types.", "Fast."]

    @classmethod
    def _weaknesses(cls):
        return ["High variance.", "Greedy splits not globally optimal.", "Sensitive to small data changes."]

    @classmethod
    def _use_cases(cls):
        return ["Explainability-first applications", "Rule extraction", "Feature importance"]


# ────────────────────────────────────────────────────────────────────
# C-11  Random Forest Classifier
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class RandomForestClassifierAlgo(BaseAlgorithm):
    """
    Random Forest Classifier.
    Bagging of decision trees with feature randomisation — low variance, high accuracy.
    """
    name = "Random Forest Classifier"
    family = "supervised"
    task_type = "classification"
    tags = ["ensemble", "bagging", "non-linear", "robust", "feature-importance"]
    complexity_train = "O(n·trees·d·log n)"
    complexity_predict = "O(trees·depth)"

    def _build_model(self):
        return RandomForestClassifier(
            n_estimators=self._hyperparams.get("n_estimators", 100),
            max_depth=self._hyperparams.get("max_depth", None),
            criterion=self._hyperparams.get("criterion", "gini"),
            max_features=self._hyperparams.get("max_features", "sqrt"),
            min_samples_split=self._hyperparams.get("min_samples_split", 2),
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
            {"name": "criterion", "type": "select", "default": "gini", "options": ["gini", "entropy"]},
            {"name": "max_features", "type": "select", "default": "sqrt",
             "options": ["sqrt", "log2", "none"]},
        ]

    @classmethod
    def _math_intuition(cls):
        return "ŷ = majority vote of T trees, each trained on bootstrap sample with d' random features."

    @classmethod
    def _assumptions(cls):
        return ["None (non-parametric)."]

    @classmethod
    def _strengths(cls):
        return ["Very accurate.", "Feature importance.", "Handles high dimensionality.", "Robust to noise."]

    @classmethod
    def _weaknesses(cls):
        return ["Less interpretable than single tree.", "Slow for many trees.", "Memory-heavy."]

    @classmethod
    def _use_cases(cls):
        return ["General-purpose classification", "Feature selection", "Medical diagnosis"]


# ────────────────────────────────────────────────────────────────────
# C-12  Gradient Boosting Classifier
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class GradientBoostingClassifierAlgo(BaseAlgorithm):
    """
    Gradient Boosting Classifier (sklearn GBT).
    Sequentially trains trees to correct predecessor errors using gradient descent.
    """
    name = "Gradient Boosting Classifier"
    family = "supervised"
    task_type = "classification"
    tags = ["ensemble", "boosting", "non-linear", "powerful"]
    complexity_train = "O(n·trees·d·log n)"
    complexity_predict = "O(trees·depth)"

    def _build_model(self):
        return GradientBoostingClassifier(
            n_estimators=self._hyperparams.get("n_estimators", 100),
            learning_rate=self._hyperparams.get("learning_rate", 0.1),
            max_depth=self._hyperparams.get("max_depth", 3),
            subsample=self._hyperparams.get("subsample", 1.0),
            random_state=self.random_state,
        )

    def score(self, X, y=None):
        proba = self.predict_proba(X)
        return classification_metrics(y, self.predict(X), proba)

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "n_estimators", "type": "int", "default": 100, "min": 10, "max": 500},
            {"name": "learning_rate", "type": "float", "default": 0.1, "min": 0.001, "max": 1.0},
            {"name": "max_depth", "type": "int", "default": 3, "min": 1, "max": 10},
            {"name": "subsample", "type": "float", "default": 1.0, "min": 0.1, "max": 1.0},
        ]

    @classmethod
    def _math_intuition(cls):
        return "Uses deviance (log-loss) as objective. F_m = F_{m-1} + η·argmin_h Σ L(yᵢ, F_{m-1}(xᵢ)+h(xᵢ))."

    @classmethod
    def _assumptions(cls):
        return ["None (non-parametric)."]

    @classmethod
    def _strengths(cls):
        return ["Highly accurate.", "Handles mixed features.", "Built-in feature importance."]

    @classmethod
    def _weaknesses(cls):
        return ["Sequential — slow to train.", "Many hyperparameters.", "Prone to overfitting."]

    @classmethod
    def _use_cases(cls):
        return ["Tabular classification", "Fraud detection", "Customer churn"]


# ────────────────────────────────────────────────────────────────────
# C-13  XGBoost Classifier
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class XGBoostClassifierAlgo(BaseAlgorithm):
    """
    XGBoost Classifier.
    Extreme Gradient Boosting — regularised boosting with approximate tree learning.
    """
    name = "XGBoost Classifier"
    family = "supervised"
    task_type = "classification"
    tags = ["ensemble", "boosting", "xgboost", "powerful", "fast"]
    complexity_train = "O(n·trees·d)"
    complexity_predict = "O(trees·depth)"

    def fit(self, X_train: np.ndarray, y_train: np.ndarray | None = None):
        # XGBoost needs the correct objective before the estimator is built.
        # BaseAlgorithm.fit() builds the model first, so resolve the class count
        # from the observed training labels here.
        if y_train is not None:
            self._hyperparams["_n_classes_"] = int(len(np.unique(y_train)))
        return super().fit(X_train, y_train)

    def _build_model(self):
        XGBClassifier = _xgb_classifier()
        n_cls = int(self._hyperparams.get("_n_classes_", 2))
        objective = "binary:logistic" if n_cls <= 2 else "multi:softprob"
        kwargs = dict(
            n_estimators=self._hyperparams.get("n_estimators", 100),
            learning_rate=self._hyperparams.get("learning_rate", 0.1),
            max_depth=self._hyperparams.get("max_depth", 6),
            subsample=self._hyperparams.get("subsample", 1.0),
            colsample_bytree=self._hyperparams.get("colsample_bytree", 1.0),
            reg_alpha=self._hyperparams.get("reg_alpha", 0.0),
            reg_lambda=self._hyperparams.get("reg_lambda", 1.0),
            objective=objective,
            random_state=self.random_state,
            verbosity=0,
            n_jobs=self._n_jobs(),
            eval_metric="logloss",
        )
        if n_cls > 2:
            kwargs["num_class"] = n_cls
        return XGBClassifier(**kwargs)

    def score(self, X, y=None):
        proba = self.predict_proba(X)
        return classification_metrics(y, self.predict(X), proba)

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "n_estimators", "type": "int", "default": 100, "min": 10, "max": 500},
            {"name": "learning_rate", "type": "float", "default": 0.1, "min": 0.001, "max": 1.0},
            {"name": "max_depth", "type": "int", "default": 6, "min": 1, "max": 15},
            {"name": "subsample", "type": "float", "default": 1.0, "min": 0.1, "max": 1.0},
            {"name": "colsample_bytree", "type": "float", "default": 1.0, "min": 0.1, "max": 1.0},
            {"name": "reg_alpha", "type": "float", "default": 0.0, "min": 0.0, "max": 10.0},
            {"name": "reg_lambda", "type": "float", "default": 1.0, "min": 0.0, "max": 10.0},
        ]

    @classmethod
    def _math_intuition(cls):
        return "Binary: log-loss objective. Multi-class: softmax with one-vs-rest. Adds Ω(f)=γT+λ‖w‖² regularisation."

    @classmethod
    def _assumptions(cls):
        return ["None (non-parametric)."]

    @classmethod
    def _strengths(cls):
        return ["State-of-the-art on tabular data.", "Built-in regularisation.", "Handles NaN."]

    @classmethod
    def _weaknesses(cls):
        return ["Many hyperparameters.", "Not suited for sequential/text data directly."]

    @classmethod
    def _use_cases(cls):
        return ["Competition ML", "Fraud detection", "Click-through rate prediction"]


# ────────────────────────────────────────────────────────────────────
# C-14  LightGBM Classifier
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class LightGBMClassifierAlgo(BaseAlgorithm):
    """
    LightGBM Classifier.
    Leaf-wise gradient boosted trees — very fast, accurate on large datasets.
    """
    name = "LightGBM Classifier"
    family = "supervised"
    task_type = "classification"
    tags = ["ensemble", "boosting", "lightgbm", "fast", "large-data"]
    complexity_train = "O(n·trees·bins)"
    complexity_predict = "O(trees·depth)"

    def _build_model(self):
        LGBMClassifier = _lgb_classifier()
        return LGBMClassifier(
            n_estimators=self._hyperparams.get("n_estimators", 100),
            learning_rate=self._hyperparams.get("learning_rate", 0.1),
            max_depth=self._hyperparams.get("max_depth", -1),
            num_leaves=self._hyperparams.get("num_leaves", 31),
            subsample=self._hyperparams.get("subsample", 1.0),
            colsample_bytree=self._hyperparams.get("colsample_bytree", 1.0),
            random_state=self.random_state,
            verbosity=-1,
            n_jobs=self._n_jobs(),
        )

    def score(self, X, y=None):
        proba = self.predict_proba(X)
        return classification_metrics(y, self.predict(X), proba)

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "n_estimators", "type": "int", "default": 100, "min": 10, "max": 500},
            {"name": "learning_rate", "type": "float", "default": 0.1, "min": 0.001, "max": 1.0},
            {"name": "num_leaves", "type": "int", "default": 31, "min": 2, "max": 300},
            {"name": "max_depth", "type": "int", "default": -1, "min": -1, "max": 50},
            {"name": "subsample", "type": "float", "default": 1.0, "min": 0.1, "max": 1.0},
        ]

    @classmethod
    def _math_intuition(cls):
        return "Leaf-wise tree growth + histogram bins. Same GBT objective but 10-20× faster split search."

    @classmethod
    def _assumptions(cls):
        return ["None (non-parametric)."]

    @classmethod
    def _strengths(cls):
        return ["Very fast training.", "State-of-the-art accuracy.", "Low memory footprint."]

    @classmethod
    def _weaknesses(cls):
        return ["Leaf-wise can overfit on small N.", "Verbose hyperparameter space."]

    @classmethod
    def _use_cases(cls):
        return ["Large-scale classification", "Production systems", "Feature importance"]


# ────────────────────────────────────────────────────────────────────
# C-15  Ridge Classifier
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class RidgeClassifierAlgo(BaseAlgorithm):
    """
    Ridge Classifier.
    Converts multi-class labels to {-1, 1} and applies Ridge regression.
    Equivalent to LDA under certain conditions. Very fast.
    """
    name = "Ridge Classifier"
    family = "supervised"
    task_type = "classification"
    tags = ["linear", "fast", "regularisation", "l2"]
    complexity_train = "O(n·d²)"
    complexity_predict = "O(d)"

    def _build_model(self):
        return RidgeClassifier(
            alpha=self._hyperparams.get("alpha", 1.0),
        )

    def predict_proba(self, X):
        # RidgeClassifier doesn't natively support predict_proba
        return None

    def score(self, X, y=None):
        return classification_metrics(y, self.predict(X))

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "alpha", "type": "float", "default": 1.0, "min": 0.0001, "max": 1000.0,
             "description": "L2 regularisation strength."},
        ]

    @classmethod
    def _math_intuition(cls):
        return "One-hot encode y, solve Ridge regression for each class, predict via argmax."

    @classmethod
    def _assumptions(cls):
        return ["Linear decision boundary.", "Scale features."]

    @classmethod
    def _strengths(cls):
        return ["Extremely fast (closed-form).", "Handles multicollinearity.", "Good baseline."]

    @classmethod
    def _weaknesses(cls):
        return ["No probability output.", "Linear only."]

    @classmethod
    def _use_cases(cls):
        return ["Fast baseline", "Text classification with TF-IDF", "High-dimensional sparse data"]


# ────────────────────────────────────────────────────────────────────
# C-16  Perceptron
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class PerceptronAlgo(BaseAlgorithm):
    """
    Perceptron.
    The original linear classification algorithm (Rosenblatt, 1958).
    Online update rule: w ← w + η·yᵢ·xᵢ on misclassification.
    """
    name = "Perceptron"
    family = "supervised"
    task_type = "classification"
    tags = ["linear", "online", "neural-network", "historical"]
    complexity_train = "O(n·d·iter)"
    complexity_predict = "O(d)"

    def _build_model(self):
        return Perceptron(
            penalty=self._hyperparams.get("penalty", None),
            alpha=self._hyperparams.get("alpha", 0.0001),
            max_iter=self._hyperparams.get("max_iter", 1000),
            tol=self._hyperparams.get("tol", 1e-3),
            random_state=self.random_state,
        )

    def predict_proba(self, X):
        return None  # Perceptron has no probability output

    def score(self, X, y=None):
        return classification_metrics(y, self.predict(X))

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "penalty", "type": "select", "default": None,
             "options": [None, "l1", "l2", "elasticnet"]},
            {"name": "alpha", "type": "float", "default": 0.0001, "min": 1e-6, "max": 1.0},
            {"name": "max_iter", "type": "int", "default": 1000, "min": 10, "max": 5000},
        ]

    @classmethod
    def _math_intuition(cls):
        return "Update: w ← w + η·(y - ŷ)·x. Converges if data is linearly separable."

    @classmethod
    def _assumptions(cls):
        return ["Data should be linearly separable for convergence.", "Scale features."]

    @classmethod
    def _strengths(cls):
        return ["Historical importance.", "Online learning.", "Very fast.", "Foundation of neural networks."]

    @classmethod
    def _weaknesses(cls):
        return ["No probability output.", "No convergence on non-separable data.", "Linear only."]

    @classmethod
    def _use_cases(cls):
        return ["Educational / historical", "Very large online learning tasks"]


# ────────────────────────────────────────────────────────────────────
# C-17  SGD Classifier
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class SGDClassifierAlgo(BaseAlgorithm):
    """
    SGD Classifier.
    General-purpose linear classifier trained via Stochastic Gradient Descent.
    Supports many loss functions — equivalent to SVM, LR, Perceptron by changing loss.
    """
    name = "SGD Classifier"
    family = "supervised"
    task_type = "classification"
    tags = ["linear", "online", "sgd", "scalable"]
    complexity_train = "O(n·d·iter)"
    complexity_predict = "O(d)"

    def _build_model(self):
        return SGDClassifier(
            loss=self._hyperparams.get("loss", "hinge"),
            penalty=self._hyperparams.get("penalty", "l2"),
            alpha=self._hyperparams.get("alpha", 0.0001),
            max_iter=self._hyperparams.get("max_iter", 1000),
            tol=self._hyperparams.get("tol", 1e-3),
            random_state=self.random_state,
        )

    def predict_proba(self, X):
        if hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(X)
        return None

    def score(self, X, y=None):
        return classification_metrics(y, self.predict(X))

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "loss", "type": "select", "default": "hinge",
             "options": ["hinge", "log_loss", "modified_huber", "squared_hinge", "perceptron"],
             "description": "'hinge'=SVM, 'log_loss'=Logistic, 'perceptron'=Perceptron."},
            {"name": "penalty", "type": "select", "default": "l2", "options": ["l1", "l2", "elasticnet"]},
            {"name": "alpha", "type": "float", "default": 0.0001, "min": 1e-6, "max": 1.0},
            {"name": "max_iter", "type": "int", "default": 1000, "min": 10, "max": 5000},
        ]

    @classmethod
    def _math_intuition(cls):
        return "w ← w - η·∇L(w;xᵢ,yᵢ) - η·λ·w. Mini-batch SGD approximates batch gradient."

    @classmethod
    def _assumptions(cls):
        return ["Linear relationship.", "Scale features."]

    @classmethod
    def _strengths(cls):
        return ["Scalable to very large datasets.", "Supports many loss functions.", "Online learning."]

    @classmethod
    def _weaknesses(cls):
        return ["Sensitive to learning rate.", "Linear only.", "Noisy convergence."]

    @classmethod
    def _use_cases(cls):
        return ["Very large text datasets", "Online streaming classification"]


# ────────────────────────────────────────────────────────────────────
# C-18  MLP Classifier
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class MLPClassifierAlgo(BaseAlgorithm):
    """
    Multi-Layer Perceptron Classifier (sklearn).
    Feedforward neural network with backpropagation. Universal approximator.
    """
    name = "MLP Classifier"
    family = "supervised"
    task_type = "classification"
    tags = ["neural-network", "mlp", "non-linear", "deep"]
    complexity_train = "O(n·epochs·layers)"
    complexity_predict = "O(layers·units)"

    def _build_model(self):
        return MLPClassifier(
            hidden_layer_sizes=tuple(self._hyperparams.get("hidden_layer_sizes", [100])),
            activation=self._hyperparams.get("activation", "relu"),
            solver=self._hyperparams.get("solver", "adam"),
            alpha=self._hyperparams.get("alpha", 0.0001),
            learning_rate_init=self._hyperparams.get("learning_rate_init", 0.001),
            max_iter=self._hyperparams.get("max_iter", 200),
            random_state=self.random_state,
        )

    def score(self, X, y=None):
        proba = self.predict_proba(X)
        return classification_metrics(y, self.predict(X), proba)

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "hidden_layer_sizes", "type": "list_int", "default": [100]},
            {"name": "activation", "type": "select", "default": "relu",
             "options": ["relu", "tanh", "logistic"]},
            {"name": "solver", "type": "select", "default": "adam",
             "options": ["adam", "sgd", "lbfgs"]},
            {"name": "alpha", "type": "float", "default": 0.0001, "min": 1e-6, "max": 1.0},
            {"name": "max_iter", "type": "int", "default": 200, "min": 10, "max": 2000},
            {"name": "learning_rate_init", "type": "float", "default": 0.001, "min": 1e-5, "max": 0.1},
        ]

    @classmethod
    def _math_intuition(cls):
        return "Softmax output layer. Minimise cross-entropy loss via backpropagation + Adam optimiser."

    @classmethod
    def _assumptions(cls):
        return ["Scale features.", "Enough training data for deep features."]

    @classmethod
    def _strengths(cls):
        return ["Universal approximator.", "Handles complex non-linear patterns.", "Multi-class natively."]

    @classmethod
    def _weaknesses(cls):
        return ["Black box.", "Needs scaling.", "Many hyperparameters.", "Can overfit."]

    @classmethod
    def _use_cases(cls):
        return ["Image classification (small)", "Complex tabular classification", "Baseline neural network"]


# ────────────────────────────────────────────────────────────────────
# C-19  Passive Aggressive Classifier
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class PassiveAggressiveClassifierAlgo(BaseAlgorithm):
    """
    Passive Aggressive Classifier.
    Online learning: passive on correct predictions, aggressive updates on errors.
    Implemented as SGDClassifier with PA update rule.
    """
    name = "Passive Aggressive Classifier"
    family = "supervised"
    task_type = "classification"
    tags = ["online", "linear", "streaming"]
    complexity_train = "O(n·d)"
    complexity_predict = "O(d)"

    def _build_model(self):
        # Use SGDClassifier with PA hinge loss as the canonical sklearn 1.9+ approach
        return SGDClassifier(
            loss="hinge",
            penalty=None,
            learning_rate="pa1",
            eta0=1.0,
            max_iter=self._hyperparams.get("max_iter", 1000),
            tol=self._hyperparams.get("tol", 1e-3),
            random_state=self.random_state,
        )

    def predict_proba(self, X):
        return None

    def score(self, X, y=None):
        return classification_metrics(y, self.predict(X))

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "C", "type": "float", "default": 1.0, "min": 0.0001, "max": 1000.0},
            {"name": "max_iter", "type": "int", "default": 1000, "min": 10, "max": 5000},
        ]

    @classmethod
    def _math_intuition(cls):
        return "Update: if hinge_loss > 0 then w ← w + τ·yᵢ·xᵢ where τ = min(C, loss/‖xᵢ‖²)."

    @classmethod
    def _assumptions(cls):
        return ["Linear relationship.", "Scale features."]

    @classmethod
    def _strengths(cls):
        return ["Very fast.", "Online/streaming.", "Handles large datasets."]

    @classmethod
    def _weaknesses(cls):
        return ["Linear only.", "No probability output.", "Order-dependent."]

    @classmethod
    def _use_cases(cls):
        return ["Text stream classification", "Online sentiment analysis"]


# ────────────────────────────────────────────────────────────────────
# C-20  Calibrated Classifier
# ────────────────────────────────────────────────────────────────────
@AlgorithmRegistry.register
class CalibratedClassifierAlgo(BaseAlgorithm):
    """
    Calibrated Classifier (Platt Scaling / Isotonic Regression).
    Wraps a base classifier and calibrates its probability outputs to be
    well-calibrated — actual probability = predicted probability.
    """
    name = "Calibrated Classifier (SVM+Platt)"
    family = "supervised"
    task_type = "classification"
    tags = ["calibration", "probabilistic", "wrapper"]
    complexity_train = "O(base + calibration)"
    complexity_predict = "O(base)"

    def _build_model(self):
        from sklearn.svm import SVC as _SVC
        base = _SVC(kernel="rbf", random_state=self.random_state)
        return CalibratedClassifierCV(
            estimator=base,
            method=self._hyperparams.get("method", "sigmoid"),
            cv=self._hyperparams.get("cv", 5),
        )

    def score(self, X, y=None):
        proba = self.predict_proba(X)
        return classification_metrics(y, self.predict(X), proba)

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "method", "type": "select", "default": "sigmoid",
             "options": ["sigmoid", "isotonic"],
             "description": "'sigmoid'=Platt scaling, 'isotonic'=monotone calibration."},
            {"name": "cv", "type": "int", "default": 5, "min": 2, "max": 10,
             "description": "Number of cross-validation folds for calibration."},
        ]

    @classmethod
    def _math_intuition(cls):
        return "Platt: P(y=1|f) = 1/(1+exp(Af+B)) fitted on held-out scores. Isotonic: monotone regression."

    @classmethod
    def _assumptions(cls):
        return ["Base classifier provides useful (if not calibrated) scores.", "Enough data for calibration CV."]

    @classmethod
    def _strengths(cls):
        return ["Well-calibrated probabilities.", "Works with any base classifier.", "Improved reliability."]

    @classmethod
    def _weaknesses(cls):
        return ["Additional fitting time for calibration.", "Isotonic overfits with small data."]

    @classmethod
    def _use_cases(cls):
        return ["Medical decision support", "Risk scoring", "Any application needing reliable probabilities"]



