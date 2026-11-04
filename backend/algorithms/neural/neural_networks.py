"""
Neural Network Algorithms for ShiftLab — Phase 3B.

Pure-NumPy custom implementations:
  N-01: PerceptronCustomAlgo     — single-layer perceptron, manual weight updates
  N-02: MLPCustomAlgo            — full backprop MLP (configurable layers + activations)
  N-03: RBFNetworkAlgo           — radial basis function network
  N-04: AutoencoderAlgo          — tied-weight autoencoder for anomaly scoring / dim-reduction

Optimizers available for MLP (all pure NumPy):
  "sgd", "sgd_momentum", "rmsprop", "adam"

Activations available:
  "sigmoid", "relu", "tanh", "leaky_relu", "linear"

All algorithms register with family="neural".
No torch, no tensorflow, no keras — CPU-friendly, 8 GB RAM.
"""

from __future__ import annotations

import numpy as np

from backend.core.base_algorithm import BaseAlgorithm
from backend.core.metrics import regression_metrics, classification_metrics
from backend.core.registry import AlgorithmRegistry


# ============================================================
# Shared activation / gradient utilities
# ============================================================

def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid. np.where evaluates both branches, so the
    # exp() calls are clipped to avoid overflow for large |z|.
    z = np.clip(z, -500.0, 500.0)
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def _sigmoid_grad(a: np.ndarray) -> np.ndarray:
    return a * (1.0 - a)


def _relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def _relu_grad(a: np.ndarray) -> np.ndarray:
    return (a > 0).astype(float)


def _leaky_relu(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(z > 0, z, alpha * z)


def _leaky_relu_grad(a: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(a > 0, 1.0, alpha)


def _tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)


def _tanh_grad(a: np.ndarray) -> np.ndarray:
    return 1.0 - a ** 2


def _linear(z: np.ndarray) -> np.ndarray:
    return z


def _linear_grad(a: np.ndarray) -> np.ndarray:
    return np.ones_like(a)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(np.clip(z, -500.0, 0.0))
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)


def _clip_gradients(grads: list[np.ndarray], max_norm: float = 5.0) -> list[np.ndarray]:
    """Global-norm gradient clipping. Prevents exploding weights on unstable
    configurations (e.g. high learning rate + momentum regression)."""
    if max_norm is None or max_norm <= 0:
        return grads
    total = 0.0
    for g in grads:
        total += float(np.sum(np.square(g)))
    total = float(np.sqrt(total))
    if not np.isfinite(total):
        # A non-finite gradient has already destroyed the step — zero it.
        return [np.zeros_like(g) for g in grads]
    if total > max_norm:
        scale = max_norm / (total + 1e-12)
        return [g * scale for g in grads]
    return grads


ACTIVATIONS = {
    "sigmoid": (_sigmoid, _sigmoid_grad),
    "relu": (_relu, _relu_grad),
    "leaky_relu": (_leaky_relu, _leaky_relu_grad),
    "tanh": (_tanh, _tanh_grad),
    "linear": (_linear, _linear_grad),
}


# ============================================================
# Optimizer state holders
# ============================================================

class _SGD:
    def __init__(self, lr: float, **_):
        self.lr = lr

    def update(self, param, grad, state_key=""):
        return param - self.lr * grad, {}


class _SGDMomentum:
    def __init__(self, lr: float, momentum: float = 0.9, **_):
        self.lr = lr
        self.momentum = momentum
        self._v: dict[str, np.ndarray] = {}

    def update(self, param, grad, state_key=""):
        v = self._v.get(state_key, np.zeros_like(param))
        v = self.momentum * v + self.lr * grad
        self._v[state_key] = v
        return param - v, {}


class _RMSProp:
    def __init__(self, lr: float, rho: float = 0.9, eps: float = 1e-8, **_):
        self.lr = lr
        self.rho = rho
        self.eps = eps
        self._eg2: dict[str, np.ndarray] = {}

    def update(self, param, grad, state_key=""):
        eg2 = self._eg2.get(state_key, np.zeros_like(param))
        eg2 = self.rho * eg2 + (1 - self.rho) * grad ** 2
        self._eg2[state_key] = eg2
        return param - self.lr * grad / (np.sqrt(eg2) + self.eps), {}


class _Adam:
    def __init__(self, lr: float, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8, **_):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._m: dict[str, np.ndarray] = {}
        self._v: dict[str, np.ndarray] = {}
        self._t: dict[str, int] = {}

    def update(self, param, grad, state_key=""):
        m = self._m.get(state_key, np.zeros_like(param))
        v = self._v.get(state_key, np.zeros_like(param))
        t = self._t.get(state_key, 0) + 1
        m = self.beta1 * m + (1 - self.beta1) * grad
        v = self.beta2 * v + (1 - self.beta2) * grad ** 2
        m_hat = m / (1 - self.beta1 ** t)
        v_hat = v / (1 - self.beta2 ** t)
        self._m[state_key] = m
        self._v[state_key] = v
        self._t[state_key] = t
        return param - self.lr * m_hat / (np.sqrt(v_hat) + self.eps), {}


def _build_optimizer(name: str, lr: float, **kwargs):
    opts = {
        "sgd": _SGD,
        "sgd_momentum": _SGDMomentum,
        "rmsprop": _RMSProp,
        "adam": _Adam,
    }
    cls = opts.get(name, _Adam)
    return cls(lr=lr, **kwargs)


# ============================================================
# N-01: Perceptron (single-layer, manual weight updates)
# ============================================================

@AlgorithmRegistry.register
class PerceptronCustomAlgo(BaseAlgorithm):
    """
    Custom Single-Layer Perceptron (NumPy).
    The classical Rosenblatt Perceptron (1958): a linear threshold classifier
    with online weight updates. Converges on linearly separable data.
    Uses step activation (sign of dot product). For multi-class problems,
    one-vs-rest strategy is applied manually.

    Trained with the Perceptron update rule:
      w ← w + lr · (y_true - y_pred) · x
    No probabilistic output — hard assignments only.
    """
    name = "Perceptron (Custom NumPy)"
    family = "neural"
    task_type = "classification"
    sklearn_compatible = False
    tags = ["perceptron", "linear", "online-learning", "numpy", "custom"]
    complexity_train = "O(epochs · n · d)"
    complexity_predict = "O(n · d)"

    # Fitted attributes
    _weights: np.ndarray      # (n_classes, n_features)
    _biases: np.ndarray       # (n_classes,)
    _classes: np.ndarray      # unique class labels

    def _build_model(self):
        # No external model object — fully self-contained
        return None

    def _fit_impl(self, X_train: np.ndarray, y_train: np.ndarray | None) -> None:
        rng = np.random.RandomState(self.random_state)
        lr = self._hyperparams.get("learning_rate", 0.01)
        epochs = self._hyperparams.get("epochs", 100)

        self._classes = np.unique(y_train)
        n_classes = len(self._classes)
        n_features = X_train.shape[1]

        # OVR: one weight vector per class
        self._weights = rng.randn(n_classes, n_features) * 0.01
        self._biases = np.zeros(n_classes)

        # Encode y as ±1 per class (OVR)
        for epoch in range(epochs):
            for i in range(len(X_train)):
                xi = X_train[i]
                for ci, c in enumerate(self._classes):
                    yi = 1 if y_train[i] == c else -1
                    # Linear score
                    score = np.dot(self._weights[ci], xi) + self._biases[ci]
                    y_hat = 1 if score >= 0 else -1
                    if y_hat != yi:
                        self._weights[ci] += lr * yi * xi
                        self._biases[ci] += lr * yi

    def _predict_impl(self, X: np.ndarray) -> np.ndarray:
        # Score all classes; pick argmax
        scores = X @ self._weights.T + self._biases  # (n_samples, n_classes)
        return self._classes[np.argmax(scores, axis=1)]

    def score(self, X, y=None):
        preds = self.predict(X)
        return classification_metrics(y, preds)

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "learning_rate", "type": "float", "default": 0.01,
             "min": 1e-4, "max": 1.0,
             "description": "Step size for weight updates."},
            {"name": "epochs", "type": "int", "default": 100,
             "min": 1, "max": 1000,
             "description": "Number of passes over the training data."},
        ]

    @classmethod
    def _math_intuition(cls) -> str:
        return (
            "Update rule: w ← w + η·(y−ŷ)·x where y,ŷ ∈ {-1,+1}. "
            "Converges in finite steps if data is linearly separable "
            "(Perceptron Convergence Theorem)."
        )

    @classmethod
    def _assumptions(cls):
        return ["Data is (or nearly is) linearly separable.",
                "Features are on comparable scales."]

    @classmethod
    def _strengths(cls):
        return ["Extremely fast online learning.",
                "Simple and interpretable weight vectors.",
                "No probabilistic assumptions required."]

    @classmethod
    def _weaknesses(cls):
        return ["Fails on non-linearly separable data.",
                "No probabilistic output.",
                "Sensitive to feature scale.",
                "Does not converge on non-separable data."]

    @classmethod
    def _use_cases(cls):
        return ["Binary / multi-class classification baselines.",
                "Educational demonstration of neural weight updates.",
                "Online streaming classification."]


# ============================================================
# N-02: MLP with full backpropagation (configurable)
# ============================================================

class _MLPEngine:
    """
    Pure-NumPy multi-layer perceptron engine used by MLPCustomAlgo.
    Supports both classification and regression.
    """

    def __init__(self, layer_sizes: list[int], activation: str,
                 output_activation: str, task: str,
                 optimizer: str, lr: float, epochs: int,
                 batch_size: int, l2: float, rng: np.random.RandomState,
                 grad_clip: float = 5.0):
        self.layer_sizes = layer_sizes
        self.act_name = activation
        self.out_act_name = output_activation
        self.task = task
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.grad_clip = grad_clip
        self.rng = rng
        self.opt = _build_optimizer(optimizer, lr)

        self.act_fn, self.act_grad = ACTIVATIONS[activation]

        # Weights and biases: list of (W, b) per layer
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        self._init_done = False

    def _init_params(self, n_in: int):
        self.weights = []
        self.biases = []
        sizes = [n_in] + self.layer_sizes
        for i in range(len(sizes) - 1):
            # He init for relu, Xavier for others
            if self.act_name == "relu":
                scale = np.sqrt(2.0 / sizes[i])
            else:
                scale = np.sqrt(1.0 / sizes[i])
            W = self.rng.randn(sizes[i], sizes[i + 1]) * scale
            b = np.zeros(sizes[i + 1])
            self.weights.append(W)
            self.biases.append(b)
        self._init_done = True

    def _forward(self, X: np.ndarray):
        """Return list of (pre-activation z, post-activation a) per layer."""
        activations = []
        a = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ W + b
            is_last = (i == len(self.weights) - 1)
            if is_last:
                if self.out_act_name == "softmax":
                    a = _softmax(z)
                elif self.out_act_name == "sigmoid":
                    a = _sigmoid(z)
                elif self.out_act_name == "linear":
                    a = z
                else:
                    a = self.act_fn(z)
            else:
                a = self.act_fn(z)
            activations.append((z, a))
        return activations

    def _loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        eps = 1e-12
        if self.task == "classification":
            if self.out_act_name == "softmax":
                # Categorical cross-entropy
                n = len(y_true)
                log_pred = np.log(y_pred + eps)
                return -float(np.sum(log_pred[np.arange(n), y_true.astype(int)]) / n)
            else:
                # Binary cross-entropy
                return -float(np.mean(y_true * np.log(y_pred + eps)
                                      + (1 - y_true) * np.log(1 - y_pred + eps)))
        else:
            # MSE for regression
            return float(np.mean((y_true - y_pred.ravel()) ** 2))

    def _backward(self, X: np.ndarray, y: np.ndarray,
                  activations: list[tuple]) -> tuple[list, list]:
        """Return gradients dW, db for each layer (reversed → forward order)."""
        n = len(X)
        grads_W: list[np.ndarray | None] = [None] * len(self.weights)
        grads_b: list[np.ndarray | None] = [None] * len(self.weights)

        # Output layer delta
        _, a_out = activations[-1]
        if self.task == "classification" and self.out_act_name == "softmax":
            # Cross-entropy + softmax: delta = a - one_hot(y)
            one_hot = np.zeros_like(a_out)
            one_hot[np.arange(n), y.astype(int)] = 1
            delta = (a_out - one_hot) / n
        elif self.task == "classification" and self.out_act_name == "sigmoid":
            # Binary cross-entropy + sigmoid: delta = a - y
            delta = (a_out.ravel() - y).reshape(-1, 1) / n
        else:
            # MSE + linear output: delta = 2(a - y)
            delta = (a_out.ravel() - y).reshape(-1, 1) * 2 / n

        for i in reversed(range(len(self.weights))):
            a_in = X if i == 0 else activations[i - 1][1]
            grads_W[i] = a_in.T @ delta + self.l2 * self.weights[i]
            grads_b[i] = delta.sum(axis=0)

            if i > 0:
                _, a_prev = activations[i - 1]
                delta = (delta @ self.weights[i].T) * self.act_grad(a_prev)

        # Clip the whole gradient set with a shared norm so relative layer
        # magnitudes are preserved.
        flat = [g for g in grads_W] + [g for g in grads_b]
        clipped = _clip_gradients(flat, self.grad_clip)
        n = len(self.weights)
        grads_W = clipped[:n]
        grads_b = clipped[n:]
        return grads_W, grads_b

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_in = X.shape[1]
        if not self._init_done:
            self._init_params(n_in)

        self.loss_curve_: list[float] = []
        n = len(X)

        for epoch in range(self.epochs):
            idx = self.rng.permutation(n)
            X_s, y_s = X[idx], y[idx]
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n, self.batch_size):
                Xb = X_s[start:start + self.batch_size]
                yb = y_s[start:start + self.batch_size]
                acts = self._forward(Xb)
                _, a_out = acts[-1]
                epoch_loss += self._loss(yb, a_out)
                n_batches += 1
                gW, gb = self._backward(Xb, yb, acts)
                for i in range(len(self.weights)):
                    self.weights[i], _ = self.opt.update(
                        self.weights[i], gW[i], state_key=f"W{i}"
                    )
                    self.biases[i], _ = self.opt.update(
                        self.biases[i], gb[i], state_key=f"b{i}"
                    )
            self.loss_curve_.append(epoch_loss / max(n_batches, 1))

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        acts = self._forward(X)
        return acts[-1][1]


# ────────────────────────────────────────────────────────────

@AlgorithmRegistry.register
class MLPClassifierCustomAlgo(BaseAlgorithm):
    """
    Custom MLP Classifier (NumPy Backpropagation).
    Fully connected feedforward network trained with mini-batch backpropagation.
    Configurable hidden layer sizes, activation functions, and optimizer.

    Supports hidden activations: sigmoid, relu, tanh, leaky_relu.
    Uses softmax output for multi-class and sigmoid for binary classification.
    Optimizers: sgd, sgd_momentum, rmsprop, adam.
    No torch, no keras — pure NumPy forward/backward passes.
    """
    name = "MLP Classifier (Custom NumPy)"
    family = "neural"
    task_type = "classification"
    sklearn_compatible = False
    tags = ["mlp", "backpropagation", "neural-network", "numpy", "custom",
            "softmax", "relu", "adam"]
    complexity_train = "O(epochs · n · d · hidden)"
    complexity_predict = "O(n · d · hidden)"

    _engine: _MLPEngine
    _classes: np.ndarray

    def _build_model(self):
        return None  # engine built in _fit_impl

    def _fit_impl(self, X_train: np.ndarray, y_train: np.ndarray | None) -> None:
        rng = np.random.RandomState(self.random_state)
        self._classes = np.unique(y_train)
        n_classes = len(self._classes)

        # Map labels to 0..n_classes-1
        label_map = {c: i for i, c in enumerate(self._classes)}
        y_enc = np.array([label_map[yi] for yi in y_train])

        hidden = list(self._hyperparams.get("hidden_layer_sizes", [64, 32]))
        layer_sizes = hidden + [n_classes]
        out_act = "sigmoid" if n_classes == 2 else "softmax"
        if n_classes == 2:
            layer_sizes = hidden + [1]

        self._engine = _MLPEngine(
            layer_sizes=layer_sizes,
            activation=self._hyperparams.get("activation", "relu"),
            output_activation=out_act,
            task="classification",
            optimizer=self._hyperparams.get("optimizer", "adam"),
            lr=self._hyperparams.get("learning_rate", 0.001),
            epochs=self._hyperparams.get("epochs", 100),
            batch_size=self._hyperparams.get("batch_size", 32),
            l2=self._hyperparams.get("l2", 1e-4),
            grad_clip=self._hyperparams.get("grad_clip", 5.0),
            rng=rng,
        )
        y_fit = y_enc if n_classes > 2 else y_enc.astype(float)
        self._engine.fit(X_train, y_fit)
        self._n_classes = n_classes

    def _predict_impl(self, X: np.ndarray) -> np.ndarray:
        raw = self._engine.predict_raw(X)
        if self._n_classes == 2:
            idx = (raw.ravel() >= 0.5).astype(int)
        else:
            idx = np.argmax(raw, axis=1)
        return self._classes[idx]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._assert_fitted()
        raw = self._engine.predict_raw(X)
        if self._n_classes == 2:
            p = raw.ravel().reshape(-1, 1)
            return np.hstack([1 - p, p])
        return raw  # softmax output

    def score(self, X, y=None):
        preds = self.predict(X)
        proba = self.predict_proba(X)
        return classification_metrics(y, preds, proba)

    def get_loss_curve(self) -> list[float]:
        """Return per-epoch training loss."""
        return self._engine.loss_curve_ if hasattr(self._engine, "loss_curve_") else []

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "hidden_layer_sizes", "type": "list[int]",
             "default": [64, 32], "description": "List of hidden layer sizes."},
            {"name": "activation", "type": "select", "default": "relu",
             "options": ["relu", "sigmoid", "tanh", "leaky_relu"],
             "description": "Hidden layer activation function."},
            {"name": "optimizer", "type": "select", "default": "adam",
             "options": ["sgd", "sgd_momentum", "rmsprop", "adam"],
             "description": "Weight update optimizer."},
            {"name": "learning_rate", "type": "float", "default": 0.001,
             "min": 1e-5, "max": 1.0,
             "description": "Learning rate for optimizer."},
            {"name": "epochs", "type": "int", "default": 100,
             "min": 1, "max": 2000,
             "description": "Number of training epochs."},
            {"name": "batch_size", "type": "int", "default": 32,
             "min": 1, "max": 512,
             "description": "Mini-batch size."},
            {"name": "l2", "type": "float", "default": 1e-4,
             "min": 0.0, "max": 1.0,
             "description": "L2 regularization weight."},
        ]

    @classmethod
    def _math_intuition(cls) -> str:
        return (
            "Forward pass: a^(l) = σ(W^(l)·a^(l-1) + b^(l)). "
            "Loss: categorical cross-entropy. "
            "Backward pass: δ^(l) = (δ^(l+1)·W^(l+1)ᵀ) ⊙ σ'(z^(l)). "
            "Weight update via chosen optimizer (SGD / Momentum / RMSProp / Adam)."
        )

    @classmethod
    def _assumptions(cls):
        return ["Features are normalized (zero-mean, unit-variance recommended).",
                "Sufficient training data relative to network capacity."]

    @classmethod
    def _strengths(cls):
        return ["Universal function approximation.",
                "Handles non-linearly separable data.",
                "Multiple optimizer choices.",
                "Configurable depth and width."]

    @classmethod
    def _weaknesses(cls):
        return ["Sensitive to hyperparameters (lr, architecture).",
                "Slower than sklearn MLP for large datasets.",
                "No GPU acceleration (pure NumPy)."]

    @classmethod
    def _use_cases(cls):
        return ["Classification with complex decision boundaries.",
                "Teaching backpropagation internals.",
                "Comparing optimizer convergence behaviour."]


@AlgorithmRegistry.register
class MLPRegressorCustomAlgo(BaseAlgorithm):
    """
    Custom MLP Regressor (NumPy Backpropagation).
    Feedforward network trained with mini-batch backpropagation using MSE loss.
    Linear output activation — no squashing at the final layer.
    Configurable hidden layers, activation, and optimizer.
    """
    name = "MLP Regressor (Custom NumPy)"
    family = "neural"
    task_type = "regression"
    sklearn_compatible = False
    tags = ["mlp", "backpropagation", "neural-network", "numpy", "custom",
            "regression", "adam"]
    complexity_train = "O(epochs · n · d · hidden)"
    complexity_predict = "O(n · d · hidden)"

    _engine: _MLPEngine

    def _build_model(self):
        return None

    def _fit_impl(self, X_train: np.ndarray, y_train: np.ndarray | None) -> None:
        rng = np.random.RandomState(self.random_state)
        hidden = list(self._hyperparams.get("hidden_layer_sizes", [64, 32]))
        layer_sizes = hidden + [1]

        self._engine = _MLPEngine(
            layer_sizes=layer_sizes,
            activation=self._hyperparams.get("activation", "relu"),
            output_activation="linear",
            task="regression",
            optimizer=self._hyperparams.get("optimizer", "adam"),
            lr=self._hyperparams.get("learning_rate", 0.001),
            epochs=self._hyperparams.get("epochs", 100),
            batch_size=self._hyperparams.get("batch_size", 32),
            l2=self._hyperparams.get("l2", 1e-4),
            grad_clip=self._hyperparams.get("grad_clip", 5.0),
            rng=rng,
        )
        self._engine.fit(X_train, y_train.astype(float))

    def _predict_impl(self, X: np.ndarray) -> np.ndarray:
        return self._engine.predict_raw(X).ravel()

    def score(self, X, y=None):
        return regression_metrics(y, self.predict(X))

    def get_loss_curve(self) -> list[float]:
        return self._engine.loss_curve_ if hasattr(self._engine, "loss_curve_") else []

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "hidden_layer_sizes", "type": "list[int]",
             "default": [64, 32], "description": "List of hidden layer sizes."},
            {"name": "activation", "type": "select", "default": "relu",
             "options": ["relu", "sigmoid", "tanh", "leaky_relu"],
             "description": "Hidden layer activation function."},
            {"name": "optimizer", "type": "select", "default": "adam",
             "options": ["sgd", "sgd_momentum", "rmsprop", "adam"],
             "description": "Weight update optimizer."},
            {"name": "learning_rate", "type": "float", "default": 0.001,
             "min": 1e-5, "max": 1.0, "description": "Learning rate."},
            {"name": "epochs", "type": "int", "default": 100,
             "min": 1, "max": 2000, "description": "Training epochs."},
            {"name": "batch_size", "type": "int", "default": 32,
             "min": 1, "max": 512, "description": "Mini-batch size."},
            {"name": "l2", "type": "float", "default": 1e-4,
             "min": 0.0, "max": 1.0, "description": "L2 regularization."},
        ]

    @classmethod
    def _math_intuition(cls) -> str:
        return (
            "Loss: MSE = (1/n)Σ(yᵢ - ŷᵢ)². "
            "Output: linear activation (unbounded). "
            "Gradients flow through hidden relu/tanh layers via chain rule."
        )

    @classmethod
    def _assumptions(cls):
        return ["Target is continuous.", "Features normalized."]

    @classmethod
    def _strengths(cls):
        return ["Learns non-linear input-output mappings.",
                "Configurable optimizer (SGD/Adam/RMSProp)."]

    @classmethod
    def _weaknesses(cls):
        return ["Requires hyperparameter tuning.",
                "Slower than linear models for simple tasks."]

    @classmethod
    def _use_cases(cls):
        return ["Non-linear regression.", "Function approximation."]


# ============================================================
# N-03: RBF Network
# ============================================================

@AlgorithmRegistry.register
class RBFNetworkAlgo(BaseAlgorithm):
    """
    Radial Basis Function (RBF) Network.
    A two-layer network where the hidden layer computes Gaussian kernel responses
    centred on prototype vectors (cluster centres). Output layer is a linear
    combination of RBF activations, fitted with Ridge regression.

    Architecture:
      Input → RBF hidden layer (Gaussian kernels) → Linear output (Ridge).

    Centre selection: K-Means on training data.
    Width (σ): heuristic from max pairwise centre distance, or user-specified.
    """
    name = "RBF Network"
    family = "neural"
    task_type = "classification"
    sklearn_compatible = False
    tags = ["rbf", "kernel", "radial-basis", "numpy", "custom"]
    complexity_train = "O(k·n·d + n·k²)"
    complexity_predict = "O(n · k)"

    _centres: np.ndarray      # (k, d)
    _sigma: float
    _output_weights: np.ndarray   # (k+1, n_classes) — includes bias
    _classes: np.ndarray

    def _build_model(self):
        return None

    def _rbf_features(self, X: np.ndarray) -> np.ndarray:
        """Compute RBF activations: φᵢ(x) = exp(-||x - cᵢ||² / (2σ²))."""
        # X: (n, d), centres: (k, d) → out: (n, k)
        diff = X[:, None, :] - self._centres[None, :, :]   # (n, k, d)
        dist2 = np.sum(diff ** 2, axis=2)                   # (n, k)
        return np.exp(-dist2 / (2 * self._sigma ** 2))

    def _fit_impl(self, X_train: np.ndarray, y_train: np.ndarray | None) -> None:
        from sklearn.cluster import KMeans
        from sklearn.linear_model import Ridge

        rng = np.random.RandomState(self.random_state)
        k = self._hyperparams.get("n_centers", 20)
        l2 = self._hyperparams.get("l2", 1.0)

        # Step 1: select centres via KMeans
        k = min(k, len(X_train))
        km = KMeans(n_clusters=k, random_state=self.random_state, n_init=5)
        km.fit(X_train)
        self._centres = km.cluster_centers_

        # Step 2: compute σ as mean nearest-centre distance (heuristic)
        user_sigma = self._hyperparams.get("sigma", None)
        if user_sigma is not None:
            self._sigma = float(user_sigma)
        else:
            from sklearn.metrics import pairwise_distances
            D = pairwise_distances(self._centres)
            np.fill_diagonal(D, np.inf)
            self._sigma = float(np.mean(np.min(D, axis=1))) or 1.0

        # Step 3: compute RBF feature matrix
        Phi = self._rbf_features(X_train)  # (n, k)

        # Step 4: fit output layer with Ridge
        self._classes = np.unique(y_train)
        n_classes = len(self._classes)

        if n_classes == 2:
            self._output_ridge = Ridge(alpha=l2)
            self._output_ridge.fit(Phi, (y_train == self._classes[1]).astype(float))
            self._mode = "binary"
        else:
            # One-hot encode
            n = len(y_train)
            Y_oh = np.zeros((n, n_classes))
            for ci, c in enumerate(self._classes):
                Y_oh[y_train == c, ci] = 1.0
            self._output_ridge = Ridge(alpha=l2)
            self._output_ridge.fit(Phi, Y_oh)
            self._mode = "multiclass"

    def _predict_impl(self, X: np.ndarray) -> np.ndarray:
        Phi = self._rbf_features(X)
        out = self._output_ridge.predict(Phi)
        if self._mode == "binary":
            idx = (out >= 0.5).astype(int)
            return self._classes[idx]
        else:
            idx = np.argmax(out, axis=1)
            return self._classes[idx]

    def score(self, X, y=None):
        return classification_metrics(y, self.predict(X))

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "n_centers", "type": "int", "default": 20,
             "min": 2, "max": 200,
             "description": "Number of RBF centres (K-Means clusters)."},
            {"name": "sigma", "type": "float", "default": None,
             "description": "Gaussian kernel width σ. None = auto from centre distances."},
            {"name": "l2", "type": "float", "default": 1.0,
             "min": 1e-4, "max": 100.0,
             "description": "Ridge regularization for output layer."},
        ]

    @classmethod
    def _math_intuition(cls) -> str:
        return (
            "φᵢ(x) = exp(−‖x − cᵢ‖² / 2σ²). "
            "Output: ŷ = Σᵢ wᵢ·φᵢ(x) + b  (fitted by Ridge regression). "
            "Universal approximator for smooth functions."
        )

    @classmethod
    def _assumptions(cls):
        return ["Centres capture input distribution (KMeans gives good coverage).",
                "σ reflects local neighbourhood scale."]

    @classmethod
    def _strengths(cls):
        return ["Fast training (one-shot Ridge solve after KMeans).",
                "Interpretable hidden layer (Gaussian kernel responses).",
                "Good for smooth, local decision boundaries."]

    @classmethod
    def _weaknesses(cls):
        return ["Scalability: large k increases feature matrix size.",
                "Kernel width σ critical — poor choice degrades performance.",
                "Not naturally suited to very high-dimensional sparse data."]

    @classmethod
    def _use_cases(cls):
        return ["Function approximation.", "Low/medium-dimensional classification.",
                "Comparison against MLP for non-linear tasks."]


# ============================================================
# N-04: Autoencoder (pure NumPy, tied weights)
# ============================================================

@AlgorithmRegistry.register
class AutoencoderAlgo(BaseAlgorithm):
    """
    Tied-Weight Autoencoder (NumPy).
    Encoder: Input → latent (bottleneck). Decoder: latent → reconstruction.
    Trained to minimise reconstruction MSE. Anomaly score = reconstruction error.
    Can also expose the latent encoding as a dimensionality reduction.

    Tied weights: decoder weights = encoder weights transposed (fewer parameters).
    Supports configurable latent dimension, hidden size, and optimizer.
    """
    name = "Autoencoder (Custom NumPy)"
    family = "neural"
    task_type = "anomaly_detection"
    sklearn_compatible = False
    tags = ["autoencoder", "reconstruction", "anomaly-detection",
            "dimensionality-reduction", "numpy", "custom", "unsupervised"]
    complexity_train = "O(epochs · n · d · hidden)"
    complexity_predict = "O(n · d · hidden)"

    # Encoder parameters (decoder = tied)
    _W1: np.ndarray    # (d_in, hidden)
    _b1: np.ndarray    # (hidden,)
    _W2: np.ndarray    # (hidden, latent)
    _b2: np.ndarray    # (latent,)
    _b3: np.ndarray    # (hidden,)   — decoder bias (tied W2.T)
    _b4: np.ndarray    # (d_in,)     — reconstruction bias
    _threshold: float  # anomaly threshold (mean + k*std on training errors)

    def _build_model(self):
        return None

    def _encode(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Encoder: X → hidden → latent. Returns (h, z)."""
        act_fn = ACTIVATIONS[self._act][0]
        h = act_fn(X @ self._W1 + self._b1)
        z = act_fn(h @ self._W2 + self._b2)
        return h, z

    def _decode(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Decoder (tied weights): latent → hidden → reconstruction."""
        act_fn = ACTIVATIONS[self._act][0]
        h_dec = act_fn(z @ self._W2.T + self._b3)
        x_rec = h_dec @ self._W1.T + self._b4
        return h_dec, x_rec

    def _forward_ae(self, X: np.ndarray):
        h, z = self._encode(X)
        h_dec, x_rec = self._decode(z)
        return h, z, h_dec, x_rec

    def _fit_impl(self, X_train: np.ndarray, y_train: np.ndarray | None) -> None:
        rng = np.random.RandomState(self.random_state)
        d_in = X_train.shape[1]
        hidden = self._hyperparams.get("hidden_size", 32)
        latent = self._hyperparams.get("latent_dim", 8)
        latent = min(latent, hidden, d_in)
        hidden = min(hidden, d_in)
        epochs = self._hyperparams.get("epochs", 100)
        batch_size = self._hyperparams.get("batch_size", 32)
        lr = self._hyperparams.get("learning_rate", 0.001)
        self._act = self._hyperparams.get("activation", "relu")
        threshold_k = self._hyperparams.get("threshold_k", 2.0)

        # Initialize tied weights
        scale = np.sqrt(2.0 / d_in)
        self._W1 = rng.randn(d_in, hidden) * scale
        self._b1 = np.zeros(hidden)
        self._W2 = rng.randn(hidden, latent) * scale
        self._b2 = np.zeros(latent)
        self._b3 = np.zeros(hidden)   # decoder hidden bias
        self._b4 = np.zeros(d_in)     # reconstruction bias

        act_fn, act_grad = ACTIVATIONS[self._act]
        opt = _build_optimizer(
            self._hyperparams.get("optimizer", "adam"), lr
        )

        loss_curve = []
        n = len(X_train)

        for epoch in range(epochs):
            idx = rng.permutation(n)
            Xs = X_train[idx]
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n, batch_size):
                Xb = Xs[start:start + batch_size]
                nb = len(Xb)

                # ─── Forward ───
                h = act_fn(Xb @ self._W1 + self._b1)          # (nb, hidden)
                z = act_fn(h @ self._W2 + self._b2)            # (nb, latent)
                h_dec = act_fn(z @ self._W2.T + self._b3)      # (nb, hidden)
                x_rec = h_dec @ self._W1.T + self._b4          # (nb, d_in)

                loss = float(np.mean((Xb - x_rec) ** 2))
                epoch_loss += loss
                n_batches += 1

                # ─── Backward (MSE loss) ───
                # δ_rec = dL/dx_rec = -2(Xb - x_rec)/nb
                d_xrec = -2 * (Xb - x_rec) / nb               # (nb, d_in)

                # Decoder reconstruction layer
                dW1_dec = h_dec.T @ d_xrec                     # (hidden, d_in)
                db4 = d_xrec.sum(axis=0)
                d_hdec_in = d_xrec @ self._W1                  # (nb, hidden)
                d_hdec = d_hdec_in * act_grad(h_dec)

                # Decoder hidden layer (tied W2.T)
                dW2_dec_T = z.T @ d_hdec                       # (latent, hidden)
                db3 = d_hdec.sum(axis=0)
                d_z_in = d_hdec @ self._W2                     # (nb, latent)
                d_z = d_z_in * act_grad(z)

                # Encoder latent layer
                dW2_enc = h.T @ d_z                            # (hidden, latent)
                db2 = d_z.sum(axis=0)
                d_h_in = d_z @ self._W2.T                      # (nb, hidden)
                d_h = d_h_in * act_grad(h)

                # Encoder hidden layer
                dW1_enc = Xb.T @ d_h                           # (d_in, hidden)
                db1 = d_h.sum(axis=0)

                # Combined W1 gradient (encoder + decoder tied)
                dW1 = dW1_enc + dW1_dec.T
                dW2 = dW2_enc + dW2_dec_T.T

                # Gradient clipping keeps reconstruction training stable with
                # aggressive optimizers / learning rates.
                grads = _clip_gradients(
                    [dW1, db1, dW2, db2, db3, db4],
                    self._hyperparams.get("grad_clip", 5.0),
                )
                dW1, db1, dW2, db2, db3, db4 = grads

                # Update
                self._W1, _ = opt.update(self._W1, dW1, "W1")
                self._b1, _ = opt.update(self._b1, db1, "b1")
                self._W2, _ = opt.update(self._W2, dW2, "W2")
                self._b2, _ = opt.update(self._b2, db2, "b2")
                self._b3, _ = opt.update(self._b3, db3, "b3")
                self._b4, _ = opt.update(self._b4, db4, "b4")

            loss_curve.append(epoch_loss / max(n_batches, 1))

        self.loss_curve_ = loss_curve

        # Compute anomaly threshold on training data
        train_errors = self._reconstruction_error(X_train)
        self._threshold = float(np.mean(train_errors) + threshold_k * np.std(train_errors))

    def _reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Per-sample MSE reconstruction error."""
        act_fn = ACTIVATIONS[self._act][0]
        h = act_fn(X @ self._W1 + self._b1)
        z = act_fn(h @ self._W2 + self._b2)
        h_dec = act_fn(z @ self._W2.T + self._b3)
        x_rec = h_dec @ self._W1.T + self._b4
        return np.mean((X - x_rec) ** 2, axis=1)

    def _predict_impl(self, X: np.ndarray) -> np.ndarray:
        """Return 1 (anomaly) or 0 (normal) based on reconstruction error."""
        errors = self._reconstruction_error(X)
        return (errors > self._threshold).astype(int)

    def get_embedding(self, X: np.ndarray) -> np.ndarray:
        """Return latent-space encoding for dimensionality reduction."""
        self._assert_fitted()
        act_fn = ACTIVATIONS[self._act][0]
        h = act_fn(X @ self._W1 + self._b1)
        return act_fn(h @ self._W2 + self._b2)

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Return raw reconstruction error per sample (higher = more anomalous)."""
        self._assert_fitted()
        return self._reconstruction_error(X)

    def score(self, X, y=None):
        from backend.core.metrics import anomaly_metrics
        preds = self.predict(X)
        scores = self.anomaly_scores(X)
        if y is not None:
            return anomaly_metrics(y, preds, scores)
        # Unsupervised: report mean reconstruction error only
        return {"mean_recon_error": float(np.mean(scores)),
                "threshold": float(self._threshold)}

    @classmethod
    def hyperparameter_schema(cls):
        return [
            {"name": "hidden_size", "type": "int", "default": 32,
             "min": 4, "max": 256,
             "description": "Hidden layer size in encoder/decoder."},
            {"name": "latent_dim", "type": "int", "default": 8,
             "min": 2, "max": 64,
             "description": "Bottleneck (latent space) dimension."},
            {"name": "activation", "type": "select", "default": "relu",
             "options": ["relu", "sigmoid", "tanh"],
             "description": "Hidden layer activation."},
            {"name": "optimizer", "type": "select", "default": "adam",
             "options": ["sgd", "sgd_momentum", "rmsprop", "adam"],
             "description": "Optimizer for weight updates."},
            {"name": "learning_rate", "type": "float", "default": 0.001,
             "min": 1e-5, "max": 0.1, "description": "Learning rate."},
            {"name": "epochs", "type": "int", "default": 100,
             "min": 10, "max": 1000, "description": "Training epochs."},
            {"name": "batch_size", "type": "int", "default": 32,
             "min": 8, "max": 256, "description": "Mini-batch size."},
            {"name": "threshold_k", "type": "float", "default": 2.0,
             "min": 0.5, "max": 5.0,
             "description": "Anomaly threshold = mean + k·std of training errors."},
        ]

    @classmethod
    def _math_intuition(cls) -> str:
        return (
            "Encoder: h=σ(W₁x+b₁), z=σ(W₂h+b₂). "
            "Decoder (tied): ĥ=σ(W₂ᵀz+b₃), x̂=W₁ᵀĥ+b₄. "
            "Loss: MSE(x, x̂). Anomaly score = ‖x − x̂‖². "
            "High reconstruction error ⟹ point lies off the data manifold ⟹ anomaly."
        )

    @classmethod
    def _assumptions(cls):
        return ["Normal data lies on a low-dimensional manifold.",
                "Anomalies are rare and differ significantly from the normal manifold."]

    @classmethod
    def _strengths(cls):
        return ["Unsupervised — no anomaly labels required.",
                "Latent space usable for dimensionality reduction.",
                "Tied weights reduce parameter count."]

    @classmethod
    def _weaknesses(cls):
        return ["Threshold k requires tuning (or validation labels).",
                "Slow on large datasets without GPU.",
                "May miss anomalies that lie on the manifold."]

    @classmethod
    def _use_cases(cls):
        return ["Fraud detection.", "Network intrusion detection.",
                "Industrial sensor anomaly detection.",
                "Latent feature extraction."]
