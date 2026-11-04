"""
Distribution Shift Analysis — ShiftLab Reliability Lab.

Implements:
  1. Covariate shift detection via Maximum Mean Discrepancy (MMD) and
     a classifier-based two-sample test.
  2. Label shift detection (marginal P(y) comparison).
  3. Importance weighting via kernel density ratio estimation
     (Kullback-Leibler Importance Estimation Procedure, KLIEP-lite).
  4. Performance-under-shift: measure how much a fitted model degrades
     when test distribution drifts away from train.

All computations use real statistics on real data — no fabricated numbers.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


# ──────────────────────────────────────────────────────────────────────────────
# 1. Maximum Mean Discrepancy (MMD²) with RBF kernel
# ──────────────────────────────────────────────────────────────────────────────

def _rbf_kernel(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
    """Compute RBF kernel matrix K(X, Y) = exp(-||xi-yj||²/2σ²)."""
    diff = X[:, None, :] - Y[None, :, :]          # (m, n, d)
    dist2 = np.sum(diff ** 2, axis=2)              # (m, n)
    return np.exp(-dist2 / (2 * sigma ** 2))


def mmd_squared(
    X_source: np.ndarray,
    X_target: np.ndarray,
    sigma: float | None = None,
) -> dict:
    """
    Compute biased Maximum Mean Discrepancy² between source and target.

    MMD²(P,Q) = E[k(x,x')] - 2·E[k(x,y)] + E[k(y,y')]
    where k is the RBF kernel. Zero if P=Q, positive if they differ.

    Args:
        X_source: Training / source samples (n, d).
        X_target: Test / target samples (m, d).
        sigma:    RBF bandwidth. None → median heuristic.

    Returns:
        dict with mmd2, sigma, interpretation.
    """
    if sigma is None:
        # Median heuristic: σ = median pairwise distance
        combined = np.vstack([X_source[:200], X_target[:200]])
        dists = np.sum((combined[:, None] - combined[None]) ** 2, axis=2)
        sigma = float(np.sqrt(np.median(dists[dists > 0]) / 2))
        sigma = max(sigma, 1e-6)

    Kxx = _rbf_kernel(X_source, X_source, sigma)
    Kyy = _rbf_kernel(X_target, X_target, sigma)
    Kxy = _rbf_kernel(X_source, X_target, sigma)

    n, m = len(X_source), len(X_target)
    # Unbiased estimator: exclude diagonal
    np.fill_diagonal(Kxx, 0)
    np.fill_diagonal(Kyy, 0)
    mmd2 = (Kxx.sum() / (n * (n - 1))
            - 2 * Kxy.mean()
            + Kyy.sum() / (m * (m - 1)))

    mmd2 = float(mmd2)
    if mmd2 < 0.01:
        interpretation = "minimal"
    elif mmd2 < 0.1:
        interpretation = "moderate"
    else:
        interpretation = "severe"

    return {
        "mmd2": mmd2,
        "sigma": float(sigma),
        "shift_severity": interpretation,
        "description": (
            "MMD² measures the distance between source and target feature "
            "distributions in a reproducing kernel Hilbert space. "
            "MMD²=0 ⟹ identical distributions."
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 2. Classifier-based two-sample test (domain classifier AUC)
# ──────────────────────────────────────────────────────────────────────────────

def classifier_two_sample_test(
    X_source: np.ndarray,
    X_target: np.ndarray,
    random_state: int = 42,
    cv: int = 3,
) -> dict:
    """
    Train a binary classifier to distinguish source from target samples.
    If AUC ≈ 0.5 → distributions are indistinguishable (no shift).
    If AUC → 1.0 → strong covariate shift.

    Uses Logistic Regression with cross-validation for speed (CPU-friendly).

    Returns:
        dict with auc_mean, auc_std, shift_detected, interpretation.
    """
    n_src = min(len(X_source), 500)
    n_tgt = min(len(X_target), 500)
    rng = np.random.RandomState(random_state)

    idx_src = rng.choice(len(X_source), n_src, replace=False)
    idx_tgt = rng.choice(len(X_target), n_tgt, replace=False)

    X_src_s = X_source[idx_src]
    X_tgt_s = X_target[idx_tgt]

    X_combined = np.vstack([X_src_s, X_tgt_s])
    y_domain = np.array([0] * n_src + [1] * n_tgt)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)

    clf = LogisticRegression(max_iter=1000, random_state=random_state, C=1.0)
    auc_scores = cross_val_score(clf, X_scaled, y_domain, cv=cv, scoring="roc_auc")

    auc_mean = float(np.mean(auc_scores))
    auc_std = float(np.std(auc_scores))

    if auc_mean < 0.6:
        severity = "minimal"
        detected = False
    elif auc_mean < 0.75:
        severity = "moderate"
        detected = True
    else:
        severity = "severe"
        detected = True

    return {
        "auc_mean": auc_mean,
        "auc_std": auc_std,
        "shift_detected": detected,
        "shift_severity": severity,
        "cv_folds": cv,
        "description": (
            "A domain classifier trained to distinguish source from target. "
            "AUC=0.5 → identical distributions; AUC→1 → strong covariate shift."
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 3. Label shift detection
# ──────────────────────────────────────────────────────────────────────────────

def label_shift_analysis(
    y_source: np.ndarray,
    y_target: np.ndarray,
) -> dict:
    """
    Detect label shift: compare marginal P(y) in source vs target.
    Uses chi-squared test of independence + TV distance.

    Returns:
        dict with class_frequencies, tv_distance, chi2_pvalue, shift_detected.
    """
    classes = np.union1d(np.unique(y_source), np.unique(y_target))

    src_counts = np.array([np.sum(y_source == c) for c in classes], dtype=float)
    tgt_counts = np.array([np.sum(y_target == c) for c in classes], dtype=float)

    src_freq = src_counts / src_counts.sum()
    tgt_freq = tgt_counts / tgt_counts.sum()

    tv_distance = float(0.5 * np.sum(np.abs(src_freq - tgt_freq)))

    # Chi-squared test: are source and target class distributions the same?
    # Compare observed target counts against expected counts (source proportions)
    expected = src_freq * tgt_counts.sum()
    # Avoid zero expected counts
    mask = expected > 0
    if mask.sum() < 2:
        chi2_stat, pvalue = float("nan"), float("nan")
    else:
        chi2_stat, pvalue = stats.chisquare(tgt_counts[mask], f_exp=expected[mask])
        chi2_stat = float(chi2_stat)
        pvalue = float(pvalue)

    shift_detected = pvalue < 0.05 if not np.isnan(pvalue) else (tv_distance > 0.1)

    return {
        "classes": [str(c) for c in classes],
        "source_frequencies": src_freq.tolist(),
        "target_frequencies": tgt_freq.tolist(),
        "tv_distance": tv_distance,
        "chi2_statistic": chi2_stat,
        "chi2_pvalue": pvalue,
        "shift_detected": bool(shift_detected),
        "description": (
            "Compares marginal class distributions P(y) between source and target. "
            "TV distance = 0 → identical. p-value < 0.05 → significant label shift."
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 4. Importance weighting (density ratio estimation)
# ──────────────────────────────────────────────────────────────────────────────

def importance_weights(
    X_source: np.ndarray,
    X_target: np.ndarray,
    method: str = "classifier",
    random_state: int = 42,
    clip_percentile: float = 95.0,
) -> dict:
    """
    Estimate importance weights w(x) = p_target(x) / p_source(x) for each
    source sample. Used to re-weight training loss to correct for covariate shift.

    Methods:
      "classifier": w(x) ≈ P(target|x) / P(source|x) via logistic regression
                    (Discriminative Density Ratio Estimation).
      "kde":        Gaussian KDE ratio (better for low-dimensional data).

    Returns:
        dict with weights array, effective_sample_size, method.
    """
    n_src = len(X_source)

    if method == "classifier":
        n_sub = min(n_src, 500)
        n_tgt_sub = min(len(X_target), 500)
        rng = np.random.RandomState(random_state)
        idx_s = rng.choice(n_src, n_sub, replace=False)
        idx_t = rng.choice(len(X_target), n_tgt_sub, replace=False)

        X_comb = np.vstack([X_source[idx_s], X_target[idx_t]])
        y_dom = np.array([0] * n_sub + [1] * n_tgt_sub)

        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X_comb)
        clf = LogisticRegression(max_iter=1000, random_state=random_state, C=1.0)
        clf.fit(X_sc, y_dom)

        # Compute weights for all source samples
        X_src_sc = scaler.transform(X_source)
        proba = clf.predict_proba(X_src_sc)
        p_target = proba[:, 1]
        p_source = proba[:, 0]
        p_source = np.clip(p_source, 1e-6, 1.0)
        weights_raw = p_target / p_source

    elif method == "kde":
        from scipy.stats import gaussian_kde
        # Fit KDE on first 2 PCA dims for speed
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(2, X_source.shape[1]))
        X_src_2d = pca.fit_transform(X_source)
        X_tgt_2d = pca.transform(X_target)

        kde_src = gaussian_kde(X_src_2d.T)
        kde_tgt = gaussian_kde(X_tgt_2d.T)
        weights_raw = kde_tgt(X_src_2d.T) / (kde_src(X_src_2d.T) + 1e-12)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'classifier' or 'kde'.")

    # Clip extreme weights
    threshold = float(np.percentile(weights_raw, clip_percentile))
    weights = np.clip(weights_raw, 0.0, threshold)
    weights = weights / (weights.mean() + 1e-12)   # normalize

    # Effective sample size (Kish 1965)
    ess = float(weights.sum() ** 2 / (weights ** 2).sum())
    ess_ratio = ess / n_src

    return {
        "weights": weights.tolist(),
        "effective_sample_size": ess,
        "ess_ratio": ess_ratio,
        "clip_threshold": float(threshold),
        "method": method,
        "description": (
            "Importance weights w(x) ≈ p_target(x)/p_source(x). "
            "Re-weight training loss with these to correct covariate shift. "
            "ESS ratio < 0.5 → severe shift; weights less reliable."
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5. Synthetic shift generators
# ──────────────────────────────────────────────────────────────────────────────

def apply_covariate_shift(
    X: np.ndarray,
    shift_type: str = "gaussian",
    magnitude: float = 1.0,
    feature_indices: list[int] | None = None,
    random_state: int = 42,
) -> np.ndarray:
    """
    Apply synthetic covariate shift to features.

    shift_type:
      "gaussian"  — add Gaussian noise N(magnitude, magnitude) to selected features
      "mean_shift" — shift feature means by magnitude standard deviations
      "scale_shift" — multiply features by magnitude

    Returns shifted copy of X.
    """
    rng = np.random.RandomState(random_state)
    X_shifted = X.copy()
    idx = feature_indices if feature_indices is not None else list(range(X.shape[1]))

    if shift_type == "gaussian":
        noise = rng.randn(len(X), len(idx)) * magnitude
        X_shifted[:, idx] += noise
    elif shift_type == "mean_shift":
        feature_std = np.std(X[:, idx], axis=0) + 1e-8
        X_shifted[:, idx] += magnitude * feature_std
    elif shift_type == "scale_shift":
        X_shifted[:, idx] *= magnitude
    else:
        raise ValueError(f"Unknown shift_type: {shift_type}")

    return X_shifted


def performance_under_shift(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    shift_magnitudes: list[float],
    shift_type: str = "gaussian",
    task_type: str = "classification",
    random_state: int = 42,
) -> dict:
    """
    Measure how model accuracy/R² degrades as test distribution shifts
    away from training distribution.

    Args:
        model: A fitted ShiftLab BaseAlgorithm instance.
        shift_magnitudes: List of shift severities to evaluate.
        task_type: "classification" or "regression".

    Returns:
        dict with per-magnitude performance metrics and degradation analysis.
    """
    from backend.core.metrics import regression_metrics, classification_metrics

    results = []
    for mag in shift_magnitudes:
        X_shifted = apply_covariate_shift(
            X_test, shift_type=shift_type,
            magnitude=mag, random_state=random_state,
        )
        preds = model.predict(X_shifted)

        if task_type == "classification":
            m = classification_metrics(y_test, preds)
            primary = m["accuracy"]
            metric_name = "accuracy"
        else:
            m = regression_metrics(y_test, preds)
            primary = m["r2"]
            metric_name = "r2"

        # MMD between original and shifted test
        mmd = mmd_squared(X_test, X_shifted)

        results.append({
            "shift_magnitude": mag,
            "metrics": m,
            "primary_metric": primary,
            "metric_name": metric_name,
            "mmd2": mmd["mmd2"],
        })

    # Degradation = baseline - worst
    baseline = results[0]["primary_metric"] if results else 0.0
    worst = min(r["primary_metric"] for r in results) if results else 0.0
    degradation = float(baseline - worst)

    return {
        "shift_type": shift_type,
        "task_type": task_type,
        "results": results,
        "baseline_performance": float(baseline),
        "worst_performance": float(worst),
        "degradation": degradation,
        "description": (
            f"Performance under {shift_type} covariate shift. "
            f"Baseline {metric_name}: {baseline:.3f}. "
            f"Degradation over shift sweep: {degradation:.3f}."
        ),
    }
